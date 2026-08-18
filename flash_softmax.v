// =============================================================================
// flash_softmax.v
// True flash-attention streaming softmax, per query row:
//   - maintains running max m_i and running sum l_i across K/V tiles
//   - on each new tile: computes local max of the tile's scores, forms
//     new running max m_new = max(m_old, local_max)
//   - computes rescale factor alpha = exp(m_old - m_new) to correct the
//     previously-accumulated l_i and the output_accumulator's partial V-sum
//   - adds the new tile's contributions: exp(score_j - m_new) for each score
//   - never materializes a full N x N score matrix -- operates one tile of
//     scores at a time, one row at a time.
//
// Reuses bf16_exp (extended with mantissa bits + linear interpolation per
// the handoff notes) and reciprocal_unit for final normalization at EOF.
// =============================================================================
`timescale 1ns / 1ps

module flash_softmax #(
    parameter DATA_WIDTH  = 16,
    parameter KV_BLOCK_LEN    = 16   // scores per tile (one per K position in tile)
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // New tile of raw (pre-softmax, already-scaled) attention scores for
    // the current query row, KV_BLOCK_LEN scores wide, one tile per cycle group.
    input  wire                              tile_valid,
    input  wire                              row_first_tile,   // first tile for this query row (reset m_i/l_i)
    input  wire                              row_last_tile,     // last tile for this row (trigger final normalize)
    input  wire [KV_BLOCK_LEN*DATA_WIDTH-1:0]    scores_in,
    input  wire [KV_BLOCK_LEN-1:0]               score_valid_mask,

    // Per-score exp(score - m_new) output, to be multiplied by V and fed
    // into output_accumulator alongside 'rescale_alpha'.
    output wire                              probs_valid,
    output wire [KV_BLOCK_LEN*DATA_WIDTH-1:0]    probs_out,
    output wire                              probs_row_first,
    output wire                              probs_row_last,

    // Correction factor for output_accumulator's previous partial sum,
    // valid one cycle (aligned) with probs_out for the same tile.
    output wire                              rescale_valid,
    output wire [DATA_WIDTH-1:0]             rescale_alpha,

    // Final normalization control (1/l_i), asserted after row_last_tile
    // has been fully processed; output_accumulator multiplies its final
    // accumulated sum by inv_l to get the true softmax-weighted output.
    output wire                              inv_l_valid,
    output wire [DATA_WIDTH-1:0]             inv_l
);

    genvar g;

    // -----------------------------------------------------------------
    // Declared here (moved up from further down the file) because the
    // PROB_LANES generate block below references both PROB_LATENCY and
    // mask_pipe. Verilog module-scope declarations are visible regardless
    // of textual order, so this was not a functional bug -- only a
    // "used before declared" lint warning -- but declaring it before use
    // is clearer and avoids the warning entirely.
    // -----------------------------------------------------------------
    // Softmax registers its subtractor operands to isolate the high-fanout
    // running maximum from the exponent-alignment logic.  The five-cycle
    // subtractor plus registered exp give a six-cycle probability latency.
    localparam PROB_LATENCY = 6;
    reg [KV_BLOCK_LEN-1:0] mask_pipe [0:PROB_LATENCY-1];
`ifdef DEBUG
  // Debug waveform dump for flash_softmax internals
  initial begin
    $dumpfile("flash_softmax_debug.vcd");
    $dumpvars(0, flash_softmax);
    $dumpvars(0, m_i);
    $dumpvars(0, l_i);
    $dumpvars(0, rescale_alpha);
    $dumpvars(0, probs_out);
    $dumpvars(0, probs_valid);
    $dumpvars(0, inv_l);
  end
`endif

    // -----------------------------------------------------------------
    // Stages A0-A3: pipelined local-max reduction.
    //
    // Do not turn this into a procedural accumulator.  The old form was a
    // 15-comparator dependency chain for KV_BLOCK_LEN=16, followed directly
    // by the first subtractor stage.  It produced an 82-level/66 ns path at
    // 100 MHz.  This is a balanced tree with one compare level per cycle.
    // Four physical stages are retained for every supported power-of-two
    // block length (4/8/16), so the score vector and its tags have a fixed,
    // simple alignment contract downstream.
    // -----------------------------------------------------------------
    function [DATA_WIDTH-1:0] bf16_max;
        input [DATA_WIDTH-1:0] x, y;
        begin
            // BF16 magnitude-then-sign compare: for same-sign values the raw
            // bit pattern (excl. sign) already compares like an unsigned int
            // for positive numbers; for a fully general compare a dedicated
            // bf16_compare primitive should be substituted here.
            if (x[DATA_WIDTH-1] != y[DATA_WIDTH-1])
                bf16_max = x[DATA_WIDTH-1] ? y : x; // negative sign bit = smaller
            else if (x[DATA_WIDTH-1] == 1'b0)
                bf16_max = (x[DATA_WIDTH-2:0] > y[DATA_WIDTH-2:0]) ? x : y;
            else
                bf16_max = (x[DATA_WIDTH-2:0] < y[DATA_WIDTH-2:0]) ? x : y;
        end
    endfunction

    localparam [DATA_WIDTH-1:0] NEG_INF = 16'hFF80;
    reg [DATA_WIDTH-1:0] max_l1 [0:KV_BLOCK_LEN-1];
    reg [DATA_WIDTH-1:0] max_l2 [0:KV_BLOCK_LEN-1];
    reg [DATA_WIDTH-1:0] max_l3 [0:KV_BLOCK_LEN-1];
    reg [DATA_WIDTH-1:0] max_l4 [0:KV_BLOCK_LEN-1];
    reg max_l1_valid, max_l2_valid, max_l3_valid, max_l4_valid;
    reg [KV_BLOCK_LEN*DATA_WIDTH-1:0] scores_pipe1, scores_pipe2;
    reg [KV_BLOCK_LEN*DATA_WIDTH-1:0] scores_pipe3, scores_pipe4;
    reg [KV_BLOCK_LEN-1:0] mask_pipe1, mask_pipe2, mask_pipe3, mask_pipe4;
    reg first_pipe1, first_pipe2, first_pipe3, first_pipe4;
    reg last_pipe1,  last_pipe2,  last_pipe3,  last_pipe4;
    integer ti;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_l1_valid <= 1'b0;
            max_l2_valid <= 1'b0;
            max_l3_valid <= 1'b0;
            max_l4_valid <= 1'b0;
            scores_pipe1 <= 0; scores_pipe2 <= 0;
            scores_pipe3 <= 0; scores_pipe4 <= 0;
            mask_pipe1 <= 0; mask_pipe2 <= 0; mask_pipe3 <= 0; mask_pipe4 <= 0;
            first_pipe1 <= 1'b0; first_pipe2 <= 1'b0;
            first_pipe3 <= 1'b0; first_pipe4 <= 1'b0;
            last_pipe1 <= 1'b0; last_pipe2 <= 1'b0;
            last_pipe3 <= 1'b0; last_pipe4 <= 1'b0;
            for (ti = 0; ti < KV_BLOCK_LEN; ti = ti + 1) begin
                max_l1[ti] <= NEG_INF; max_l2[ti] <= NEG_INF;
                max_l3[ti] <= NEG_INF; max_l4[ti] <= NEG_INF;
            end
        end else begin
            // Level 1: KV_BLOCK_LEN -> KV_BLOCK_LEN/2.
            max_l1_valid <= tile_valid;
            if (tile_valid) begin
                for (ti = 0; ti < KV_BLOCK_LEN/2; ti = ti + 1)
                    max_l1[ti] <= bf16_max(
                        score_valid_mask[2*ti] ? scores_in[(2*ti+1)*DATA_WIDTH-1 -: DATA_WIDTH] : NEG_INF,
                        score_valid_mask[2*ti+1] ? scores_in[(2*ti+2)*DATA_WIDTH-1 -: DATA_WIDTH] : NEG_INF);
            end
            scores_pipe1 <= scores_in; mask_pipe1 <= score_valid_mask;
            first_pipe1 <= row_first_tile; last_pipe1 <= row_last_tile;

            // Level 2: ... -> KV_BLOCK_LEN/4.
            max_l2_valid <= max_l1_valid;
            if (max_l1_valid) begin
                for (ti = 0; ti < KV_BLOCK_LEN/4; ti = ti + 1)
                    max_l2[ti] <= bf16_max(max_l1[2*ti], max_l1[2*ti+1]);
            end
            scores_pipe2 <= scores_pipe1; mask_pipe2 <= mask_pipe1;
            first_pipe2 <= first_pipe1; last_pipe2 <= last_pipe1;

            // Level 3 is a reduction for 8/16-wide blocks and a pass stage
            // for the 4-wide regression configuration.
            max_l3_valid <= max_l2_valid;
            if (max_l2_valid) begin
                if (KV_BLOCK_LEN <= 4)
                    max_l3[0] <= max_l2[0];
                else begin
                    for (ti = 0; ti < KV_BLOCK_LEN/8; ti = ti + 1)
                        max_l3[ti] <= bf16_max(max_l2[2*ti], max_l2[2*ti+1]);
                end
            end
            scores_pipe3 <= scores_pipe2; mask_pipe3 <= mask_pipe2;
            first_pipe3 <= first_pipe2; last_pipe3 <= last_pipe2;

            // Level 4 completes a 16-wide tree; smaller configurations pass
            // their completed maximum through this alignment stage.
            max_l4_valid <= max_l3_valid;
            if (max_l3_valid) begin
                if (KV_BLOCK_LEN <= 8)
                    max_l4[0] <= max_l3[0];
                else begin
                    for (ti = 0; ti < KV_BLOCK_LEN/16; ti = ti + 1)
                        max_l4[ti] <= bf16_max(max_l3[2*ti], max_l3[2*ti+1]);
                end
            end
            scores_pipe4 <= scores_pipe3; mask_pipe4 <= mask_pipe3;
            first_pipe4 <= first_pipe3; last_pipe4 <= last_pipe3;
        end
    end

    // -----------------------------------------------------------------
    // Stage B: register m_new before it fans out to all score subtractors.
    // m_old is captured with it, so alpha starts in the same cycle as the
    // subtractors and retains the original rescale/probability alignment.
    // -----------------------------------------------------------------
    reg [DATA_WIDTH-1:0] m_i, l_i;
    // Fan-out isolation for the running max, feeding both u_m_diff_sub (1
    // load) and all KV_BLOCK_LEN score subtractors (16 loads at default
    // sizing).
    //
    // FIX (timing): this used to be a single (* keep = "true" *) register.
    // "keep" is a synthesis-only directive -- opt_design/phys_opt_design
    // (implementation-stage optimization) is free to ignore it and merge
    // m_new_reg into m_i anyway, since both are loaded with the identical
    // expression m_new_next on the identical condition. That merge is
    // exactly what happened: the timing report showed the critical path
    // sourced from "m_i_reg[7]", not "m_new_reg[7]", meaning the isolation
    // register the comment describes was gone by implementation, and the
    // resulting single physical FF was carrying BOTH the m_i feedback load
    // AND the full 16-lane subtractor fanout -- the dominant (67%) ROUTE
    // component of the reported 12.111 ns path.
    //
    // Two changes fix this without touching any latency/tag arithmetic
    // (bit-exact same values, same cycle, same handshake -- simulation is
    // unaffected):
    //   1. dont_touch (not keep) -- this DOES persist through
    //      implementation optimization, so the register can no longer be
    //      silently merged away.
    //   2. Replicate m_new_reg into MNEW_COPIES physically-independent
    //      registers, all loaded with the same value, so no single FF has
    //      to drive all 16 subtractor instances (+ the alpha subtractor)
    //      at once. This directly targets the route-delay component.
    localparam MNEW_GROUP_SIZE = 1; // was 4 -- still left ~33 fanout per
                                     // copy (report: High Fanout=33 on
                                     // m_new_reg[2][8]); 1 = dedicated
                                     // register per PROB_LANES instance
    localparam MNEW_COPIES     = (KV_BLOCK_LEN + MNEW_GROUP_SIZE - 1) / MNEW_GROUP_SIZE + 1; // +1 = dedicated copy for u_m_diff_sub
    localparam MNEW_ALPHA_IDX  = MNEW_COPIES - 1;

    (* dont_touch = "true" *) reg [DATA_WIDTH-1:0] m_new_reg [0:MNEW_COPIES-1];
    (* dont_touch = "true" *) reg [DATA_WIDTH-1:0] m_old_reg;
    reg softmax_tile_valid, softmax_row_first, softmax_row_last;
    reg [KV_BLOCK_LEN*DATA_WIDTH-1:0] softmax_scores;
    reg [KV_BLOCK_LEN-1:0] softmax_score_mask;
    wire [DATA_WIDTH-1:0] m_new_next = first_pipe4 ? max_l4[0] : bf16_max(m_i, max_l4[0]);
    integer mn;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_i <= 0; m_old_reg <= 0;
            for (mn = 0; mn < MNEW_COPIES; mn = mn + 1) m_new_reg[mn] <= 0;
            softmax_tile_valid <= 1'b0;
            softmax_row_first <= 1'b0; softmax_row_last <= 1'b0;
            softmax_scores <= 0; softmax_score_mask <= 0;
        end else begin
            softmax_tile_valid <= max_l4_valid;
            if (max_l4_valid) begin
                m_old_reg <= m_i;
                for (mn = 0; mn < MNEW_COPIES; mn = mn + 1) m_new_reg[mn] <= m_new_next;
                m_i <= m_new_next;
                softmax_row_first <= first_pipe4;
                softmax_row_last <= last_pipe4;
                softmax_scores <= scores_pipe4;
                softmax_score_mask <= mask_pipe4;
            end
        end
    end

    // -----------------------------------------------------------------
    // Stage C: rescale factor alpha = exp(m_old - m_new). On the first
    // tile of a row there is no "old" accumulator to rescale, so alpha
    // should be treated as 1.0 by output_accumulator (gated by row_first_tile).
    // -----------------------------------------------------------------
    wire [DATA_WIDTH-1:0] m_diff;
    wire                  m_diff_valid;

    bf16_subtractor #(.INPUT_PIPE_STAGE(1)) u_m_diff_sub (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (softmax_tile_valid & ~softmax_row_first),
        .a         (m_old_reg),
        .b         (m_new_reg[MNEW_ALPHA_IDX]),
        .valid_out (m_diff_valid),
        .result    (m_diff)
    );

    wire alpha_valid;
    wire [DATA_WIDTH-1:0] alpha_val;
    bf16_exp u_alpha_exp (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (m_diff_valid),
        .x         (m_diff),
        .valid_out (alpha_valid),
        .y         (alpha_val)
    );

    assign rescale_valid = alpha_valid;
    assign rescale_alpha = alpha_val;

    // -----------------------------------------------------------------
    // Stage D: per-score exp(score_j - m_new), KV_BLOCK_LEN lanes in parallel
    // -----------------------------------------------------------------
    wire [DATA_WIDTH-1:0] score_diff  [0:KV_BLOCK_LEN-1];
    wire                  diff_valid  [0:KV_BLOCK_LEN-1];
    wire [DATA_WIDTH-1:0] exp_result  [0:KV_BLOCK_LEN-1];
    wire [DATA_WIDTH-1:0] masked_exp_result [0:KV_BLOCK_LEN-1];
    wire                  exp_valid   [0:KV_BLOCK_LEN-1];

    generate
        for (g = 0; g < KV_BLOCK_LEN; g = g + 1) begin : PROB_LANES
            bf16_subtractor #(.INPUT_PIPE_STAGE(1)) u_score_sub (
                .clk       (clk),
                .rst_n     (rst_n),
                .valid_in  (softmax_tile_valid),
                .a         (softmax_scores[(g+1)*DATA_WIDTH-1 -: DATA_WIDTH]),
                .b         (m_new_reg[g/MNEW_GROUP_SIZE]),
                .valid_out (diff_valid[g]),
                .result    (score_diff[g])
            );

            bf16_exp u_score_exp (
                .clk       (clk),
                .rst_n     (rst_n),
                .valid_in  (diff_valid[g]),
                .x         (score_diff[g]),
                .valid_out (exp_valid[g]),
                .y         (exp_result[g])
            );

            // Mask after exp rather than relying on IEEE -inf behavior in
            // the approximate BF16 exp LUT. This guarantees masked K/V
            // positions make no contribution to l_i or O.  The masked
            // value is also used by the running-sum chain below; masking
            // only probs_out biases the final normalization denominator.
            assign masked_exp_result[g] = mask_pipe[PROB_LATENCY-1][g] ? exp_result[g] : 16'h0000;
            assign probs_out[(g+1)*DATA_WIDTH-1 -: DATA_WIDTH] = masked_exp_result[g];
        end
    endgenerate

    assign probs_valid = exp_valid[0]; // all lanes share identical pipeline depth

    // bf16_subtractor's isolated five-cycle core and bf16_exp's registered
    // output produce a six-cycle visible probability latency.  The tag pipeline
    // below must stay exactly matched to that pulse; changing BF16
    // primitive latency requires updating this value and the associated
    // tag/data timing together.
    // Carry the row tags alongside the probability vector rather than using
    // the raw tile-time control pulse at the accumulator many cycles later.
    reg [PROB_LATENCY-1:0] probs_first_pipe, probs_last_pipe;
    integer mi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            probs_first_pipe <= 0;
            probs_last_pipe  <= 0;
            for (mi = 0; mi < PROB_LATENCY; mi = mi + 1) mask_pipe[mi] <= 0;
        end else begin
            probs_first_pipe <= {probs_first_pipe[PROB_LATENCY-2:0], softmax_tile_valid & softmax_row_first};
            probs_last_pipe  <= {probs_last_pipe[PROB_LATENCY-2:0],  softmax_tile_valid & softmax_row_last};
            mask_pipe[0] <= softmax_score_mask;
            for (mi = 1; mi < PROB_LATENCY; mi = mi + 1) mask_pipe[mi] <= mask_pipe[mi-1];
        end
    end
    assign probs_row_first = probs_first_pipe[PROB_LATENCY-1];
    assign probs_row_last  = probs_last_pipe[PROB_LATENCY-1];

    // -----------------------------------------------------------------
    // Stage E: running sum update l_i = l_i*alpha + sum(exp_result)
    // Tile-local sum via balanced binary adder tree, then combine with rescaled l_i.
    // -----------------------------------------------------------------
    localparam SUM_LEVELS  = (KV_BLOCK_LEN <= 1) ? 0 : $clog2(KV_BLOCK_LEN);
    localparam ADD_LATENCY = 6; // bf16_adder with SPLIT_FINAL_STAGE=1, SPLIT_ARITH_STAGE=1

    function automatic integer sum_level_width(input integer lv2);
        integer w, l;
        begin
            w = KV_BLOCK_LEN;
            for (l = 0; l < lv2; l = l + 1)
                w = (w + 1) / 2;
            sum_level_width = w;
        end
    endfunction

    wire [DATA_WIDTH-1:0] level_data  [0:SUM_LEVELS][0:KV_BLOCK_LEN-1];
    wire                  level_valid [0:SUM_LEVELS][0:KV_BLOCK_LEN-1];

    for (g = 0; g < KV_BLOCK_LEN; g = g + 1) begin : LEVEL0_TAP
        assign level_data[0][g]  = masked_exp_result[g];
        assign level_valid[0][g] = exp_valid[g];
    end

    genvar lvl, p;
    generate
        for (lvl = 0; lvl < SUM_LEVELS; lvl = lvl + 1) begin : TREE_LEVEL
            localparam IN_CNT  = sum_level_width(lvl);
            localparam PAIRS   = IN_CNT / 2;
            localparam HAS_ODD = IN_CNT[0];
            for (p = 0; p < PAIRS; p = p + 1) begin : PAIR_ADD
                bf16_adder #(
                    .SPLIT_FINAL_STAGE(1),
                    .SPLIT_ARITH_STAGE(1)
                ) u_sum_tree_add (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .valid_in  (level_valid[lvl][2*p]),
                    .a         (level_data[lvl][2*p]),
                    .b         (level_data[lvl][2*p+1]),
                    .valid_out (level_valid[lvl+1][p]),
                    .result    (level_data[lvl+1][p])
                );
            end
            if (HAS_ODD) begin : ODD_PASSTHROUGH
                reg [DATA_WIDTH-1:0] odd_data_sr  [0:ADD_LATENCY-1];
                reg                  odd_valid_sr [0:ADD_LATENCY-1];
                integer di;
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (di = 0; di < ADD_LATENCY; di = di + 1) begin
                            odd_data_sr[di]  <= {DATA_WIDTH{1'b0}};
                            odd_valid_sr[di] <= 1'b0;
                        end
                    end else begin
                        odd_data_sr[0]  <= level_data[lvl][IN_CNT-1];
                        odd_valid_sr[0] <= level_valid[lvl][IN_CNT-1];
                        for (di = 1; di < ADD_LATENCY; di = di + 1) begin
                            odd_data_sr[di]  <= odd_data_sr[di-1];
                            odd_valid_sr[di] <= odd_valid_sr[di-1];
                        end
                    end
                end
                assign level_data[lvl+1][PAIRS]  = odd_data_sr[ADD_LATENCY-1];
                assign level_valid[lvl+1][PAIRS] = odd_valid_sr[ADD_LATENCY-1];
            end
        end
    endgenerate

    wire [DATA_WIDTH-1:0] tile_sum       = level_data[SUM_LEVELS][0];
    wire                  tile_sum_valid = level_valid[SUM_LEVELS][0];

    // l_old * alpha (skip multiply on first tile of row -- l_old is 0)
    // NOTE: requires a bf16_multiplier instance; per handoff notes this
    // primitive exists as bf16_multiplier.v with a 2-stage pipeline.
    wire [DATA_WIDTH-1:0] l_old_scaled;
    wire                  l_old_scaled_valid;
    bf16_multiplier u_l_scale_mult (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (alpha_valid),
        .a         (l_i),
        .b         (alpha_val),
        .valid_out (l_old_scaled_valid),
        .result    (l_old_scaled)
    );

    localparam TILE_SUM_LATENCY = PROB_LATENCY + SUM_LEVELS * ADD_LATENCY;
    reg [TILE_SUM_LATENCY-1:0] row_first_pipe;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) row_first_pipe <= 0;
        else row_first_pipe <= {row_first_pipe[TILE_SUM_LATENCY-2:0], softmax_tile_valid & softmax_row_first};
    end
    wire row_first_for_l = row_first_pipe[TILE_SUM_LATENCY-1];

    // alpha's path is much shorter than the exponent-sum tree.
    // Keep l_old*alpha until its matching tile_sum arrives instead of
    // assuming a fixed implementation-dependent latency difference.
    reg [DATA_WIDTH-1:0] l_old_scaled_saved;
    reg                  l_old_scaled_pending;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l_old_scaled_saved   <= {DATA_WIDTH{1'b0}};
            l_old_scaled_pending <= 1'b0;
        end else begin
            if (l_old_scaled_valid) begin
                l_old_scaled_saved   <= l_old_scaled;
                l_old_scaled_pending <= 1'b1;
            end
            if (tile_sum_valid && !row_first_for_l && l_old_scaled_pending)
                l_old_scaled_pending <= 1'b0;
        end
    end
    wire [DATA_WIDTH-1:0] l_combine_a = row_first_for_l ? {DATA_WIDTH{1'b0}} : l_old_scaled_saved;
    wire l_combine_valid = row_first_for_l ? tile_sum_valid :
                           (tile_sum_valid & l_old_scaled_pending);

    wire [DATA_WIDTH-1:0] l_new;
    wire                  l_new_valid;
    bf16_adder #(.SPLIT_FINAL_STAGE(1), .SPLIT_ARITH_STAGE(1)) u_l_update_add (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (l_combine_valid),
        .a         (l_combine_a),
        .b         (tile_sum),
        .valid_out (l_new_valid),
        .result    (l_new)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            l_i <= {DATA_WIDTH{1'b0}};
        else if (l_new_valid)
            l_i <= l_new;
    end

    // -----------------------------------------------------------------
    // Stage F: final reciprocal 1/l_i, triggered after row_last_tile's
    // l_new_valid pulse.
    // -----------------------------------------------------------------
    reg row_last_pending;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) row_last_pending <= 1'b0;
        else if (softmax_tile_valid && softmax_row_last) row_last_pending <= 1'b1;
        else if (l_new_valid && row_last_pending) row_last_pending <= 1'b0;
    end

    wire final_l_valid = l_new_valid & row_last_pending;

    reciprocal_unit u_reciprocal (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (final_l_valid),
        .sum_in    (l_new),
        .valid_out (inv_l_valid),
        .inv_sum   (inv_l)
    );

endmodule
