// =============================================================================
// output_accumulator.v  (REV 2 -- time-multiplexed)
//
// Running weighted-V accumulator per query row (per HEAD_DIM output element).
// On each tile: acc[k] = acc[k]*rescale_alpha + sum_j( prob_j * V_j[k] )
// At row_last_tile, once flash_softmax's inv_l is valid, applies the final
// normalization: out[k] = acc[k] * inv_l.
//
// REV 2: the previous revision fully unrolled BOTH the ACC_WIDTH (normally
// HEAD_DIM=128) output dimension AND the KV_BLOCK_LEN reduction dimension
// into parallel hardware -- ACC_WIDTH*KV_BLOCK_LEN=2048 real bf16_multiplier
// instances plus ~1920 bf16_adder instances for the P.V reduction alone, per
// lane. Times GROUP_SIZE=4 lanes in the wrapper, that's ~17,400 primitive
// instances from this one module -- impractical to elaborate/simulate and
// far beyond any realistic Zynq-7000 target's LUT budget.
//
// This revision time-multiplexes the ACC_WIDTH dimension into TILE_DIM-wide
// chunks processed over N_PASSES=ACC_WIDTH/TILE_DIM back-to-back cycles,
// reusing one shared bank of TILE_DIM lanes -- the same pattern
// bf16_dot_product_mac.v already uses for the Q.K reduction (TILE_DIM lanes
// reused across N_PASSES=HEAD_DIM/TILE_DIM passes). This cuts instance count
// ~N_PASSES-fold (8x for the default 128/16 sizing): ~17,400 -> ~2,176
// total, at the cost of additional per-tile latency (pipelined back-to-back
// per stage, not stop-and-wait, so materially less than N_PASSES x
// pipeline-depth).
//
// Internally split into two back-to-back stages to keep each individually
// simple/auditable rather than fusing rescale+reduce+accumulate into one
// pass-tagged pipeline:
//   STAGE 1: for each of N_PASSES chunks, compute tile_contrib[k] via the
//            shared TILE_DIM-lane multiply + balanced-tree reduction, tag
//            results by (delayed) chunk index into contrib_buf[ACC_WIDTH].
//   STAGE 2: for each of N_PASSES chunks, rescale the OLD acc[] window
//            (discarded/zeroed on row_first) and add contrib_buf's window,
//            writing back to acc[]. Also handles the (separately-triggered,
//            once-per-row) final normalize into out_row.
//
// `busy` stays asserted for the full STAGE 1 + STAGE 2 duration of a tile;
// an upstream scheduler must not present a new tile_valid while busy is high
// (matches gqa_controller.v's existing wait-for-drain philosophy already
// used for bf16_dot_product_mac.v's own busy signal).
// =============================================================================
`timescale 1ns / 1ps

module output_accumulator #(
    parameter DATA_WIDTH   = 16,
    parameter KV_BLOCK_LEN = 16,  // number of K/V rows (probs) contributing per tile
    parameter ACC_WIDTH    = 128, // number of output (HEAD_DIM) elements this instance holds
    parameter TILE_DIM     = 16   // shared-lane width; must evenly divide ACC_WIDTH
) (
    input  wire                                         clk,
    input  wire                                         rst_n,

    input  wire                                         tile_valid,
    input  wire                                         row_first_tile,
    input  wire                                         row_last_tile,
    input  wire [KV_BLOCK_LEN*DATA_WIDTH-1:0]           probs_in,      // exp(score - m_new), from flash_softmax
    input  wire [KV_BLOCK_LEN*TILE_DIM*DATA_WIDTH-1:0]  v_slice_in,    // V[j][k] for the current output pass

    input  wire                                         rescale_valid,
    input  wire [DATA_WIDTH-1:0]                        rescale_alpha, // 1.0 implied when row_first_tile

    input  wire                                         inv_l_valid,
    input  wire [DATA_WIDTH-1:0]                        inv_l,

    output wire                                         out_valid,
    output wire [ACC_WIDTH*DATA_WIDTH-1:0]              out_row,       // final normalized output row
    output wire [$clog2(ACC_WIDTH/TILE_DIM)-1:0]        v_pass_idx,

    // Pulses when stage 1 has consumed every V slice for this tile.  This
    // permits the upstream single V-block buffer to be safely reused.
    output wire                                         tile_consumed,

    // High from tile_valid until this tile's stage 1 + stage 2 processing
    // has fully completed. The upstream scheduler must hold this tile's
    // data stable and must not present a new tile_valid while busy is high.
    output wire                                         busy
);
    localparam N_PASSES     = ACC_WIDTH / TILE_DIM;
    localparam PASS_BITS    = (N_PASSES <= 1) ? 1 : $clog2(N_PASSES);
    localparam MULT_LATENCY = 2; // bf16_multiplier pipeline depth
    localparam ADD_LATENCY  = 5; // bf16_adder with split arithmetic stage
    localparam SUM_LEVELS   = (KV_BLOCK_LEN <= 1) ? 0 : $clog2(KV_BLOCK_LEN);
    // Only the first reduction level is timing-isolated (see u_pv_sum),
    // adding one clock to this tree but avoiding the congestion caused by
    // globally deepening every BF16 adder in the design.
    // The contribution result is deliberately tagged at its completion
    // point below.  Do not couple this to a fixed arithmetic latency: the
    // P*V tree has optional timing stages and its depth varies with
    // KV_BLOCK_LEN.
    localparam CONTRIB_TAG_SKEW = 2;
    localparam ACCUM_LATENCY  = MULT_LATENCY + ADD_LATENCY;             // pass2 issue -> acc write valid

    genvar t, jx, lvl, p, jv, pv, lv;
    integer ii;

    function automatic integer sum_level_width(input integer lv2);
        integer w, l;
        begin
            w = KV_BLOCK_LEN;
            for (l = 0; l < lv2; l = l + 1)
                w = (w + 1) / 2; // ceil(w/2)
            sum_level_width = w;
        end
    endfunction

    // -----------------------------------------------------------------
    // Persistent per-output-element accumulator, banked by lane.  Each
    // lane owns N_PASSES elements and has exactly one write process below.
    // Keeping all lanes in one ACC_WIDTH-deep array made the generated lane
    // writers look like TILE_DIM write ports on a single RAM to Vivado.
    // -----------------------------------------------------------------
    // These banks are only N_PASSES deep (8 entries at the default
    // geometry), so registers are both cheaper to control and avoid asking
    // Vivado to map the independent lane accesses onto a BRAM primitive.
    (* ram_style = "registers" *) reg [DATA_WIDTH-1:0] acc [0:TILE_DIM-1][0:N_PASSES-1];

    // -----------------------------------------------------------------
    // Tile latch + stage 1/2 sequencer
    // -----------------------------------------------------------------
    reg [KV_BLOCK_LEN*DATA_WIDTH-1:0] probs_r;
    reg row_first_r, row_last_r;
    reg [DATA_WIDTH-1:0] rescale_alpha_r;

    reg                  stage1_busy;
    reg [PASS_BITS-1:0]  pass_cnt;
    reg                  stage2_busy;
    reg [PASS_BITS-1:0]  pass_cnt2;

    // v_tile_buffer uses BRAM and therefore returns the requested V window
    // one clock later.  Delay the issue tag/valid by one cycle so the
    // multiplier bank sees the matching window rather than the previous
    // pass's data.
    reg                 v_read_valid_d;
    reg [PASS_BITS-1:0] v_read_tag_d;
    reg                 v_read_last_d;

    // `v_slice_in` is driven by the registered output of v_tile_buffer's
    // BRAMs.  Register it once more before the multipliers: on XC7Z020 the
    // direct BRAM-to-multiplier path exceeded 10 ns by more than 2 ns.
    reg                 v_mult_valid_d;
    reg [PASS_BITS-1:0] v_mult_tag_d;
    reg                 v_mult_last_d;
    reg [TILE_DIM*DATA_WIDTH-1:0] v_slice_r [0:KV_BLOCK_LEN-1];
    // Unpack the selected V windows before the capture process below.  The
    // declaration must precede its use for Vivado OOC synthesis; otherwise
    // it creates an implicit net and emits Synth 8-6901.
    wire [TILE_DIM*DATA_WIDTH-1:0] v_slice [0:KV_BLOCK_LEN-1];
    generate
        for (jv = 0; jv < KV_BLOCK_LEN; jv = jv + 1) begin : V2D_J
            assign v_slice[jv] = v_slice_in[(jv+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH];
        end
    endgenerate
    integer v_capture_i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_busy <= 1'b0;
            pass_cnt    <= {PASS_BITS{1'b0}};
        end else begin
            if (tile_valid && !stage1_busy && !stage2_busy) begin
                stage1_busy     <= 1'b1;
                pass_cnt        <= {PASS_BITS{1'b0}};
                probs_r         <= probs_in;
                row_first_r     <= row_first_tile;
                row_last_r      <= row_last_tile;
                // `tile_valid` and `rescale_valid` describe the same
                // non-first softmax tile.  Latching a separate registered
                // copy here used the *previous* alpha because nonblocking
                // assignments update after this capture edge.  That scales
                // the prior partial V sum with a stale tile's max correction
                // whenever a row spans more than one KV block.
                //
                // A first tile never has rescale_valid and stage 2 discards
                // the old accumulator for row_first_r, so its don't-care
                // alpha is explicitly set to 1.0 for deterministic behavior.
                rescale_alpha_r <= row_first_tile ? 16'h3F80 : rescale_alpha;
            end else if (stage1_busy) begin
                if (pass_cnt == N_PASSES-1)
                    stage1_busy <= 1'b0;
                else
                    pass_cnt <= pass_cnt + 1'b1;
            end
        end
    end
    wire pass1_fire      = stage1_busy;
    wire pass1_last_fire = stage1_busy && (pass_cnt == N_PASSES-1);
    // v_tile_buffer has a registered word and sub-word select.  Together
    // with the accumulator's V capture register, its visible data is two
    // pass requests ahead of the issue tag unless the read address is
    // compensated here.  Without this correction a simple one-key smoke
    // row is rotated by two elements (out[d] receives V[d+2]).
    //
    // N_PASSES is a power of two for the supported HEAD_DIM/TILE_DIM
    // geometries, so this fixed-width subtraction naturally wraps the
    // initial prefetch requests to the final two passes.
    assign v_pass_idx    = pass_cnt - 2'd2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_read_valid_d <= 1'b0;
            v_read_tag_d   <= {PASS_BITS{1'b0}};
            v_read_last_d  <= 1'b0;
            v_mult_valid_d <= 1'b0;
            v_mult_tag_d   <= {PASS_BITS{1'b0}};
            v_mult_last_d  <= 1'b0;
            // v_slice_r is deliberately not reset.  v_mult_valid_d is reset
            // and is the only enable that can consume this data, so clearing
            // a 256-bit pipeline payload gains nothing.  Leaving payload
            // flops reset-less also reduces reset fanout/control-set packing
            // pressure on the resource-constrained XC7Z020.
        end else begin
            v_read_valid_d <= pass1_fire;
            v_read_tag_d   <= pass_cnt;
            v_read_last_d  <= pass1_last_fire;
            v_mult_valid_d <= v_read_valid_d;
            v_mult_tag_d   <= v_read_tag_d;
            v_mult_last_d  <= v_read_last_d;
            if (v_read_valid_d) begin
                for (v_capture_i = 0; v_capture_i < KV_BLOCK_LEN; v_capture_i = v_capture_i + 1)
                    v_slice_r[v_capture_i] <= v_slice[v_capture_i];
            end
        end
    end

    // -----------------------------------------------------------------
    // STAGE 1: shared TILE_DIM-lane bank. For the CURRENT pass_cnt's
    // TILE_DIM-wide output chunk, compute
    //   tile_contrib[lane] = sum_j( probs_r[j] * V[j][chunk*TILE_DIM+lane] )
    // via a balanced binary adder tree per lane (same tree structure as the
    // single-pass version, just TILE_DIM instances instead of ACC_WIDTH).
    // Passes are issued back-to-back (no cross-pass state hazard: each pass
    // touches an independent, non-overlapping set of lanes/tree instances),
    // tagged by a pass_cnt shift register so results land in the correct
    // ACC_WIDTH-wide slice of contrib_buf.
    // -----------------------------------------------------------------
    wire [DATA_WIDTH-1:0] mult_result [0:TILE_DIM-1][0:KV_BLOCK_LEN-1];
    wire                  mult_valid  [0:TILE_DIM-1][0:KV_BLOCK_LEN-1];
    wire [DATA_WIDTH-1:0] tile_contrib_lane       [0:TILE_DIM-1];
    wire                  tile_contrib_lane_valid [0:TILE_DIM-1];
    wire [DATA_WIDTH-1:0] tile_contrib_aligned       [0:TILE_DIM-1];
    wire                  tile_contrib_aligned_valid [0:TILE_DIM-1];

    generate
        for (t = 0; t < TILE_DIM; t = t + 1) begin : LANE
            for (jx = 0; jx < KV_BLOCK_LEN; jx = jx + 1) begin : PVMULT
                bf16_multiplier u_pv_mult (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .valid_in  (v_mult_valid_d),
                    .a         (probs_r[(jx+1)*DATA_WIDTH-1 -: DATA_WIDTH]),
                    .b         (v_slice_r[jx][(t+1)*DATA_WIDTH-1 -: DATA_WIDTH]),
                    .valid_out (mult_valid[t][jx]),
                    .result    (mult_result[t][jx])
                );
            end

            wire [DATA_WIDTH-1:0] level_data  [0:SUM_LEVELS][0:KV_BLOCK_LEN-1];
            wire                  level_valid [0:SUM_LEVELS][0:KV_BLOCK_LEN-1];
            for (jx = 0; jx < KV_BLOCK_LEN; jx = jx + 1) begin : LEVEL0_TAP
                assign level_data[0][jx]  = mult_result[t][jx];
                assign level_valid[0][jx] = mult_valid[t][jx];
            end
            for (lvl = 0; lvl < SUM_LEVELS; lvl = lvl + 1) begin : TREE_LEVEL
                localparam IN_CNT  = sum_level_width(lvl);
                localparam PAIRS   = IN_CNT / 2;
                localparam HAS_ODD = IN_CNT[0];
                for (p = 0; p < PAIRS; p = p + 1) begin : PAIR_ADD
                // The routed critical path reported by implementation is
                // this level-0 P*V reduction adder.  Add a local output
                // register only here; later tree levels retain the compact
                // four-cycle primitive and no unrelated datapath grows.
                bf16_adder #(
                    .EXTRA_PIPE_STAGE(lvl == 0),
                    .SPLIT_ARITH_STAGE(1)
                ) u_pv_sum (
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
                localparam LEVEL_ADD_LATENCY = (lvl == 0) ? ADD_LATENCY + 1 : ADD_LATENCY;
                reg [DATA_WIDTH-1:0] odd_data_sr  [0:LEVEL_ADD_LATENCY-1];
                reg                  odd_valid_sr [0:LEVEL_ADD_LATENCY-1];
                    integer di;
                    always @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            for (di = 0; di < LEVEL_ADD_LATENCY; di = di + 1) begin
                                odd_data_sr[di]  <= {DATA_WIDTH{1'b0}};
                                odd_valid_sr[di] <= 1'b0;
                            end
                        end else begin
                            odd_data_sr[0]  <= level_data[lvl][IN_CNT-1];
                            odd_valid_sr[0] <= level_valid[lvl][IN_CNT-1];
                            for (di = 1; di < LEVEL_ADD_LATENCY; di = di + 1) begin
                                odd_data_sr[di]  <= odd_data_sr[di-1];
                                odd_valid_sr[di] <= odd_valid_sr[di-1];
                            end
                        end
                    end
                    assign level_data[lvl+1][PAIRS]  = odd_data_sr[LEVEL_ADD_LATENCY-1];
                    assign level_valid[lvl+1][PAIRS] = odd_valid_sr[LEVEL_ADD_LATENCY-1];
                end
            end

            assign tile_contrib_lane[t]       = level_data[SUM_LEVELS][0];
            assign tile_contrib_lane_valid[t] = level_valid[SUM_LEVELS][0];

            reg [DATA_WIDTH-1:0] contrib_data_skid [0:CONTRIB_TAG_SKEW-1];
            reg                  contrib_valid_skid[0:CONTRIB_TAG_SKEW-1];
            integer skid_i;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (skid_i = 0; skid_i < CONTRIB_TAG_SKEW; skid_i = skid_i + 1) begin
                        contrib_data_skid[skid_i]  <= {DATA_WIDTH{1'b0}};
                        contrib_valid_skid[skid_i] <= 1'b0;
                    end
                end else begin
                    contrib_data_skid[0]  <= tile_contrib_lane[t];
                    contrib_valid_skid[0] <= tile_contrib_lane_valid[t];
                    for (skid_i = 1; skid_i < CONTRIB_TAG_SKEW; skid_i = skid_i + 1) begin
                        contrib_data_skid[skid_i]  <= contrib_data_skid[skid_i-1];
                        contrib_valid_skid[skid_i] <= contrib_valid_skid[skid_i-1];
                    end
                end
            end
            assign tile_contrib_aligned[t]       = contrib_data_skid[CONTRIB_TAG_SKEW-1];
            assign tile_contrib_aligned_valid[t] = contrib_valid_skid[CONTRIB_TAG_SKEW-1];
        end
    endgenerate

    // Keep issue tags in order and retire one only when its reduction
    // payload actually emerges.  This makes pass-to-output-slice mapping
    // independent of primitive latencies and of future timing pipelining.
    localparam CONTRIB_COUNT_BITS = (N_PASSES <= 1) ? 1 : $clog2(N_PASSES + 1);
    reg [PASS_BITS-1:0] contrib_tag_fifo [0:N_PASSES-1];
    reg [PASS_BITS-1:0] contrib_tag_wr, contrib_tag_rd;
    reg [CONTRIB_COUNT_BITS-1:0] contrib_tag_count;
    wire contrib_payload_valid = tile_contrib_aligned_valid[0];
    wire contrib_tag_valid = contrib_payload_valid && (contrib_tag_count != 0);
    wire [PASS_BITS-1:0] contrib_tag = contrib_tag_fifo[contrib_tag_rd];
    wire stage1_done = contrib_tag_valid && (contrib_tag == N_PASSES-1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            contrib_tag_wr    <= {PASS_BITS{1'b0}};
            contrib_tag_rd    <= {PASS_BITS{1'b0}};
            contrib_tag_count <= {CONTRIB_COUNT_BITS{1'b0}};
        end else begin
            case ({v_mult_valid_d, contrib_tag_valid})
                2'b10: contrib_tag_count <= contrib_tag_count + 1'b1;
                2'b01: contrib_tag_count <= contrib_tag_count - 1'b1;
                default: contrib_tag_count <= contrib_tag_count;
            endcase
            if (v_mult_valid_d) begin
                contrib_tag_fifo[contrib_tag_wr] <= v_mult_tag_d;
                contrib_tag_wr <= (contrib_tag_wr == N_PASSES-1) ?
                                  {PASS_BITS{1'b0}} : contrib_tag_wr + 1'b1;
            end
            if (contrib_tag_valid)
                contrib_tag_rd <= (contrib_tag_rd == N_PASSES-1) ?
                                  {PASS_BITS{1'b0}} : contrib_tag_rd + 1'b1;
        end
    end
    assign tile_consumed = stage1_done;

    // One contribution RAM/register bank per lane.  A stage-1 lane writes
    // its own bank while the matching stage-2 lane reads it; no bank has
    // multiple generated writers.
    (* ram_style = "registers" *) reg [DATA_WIDTH-1:0] contrib_buf [0:TILE_DIM-1][0:N_PASSES-1];
    generate
        for (t = 0; t < TILE_DIM; t = t + 1) begin : CONTRIB_WRITE
            integer pp;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (pp = 0; pp < N_PASSES; pp = pp + 1)
                        contrib_buf[t][pp] <= {DATA_WIDTH{1'b0}};
                end else if (contrib_tag_valid && tile_contrib_aligned_valid[t]) begin
                    contrib_buf[t][contrib_tag] <= tile_contrib_aligned[t];
                end
            end
        end
    endgenerate

    // -----------------------------------------------------------------
    // STAGE 2: rescale (acc window * alpha, discarded on row_first) + add
    // contrib_buf's window, write back to acc[]. Starts the cycle stage 1
    // fully drains for this tile. Always pays the full rescale-then-add
    // latency (uniform per-pass timing) even on row_first passes; the
    // rescale result is simply muxed to zero rather than skipped, trading
    // a redundant multiply for simpler, single-latency tagging.
    // -----------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage2_busy <= 1'b0;
            pass_cnt2   <= {PASS_BITS{1'b0}};
        end else begin
            if (stage1_done && !stage2_busy) begin
                stage2_busy <= 1'b1;
                pass_cnt2   <= {PASS_BITS{1'b0}};
            end else if (stage2_busy) begin
                if (pass_cnt2 == N_PASSES-1)
                    stage2_busy <= 1'b0;
                else
                    pass_cnt2 <= pass_cnt2 + 1'b1;
            end
        end
    end
    wire pass2_fire      = stage2_busy;
    wire pass2_last_fire = stage2_busy && (pass_cnt2 == N_PASSES-1);

    assign busy = tile_valid || stage1_busy || stage2_busy;

    wire [DATA_WIDTH-1:0] acc_window     [0:TILE_DIM-1];
    wire [DATA_WIDTH-1:0] contrib_window [0:TILE_DIM-1];
    generate
        for (t = 0; t < TILE_DIM; t = t + 1) begin : WINDOW_TAP
            assign acc_window[t]     = acc[t][pass_cnt2];
            assign contrib_window[t] = contrib_buf[t][pass_cnt2];
        end
    endgenerate

    // Register the variable-indexed accumulator/contribution reads before
    // the BF16 rescale multiplier.  Without this boundary pass_cnt2 drives
    // a wide mux and then the multiplier's exponent logic in one 100-MHz
    // cycle (the former -2.947 ns critical path).  This adds one cycle of
    // latency but keeps the datapath and tag/valid streams aligned.
    reg                  stage2_issue_valid;
    reg                  stage2_issue_row_first;
    reg [PASS_BITS-1:0]  stage2_issue_tag;
    reg                  stage2_issue_last;
    reg [DATA_WIDTH-1:0] stage2_acc_r     [0:TILE_DIM-1];
    reg [DATA_WIDTH-1:0] stage2_contrib_r [0:TILE_DIM-1];
    reg [DATA_WIDTH-1:0] stage2_alpha_r;
    integer stage2_capture_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage2_issue_valid     <= 1'b0;
            stage2_issue_row_first <= 1'b0;
            stage2_issue_tag       <= {PASS_BITS{1'b0}};
            stage2_issue_last      <= 1'b0;
            stage2_alpha_r         <= {DATA_WIDTH{1'b0}};
            for (stage2_capture_i = 0; stage2_capture_i < TILE_DIM; stage2_capture_i = stage2_capture_i + 1) begin
                stage2_acc_r[stage2_capture_i]     <= {DATA_WIDTH{1'b0}};
                stage2_contrib_r[stage2_capture_i] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            stage2_issue_valid     <= pass2_fire;
            stage2_issue_row_first <= row_first_r & pass2_fire;
            stage2_issue_tag       <= pass_cnt2;
            stage2_issue_last      <= pass2_last_fire;
            if (pass2_fire) begin
                stage2_alpha_r <= rescale_alpha_r;
                for (stage2_capture_i = 0; stage2_capture_i < TILE_DIM; stage2_capture_i = stage2_capture_i + 1) begin
                    stage2_acc_r[stage2_capture_i]     <= acc_window[stage2_capture_i];
                    stage2_contrib_r[stage2_capture_i] <= contrib_window[stage2_capture_i];
                end
            end
        end
    end

    // Align row_first_r with when acc_scaled becomes valid (MULT_LATENCY
    // cycles after the registered stage-2 issue), so we know whether to
    // keep or discard it.
    reg [MULT_LATENCY-1:0] row_first_mult_sr;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) row_first_mult_sr <= {MULT_LATENCY{1'b0}};
        else        row_first_mult_sr <= {row_first_mult_sr[MULT_LATENCY-2:0], stage2_issue_row_first};
    end

    // Tag pipeline for stage 2: carries pass_cnt2 + "last pass" through the
    // ACCUM_LATENCY-deep rescale-then-add pipeline.
    reg [PASS_BITS-1:0] accum_tag_sr  [0:ACCUM_LATENCY-1];
    reg                 accum_last_sr [0:ACCUM_LATENCY-1];
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ii = 0; ii < ACCUM_LATENCY; ii = ii + 1) begin
                accum_tag_sr[ii]  <= {PASS_BITS{1'b0}};
                accum_last_sr[ii] <= 1'b0;
            end
        end else begin
            accum_tag_sr[0]  <= stage2_issue_tag;
            accum_last_sr[0] <= stage2_issue_last;
            for (ii = 1; ii < ACCUM_LATENCY; ii = ii + 1) begin
                accum_tag_sr[ii]  <= accum_tag_sr[ii-1];
                accum_last_sr[ii] <= accum_last_sr[ii-1];
            end
        end
    end
    wire [PASS_BITS-1:0] accum_tag = accum_tag_sr[ACCUM_LATENCY-1];

    wire [DATA_WIDTH-1:0] acc_scaled       [0:TILE_DIM-1];
    wire                  acc_scaled_valid [0:TILE_DIM-1];
    wire [DATA_WIDTH-1:0] acc_next         [0:TILE_DIM-1];
    wire                  acc_next_valid   [0:TILE_DIM-1];

    generate
        for (t = 0; t < TILE_DIM; t = t + 1) begin : STAGE2_LANE
            bf16_multiplier u_rescale_mult (
                .clk       (clk),
                .rst_n     (rst_n),
                .valid_in  (stage2_issue_valid),
                .a         (stage2_acc_r[t]),
                .b         (stage2_alpha_r),
                .valid_out (acc_scaled_valid[t]),
                .result    (acc_scaled[t])
            );

            wire [DATA_WIDTH-1:0] add_a = row_first_mult_sr[MULT_LATENCY-1] ? {DATA_WIDTH{1'b0}} : acc_scaled[t];

            bf16_adder #(.SPLIT_ARITH_STAGE(1)) u_acc_add (
                .clk       (clk),
                .rst_n     (rst_n),
                .valid_in  (acc_scaled_valid[t]),
                .a         (add_a),
                .b         (stage2_contrib_r[t]),
                .valid_out (acc_next_valid[t]),
                .result    (acc_next[t])
            );

            // acc[] is deliberately not reset.  A row's first tile selects
            // a zero addend (add_a above), then writes every pass before the
            // value can be read by a later tile.  Resetting a
            // variable-indexed entry only resets one inferred register and
            // makes Vivado infer conflicting set/reset controls.
            always @(posedge clk) begin
                if (acc_next_valid[t])
                    acc[t][accum_tag] <= acc_next[t];
            end
        end
    endgenerate

    wire stage2_done = accum_last_sr[ACCUM_LATENCY-1] & acc_next_valid[0];

    // -----------------------------------------------------------------
    // Final normalize: only after BOTH the final accumulation and the
    // reciprocal have arrived (the reciprocal can precede or follow the
    // final acc[] write by one or more cycles), and only on the row's last
    // tile. Time-multiplexed the same way as stage 1/2.
    // -----------------------------------------------------------------
    reg final_tile_pending, final_acc_committed, inv_pending, norm_start;
    reg [DATA_WIDTH-1:0] inv_saved;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            final_tile_pending <= 1'b0; final_acc_committed <= 1'b0;
            inv_pending <= 1'b0; inv_saved <= {DATA_WIDTH{1'b0}}; norm_start <= 1'b0;
        end else begin
            norm_start <= 1'b0;
            if (row_last_r && stage2_done)
                final_tile_pending <= 1'b1;
            if (final_tile_pending) begin
                final_tile_pending  <= 1'b0;
                final_acc_committed <= 1'b1;
            end
            if (inv_l_valid) begin
                inv_pending <= 1'b1;
                inv_saved   <= inv_l;
            end
            if (inv_pending && final_acc_committed) begin
                norm_start           <= 1'b1;
                inv_pending          <= 1'b0;
                final_acc_committed  <= 1'b0;
            end
        end
    end

    reg                 norm_busy;
    reg [PASS_BITS-1:0] pass_cnt3;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            norm_busy <= 1'b0;
            pass_cnt3 <= {PASS_BITS{1'b0}};
        end else begin
            if (norm_start && !norm_busy) begin
                norm_busy <= 1'b1;
                pass_cnt3 <= {PASS_BITS{1'b0}};
            end else if (norm_busy) begin
                if (pass_cnt3 == N_PASSES-1)
                    norm_busy <= 1'b0;
                else
                    pass_cnt3 <= pass_cnt3 + 1'b1;
            end
        end
    end
    wire pass3_fire      = norm_busy;
    wire pass3_last_fire = norm_busy && (pass_cnt3 == N_PASSES-1);

    // Register the variable-indexed accumulator read before the final BF16
    // multiplier.  This removes the acc[] mux plus multiplier exponent
    // logic from one cycle on the critical normalization path.
    reg                 norm_issue_valid;
    reg [PASS_BITS-1:0] norm_issue_tag;
    reg                 norm_issue_last;
    reg [DATA_WIDTH-1:0] norm_issue_inv;
    reg [DATA_WIDTH-1:0] norm_issue_data [0:TILE_DIM-1];
    integer norm_capture_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            norm_issue_valid <= 1'b0;
            norm_issue_tag   <= {PASS_BITS{1'b0}};
            norm_issue_last  <= 1'b0;
            norm_issue_inv   <= {DATA_WIDTH{1'b0}};
            for (norm_capture_i = 0; norm_capture_i < TILE_DIM; norm_capture_i = norm_capture_i + 1)
                norm_issue_data[norm_capture_i] <= {DATA_WIDTH{1'b0}};
        end else begin
            norm_issue_valid <= pass3_fire;
            norm_issue_tag   <= pass_cnt3;
            norm_issue_last  <= pass3_last_fire;
            if (pass3_fire) begin
                norm_issue_inv <= inv_saved;
                for (norm_capture_i = 0; norm_capture_i < TILE_DIM; norm_capture_i = norm_capture_i + 1)
                    norm_issue_data[norm_capture_i] <= acc[norm_capture_i][pass_cnt3];
            end
        end
    end

    reg [ACC_WIDTH*DATA_WIDTH-1:0] out_row_reg;
    wire [DATA_WIDTH-1:0] out_elem [0:TILE_DIM-1];
    wire                  out_elem_valid [0:TILE_DIM-1];
    generate
        for (t = 0; t < TILE_DIM; t = t + 1) begin : NORMALIZE
            bf16_multiplier u_norm_mult (
                .clk       (clk),
                .rst_n     (rst_n),
                .valid_in  (norm_issue_valid),
                .a         (norm_issue_data[t]),
                .b         (norm_issue_inv),
                .valid_out (out_elem_valid[t]),
                .result    (out_elem[t])
            );
        end
    endgenerate

    // The normalization requests and multiplier results are strictly ordered.
    // Address the completed results with an output-side counter instead of a
    // second, independently delayed copy of the request tag.  The latter
    // could drift at the start/end of a row and overwrite the penultimate
    // element, leaving stale data in the final output slots.
    reg [PASS_BITS-1:0] norm_write_pass;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            norm_write_pass <= {PASS_BITS{1'b0}};
        else if (norm_start)
            norm_write_pass <= {PASS_BITS{1'b0}};
        else if (out_elem_valid[0])
            norm_write_pass <= norm_write_pass + 1'b1;
    end

    // All slices of out_row_reg must be written by one sequential process.
    // With one clocked process per lane and a runtime norm_tag index, each
    // process could legally select any slice.  Synthesis therefore saw
    // multiple drivers for every bit and replaced the computed result with
    // a constant.  A single writer preserves the intended independent lane
    // updates while making the ownership unambiguous.
    integer norm_write_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_row_reg <= {ACC_WIDTH*DATA_WIDTH{1'b0}};
        end else begin
            for (norm_write_i = 0; norm_write_i < TILE_DIM; norm_write_i = norm_write_i + 1) begin
                if (out_elem_valid[norm_write_i])
                    out_row_reg[(norm_write_pass*TILE_DIM+norm_write_i+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= out_elem[norm_write_i];
            end
        end
    end

    // out_valid is delayed one extra cycle past out_elem_valid[0] &&
    // (last pass): the last pass's own out_row_reg write is a non-blocking
    // assignment landing on THIS edge, not visible until next cycle, so
    // asserting out_valid on the same cycle as that write would let a
    // combinational reader sample out_row_reg one cycle too early (missing
    // the final TILE_DIM-wide chunk).
    reg out_valid_r;
    wire stage3_last_valid = out_elem_valid[0] && (norm_write_pass == N_PASSES-1);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) out_valid_r <= 1'b0;
        else        out_valid_r <= stage3_last_valid;
    end

    assign out_row   = out_row_reg;
    assign out_valid = out_valid_r;

endmodule
