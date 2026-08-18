// =============================================================================
// bf16_dot_product_mac.v
// TILE_DIM-wide BF16 MAC lane array performing a genuine reduction dot-product
// over HEAD_DIM elements, accumulating across HEAD_DIM/TILE_DIM passes.
//
// Each lane instantiates one bf16_mac primitive: result = c + a*b.
// Lane i handles elements {i, i+TILE_DIM, i+2*TILE_DIM, ...} of the HEAD_DIM
// vectors (a "vertical strip" of the reduction), then a final adder tree
// combines the TILE_DIM partial sums into the single dot-product scalar.
//
// bf16_mac enables the BF16 adder's split arithmetic stage.  The matching
// MAC_LATENCY value keeps the pass_last tag aligned with the lane result.
// =============================================================================
`timescale 1ns / 1ps

module bf16_dot_product_mac #(
    parameter HEAD_DIM     = 128,
    parameter TILE_DIM     = 16,               // must divide HEAD_DIM
    parameter DATA_WIDTH   = 16,
    parameter MULT_LATENCY = 2,                // bf16_multiplier pipeline stages
    parameter MAC_LATENCY  = MULT_LATENCY + 1 + 5,
    parameter N_PASSES     = HEAD_DIM / TILE_DIM
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // Streaming input: one TILE_DIM-wide slice of Q (a_vec) and K/V (b_vec)
    // per cycle, for N_PASSES consecutive cycles per query-key pair.
    input  wire                              valid_in,
    input  wire                              pass_first,   // asserted on pass 0 (clears accumulator)
    input  wire                              pass_last,    // asserted on final pass (result is final)
    input  wire [TILE_DIM*DATA_WIDTH-1:0]    a_vec,        // Q slice, packed
    input  wire [TILE_DIM*DATA_WIDTH-1:0]    b_vec,        // K/V slice, packed

    output wire                              valid_out,
    output wire                              result_valid_final, // dot product complete
    output wire [DATA_WIDTH-1:0]             result,

    // Backpressure for the controller: pass_ack pulses once THIS pass's
    // per-lane accumulate has actually landed (safe to issue the next
    // pass_first/valid_in); busy stays high from pass_first until
    // result_valid_final (safe to start the NEXT K-position's passes
    // only once busy has gone back low -- this instance is reused
    // sequentially across K-positions and has no internal queuing).
    output wire                              pass_ack,
    output wire                              busy
);

    genvar g;

    // ---------------------------------------------------------------
    // Per-lane running accumulator (fed back as 'c' into bf16_mac).
    // Accumulator register holds each lane's partial sum across passes.
    // ---------------------------------------------------------------
    wire [DATA_WIDTH-1:0] lane_c_in   [0:TILE_DIM-1];
    wire [DATA_WIDTH-1:0] lane_result [0:TILE_DIM-1];
    wire                  lane_valid_out [0:TILE_DIM-1]; // FIX: per-lane array, was a single
                                                          // shared wire driven by TILE_DIM
                                                          // separate bf16_mac instances --
                                                          // a multi-driver conflict.
    reg  [DATA_WIDTH-1:0] lane_acc    [0:TILE_DIM-1];
    reg                   pass_last_d [0:MAC_LATENCY-1]; // shift reg to align pass_last with lane latency

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < MAC_LATENCY; i = i + 1)
                pass_last_d[i] <= 1'b0;
        end else begin
            pass_last_d[0] <= pass_last & valid_in;
            for (i = 1; i < MAC_LATENCY; i = i + 1)
                pass_last_d[i] <= pass_last_d[i-1];
        end
    end

    generate
        for (g = 0; g < TILE_DIM; g = g + 1) begin : LANES
            // c input: 0 on first pass (start fresh dot product for new row),
            // else the running per-lane accumulator from the previous pass.
            assign lane_c_in[g] = pass_first ? {DATA_WIDTH{1'b0}} : lane_acc[g];

            bf16_mac #(
                .MULT_LATENCY(MULT_LATENCY)
            ) u_bf16_mac (
                .clk       (clk),
                .rst_n     (rst_n),
                .valid_in  (valid_in),
                .a         (a_vec[(g+1)*DATA_WIDTH-1 -: DATA_WIDTH]),
                .b         (b_vec[(g+1)*DATA_WIDTH-1 -: DATA_WIDTH]),
                .c         (lane_c_in[g]),
                .valid_out (lane_valid_out[g]),
                .result    (lane_result[g])
            );

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    lane_acc[g] <= {DATA_WIDTH{1'b0}};
                else if (lane_valid_out[g])
                    lane_acc[g] <= lane_result[g];
            end
        end
    endgenerate

    // ---------------------------------------------------------------
    // Adder tree: combine TILE_DIM lane results into one scalar, only
    // meaningful/latched on the final pass. Genuine balanced binary
    // reduction tree (log2(TILE_DIM) levels of bf16_adder, 5-cycle
    // pipeline each), not a linear chain -- a linear chain here would
    // cost (TILE_DIM-1)*ADD_LATENCY cycles (60 for TILE_DIM=16) instead
    // of ceil(log2(TILE_DIM))*ADD_LATENCY (20 for TILE_DIM=16).
    // Handles non-power-of-2 TILE_DIM: any odd leftover at a level is
    // passed through a delay-matched shift register instead of paired.
    // ---------------------------------------------------------------
    localparam ADD_LATENCY = 5;
    localparam NUM_LEVELS  = (TILE_DIM <= 1) ? 0 : $clog2(TILE_DIM);

    function automatic integer level_width(input integer lvl);
        integer w, l;
        begin
            w = TILE_DIM;
            for (l = 0; l < lvl; l = l + 1) begin
                w = (w + 1) / 2; // ceil(w/2)
            end
            level_width = w;
        end
    endfunction

    wire final_pass_valid = pass_last_d[MAC_LATENCY-1];

    wire [DATA_WIDTH-1:0] level_data  [0:NUM_LEVELS][0:TILE_DIM-1];
    wire                  level_valid [0:NUM_LEVELS][0:TILE_DIM-1];

    generate
        for (g = 0; g < TILE_DIM; g = g + 1) begin : LEVEL0_TAP
            assign level_data[0][g]  = lane_result[g];
            assign level_valid[0][g] = final_pass_valid;
        end
    endgenerate

    genvar lvl, p;
    generate
        for (lvl = 0; lvl < NUM_LEVELS; lvl = lvl + 1) begin : TREE_LEVEL
            localparam IN_CNT  = level_width(lvl);
            localparam PAIRS   = IN_CNT / 2;
            localparam HAS_ODD = IN_CNT[0]; // IN_CNT % 2

            for (p = 0; p < PAIRS; p = p + 1) begin : PAIR_ADD
                bf16_adder #(.SPLIT_ARITH_STAGE(1)) u_bf16_adder (
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
                // Odd leftover at this level: delay-match it by ADD_LATENCY
                // cycles (same latency the paired adders take) so it lands
                // at level lvl+1 in lockstep with the paired results.
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

    assign result              = level_data[NUM_LEVELS][0];
    assign result_valid_final  = level_valid[NUM_LEVELS][0];
    assign valid_out            = lane_valid_out[0]; // all lanes share identical, data-independent
                                                       // latency, so any single lane's valid_out is
                                                       // representative of all TILE_DIM lanes.

    // ---------------------------------------------------------------
    // Backpressure tracking for the controller (see port comments above).
    // ---------------------------------------------------------------
    assign pass_ack = lane_valid_out[0];

    reg mac_busy;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_busy <= 1'b0;
        end else begin
            if (valid_in && pass_first) begin
                mac_busy <= 1'b1;
            end else if (result_valid_final) begin
                mac_busy <= 1'b0;
            end
        end
    end
    assign busy = mac_busy;

endmodule
