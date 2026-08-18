// =============================================================================
// gqa_kv_broadcast.v
// Holds the current KV-head's tile (K and V slices for one tile_scheduler
// burst) and fans it out combinationally to GROUP_SIZE parallel Q-head
// compute paths. K/V is computed/streamed once per KV-head and reused by
// all GROUP_SIZE query heads that share it -- no per-Q-head duplication.
// =============================================================================
`timescale 1ns / 1ps

module gqa_kv_broadcast #(
    parameter GROUP_SIZE  = 4,
    parameter TILE_DIM    = 16,
    parameter DATA_WIDTH  = 16
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // Load side: tile_scheduler pushes one new K/V tile slice per cycle
    input  wire                              load_valid,
    input  wire [TILE_DIM*DATA_WIDTH-1:0]    k_slice_in,
    input  wire [TILE_DIM*DATA_WIDTH-1:0]    v_slice_in,
    input  wire                              load_pass_first,
    input  wire                              load_pass_last,

    // Broadcast side: GROUP_SIZE consumers read the same registered slice.
    // consumer_req[i] pulses when Q-head i's compute path is ready to accept
    // this cycle's slice (all GROUP_SIZE heads consume the same KV data in
    // lockstep, since they share the same KV-head).
    input  wire [GROUP_SIZE-1:0]             consumer_ready,

    output reg  [TILE_DIM*DATA_WIDTH-1:0]    k_slice_out,
    output reg  [TILE_DIM*DATA_WIDTH-1:0]    v_slice_out,
    output reg                               slice_pass_first,
    output reg                               slice_pass_last,
    output reg  [GROUP_SIZE-1:0]             consumer_valid
);

    // Broadcast only advances once every consumer in the group has accepted
    // the current slice (lockstep fan-out), preventing a slow Q-head lane
    // from causing others to silently skip data.
    wire all_consumers_done = &consumer_ready | ~(|consumer_valid);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            k_slice_out      <= {(TILE_DIM*DATA_WIDTH){1'b0}};
            v_slice_out      <= {(TILE_DIM*DATA_WIDTH){1'b0}};
            slice_pass_first <= 1'b0;
            slice_pass_last  <= 1'b0;
            consumer_valid   <= {GROUP_SIZE{1'b0}};
        end else if (load_valid && all_consumers_done) begin
            k_slice_out      <= k_slice_in;
            v_slice_out      <= v_slice_in;
            slice_pass_first <= load_pass_first;
            slice_pass_last  <= load_pass_last;
            consumer_valid   <= {GROUP_SIZE{1'b1}};
        end else if (all_consumers_done) begin
            consumer_valid   <= {GROUP_SIZE{1'b0}};
        end
    end

endmodule
