// =============================================================================
// v_row_assembler.v
// gqa_kv_broadcast delivers V in the same TILE_DIM-wide, N_PASSES-chunk
// fashion as K (since both stream alongside the MAC's HEAD_DIM reduction
// passes). Before V can be buffered per K-position for output_accumulator,
// the N_PASSES chunks belonging to one K position must be re-assembled
// into a single contiguous HEAD_DIM-wide row. This module does that
// assembly; its output feeds v_tile_buffer.v.
// =============================================================================
`timescale 1ns / 1ps

module v_row_assembler #(
    parameter HEAD_DIM   = 128,
    parameter TILE_DIM   = 16,
    parameter DATA_WIDTH = 16,
    parameter N_PASSES   = HEAD_DIM / TILE_DIM
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                              chunk_valid,
    input  wire                              pass_first,   // pass 0 of this K position's V row
    input  wire                              pass_last,    // final pass -> row_out is complete
    input  wire [TILE_DIM*DATA_WIDTH-1:0]    chunk_in,

    output reg                               row_valid,
    output reg [HEAD_DIM*DATA_WIDTH-1:0]     row_out
);

    reg [$clog2(N_PASSES):0] pass_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pass_cnt <= {($clog2(N_PASSES)+1){1'b0}};
            row_valid <= 1'b0;
            row_out   <= {(HEAD_DIM*DATA_WIDTH){1'b0}};
        end else begin
            row_valid <= 1'b0;
            if (chunk_valid) begin
                row_out[(pass_cnt+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH] <= chunk_in;
                if (pass_last) begin
                    row_valid <= 1'b1;
                    pass_cnt  <= {($clog2(N_PASSES)+1){1'b0}};
                end else begin
                    pass_cnt <= pass_first ? {{($clog2(N_PASSES)){1'b0}}, 1'b1} : (pass_cnt + 1'b1);
                end
            end
        end
    end

endmodule
