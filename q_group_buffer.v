`timescale 1ns / 1ps
// Buffers the four query heads that share a KV head.  Each MAC pass exposes
// the same dimension slice from all four rows, so the lanes no longer reuse
// one Q vector accidentally.
module q_group_buffer #(
    parameter GROUP_SIZE = 4, HEAD_DIM = 128, TILE_DIM = 16,
    parameter DATA_WIDTH = 16, N_PASSES = HEAD_DIM / TILE_DIM,
    parameter PASS_BITS = (N_PASSES <= 1) ? 1 : $clog2(N_PASSES)
) (
    input wire clk, input wire rst_n,
    input wire q_wr_valid,
    input wire [HEAD_DIM*DATA_WIDTH-1:0] q_wr_data,
    output wire q_wr_ready,
    input wire q_req, output reg q_ack, input wire q_release,
    input wire [PASS_BITS-1:0] pass_idx,
    output wire [GROUP_SIZE*TILE_DIM*DATA_WIDTH-1:0] q_slices
);
    localparam FILL_BITS = (GROUP_SIZE <= 1) ? 1 : $clog2(GROUP_SIZE);
    reg [HEAD_DIM*DATA_WIDTH-1:0] q_mem [0:GROUP_SIZE-1];
    reg [FILL_BITS-1:0] fill_idx;
    reg group_ready, group_active;
    integer i;
    assign q_wr_ready = !group_ready && !group_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fill_idx <= 0; group_ready <= 0; group_active <= 0; q_ack <= 0;
            for (i = 0; i < GROUP_SIZE; i = i + 1) q_mem[i] <= 0;
        end else begin
            q_ack <= 1'b0;
            if (q_wr_valid && q_wr_ready) begin
                q_mem[fill_idx] <= q_wr_data;
                if (fill_idx == GROUP_SIZE-1) begin
                    fill_idx <= 0;
                    group_ready <= 1'b1;
                end else fill_idx <= fill_idx + 1'b1;
            end
            if (q_req && group_ready && !group_active) begin
                group_active <= 1'b1;
                q_ack <= 1'b1;
            end
            if (q_release) begin
                fill_idx <= 0;
                group_ready <= 1'b0;
                group_active <= 1'b0;
            end
        end
    end

    genvar g;
    generate for (g = 0; g < GROUP_SIZE; g = g + 1) begin : GEN_SLICE
        assign q_slices[(g+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH] =
            q_mem[g][(pass_idx+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH];
    end endgenerate
endmodule
