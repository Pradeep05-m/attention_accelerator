// =============================================================================
// score_tile_buffer.v
// Collects KV_BLOCK_LEN scalar scores -- one produced per K position by
// bf16_dot_product_mac + scale_unit -- into a single KV_BLOCK_LEN-wide
// vector, then asserts block_valid once full so flash_softmax can consume
// a whole flash-attention block at once. This is the piece that was
// missing: bf16_dot_product_mac's own TILE_DIM (MAC lanes across HEAD_DIM)
// is unrelated to KV_BLOCK_LEN (K/V positions per flash-attention block);
// conflating them was the bug in the original score_tile_stub wiring.
//
// Companion module v_tile_buffer.v performs the equivalent job for V rows.
// =============================================================================
`timescale 1ns / 1ps

module score_tile_buffer #(
    parameter KV_BLOCK_LEN = 16,
    parameter DATA_WIDTH   = 16
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                    score_valid,   // one scalar score, valid this cycle
    input  wire                    key_block_first,
    input  wire                    key_row_first,
    input  wire                    key_row_last,
    // 0 for a causal-masked or padding entry.  Invalid entries retain a
    // slot for block alignment but contribute exactly zero to softmax.
    input  wire                    key_valid,
    input  wire [DATA_WIDTH-1:0]   score_in,

    output reg                     block_valid,   // pulses once KV_BLOCK_LEN scores collected
    output wire [KV_BLOCK_LEN*DATA_WIDTH-1:0] block_out,
    output reg                     row_first_tile,
    output reg                     row_last_tile,
    output wire [KV_BLOCK_LEN-1:0] block_valid_mask
);

    reg [$clog2(KV_BLOCK_LEN):0] fill_cnt;
    reg [DATA_WIDTH-1:0] slot [0:KV_BLOCK_LEN-1];
    reg row_first_pending;
    reg valid_slot [0:KV_BLOCK_LEN-1];
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fill_cnt    <= {($clog2(KV_BLOCK_LEN)+1){1'b0}};
            block_valid <= 1'b0;
            row_first_tile <= 1'b0;
            row_last_tile <= 1'b0;
            row_first_pending <= 1'b0;
            for (i = 0; i < KV_BLOCK_LEN; i = i + 1) begin
                slot[i] <= {DATA_WIDTH{1'b0}};
                valid_slot[i] <= 1'b0;
            end
        end else begin
            block_valid <= 1'b0;
            row_first_tile <= 1'b0;
            row_last_tile  <= 1'b0;

            if (score_valid) begin
                // Fill count, rather than a raw controller strobe, defines
                // the block boundary.  Score data arrives after RoPE/MAC/
                // scale latency, so raw block_first is not time-aligned.
                slot[fill_cnt] <= score_in;
                valid_slot[fill_cnt] <= key_valid;
                if (key_block_first)
                    row_first_pending <= key_row_first;
                if (fill_cnt == KV_BLOCK_LEN - 1) begin
                    block_valid <= 1'b1;
                    row_first_tile <= row_first_pending | (key_block_first & key_row_first);
                    row_last_tile <= key_row_last;
                    fill_cnt <= {($clog2(KV_BLOCK_LEN)+1){1'b0}};
                    row_first_pending <= 1'b0;
                end else begin
                    fill_cnt <= fill_cnt + 1'b1;
                end
            end
        end
    end

    generate
        genvar g;
        for (g = 0; g < KV_BLOCK_LEN; g = g + 1) begin : PACK
            assign block_out[(g+1)*DATA_WIDTH-1 -: DATA_WIDTH] = slot[g];
            assign block_valid_mask[g] = valid_slot[g];
        end
    endgenerate

endmodule
