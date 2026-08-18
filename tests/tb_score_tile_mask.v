`timescale 1ns / 1ps

// Checks that metadata travelling with delayed scores preserves causal and
// padding masks at the flash-attention block boundary.
module tb_score_tile_mask;
    reg clk = 0, rst_n = 0, score_valid = 0;
    reg key_block_first = 0, key_row_first = 0, key_row_last = 0, key_valid = 0;
    reg [15:0] score_in = 0;
    wire block_valid, row_first_tile, row_last_tile;
    wire [63:0] block_out;
    wire [3:0] block_valid_mask;

    score_tile_buffer #(.KV_BLOCK_LEN(4)) dut (
        .clk(clk), .rst_n(rst_n), .score_valid(score_valid),
        .key_block_first(key_block_first), .key_row_first(key_row_first), .key_row_last(key_row_last),
        .key_valid(key_valid), .score_in(score_in), .block_valid(block_valid),
        .row_first_tile(row_first_tile), .row_last_tile(row_last_tile),
        .block_out(block_out), .block_valid_mask(block_valid_mask)
    );
    always #5 clk = ~clk;
    task send_score(input [15:0] value, input is_valid, input is_first, input is_last);
        begin
            @(negedge clk);
            score_in = value; key_valid = is_valid; key_block_first = is_first;
            key_row_first = is_first; key_row_last = is_last; score_valid = 1;
            @(negedge clk);
            score_valid = 0; key_block_first = 0; key_row_first = 0; key_row_last = 0;
        end
    endtask
    initial begin
        repeat (2) @(posedge clk); rst_n = 1;
        send_score(16'h3f80, 1, 1, 0);
        send_score(16'h4000, 1, 0, 0);
        send_score(16'h4040, 0, 0, 0); // causal/padding masked
        send_score(16'h4080, 0, 0, 1); // causal/padding masked
        #1;
        if (!block_valid || !row_first_tile || !row_last_tile || block_valid_mask !== 4'b0011)
            $fatal(1, "mask/tag assembly failed");
        if (block_out !== 64'h4080_4040_4000_3f80) $fatal(1, "score packing failed");
        $display("PASS: score tile causal/padding mask");
        $finish;
    end
endmodule
