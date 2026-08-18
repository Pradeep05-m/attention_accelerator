`timescale 1ns / 1ps

module tb_flash_softmax_simple;
    parameter DATA_WIDTH = 16;
    parameter KV_BLOCK_LEN = 4;

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    reg tile_valid;
    reg row_first_tile;
    reg row_last_tile;
    reg [KV_BLOCK_LEN*DATA_WIDTH-1:0] scores_in;
    reg [KV_BLOCK_LEN-1:0] score_valid_mask;

    wire probs_valid;
    wire [KV_BLOCK_LEN*DATA_WIDTH-1:0] probs_out;
    wire probs_row_first, probs_row_last;
    wire rescale_valid;
    wire [DATA_WIDTH-1:0] rescale_alpha;
    wire inv_l_valid;
    wire [DATA_WIDTH-1:0] inv_l;

    flash_softmax #(
        .DATA_WIDTH(DATA_WIDTH),
        .KV_BLOCK_LEN(KV_BLOCK_LEN)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .tile_valid(tile_valid),
        .row_first_tile(row_first_tile),
        .row_last_tile(row_last_tile),
        .scores_in(scores_in),
        .score_valid_mask(score_valid_mask),
        .probs_valid(probs_valid),
        .probs_out(probs_out),
        .probs_row_first(probs_row_first),
        .probs_row_last(probs_row_last),
        .rescale_valid(rescale_valid),
        .rescale_alpha(rescale_alpha),
        .inv_l_valid(inv_l_valid),
        .inv_l(inv_l)
    );

    function [15:0] real2bf16;
        input real r;
        reg [31:0] bits;
        begin
            bits = $realtobits(r);
            real2bf16 = {bits[31], bits[30:23], bits[22:16]};
        end
    endfunction

    real test_val0, test_val1, test_val2, test_val3;
    integer i;

    always @(posedge clk) begin
        if (probs_valid) begin
            $display("[%0t] PROBS_VALID: lane0=%h lane1=%h lane2=%h lane3=%h",
                $time, probs_out[16-1:0], probs_out[32-1:16], probs_out[48-1:32], probs_out[64-1:48]);
        end
        if (inv_l_valid) begin
            $display("[%0t] INV_L_VALID: inv_l=%h", $time, inv_l);
        end
    end

    initial begin
        test_val0 = 0.5;   test_val1 = -0.3; test_val2 = 1.2; test_val3 = 0.0;
        scores_in[1*16-1 -: 16] = real2bf16(test_val0);
        scores_in[2*16-1 -: 16] = real2bf16(test_val1);
        scores_in[3*16-1 -: 16] = real2bf16(test_val2);
        scores_in[4*16-1 -: 16] = real2bf16(test_val3);
        score_valid_mask = 4'hf;

        tile_valid = 0; row_first_tile = 0; row_last_tile = 0;
        rst_n = 0; #20; rst_n = 1; #20;
        
        @(posedge clk);
        tile_valid = 1; row_first_tile = 1; row_last_tile = 1; 
        @(posedge clk);
        tile_valid = 0; row_first_tile = 0; row_last_tile = 0;

        #300;
        $finish;
    end
endmodule
