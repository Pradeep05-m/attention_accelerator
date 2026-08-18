`timescale 1ns / 1ps

// Regression for causal/padding masking in the online-softmax denominator.
// A one-token row is the edge case that exposed the lane-0 max bug.
module tb_flash_softmax_mask;
    reg clk = 0, rst_n = 0;
    reg tile_valid = 0, row_first_tile = 0, row_last_tile = 0;
    reg [63:0] scores_in = 0;
    reg [3:0] score_valid_mask = 0;
    wire probs_valid, probs_row_first, probs_row_last, rescale_valid, inv_l_valid;
    wire [63:0] probs_out;
    wire [15:0] rescale_alpha, inv_l;
    integer seen;

    flash_softmax #(.KV_BLOCK_LEN(4)) dut (
        .clk(clk), .rst_n(rst_n), .tile_valid(tile_valid),
        .row_first_tile(row_first_tile), .row_last_tile(row_last_tile),
        .scores_in(scores_in), .score_valid_mask(score_valid_mask),
        .probs_valid(probs_valid), .probs_out(probs_out),
        .probs_row_first(probs_row_first), .probs_row_last(probs_row_last),
        .rescale_valid(rescale_valid), .rescale_alpha(rescale_alpha),
        .inv_l_valid(inv_l_valid), .inv_l(inv_l)
    );

    always #5 clk = ~clk;
    always @(posedge clk) begin
        if (probs_valid) begin
            // exp(0) must survive in lane 0; the three masked lanes must be
            // exact zero so neither V accumulation nor l_i can include them.
            if (probs_out[15:0] == 16'h0000 || probs_out[63:16] !== 48'h0)
                $fatal(1, "incorrect masked probabilities: %h", probs_out);
            seen = seen + 1;
        end
    end

    initial begin
        seen = 0;
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        scores_in = {16'h42c8, 16'h4248, 16'h4180, 16'h0000};
        score_valid_mask = 4'b0001;
        tile_valid = 1;
        row_first_tile = 1;
        row_last_tile = 1;
        @(posedge clk);
        tile_valid = 0;
        row_first_tile = 0;
        row_last_tile = 0;
        repeat (100) @(posedge clk);
        if (seen != 1) $fatal(1, "expected one probability vector, got %0d", seen);
        $display("PASS: flash_softmax masking");
        $finish;
    end
endmodule
