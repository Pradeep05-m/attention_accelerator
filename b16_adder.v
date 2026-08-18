`timescale 1ns / 1ps

module bf16_adder #(
    // Keep the compact four-cycle implementation as the default.  The
    // optional output isolation stage is used only at the known critical
    // output-accumulator reduction level, avoiding a device-wide FF and
    // routing increase on the XC7Z020.
    parameter EXTRA_PIPE_STAGE = 0,
    // Split the former normalize/round/final-select output cloud across two
    // clocks.  Enable this only on serial reductions where the compact
    // output stage is timing-critical; it adds one cycle but does not alter
    // the BF16 result.
    parameter SPLIT_FINAL_STAGE = 0,
    // Register the add/sub result before normalization.  This isolates the
    // carry chain from the leading-zero/shift logic; use only where the
    // parent also delays its associated control tag by one cycle.
    parameter SPLIT_ARITH_STAGE = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg          valid_out,
    output reg  [15:0] result
);

    function [3:0] clz11;
        input [10:0] din;
        integer k;
        begin
            clz11 = 4'd11;
            for (k = 10; k >= 0; k = k - 1) begin
                if (din[k] && (clz11 == 4'd11))
                    clz11 = 10 - k;
            end
        end
    endfunction

    reg         valid_s1;
    reg [15:0]  a_s1, b_s1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1 <= 1'b0;
            a_s1     <= 16'd0;
            b_s1     <= 16'd0;
        end else begin
            valid_s1 <= valid_in;
            a_s1     <= a;
            b_s1     <= b;
        end
    end

    wire        sign_a1 = a_s1[15];
    wire [7:0]  exp_a1  = a_s1[14:7];
    wire [6:0]  frac_a1 = a_s1[6:0];
    wire        sign_b1 = b_s1[15];
    wire [7:0]  exp_b1  = b_s1[14:7];
    wire [6:0]  frac_b1 = b_s1[6:0];
    // BF16 exponent zero represents zero/subnormal, which has no implicit
    // leading one.  Treating 0 as 1.0 here corrupted every accumulator's
    // first operation (0 + product) and biased subsequent reductions.
    // Subnormals are flushed to zero, consistent with the multiplier.
    wire [7:0]  sig_a1  = (exp_a1 == 8'd0) ? 8'd0 : {1'b1, frac_a1};
    wire [7:0]  sig_b1  = (exp_b1 == 8'd0) ? 8'd0 : {1'b1, frac_b1};

    reg         exp_ge1;
    reg [7:0]   exp_big1, exp_small1;
    reg [7:0]   sig_big1, sig_small1;
    reg         sign_big1, sign_small1;
    reg [7:0]   exp_diff1;
    reg [3:0]   shift_amt1;
    reg [10:0]  sig_big_ext1;
    reg [10:0]  sig_small_ext1;
    reg [10:0]  shifted_small1;
    reg         sticky_shift1;
    reg [10:0]  shifted_small_adj1;
    reg         signs_equal1;

    always @* begin
        exp_ge1 = (exp_a1 >= exp_b1);
        if (exp_ge1) begin
            exp_big1   = exp_a1; sig_big1   = sig_a1; sign_big1   = sign_a1;
            exp_small1 = exp_b1; sig_small1 = sig_b1; sign_small1 = sign_b1;
        end else begin
            exp_big1   = exp_b1; sig_big1   = sig_b1; sign_big1   = sign_b1;
            exp_small1 = exp_a1; sig_small1 = sig_a1; sign_small1 = sign_a1;
        end

        exp_diff1  = exp_big1 - exp_small1;
        shift_amt1 = (exp_diff1 > 8'd11) ? 4'd11 : exp_diff1[3:0];

        sig_big_ext1   = {sig_big1,   3'b000};
        sig_small_ext1 = {sig_small1, 3'b000};

        begin : GEN_STICKY_ADD
            reg [21:0] wide_small;
            wide_small = {sig_small_ext1, 11'b0} >> shift_amt1;
            shifted_small1 = wide_small[21:11];
            sticky_shift1  = |wide_small[10:0];
        end

        shifted_small_adj1    = shifted_small1;
        shifted_small_adj1[0] = shifted_small1[0] | sticky_shift1;

        signs_equal1 = (sign_big1 == sign_small1);
    end

    reg         valid_s2;
    reg [10:0]  sig_big_ext_s2;
    reg [10:0]  shifted_small_adj_s2;
    reg         signs_equal_s2;
    reg         sign_big_s2, sign_small_s2;
    reg [7:0]   exp_big_s2;
    reg [15:0]  a_s2, b_s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s2             <= 1'b0;
            sig_big_ext_s2       <= 11'd0;
            shifted_small_adj_s2 <= 11'd0;
            signs_equal_s2       <= 1'b0;
            sign_big_s2          <= 1'b0;
            sign_small_s2        <= 1'b0;
            exp_big_s2           <= 8'd0;
            a_s2                 <= 16'd0;
            b_s2                 <= 16'd0;
        end else begin
            valid_s2             <= valid_s1;
            sig_big_ext_s2       <= sig_big_ext1;
            shifted_small_adj_s2 <= shifted_small_adj1;
            signs_equal_s2       <= signs_equal1;
            sign_big_s2          <= sign_big1;
            sign_small_s2        <= sign_small1;
            exp_big_s2           <= exp_big1;
            a_s2                 <= a_s1;
            b_s2                 <= b_s1;
        end
    end

    reg [11:0]  add_result2;
    reg [11:0]  sub_result2;
    reg [11:0]  mantissa_ext_carry2;
    reg         result_sign_comb2;
    reg [10:0]  mag_result2;
    reg [10:0]  norm_pre_round2;
    reg [3:0]   lzc2;
    reg signed [9:0] exp_calc2;
    reg         is_zero_result2;

    always @* begin
        add_result2 = {1'b0, sig_big_ext_s2} + {1'b0, shifted_small_adj_s2};
        sub_result2 = {1'b0, sig_big_ext_s2} - {1'b0, shifted_small_adj_s2};

        if (signs_equal_s2) begin
            mantissa_ext_carry2 = add_result2;
            result_sign_comb2   = sign_big_s2;
        end else begin
            if (sub_result2[11]) begin
                mantissa_ext_carry2 = (~sub_result2) + 12'd1;
                result_sign_comb2   = sign_small_s2;
            end else begin
                mantissa_ext_carry2 = sub_result2;
                result_sign_comb2   = sign_big_s2;
            end
        end

        if (signs_equal_s2) begin
            if (mantissa_ext_carry2[11]) begin
                norm_pre_round2    = mantissa_ext_carry2[11:1];
                norm_pre_round2[0] = norm_pre_round2[0] | mantissa_ext_carry2[0];
                exp_calc2          = $signed({2'b00, exp_big_s2}) + 10'sd1;
            end else begin
                norm_pre_round2 = mantissa_ext_carry2[10:0];
                exp_calc2       = $signed({2'b00, exp_big_s2});
            end
            is_zero_result2 = 1'b0;
        end else begin
            mag_result2 = mantissa_ext_carry2[10:0];
            if (mag_result2 == 11'd0) begin
                norm_pre_round2 = 11'd0;
                exp_calc2       = 10'sd0;
                is_zero_result2 = 1'b1;
            end else begin
                lzc2            = clz11(mag_result2);
                norm_pre_round2 = mag_result2 << lzc2;
                exp_calc2       = $signed({2'b00, exp_big_s2}) - $signed({6'b0, lzc2});
                is_zero_result2 = 1'b0;
            end
        end
    end

    reg         valid_s3;
    reg [10:0]  norm_pre_round_s3;
    reg signed [9:0] exp_calc_s3;
    reg         result_sign_comb_s3;
    reg         is_zero_result_s3;
    reg [15:0]  a_s3, b_s3;

    generate
        if (SPLIT_ARITH_STAGE) begin : GEN_SPLIT_ARITH_STAGE
            reg         pre_valid;
            reg [11:0]  pre_mantissa;
            reg         pre_signs_equal, pre_result_sign;
            reg [7:0]   pre_exp_big;
            reg [15:0]  pre_a, pre_b;
            reg [10:0]  split_norm_pre_round;
            reg signed [9:0] split_exp_calc;
            reg         split_is_zero_result;
            reg [10:0]  split_mag;
            reg [3:0]   split_lzc;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    pre_valid       <= 1'b0;
                    pre_mantissa    <= 12'd0;
                    pre_signs_equal <= 1'b0;
                    pre_result_sign <= 1'b0;
                    pre_exp_big     <= 8'd0;
                    pre_a           <= 16'd0;
                    pre_b           <= 16'd0;
                end else begin
                    pre_valid       <= valid_s2;
                    pre_mantissa    <= mantissa_ext_carry2;
                    pre_signs_equal <= signs_equal_s2;
                    pre_result_sign <= result_sign_comb2;
                    pre_exp_big     <= exp_big_s2;
                    pre_a           <= a_s2;
                    pre_b           <= b_s2;
                end
            end

            always @* begin
                if (pre_signs_equal) begin
                    if (pre_mantissa[11]) begin
                        split_norm_pre_round    = pre_mantissa[11:1];
                        split_norm_pre_round[0] = split_norm_pre_round[0] | pre_mantissa[0];
                        split_exp_calc          = $signed({2'b00, pre_exp_big}) + 10'sd1;
                    end else begin
                        split_norm_pre_round = pre_mantissa[10:0];
                        split_exp_calc       = $signed({2'b00, pre_exp_big});
                    end
                    split_is_zero_result = 1'b0;
                end else begin
                    split_mag = pre_mantissa[10:0];
                    if (split_mag == 11'd0) begin
                        split_norm_pre_round = 11'd0;
                        split_exp_calc       = 10'sd0;
                        split_is_zero_result = 1'b1;
                    end else begin
                        split_lzc            = clz11(split_mag);
                        split_norm_pre_round = split_mag << split_lzc;
                        split_exp_calc       = $signed({2'b00, pre_exp_big}) - $signed({6'b0, split_lzc});
                        split_is_zero_result = 1'b0;
                    end
                end
            end

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_s3            <= 1'b0;
                    norm_pre_round_s3   <= 11'd0;
                    exp_calc_s3         <= 10'sd0;
                    result_sign_comb_s3 <= 1'b0;
                    is_zero_result_s3   <= 1'b0;
                    a_s3                <= 16'd0;
                    b_s3                <= 16'd0;
                end else begin
                    valid_s3            <= pre_valid;
                    norm_pre_round_s3   <= split_norm_pre_round;
                    exp_calc_s3         <= split_exp_calc;
                    result_sign_comb_s3 <= pre_result_sign;
                    is_zero_result_s3   <= split_is_zero_result;
                    a_s3                <= pre_a;
                    b_s3                <= pre_b;
                end
            end
        end else begin : GEN_COMPACT_ARITH_STAGE
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_s3            <= 1'b0;
                    norm_pre_round_s3   <= 11'd0;
                    exp_calc_s3         <= 10'sd0;
                    result_sign_comb_s3 <= 1'b0;
                    is_zero_result_s3   <= 1'b0;
                    a_s3                <= 16'd0;
                    b_s3                <= 16'd0;
                end else begin
                    valid_s3            <= valid_s2;
                    norm_pre_round_s3   <= norm_pre_round2;
                    exp_calc_s3         <= exp_calc2;
                    result_sign_comb_s3 <= result_sign_comb2;
                    is_zero_result_s3   <= is_zero_result2;
                    a_s3                <= a_s2;
                    b_s3                <= b_s2;
                end
            end
        end
    endgenerate

    wire        sign_a3 = a_s3[15];
    wire [7:0]  exp_a3  = a_s3[14:7];
    wire [6:0]  frac_a3 = a_s3[6:0];
    wire        sign_b3 = b_s3[15];
    wire [7:0]  exp_b3  = b_s3[14:7];
    wire [6:0]  frac_b3 = b_s3[6:0];
    wire        is_zero_a3 = (exp_a3 == 8'd0);
    wire        is_zero_b3 = (exp_b3 == 8'd0);

    reg         guard_bit3, round_bit3, sticky_bit3, lsb_bit3, round_up3;
    reg [7:0]   kept8_3;
    reg [8:0]   rounded9_3;
    reg signed [9:0] exp_final3;
    reg [6:0]   final_frac3;
    reg [7:0]   final_exp3;
    reg         final_sign3;
    reg         is_overflow3;
    reg [15:0]  comb_result3;

    always @* begin
        guard_bit3  = norm_pre_round_s3[2];
        round_bit3  = norm_pre_round_s3[1];
        sticky_bit3 = norm_pre_round_s3[0];
        lsb_bit3    = norm_pre_round_s3[3];
        round_up3   = guard_bit3 & (round_bit3 | sticky_bit3 | lsb_bit3);

        kept8_3    = norm_pre_round_s3[10:3];
        rounded9_3 = {1'b0, kept8_3} + {8'd0, round_up3};

        if (rounded9_3[8]) begin
            final_frac3 = 7'd0;
            exp_final3  = exp_calc_s3 + 10'sd1;
        end else begin
            final_frac3 = rounded9_3[6:0];
            exp_final3  = exp_calc_s3;
        end

        is_overflow3 = (exp_final3 >= 10'sd255);

        if (is_zero_a3 && is_zero_b3) begin
            final_exp3  = 8'd0;
            final_frac3 = 7'd0;
            final_sign3 = sign_a3 & sign_b3;
        end else if (is_zero_a3 && !is_zero_b3) begin
            final_exp3  = exp_b3;
            final_frac3 = frac_b3;
            final_sign3 = sign_b3;
        end else if (!is_zero_a3 && is_zero_b3) begin
            final_exp3  = exp_a3;
            final_frac3 = frac_a3;
            final_sign3 = sign_a3;
        end else if (is_zero_result_s3) begin
            final_exp3  = 8'd0;
            final_frac3 = 7'd0;
            final_sign3 = 1'b0;
        end else if (is_overflow3) begin
            final_exp3  = 8'hFF;
            final_frac3 = 7'd0;
            final_sign3 = result_sign_comb_s3;
        end else if (exp_final3 <= 10'sd0) begin
            final_exp3  = 8'd0;
            final_frac3 = 7'd0;
            final_sign3 = 1'b0;
        end else begin
            final_exp3  = exp_final3[7:0];
            final_sign3 = result_sign_comb_s3;
        end

        comb_result3 = {final_sign3, final_exp3, final_frac3};
    end

    generate
        if (SPLIT_FINAL_STAGE) begin : GEN_SPLIT_FINAL_STAGE
            // The reported failing path starts at norm_pre_round_s3 and ends
            // at result.  Register the round inputs first, then perform the
            // special-case/output selection in the following cycle.  Merely
            // registering comb_result3 would not shorten that path.
            reg         valid_s4;
            reg [8:0]   rounded9_s4;
            reg signed [9:0] exp_calc_s4;
            reg         result_sign_s4, is_zero_result_s4;
            reg [15:0]  a_s4, b_s4;

            wire [6:0] split_frac = rounded9_s4[8] ? 7'd0 : rounded9_s4[6:0];
            wire signed [9:0] split_exp = exp_calc_s4 +
                                           (rounded9_s4[8] ? 10'sd1 : 10'sd0);
            wire split_overflow = (split_exp >= 10'sd255);
            wire split_zero_a = (a_s4[14:7] == 8'd0);
            wire split_zero_b = (b_s4[14:7] == 8'd0);
            reg [15:0] split_result;

            always @* begin
                if (split_zero_a && split_zero_b)
                    split_result = {a_s4[15] & b_s4[15], 15'd0};
                else if (split_zero_a)
                    split_result = b_s4;
                else if (split_zero_b)
                    split_result = a_s4;
                else if (is_zero_result_s4)
                    split_result = 16'd0;
                else if (split_overflow)
                    split_result = {result_sign_s4, 8'hFF, 7'd0};
                else if (split_exp <= 10'sd0)
                    split_result = 16'd0;
                else
                    split_result = {result_sign_s4, split_exp[7:0], split_frac};
            end

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_s4          <= 1'b0;
                    rounded9_s4       <= 9'd0;
                    exp_calc_s4       <= 10'sd0;
                    result_sign_s4    <= 1'b0;
                    is_zero_result_s4 <= 1'b0;
                    a_s4              <= 16'd0;
                    b_s4              <= 16'd0;
                    valid_out         <= 1'b0;
                    result            <= 16'd0;
                end else begin
                    valid_s4          <= valid_s3;
                    rounded9_s4       <= rounded9_3;
                    exp_calc_s4       <= exp_calc_s3;
                    result_sign_s4    <= result_sign_comb_s3;
                    is_zero_result_s4 <= is_zero_result_s3;
                    a_s4              <= a_s3;
                    b_s4              <= b_s3;
                    valid_out         <= valid_s4;
                    result            <= split_result;
                end
            end
        end else if (EXTRA_PIPE_STAGE) begin : GEN_OUTPUT_ISOLATION
            // Local register boundary for the specific high-fanout,
            // high-route-delay reduction adders selected by their parent.
            reg        valid_s4;
            reg [15:0] result_s4;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_s4 <= 1'b0;
                    result_s4 <= 16'd0;
                    valid_out <= 1'b0;
                    result    <= 16'd0;
                end else begin
                    valid_s4 <= valid_s3;
                    result_s4 <= comb_result3;
                    valid_out <= valid_s4;
                    result    <= result_s4;
                end
            end
        end else begin : GEN_COMPACT_OUTPUT
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    valid_out <= 1'b0;
                    result    <= 16'd0;
                end else begin
                    valid_out <= valid_s3;
                    result    <= comb_result3;
                end
            end
        end
    endgenerate

endmodule
