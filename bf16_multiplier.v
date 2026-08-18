`timescale 1ns / 1ps
module bf16_multiplier (
input wire clk,
input wire rst_n,
input wire valid_in,
input wire [15:0] a,
input wire [15:0] b,
output reg valid_out,
output reg [15:0] result
);

wire        sign_a, sign_b, sign_r0;
wire [7:0]  exp_a, exp_b;
wire [6:0]  mant_a, mant_b;
wire        a_exp_zero, b_exp_zero;
wire        a_exp_ones, b_exp_ones;
wire        a_mant_zero, b_mant_zero;
wire        a_is_zero0, b_is_zero0;
wire        a_is_nan0,  b_is_nan0;
wire        a_is_inf0,  b_is_inf0;
wire [7:0]  mant_a_ext, mant_b_ext;
wire [15:0] product0;
wire signed [9:0] exp_ab0;
wire signed [9:0] exp_sum0;

assign sign_a = a[15];
assign sign_b = b[15];
assign exp_a  = a[14:7];
assign exp_b  = b[14:7];
assign mant_a = a[6:0];
assign mant_b = b[6:0];

assign a_exp_zero  = (exp_a == 8'd0);
assign b_exp_zero  = (exp_b == 8'd0);
assign a_exp_ones  = (exp_a == 8'hFF);
assign b_exp_ones  = (exp_b == 8'hFF);
assign a_mant_zero = (mant_a == 7'd0);
assign b_mant_zero = (mant_b == 7'd0);

assign a_is_zero0 = a_exp_zero;
assign b_is_zero0 = b_exp_zero;
assign a_is_nan0  = a_exp_ones && !a_mant_zero;
assign b_is_nan0  = b_exp_ones && !b_mant_zero;
assign a_is_inf0  = a_exp_ones && a_mant_zero;
assign b_is_inf0  = b_exp_ones && b_mant_zero;

assign sign_r0 = sign_a ^ sign_b;

assign mant_a_ext = {1'b1, mant_a};
assign mant_b_ext = {1'b1, mant_b};
assign product0 = mant_a_ext * mant_b_ext;

assign exp_ab0  = $signed({2'b00, exp_a}) + $signed({2'b00, exp_b}) - 10'sd127;
assign exp_sum0 = product0[15] ? (exp_ab0 + 10'sd1) : exp_ab0;

reg         s1_valid;
reg         s1_sign_r;
reg  [15:0] s1_product;
reg  signed [9:0] s1_exp_sum;
reg         s1_a_is_nan, s1_b_is_nan, s1_a_is_inf, s1_b_is_inf, s1_a_is_zero, s1_b_is_zero;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s1_valid     <= 1'b0;
        s1_sign_r    <= 1'b0;
        s1_product   <= 16'h0000;
        s1_exp_sum   <= 10'sd0;
        s1_a_is_nan  <= 1'b0;
        s1_b_is_nan  <= 1'b0;
        s1_a_is_inf  <= 1'b0;
        s1_b_is_inf  <= 1'b0;
        s1_a_is_zero <= 1'b0;
        s1_b_is_zero <= 1'b0;
    end else begin
        s1_valid     <= valid_in;
        s1_sign_r    <= sign_r0;
        s1_product   <= product0;
        s1_exp_sum   <= exp_sum0;
        s1_a_is_nan  <= a_is_nan0;
        s1_b_is_nan  <= b_is_nan0;
        s1_a_is_inf  <= a_is_inf0;
        s1_b_is_inf  <= b_is_inf0;
        s1_a_is_zero <= a_is_zero0;
        s1_b_is_zero <= b_is_zero0;
    end
end

wire [6:0]  mant_norm;
wire        round_bit;
wire        sticky_bit;
wire        round_up;
wire        mant_norm_allones;
wire [6:0]  mant_final;
wire        mant_carry;
wire signed [9:0] exp_final;
wire        exp_sum_ge255;
wire        exp_sum_eq254;
wire        exp_sum_lt0;
wire        exp_sum_eq0;
wire        result_ovf;
wire        result_unf;
reg  [15:0] result_comb;

// The two normalization cases have different guard/sticky positions.  The
// old common right-shift dropped product[0] when product[15] was set, so
// values just above a halfway case rounded down by one BF16 ULP.
assign mant_norm  = s1_product[15] ? s1_product[14:8] : s1_product[13:7];
assign round_bit  = s1_product[15] ? s1_product[7]    : s1_product[6];
assign sticky_bit = s1_product[15] ? |s1_product[6:0] : |s1_product[5:0];
assign round_up = round_bit && (sticky_bit || mant_norm[0]);
assign mant_norm_allones = &mant_norm;
assign mant_carry        = round_up && mant_norm_allones;
assign mant_final        = mant_norm + {6'b0, round_up};
assign exp_final = mant_carry ? (s1_exp_sum + 10'sd1) : s1_exp_sum;

assign exp_sum_ge255 = (s1_exp_sum >= 10'sd255);
assign exp_sum_eq254 = (s1_exp_sum == 10'sd254);
assign exp_sum_lt0   = (s1_exp_sum <  10'sd0);
assign exp_sum_eq0   = (s1_exp_sum == 10'sd0);
assign result_ovf = exp_sum_ge255 || (mant_carry && exp_sum_eq254);
assign result_unf = exp_sum_lt0   || (!mant_carry && exp_sum_eq0);

always @* begin
    if (s1_a_is_nan || s1_b_is_nan) begin
        result_comb = {1'b0, 8'hFF, 7'h40};
    end else if (s1_a_is_inf || s1_b_is_inf) begin
        if (s1_a_is_zero || s1_b_is_zero)
            result_comb = {1'b0, 8'hFF, 7'h40};
        else
            result_comb = {s1_sign_r, 8'hFF, 7'h00};
    end else if (s1_a_is_zero || s1_b_is_zero) begin
        result_comb = {s1_sign_r, 8'h00, 7'h00};
    end else if (result_ovf) begin
        result_comb = {s1_sign_r, 8'hFF, 7'h00};
    end else if (result_unf) begin
        result_comb = {s1_sign_r, 8'h00, 7'h00};
    end else begin
        result_comb = {s1_sign_r, exp_final[7:0], mant_final};
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_out <= 1'b0;
        result    <= 16'h0000;
    end else begin
        valid_out <= s1_valid;
        result    <= result_comb;
    end
end

endmodule
