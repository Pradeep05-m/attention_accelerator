`timescale 1ns / 1ps

module bf16_subtractor #(
parameter LATENCY = 2,
// Register the operands before alignment when the instance sits directly
// behind a high-fanout producer.  This is intentionally opt-in: most users
// retain the four-cycle core, while softmax uses it to isolate m_new from the
// exponent compare/shift cloud.
parameter INPUT_PIPE_STAGE = 0
)(
input wire clk,
input wire rst_n,
input wire valid_in,
input wire [15:0] a,
input wire [15:0] b,
output wire valid_out,
output wire [15:0] result
);

localparam CORE_LATENCY = 4 + INPUT_PIPE_STAGE; // Stage 2 (normalize+round+select) split
                              // into two registered stages, see r2_* bank below
localparam EXTRA_STAGES = (LATENCY > CORE_LATENCY) ? (LATENCY - CORE_LATENCY) : 0;

function [3:0] clz11;
    input [10:0] din;
    input        dummy;
    integer k;
    begin
        clz11 = 4'd11;
        for (k = 10; k >= 0; k = k - 1) begin
            if (din[k] && (clz11 == 4'd11))
                clz11 = 10 - k;
        end
    end
endfunction

wire        core_valid_in;
wire [15:0] core_a, core_b;

generate
    if (INPUT_PIPE_STAGE) begin : GEN_INPUT_PIPE
        reg        input_valid;
        reg [15:0] input_a, input_b;
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                input_valid <= 1'b0;
                input_a     <= 16'd0;
                input_b     <= 16'd0;
            end else begin
                input_valid <= valid_in;
                input_a     <= a;
                input_b     <= b;
            end
        end
        assign core_valid_in = input_valid;
        assign core_a        = input_a;
        assign core_b        = input_b;
    end else begin : GEN_NO_INPUT_PIPE
        assign core_valid_in = valid_in;
        assign core_a        = a;
        assign core_b        = b;
    end
endgenerate

wire        sign_a      = core_a[15];
wire [7:0]  exp_a       = core_a[14:7];
wire [6:0]  frac_a      = core_a[6:0];

wire        sign_b_raw  = core_b[15];
wire [7:0]  exp_b       = core_b[14:7];
wire [6:0]  frac_b      = core_b[6:0];
wire        bsign_eff   = ~sign_b_raw;

wire        a_is_zsub = (exp_a == 8'd0);
wire        b_is_zsub = (exp_b == 8'd0);
wire        a_is_inf  = (exp_a == 8'hFF) && (frac_a == 7'd0);
wire        a_is_nan  = (exp_a == 8'hFF) && (frac_a != 7'd0);
wire        b_is_inf  = (exp_b == 8'hFF) && (frac_b == 7'd0);
wire        b_is_nan  = (exp_b == 8'hFF) && (frac_b != 7'd0);

wire        hidden_a  = ~a_is_zsub;
wire        hidden_b  = ~b_is_zsub;
wire [7:0]  sig_a     = {hidden_a, frac_a};
wire [7:0]  sig_b     = {hidden_b, frac_b};

wire signed [8:0] exp_a_eff = a_is_zsub ? -9'sd126 : ($signed({1'b0, exp_a}) - 9'sd127);
wire signed [8:0] exp_b_eff = b_is_zsub ? -9'sd126 : ($signed({1'b0, exp_b}) - 9'sd127);

wire exp_a_ge_b = (exp_a_eff >= exp_b_eff);

wire signed [8:0] exp_big0    = exp_a_ge_b ? exp_a_eff : exp_b_eff;
wire signed [8:0] exp_small0  = exp_a_ge_b ? exp_b_eff : exp_a_eff;
wire        [7:0] sig_big0    = exp_a_ge_b ? sig_a     : sig_b;
wire        [7:0] sig_small0  = exp_a_ge_b ? sig_b     : sig_a;
wire               sign_big0   = exp_a_ge_b ? sign_a    : bsign_eff;
wire               sign_small0 = exp_a_ge_b ? bsign_eff : sign_a;

wire signed [8:0] exp_diff_s0  = exp_big0 - exp_small0;
wire        [3:0] shift_amt0   = (exp_diff_s0 > 9'sd11) ? 4'd11 : exp_diff_s0[3:0];

wire [10:0] sig_big_ext0   = {sig_big0,   3'b000};
wire [10:0] sig_small_ext0 = {sig_small0, 3'b000};

reg         ra_valid;
reg         ra_sign_big, ra_sign_small;
reg  [10:0] ra_sig_big_ext, ra_sig_small_ext;
reg  [3:0]  ra_shift_amt;
reg  signed [8:0] ra_exp_big;
reg         ra_a_is_nan, ra_b_is_nan, ra_a_is_inf, ra_b_is_inf;
reg         ra_sign_a, ra_bsign_eff;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ra_valid         <= 1'b0;
        ra_sign_big      <= 1'b0;
        ra_sign_small    <= 1'b0;
        ra_sig_big_ext   <= 11'd0;
        ra_sig_small_ext <= 11'd0;
        ra_shift_amt     <= 4'd0;
        ra_exp_big       <= 9'sd0;
        ra_a_is_nan      <= 1'b0;
        ra_b_is_nan      <= 1'b0;
        ra_a_is_inf      <= 1'b0;
        ra_b_is_inf      <= 1'b0;
        ra_sign_a        <= 1'b0;
        ra_bsign_eff     <= 1'b0;
    end
    else begin
        ra_valid         <= core_valid_in;
        ra_sign_big      <= sign_big0;
        ra_sign_small    <= sign_small0;
        ra_sig_big_ext   <= sig_big_ext0;
        ra_sig_small_ext <= sig_small_ext0;
        ra_shift_amt     <= shift_amt0;
        ra_exp_big       <= exp_big0;
        ra_a_is_nan      <= a_is_nan;
        ra_b_is_nan      <= b_is_nan;
        ra_a_is_inf      <= a_is_inf;
        ra_b_is_inf      <= b_is_inf;
        ra_sign_a        <= sign_a;
        ra_bsign_eff     <= bsign_eff;
    end
end

wire [21:0] wide_small_sub  = {ra_sig_small_ext, 11'b0} >> ra_shift_amt;
wire [10:0] shifted_small   = wide_small_sub[21:11];
wire        sticky_shift    = |wide_small_sub[10:0];
wire [10:0] shifted_small_a = {shifted_small[10:1], shifted_small[0] | sticky_shift};

wire        signs_equal = (ra_sign_big == ra_sign_small);
wire [11:0] add_result  = {1'b0, ra_sig_big_ext} + {1'b0, shifted_small_a};
wire [11:0] sub_result  = {1'b0, ra_sig_big_ext} - {1'b0, shifted_small_a};

reg  [11:0] mantissa_ext;
reg         result_sign_c;
always @* begin
    if (signs_equal) begin
        mantissa_ext  = add_result;
        result_sign_c = ra_sign_big;
    end
    else begin
        if (sub_result[11]) begin
            mantissa_ext  = (~sub_result) + 12'd1;
            result_sign_c = ra_sign_small;
        end
        else begin
            mantissa_ext  = sub_result;
            result_sign_c = ra_sign_big;
        end
    end
end

reg         r1_valid;
reg         r1_result_sign;
reg  signed [8:0]  r1_exp_big;
reg  [11:0] r1_mantissa_ext;
reg         r1_a_is_nan, r1_b_is_nan, r1_a_is_inf, r1_b_is_inf;
reg         r1_sign_a, r1_bsign_eff;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        r1_valid          <= 1'b0;
        r1_result_sign    <= 1'b0;
        r1_exp_big        <= 9'sd0;
        r1_mantissa_ext   <= 12'd0;
        r1_a_is_nan       <= 1'b0;
        r1_b_is_nan       <= 1'b0;
        r1_a_is_inf       <= 1'b0;
        r1_b_is_inf       <= 1'b0;
        r1_sign_a         <= 1'b0;
        r1_bsign_eff      <= 1'b0;
    end
    else begin
        r1_valid          <= ra_valid;
        r1_result_sign    <= result_sign_c;
        r1_exp_big        <= ra_exp_big;
        r1_mantissa_ext   <= mantissa_ext;
        r1_a_is_nan       <= ra_a_is_nan;
        r1_b_is_nan       <= ra_b_is_nan;
        r1_a_is_inf       <= ra_a_is_inf;
        r1_b_is_inf       <= ra_b_is_inf;
        r1_sign_a         <= ra_sign_a;
        r1_bsign_eff      <= ra_bsign_eff;
    end
end

reg  [10:0]       norm_pre_round_c;
reg  signed [8:0] true_exp_c;
reg               is_zero_result_c;
reg  [3:0]        lzc_v;

always @* begin
    if (r1_mantissa_ext[11]) begin
        norm_pre_round_c    = r1_mantissa_ext[11:1];
        norm_pre_round_c[0] = norm_pre_round_c[0] | r1_mantissa_ext[0];
        true_exp_c          = r1_exp_big + 9'sd1;
        is_zero_result_c    = 1'b0;
    end
    else if (r1_mantissa_ext[10:0] == 11'd0) begin
        norm_pre_round_c = 11'd0;
        true_exp_c       = -9'sd200;
        is_zero_result_c = 1'b1;
    end
    else begin
        lzc_v            = clz11(r1_mantissa_ext[10:0], 1'b0);
        norm_pre_round_c = r1_mantissa_ext[10:0] << lzc_v;
        true_exp_c       = r1_exp_big - $signed({5'b0, lzc_v});
        is_zero_result_c = 1'b0;
    end
end

// -----------------------------------------------------------------
// FIX (timing): this used to feed directly into the round/subnormal/
// final-select logic below in the SAME cycle as normalize -- one
// unregistered cloud covering LZC + shift + round + subnormal repath +
// a 7-way priority mux. Reported as a 79%-route-delay path (10.667 of
// 13.486 ns, 16 logic levels) from r1_mantissa_ext_reg straight through
// to core_result_reg: too much logic fanning out over too much area for
// one cycle. Registering the normalize output here (r2_*) splits that
// cloud into two smaller, more locally-placeable stages. This adds one
// cycle to CORE_LATENCY (3 -> 4) but changes no values -- every signal
// below is renamed from its Stage-1 source (r1_*) to its Stage-2 source
// (r2_*) with identical semantics, nothing recomputed differently.
// -----------------------------------------------------------------
reg         r2_valid;
reg         r2_result_sign;
reg  [10:0] r2_norm_pre_round;
reg  signed [8:0] r2_true_exp;
reg         r2_is_zero_result;
reg         r2_a_is_nan, r2_b_is_nan, r2_a_is_inf, r2_b_is_inf;
reg         r2_sign_a, r2_bsign_eff;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        r2_valid          <= 1'b0;
        r2_result_sign    <= 1'b0;
        r2_norm_pre_round <= 11'd0;
        r2_true_exp       <= 9'sd0;
        r2_is_zero_result <= 1'b0;
        r2_a_is_nan       <= 1'b0;
        r2_b_is_nan       <= 1'b0;
        r2_a_is_inf       <= 1'b0;
        r2_b_is_inf       <= 1'b0;
        r2_sign_a         <= 1'b0;
        r2_bsign_eff      <= 1'b0;
    end
    else begin
        r2_valid          <= r1_valid;
        r2_result_sign    <= r1_result_sign;
        r2_norm_pre_round <= norm_pre_round_c;
        r2_true_exp       <= true_exp_c;
        r2_is_zero_result <= is_zero_result_c;
        r2_a_is_nan       <= r1_a_is_nan;
        r2_b_is_nan       <= r1_b_is_nan;
        r2_a_is_inf       <= r1_a_is_inf;
        r2_b_is_inf       <= r1_b_is_inf;
        r2_sign_a         <= r1_sign_a;
        r2_bsign_eff      <= r1_bsign_eff;
    end
end

wire        n_guard  = r2_norm_pre_round[2];
wire        n_round  = r2_norm_pre_round[1];
wire        n_sticky = r2_norm_pre_round[0];
wire        n_lsb    = r2_norm_pre_round[3];
wire        n_roundup = n_guard & (n_round | n_sticky | n_lsb);
wire [7:0]  n_kept8   = r2_norm_pre_round[10:3];
wire [8:0]  n_rounded9 = {1'b0, n_kept8} + {8'd0, n_roundup};

wire signed [8:0] normal_exp_unbiased = r2_true_exp + (n_rounded9[8] ? 9'sd1 : 9'sd0);
wire signed [9:0] normal_exp_biased   = {normal_exp_unbiased[8], normal_exp_unbiased} + 10'sd127;
wire [6:0]  normal_frac = n_rounded9[8] ? 7'd0 : n_rounded9[6:0];
wire        is_overflow = (normal_exp_biased > 10'sd254);

wire signed [9:0] shift_sub_raw = (-10'sd126) - {normal_exp_unbiased[8],normal_exp_unbiased[8],r2_true_exp};
wire [3:0]  shift_sub = (shift_sub_raw > 10'sd15) ? 4'd15 : shift_sub_raw[3:0];
wire [10:0] lost_mask = (shift_sub >= 4'd11) ? 11'h7FF : ((11'h001 << shift_sub) - 11'h001);
wire        sub_sticky_lost = |(r2_norm_pre_round & lost_mask);
wire [10:0] shifted_full = (shift_sub >= 4'd11) ? 11'd0 : (r2_norm_pre_round >> shift_sub);

wire [6:0]  s_n8       = shifted_full[9:3];
wire        s_guard    = shifted_full[2];
wire        s_round    = shifted_full[1];
wire        s_sticky   = shifted_full[0] | sub_sticky_lost;
wire        s_roundup  = s_guard & (s_round | s_sticky | s_n8[0]);
wire [7:0]  s_rounded8 = {1'b0, s_n8} + {7'd0, s_roundup};

wire [7:0]  sub_final_exp_biased = s_rounded8[7] ? 8'd1  : 8'd0;
wire [6:0]  sub_final_frac       = s_rounded8[7] ? 7'd0  : s_rounded8[6:0];

wire        is_subnormal_path = (r2_true_exp < -9'sd126);

reg  [15:0] stage2_result_c;
reg         stage2_valid_c;

always @* begin
    stage2_valid_c = r2_valid;

    if (r2_a_is_nan || r2_b_is_nan) begin
        stage2_result_c = 16'h7FC0;
    end
    else if (r2_a_is_inf && r2_b_is_inf) begin
        if (r2_sign_a == r2_bsign_eff)
            stage2_result_c = {r2_sign_a, 8'hFF, 7'd0};
        else
            stage2_result_c = 16'h7FC0;
    end
    else if (r2_a_is_inf) begin
        stage2_result_c = {r2_sign_a, 8'hFF, 7'd0};
    end
    else if (r2_b_is_inf) begin
        stage2_result_c = {r2_bsign_eff, 8'hFF, 7'd0};
    end
    else if (r2_is_zero_result) begin
        stage2_result_c = 16'h0000;
    end
    else if (is_subnormal_path) begin
        stage2_result_c = {r2_result_sign, sub_final_exp_biased, sub_final_frac};
    end
    else if (is_overflow) begin
        stage2_result_c = {r2_result_sign, 8'hFF, 7'd0};
    end
    else begin
        stage2_result_c = {r2_result_sign, normal_exp_biased[7:0], normal_frac};
    end
end

reg         core_valid;
reg  [15:0] core_result;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        core_valid  <= 1'b0;
        core_result <= 16'd0;
    end
    else begin
        core_valid  <= stage2_valid_c;
        core_result <= stage2_result_c;
    end
end

generate
    if (EXTRA_STAGES == 0) begin: NO_EXTRA
        assign valid_out = core_valid;
        assign result    = core_result;
    end
    else begin: EXTRA_DELAY
        reg         ev_dly [0:EXTRA_STAGES-1];
        reg [15:0]  er_dly [0:EXTRA_STAGES-1];
        integer     k;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                for (k = 0; k < EXTRA_STAGES; k = k + 1) begin
                    ev_dly[k] <= 1'b0;
                    er_dly[k] <= 16'd0;
                end
            end
            else begin
                ev_dly[0] <= core_valid;
                er_dly[0] <= core_result;
                for (k = 1; k < EXTRA_STAGES; k = k + 1) begin
                    ev_dly[k] <= ev_dly[k-1];
                    er_dly[k] <= er_dly[k-1];
                end
            end
        end

        assign valid_out = ev_dly[EXTRA_STAGES-1];
        assign result    = er_dly[EXTRA_STAGES-1];
    end
endgenerate

endmodule
