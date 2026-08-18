`timescale 1ns / 1ps

module bf16_mac #(
    parameter MULT_LATENCY = 2
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire [15:0] c,
    output wire        valid_out,
    output wire [15:0] result
);

    wire        mult_valid_out;
    wire [15:0] mult_result;
    wire [15:0] c_aligned;

    generate
        genvar gi;
        if (MULT_LATENCY <= 0) begin : GEN_NO_DELAY
            assign c_aligned = c;
        end
        else begin : GEN_DELAY_CHAIN
            wire [15:0] c_tap [0:MULT_LATENCY];
            reg  [15:0] c_stage [0:MULT_LATENCY-1];

            assign c_tap[0] = c;
            for (gi = 0; gi < MULT_LATENCY; gi = gi + 1) begin : DELAY_STAGE
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        c_stage[gi] <= 16'h0000;
                    end
                    else begin
                        c_stage[gi] <= c_tap[gi];
                    end
                end
                assign c_tap[gi+1] = c_stage[gi];
            end

            assign c_aligned = c_tap[MULT_LATENCY];
        end
    endgenerate

    bf16_multiplier u_bf16_multiplier (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .a         (a),
        .b         (b),
        .valid_out (mult_valid_out),
        .result    (mult_result)
    );

    reg         adder_valid_in_r;
    reg  [15:0] adder_a_r;
    reg  [15:0] adder_b_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            adder_valid_in_r <= 1'b0;
            adder_a_r        <= 16'h0000;
            adder_b_r        <= 16'h0000;
        end
        else begin
            adder_valid_in_r <= mult_valid_out;
            adder_a_r        <= mult_result;
            adder_b_r        <= c_aligned;
        end
    end

    // Register the add/sub result before normalization.  bf16_dot_product_mac
    // tracks this one-cycle increase through its MAC_LATENCY parameter.
    bf16_adder #(.SPLIT_ARITH_STAGE(1)) u_bf16_adder (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (adder_valid_in_r),
        .a         (adder_a_r),
        .b         (adder_b_r),
        .valid_out (valid_out),
        .result    (result)
    );

endmodule
