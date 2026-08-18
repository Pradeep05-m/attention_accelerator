`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/22/2026 01:37:04 PM
// Design Name: 
// Module Name: scale_unit
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


// =============================================================================
// File        : scale_unit.v
// Module      : scale_unit
// Description : BF16 Scale Unit : scaled_score = attention_score * scale_factor
//               Instantiates exactly one bf16_multiplier (no arithmetic is
//               implemented in this module). scale_factor is supplied
//               externally by the wrapper (e.g. 1/sqrt(dk)) and is not
//               hardcoded, making the unit reusable across different dk.
// Target      : AMD/Xilinx FPGA, Vivado/XSIM
// Coding      : Verilog-2001, synthesizable, no latches
// =============================================================================

module scale_unit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [15:0] attention_score,
    input  wire [15:0] scale_factor,
    output wire         valid_out,
    output wire [15:0] scaled_score
);

    // -------------------------------------------------------------------
    // BF16 Multiplier : scaled_score = attention_score * scale_factor
    // -------------------------------------------------------------------
    bf16_multiplier u_bf16_multiplier (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .a         (attention_score),
        .b         (scale_factor),
        .valid_out (valid_out),
        .result    (scaled_score)
    );
`ifdef DEBUG
  // Debug waveform dump for scaling stage
  initial begin
    $dumpfile("scale_unit_debug.vcd");
    $dumpvars(0, scale_unit);
    $dumpvars(0, attention_score);
    $dumpvars(0, scale_factor);
    $dumpvars(0, scaled_score);
    $dumpvars(0, valid_in);
    $dumpvars(0, valid_out);
  end
`endif

endmodule