// =============================================================================
// File        : reciprocal_unit.v
// Module      : reciprocal_unit
// Description : LUT-based BF16 Reciprocal Approximation unit.
//               inv_sum = 1 / sum_in
//
//               Architecture (pure bit-field / integer-exponent operations,
//               no floating-point division, no real/shortreal at run time):
//
//                   BF16 input
//                       |
//                   field split : sign_in / exp_in / frac_in
//                       |
//                   address generation : addr = frac_in (mantissa bits)
//                       |
//                   ROM lookup : rom[addr] = {shift_bit, reciprocal mantissa}
//                   exponent   : exp_out = 254 - exp_in - shift_bit
//                       |
//                   zero override : sum_in == 0 -> +INF (0x7F80)
//                       |
//                   Output register
//
//               Why shift_bit is needed (this is the actual bug fix vs.
//               the previous version, which always used exp_out =
//               254 - exp_in unconditionally):
//                 Let m = 1.frac_in (the input's normalized mantissa,
//                 m in [1,2)). Then 1/m lies in (0.5, 1]. Only when
//                 frac_in == 0 (m == 1 exactly) is 1/m == 1, requiring no
//                 renormalization, so exp_out = 254 - exp_in is exact and
//                 rom[0] = 0. For every other frac_in, 1/m < 1 and must be
//                 renormalized as 2^-1 * (2/m) to restore a leading-1
//                 mantissa -- which means the true output exponent is
//                 ONE LESS than 254 - exp_in. shift_bit encodes exactly
//                 that per-address renormalization decision, precomputed
//                 offline alongside the mantissa, so the datapath below
//                 stays pure bit-field/ROM/subtract -- no run-time
//                 division or renormalization logic.
//
//               For power-of-two magnitudes (frac_in == 0), rom[0] = 0 by
//               construction, so inv_sum is exact:
//                   1.0 (0x3F80) -> 1.0   (0x3F80)
//                   2.0 (0x4000) -> 0.5   (0x3F00)
//                   4.0 (0x4080) -> 0.25  (0x3E80)
//                   8.0 (0x4100) -> 0.125 (0x3E00)
//               For non-zero mantissas the ROM contains correctly-rounded
//               reciprocal-mantissa coefficients (generated offline in
//               Python), not placeholder values.
//
//               Sign is preserved from the input (1/negative = negative);
//               a zero input (exp_in == 0, regardless of sign or mantissa)
//               is forced to +Infinity (0x7F80) per specification.
//
// Pipeline    : Single registered output stage; valid_out asserted 1 cycle
//               after valid_in. No combinational loops, no latches.
// Target      : AMD/Xilinx FPGA, Vivado/XSIM
// Coding      : Verilog-2001, synthesizable
// =============================================================================
`timescale 1ns / 1ps
module reciprocal_unit #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 7          // = mantissa width for DATA_WIDTH=16 BF16
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     valid_in,
    input  wire [DATA_WIDTH-1:0]    sum_in,
    output reg                      valid_out,
    output reg  [DATA_WIDTH-1:0]    inv_sum
);

    localparam EXP_WIDTH  = 8;
    localparam FRAC_WIDTH = DATA_WIDTH - 1 - EXP_WIDTH;
    localparam ROM_DEPTH  = (1 << ADDR_WIDTH);

    // -------------------------------------------------------------------
    // Field extraction
    // -------------------------------------------------------------------
    wire                   sign_in = sum_in[DATA_WIDTH-1];
    wire [EXP_WIDTH-1:0]   exp_in  = sum_in[DATA_WIDTH-2 -: EXP_WIDTH];
    wire [FRAC_WIDTH-1:0]  frac_in = sum_in[FRAC_WIDTH-1:0];

    wire is_zero = (exp_in == {EXP_WIDTH{1'b0}});

    // -------------------------------------------------------------------
    // Address generation : mantissa bits directly index the ROM
    // -------------------------------------------------------------------
    wire [ADDR_WIDTH-1:0] addr = frac_in[FRAC_WIDTH-1 -: ADDR_WIDTH];

    // -------------------------------------------------------------------
    // ROM / LUT storage : precomputed {shift_bit, reciprocal mantissa}
    // table (correctly rounded, generated offline in Python -- no
    // run-time arithmetic in the RTL itself).
    // -------------------------------------------------------------------
    reg [7:0] rom [0:ROM_DEPTH-1];

    integer ri;
    initial begin
        for (ri = 0; ri < ROM_DEPTH; ri = ri + 1) begin
            case (ri)
                  0: rom[ri] = 8'h00;
                  1: rom[ri] = 8'hfe;
                  2: rom[ri] = 8'hfc;
                  3: rom[ri] = 8'hfa;
                  4: rom[ri] = 8'hf8;
                  5: rom[ri] = 8'hf6;
                  6: rom[ri] = 8'hf5;
                  7: rom[ri] = 8'hf3;
                  8: rom[ri] = 8'hf1;
                  9: rom[ri] = 8'hef;
                 10: rom[ri] = 8'hed;
                 11: rom[ri] = 8'hec;
                 12: rom[ri] = 8'hea;
                 13: rom[ri] = 8'he8;
                 14: rom[ri] = 8'he7;
                 15: rom[ri] = 8'he5;
                 16: rom[ri] = 8'he4;
                 17: rom[ri] = 8'he2;
                 18: rom[ri] = 8'he0;
                 19: rom[ri] = 8'hdf;
                 20: rom[ri] = 8'hdd;
                 21: rom[ri] = 8'hdc;
                 22: rom[ri] = 8'hda;
                 23: rom[ri] = 8'hd9;
                 24: rom[ri] = 8'hd8;
                 25: rom[ri] = 8'hd6;
                 26: rom[ri] = 8'hd5;
                 27: rom[ri] = 8'hd3;
                 28: rom[ri] = 8'hd2;
                 29: rom[ri] = 8'hd1;
                 30: rom[ri] = 8'hcf;
                 31: rom[ri] = 8'hce;
                 32: rom[ri] = 8'hcd;
                 33: rom[ri] = 8'hcc;
                 34: rom[ri] = 8'hca;
                 35: rom[ri] = 8'hc9;
                 36: rom[ri] = 8'hc8;
                 37: rom[ri] = 8'hc7;
                 38: rom[ri] = 8'hc5;
                 39: rom[ri] = 8'hc4;
                 40: rom[ri] = 8'hc3;
                 41: rom[ri] = 8'hc2;
                 42: rom[ri] = 8'hc1;
                 43: rom[ri] = 8'hc0;
                 44: rom[ri] = 8'hbf;
                 45: rom[ri] = 8'hbd;
                 46: rom[ri] = 8'hbc;
                 47: rom[ri] = 8'hbb;
                 48: rom[ri] = 8'hba;
                 49: rom[ri] = 8'hb9;
                 50: rom[ri] = 8'hb8;
                 51: rom[ri] = 8'hb7;
                 52: rom[ri] = 8'hb6;
                 53: rom[ri] = 8'hb5;
                 54: rom[ri] = 8'hb4;
                 55: rom[ri] = 8'hb3;
                 56: rom[ri] = 8'hb2;
                 57: rom[ri] = 8'hb1;
                 58: rom[ri] = 8'hb0;
                 59: rom[ri] = 8'haf;
                 60: rom[ri] = 8'hae;
                 61: rom[ri] = 8'had;
                 62: rom[ri] = 8'hac;
                 63: rom[ri] = 8'hac;
                 64: rom[ri] = 8'hab;
                 65: rom[ri] = 8'haa;
                 66: rom[ri] = 8'ha9;
                 67: rom[ri] = 8'ha8;
                 68: rom[ri] = 8'ha7;
                 69: rom[ri] = 8'ha6;
                 70: rom[ri] = 8'ha5;
                 71: rom[ri] = 8'ha5;
                 72: rom[ri] = 8'ha4;
                 73: rom[ri] = 8'ha3;
                 74: rom[ri] = 8'ha2;
                 75: rom[ri] = 8'ha1;
                 76: rom[ri] = 8'ha1;
                 77: rom[ri] = 8'ha0;
                 78: rom[ri] = 8'h9f;
                 79: rom[ri] = 8'h9e;
                 80: rom[ri] = 8'h9e;
                 81: rom[ri] = 8'h9d;
                 82: rom[ri] = 8'h9c;
                 83: rom[ri] = 8'h9b;
                 84: rom[ri] = 8'h9b;
                 85: rom[ri] = 8'h9a;
                 86: rom[ri] = 8'h99;
                 87: rom[ri] = 8'h98;
                 88: rom[ri] = 8'h98;
                 89: rom[ri] = 8'h97;
                 90: rom[ri] = 8'h96;
                 91: rom[ri] = 8'h96;
                 92: rom[ri] = 8'h95;
                 93: rom[ri] = 8'h94;
                 94: rom[ri] = 8'h94;
                 95: rom[ri] = 8'h93;
                 96: rom[ri] = 8'h92;
                 97: rom[ri] = 8'h92;
                 98: rom[ri] = 8'h91;
                 99: rom[ri] = 8'h90;
                100: rom[ri] = 8'h90;
                101: rom[ri] = 8'h8f;
                102: rom[ri] = 8'h8e;
                103: rom[ri] = 8'h8e;
                104: rom[ri] = 8'h8d;
                105: rom[ri] = 8'h8d;
                106: rom[ri] = 8'h8c;
                107: rom[ri] = 8'h8b;
                108: rom[ri] = 8'h8b;
                109: rom[ri] = 8'h8a;
                110: rom[ri] = 8'h8a;
                111: rom[ri] = 8'h89;
                112: rom[ri] = 8'h89;
                113: rom[ri] = 8'h88;
                114: rom[ri] = 8'h87;
                115: rom[ri] = 8'h87;
                116: rom[ri] = 8'h86;
                117: rom[ri] = 8'h86;
                118: rom[ri] = 8'h85;
                119: rom[ri] = 8'h85;
                120: rom[ri] = 8'h84;
                121: rom[ri] = 8'h84;
                122: rom[ri] = 8'h83;
                123: rom[ri] = 8'h83;
                124: rom[ri] = 8'h82;
                125: rom[ri] = 8'h82;
                126: rom[ri] = 8'h81;
                127: rom[ri] = 8'h81;
                default: rom[ri] = 8'h00;
            endcase
        end
    end

    wire [7:0]            rom_entry = rom[addr];
    wire                  shift_bit = rom_entry[7];
    wire [FRAC_WIDTH-1:0] rom_frac  = rom_entry[FRAC_WIDTH-1:0];

    // -------------------------------------------------------------------
    // Exponent computation : exp_out = 254 - exp_in - shift_bit
    // (integer subtraction on the biased exponent field; shift_bit
    // corrects for reciprocal-mantissa renormalization, see header)
    // -------------------------------------------------------------------
    wire [EXP_WIDTH-1:0] exp_out = 8'd254 - exp_in - {7'b0, shift_bit};

    // -------------------------------------------------------------------
    // Result formation
    // -------------------------------------------------------------------
    wire [DATA_WIDTH-1:0] normal_result = {sign_in, exp_out, rom_frac};
    wire [DATA_WIDTH-1:0] comb_result   = is_zero ? 16'h7F80 : normal_result;

    // -------------------------------------------------------------------
    // Registered output stage
    // -------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            inv_sum   <= {DATA_WIDTH{1'b0}};
        end
        else begin
            valid_out <= valid_in;
            inv_sum   <= comb_result;
        end
    end

endmodule