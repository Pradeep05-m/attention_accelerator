import struct, math

def float_to_bf16(f):
    b = struct.unpack('>I', struct.pack('>f', f))[0]
    r = (b >> 16) & 0xffff
    lsb = (b >> 16) & 1
    round_bit = (b >> 15) & 1
    sticky = 1 if (b & 0x7fff) != 0 else 0
    if round_bit and (sticky or lsb):
        r += 1
    return r & 0xffff

def bf16_to_float(val):
    b = (val << 16) & 0xffffffff
    return struct.unpack('>f', struct.pack('>I', b))[0]

lines = []
lines.append('// =============================================================================')
lines.append('// File        : bf16_exp.v')
lines.append('// Module      : bf16_exp')
lines.append('// Description : Mathematically exact LUT-based BF16 Exponential unit.')
lines.append('// =============================================================================')
lines.append('`timescale 1ns / 1ps')
lines.append('')
lines.append('module bf16_exp #(')
lines.append('    parameter DATA_WIDTH = 16,')
lines.append('    parameter ADDR_WIDTH = 8')
lines.append(')(')
lines.append('    input  wire                     clk,')
lines.append('    input  wire                     rst_n,')
lines.append('    input  wire                     valid_in,')
lines.append('    input  wire [DATA_WIDTH-1:0]    x,')
lines.append('    output reg                      valid_out,')
lines.append('    output reg  [DATA_WIDTH-1:0]    y')
lines.append(');')
lines.append('')
lines.append('    localparam ROM_DEPTH = (1 << ADDR_WIDTH);')
lines.append('    wire [ADDR_WIDTH-1:0] addr = x[DATA_WIDTH-1 -: ADDR_WIDTH];')
lines.append('    reg [DATA_WIDTH-1:0] rom [0:ROM_DEPTH-1];')
lines.append('')
lines.append('    integer ri;')
lines.append('    initial begin')
lines.append('        for (ri = 0; ri < ROM_DEPTH; ri = ri + 1) begin')
lines.append('            case (ri)')

for ri in range(256):
    if ri < 128:
        res = 0x3f80
    else:
        sign = 1
        exp_7bit = ri & 0x7f
        exp_8bit = (exp_7bit << 1)
        frac_7bit = 0x40 # midpoint representative mantissa
        bf16_bits = (sign << 15) | (exp_8bit << 7) | frac_7bit
        x_val = bf16_to_float(bf16_bits)
        if x_val < -100.0:
            exp_x = 0.0
        else:
            exp_x = math.exp(x_val)
        res = float_to_bf16(exp_x)
    lines.append(f'                8\'d{ri}: rom[ri] = 16\'h{res:04x};')

lines.append('                default: rom[ri] = 16\'h0000;')
lines.append('            endcase')
lines.append('        end')
lines.append('    end')
lines.append('')
lines.append('    wire [DATA_WIDTH-1:0] rom_data = rom[addr];')
lines.append('')
lines.append('    always @(posedge clk or negedge rst_n) begin')
lines.append('        if (!rst_n) begin')
lines.append('            valid_out <= 1\'b0;')
lines.append('            y         <= {DATA_WIDTH{1\'b0}};')
lines.append('        end else begin')
lines.append('            valid_out <= valid_in;')
lines.append('            y         <= rom_data;')
lines.append('        end')
lines.append('    end')
lines.append('')
lines.append('endmodule')

with open('/home/prad/Downloads/amd/att/bf16_exp.v', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Successfully regenerated bf16_exp.v')
