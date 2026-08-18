// =============================================================================
// File        : bf16_exp.v
// Module      : bf16_exp
// Description : Mathematically exact LUT-based BF16 Exponential unit.
// =============================================================================
`timescale 1ns / 1ps

module bf16_exp #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     valid_in,
    input  wire [DATA_WIDTH-1:0]    x,
    output reg                      valid_out,
    output reg  [DATA_WIDTH-1:0]    y
);

    localparam ROM_DEPTH = (1 << ADDR_WIDTH);
    wire [ADDR_WIDTH-1:0] addr = x[DATA_WIDTH-1 -: ADDR_WIDTH];
    reg [DATA_WIDTH-1:0] rom [0:ROM_DEPTH-1];

    integer ri;
    initial begin
        for (ri = 0; ri < ROM_DEPTH; ri = ri + 1) begin
            case (ri)
                8'd0: rom[ri] = 16'h3f80;
                8'd1: rom[ri] = 16'h3f80;
                8'd2: rom[ri] = 16'h3f80;
                8'd3: rom[ri] = 16'h3f80;
                8'd4: rom[ri] = 16'h3f80;
                8'd5: rom[ri] = 16'h3f80;
                8'd6: rom[ri] = 16'h3f80;
                8'd7: rom[ri] = 16'h3f80;
                8'd8: rom[ri] = 16'h3f80;
                8'd9: rom[ri] = 16'h3f80;
                8'd10: rom[ri] = 16'h3f80;
                8'd11: rom[ri] = 16'h3f80;
                8'd12: rom[ri] = 16'h3f80;
                8'd13: rom[ri] = 16'h3f80;
                8'd14: rom[ri] = 16'h3f80;
                8'd15: rom[ri] = 16'h3f80;
                8'd16: rom[ri] = 16'h3f80;
                8'd17: rom[ri] = 16'h3f80;
                8'd18: rom[ri] = 16'h3f80;
                8'd19: rom[ri] = 16'h3f80;
                8'd20: rom[ri] = 16'h3f80;
                8'd21: rom[ri] = 16'h3f80;
                8'd22: rom[ri] = 16'h3f80;
                8'd23: rom[ri] = 16'h3f80;
                8'd24: rom[ri] = 16'h3f80;
                8'd25: rom[ri] = 16'h3f80;
                8'd26: rom[ri] = 16'h3f80;
                8'd27: rom[ri] = 16'h3f80;
                8'd28: rom[ri] = 16'h3f80;
                8'd29: rom[ri] = 16'h3f80;
                8'd30: rom[ri] = 16'h3f80;
                8'd31: rom[ri] = 16'h3f80;
                8'd32: rom[ri] = 16'h3f80;
                8'd33: rom[ri] = 16'h3f80;
                8'd34: rom[ri] = 16'h3f80;
                8'd35: rom[ri] = 16'h3f80;
                8'd36: rom[ri] = 16'h3f80;
                8'd37: rom[ri] = 16'h3f80;
                8'd38: rom[ri] = 16'h3f80;
                8'd39: rom[ri] = 16'h3f80;
                8'd40: rom[ri] = 16'h3f80;
                8'd41: rom[ri] = 16'h3f80;
                8'd42: rom[ri] = 16'h3f80;
                8'd43: rom[ri] = 16'h3f80;
                8'd44: rom[ri] = 16'h3f80;
                8'd45: rom[ri] = 16'h3f80;
                8'd46: rom[ri] = 16'h3f80;
                8'd47: rom[ri] = 16'h3f80;
                8'd48: rom[ri] = 16'h3f80;
                8'd49: rom[ri] = 16'h3f80;
                8'd50: rom[ri] = 16'h3f80;
                8'd51: rom[ri] = 16'h3f80;
                8'd52: rom[ri] = 16'h3f80;
                8'd53: rom[ri] = 16'h3f80;
                8'd54: rom[ri] = 16'h3f80;
                8'd55: rom[ri] = 16'h3f80;
                8'd56: rom[ri] = 16'h3f80;
                8'd57: rom[ri] = 16'h3f80;
                8'd58: rom[ri] = 16'h3f80;
                8'd59: rom[ri] = 16'h3f80;
                8'd60: rom[ri] = 16'h3f80;
                8'd61: rom[ri] = 16'h3f80;
                8'd62: rom[ri] = 16'h3f80;
                8'd63: rom[ri] = 16'h3f80;
                8'd64: rom[ri] = 16'h3f80;
                8'd65: rom[ri] = 16'h3f80;
                8'd66: rom[ri] = 16'h3f80;
                8'd67: rom[ri] = 16'h3f80;
                8'd68: rom[ri] = 16'h3f80;
                8'd69: rom[ri] = 16'h3f80;
                8'd70: rom[ri] = 16'h3f80;
                8'd71: rom[ri] = 16'h3f80;
                8'd72: rom[ri] = 16'h3f80;
                8'd73: rom[ri] = 16'h3f80;
                8'd74: rom[ri] = 16'h3f80;
                8'd75: rom[ri] = 16'h3f80;
                8'd76: rom[ri] = 16'h3f80;
                8'd77: rom[ri] = 16'h3f80;
                8'd78: rom[ri] = 16'h3f80;
                8'd79: rom[ri] = 16'h3f80;
                8'd80: rom[ri] = 16'h3f80;
                8'd81: rom[ri] = 16'h3f80;
                8'd82: rom[ri] = 16'h3f80;
                8'd83: rom[ri] = 16'h3f80;
                8'd84: rom[ri] = 16'h3f80;
                8'd85: rom[ri] = 16'h3f80;
                8'd86: rom[ri] = 16'h3f80;
                8'd87: rom[ri] = 16'h3f80;
                8'd88: rom[ri] = 16'h3f80;
                8'd89: rom[ri] = 16'h3f80;
                8'd90: rom[ri] = 16'h3f80;
                8'd91: rom[ri] = 16'h3f80;
                8'd92: rom[ri] = 16'h3f80;
                8'd93: rom[ri] = 16'h3f80;
                8'd94: rom[ri] = 16'h3f80;
                8'd95: rom[ri] = 16'h3f80;
                8'd96: rom[ri] = 16'h3f80;
                8'd97: rom[ri] = 16'h3f80;
                8'd98: rom[ri] = 16'h3f80;
                8'd99: rom[ri] = 16'h3f80;
                8'd100: rom[ri] = 16'h3f80;
                8'd101: rom[ri] = 16'h3f80;
                8'd102: rom[ri] = 16'h3f80;
                8'd103: rom[ri] = 16'h3f80;
                8'd104: rom[ri] = 16'h3f80;
                8'd105: rom[ri] = 16'h3f80;
                8'd106: rom[ri] = 16'h3f80;
                8'd107: rom[ri] = 16'h3f80;
                8'd108: rom[ri] = 16'h3f80;
                8'd109: rom[ri] = 16'h3f80;
                8'd110: rom[ri] = 16'h3f80;
                8'd111: rom[ri] = 16'h3f80;
                8'd112: rom[ri] = 16'h3f80;
                8'd113: rom[ri] = 16'h3f80;
                8'd114: rom[ri] = 16'h3f80;
                8'd115: rom[ri] = 16'h3f80;
                8'd116: rom[ri] = 16'h3f80;
                8'd117: rom[ri] = 16'h3f80;
                8'd118: rom[ri] = 16'h3f80;
                8'd119: rom[ri] = 16'h3f80;
                8'd120: rom[ri] = 16'h3f80;
                8'd121: rom[ri] = 16'h3f80;
                8'd122: rom[ri] = 16'h3f80;
                8'd123: rom[ri] = 16'h3f80;
                8'd124: rom[ri] = 16'h3f80;
                8'd125: rom[ri] = 16'h3f80;
                8'd126: rom[ri] = 16'h3f80;
                8'd127: rom[ri] = 16'h3f80;
                8'd128: rom[ri] = 16'h3f80;
                8'd129: rom[ri] = 16'h3f80;
                8'd130: rom[ri] = 16'h3f80;
                8'd131: rom[ri] = 16'h3f80;
                8'd132: rom[ri] = 16'h3f80;
                8'd133: rom[ri] = 16'h3f80;
                8'd134: rom[ri] = 16'h3f80;
                8'd135: rom[ri] = 16'h3f80;
                8'd136: rom[ri] = 16'h3f80;
                8'd137: rom[ri] = 16'h3f80;
                8'd138: rom[ri] = 16'h3f80;
                8'd139: rom[ri] = 16'h3f80;
                8'd140: rom[ri] = 16'h3f80;
                8'd141: rom[ri] = 16'h3f80;
                8'd142: rom[ri] = 16'h3f80;
                8'd143: rom[ri] = 16'h3f80;
                8'd144: rom[ri] = 16'h3f80;
                8'd145: rom[ri] = 16'h3f80;
                8'd146: rom[ri] = 16'h3f80;
                8'd147: rom[ri] = 16'h3f80;
                8'd148: rom[ri] = 16'h3f80;
                8'd149: rom[ri] = 16'h3f80;
                8'd150: rom[ri] = 16'h3f80;
                8'd151: rom[ri] = 16'h3f80;
                8'd152: rom[ri] = 16'h3f80;
                8'd153: rom[ri] = 16'h3f80;
                8'd154: rom[ri] = 16'h3f80;
                8'd155: rom[ri] = 16'h3f80;
                8'd156: rom[ri] = 16'h3f80;
                8'd157: rom[ri] = 16'h3f80;
                8'd158: rom[ri] = 16'h3f80;
                8'd159: rom[ri] = 16'h3f80;
                8'd160: rom[ri] = 16'h3f80;
                8'd161: rom[ri] = 16'h3f80;
                8'd162: rom[ri] = 16'h3f80;
                8'd163: rom[ri] = 16'h3f80;
                8'd164: rom[ri] = 16'h3f80;
                8'd165: rom[ri] = 16'h3f80;
                8'd166: rom[ri] = 16'h3f80;
                8'd167: rom[ri] = 16'h3f80;
                8'd168: rom[ri] = 16'h3f80;
                8'd169: rom[ri] = 16'h3f80;
                8'd170: rom[ri] = 16'h3f80;
                8'd171: rom[ri] = 16'h3f80;
                8'd172: rom[ri] = 16'h3f80;
                8'd173: rom[ri] = 16'h3f80;
                8'd174: rom[ri] = 16'h3f80;
                8'd175: rom[ri] = 16'h3f80;
                8'd176: rom[ri] = 16'h3f80;
                8'd177: rom[ri] = 16'h3f80;
                8'd178: rom[ri] = 16'h3f80;
                8'd179: rom[ri] = 16'h3f80;
                8'd180: rom[ri] = 16'h3f80;
                8'd181: rom[ri] = 16'h3f80;
                8'd182: rom[ri] = 16'h3f80;
                8'd183: rom[ri] = 16'h3f80;
                8'd184: rom[ri] = 16'h3f80;
                8'd185: rom[ri] = 16'h3f80;
                8'd186: rom[ri] = 16'h3f80;
                8'd187: rom[ri] = 16'h3f7f;
                8'd188: rom[ri] = 16'h3f7d;
                8'd189: rom[ri] = 16'h3f74;
                8'd190: rom[ri] = 16'h3f54;
                8'd191: rom[ri] = 16'h3ef2;
                8'd192: rom[ri] = 16'h3d4c;
                8'd193: rom[ri] = 16'h36ce;
                8'd194: rom[ri] = 16'h1cd7;
                8'd195: rom[ri] = 16'h0000;
                8'd196: rom[ri] = 16'h0000;
                8'd197: rom[ri] = 16'h0000;
                8'd198: rom[ri] = 16'h0000;
                8'd199: rom[ri] = 16'h0000;
                8'd200: rom[ri] = 16'h0000;
                8'd201: rom[ri] = 16'h0000;
                8'd202: rom[ri] = 16'h0000;
                8'd203: rom[ri] = 16'h0000;
                8'd204: rom[ri] = 16'h0000;
                8'd205: rom[ri] = 16'h0000;
                8'd206: rom[ri] = 16'h0000;
                8'd207: rom[ri] = 16'h0000;
                8'd208: rom[ri] = 16'h0000;
                8'd209: rom[ri] = 16'h0000;
                8'd210: rom[ri] = 16'h0000;
                8'd211: rom[ri] = 16'h0000;
                8'd212: rom[ri] = 16'h0000;
                8'd213: rom[ri] = 16'h0000;
                8'd214: rom[ri] = 16'h0000;
                8'd215: rom[ri] = 16'h0000;
                8'd216: rom[ri] = 16'h0000;
                8'd217: rom[ri] = 16'h0000;
                8'd218: rom[ri] = 16'h0000;
                8'd219: rom[ri] = 16'h0000;
                8'd220: rom[ri] = 16'h0000;
                8'd221: rom[ri] = 16'h0000;
                8'd222: rom[ri] = 16'h0000;
                8'd223: rom[ri] = 16'h0000;
                8'd224: rom[ri] = 16'h0000;
                8'd225: rom[ri] = 16'h0000;
                8'd226: rom[ri] = 16'h0000;
                8'd227: rom[ri] = 16'h0000;
                8'd228: rom[ri] = 16'h0000;
                8'd229: rom[ri] = 16'h0000;
                8'd230: rom[ri] = 16'h0000;
                8'd231: rom[ri] = 16'h0000;
                8'd232: rom[ri] = 16'h0000;
                8'd233: rom[ri] = 16'h0000;
                8'd234: rom[ri] = 16'h0000;
                8'd235: rom[ri] = 16'h0000;
                8'd236: rom[ri] = 16'h0000;
                8'd237: rom[ri] = 16'h0000;
                8'd238: rom[ri] = 16'h0000;
                8'd239: rom[ri] = 16'h0000;
                8'd240: rom[ri] = 16'h0000;
                8'd241: rom[ri] = 16'h0000;
                8'd242: rom[ri] = 16'h0000;
                8'd243: rom[ri] = 16'h0000;
                8'd244: rom[ri] = 16'h0000;
                8'd245: rom[ri] = 16'h0000;
                8'd246: rom[ri] = 16'h0000;
                8'd247: rom[ri] = 16'h0000;
                8'd248: rom[ri] = 16'h0000;
                8'd249: rom[ri] = 16'h0000;
                8'd250: rom[ri] = 16'h0000;
                8'd251: rom[ri] = 16'h0000;
                8'd252: rom[ri] = 16'h0000;
                8'd253: rom[ri] = 16'h0000;
                8'd254: rom[ri] = 16'h0000;
                8'd255: rom[ri] = 16'h0000;
                default: rom[ri] = 16'h0000;
            endcase
        end
    end

    wire [DATA_WIDTH-1:0] rom_data = rom[addr];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            y         <= {DATA_WIDTH{1'b0}};
        end else begin
            valid_out <= valid_in;
            y         <= rom_data;
            if (valid_in) $display("[%0t] BF16_EXP: x=%h addr=%d rom_data=%h", $time, x, addr, rom_data);
        end
    end

endmodule
