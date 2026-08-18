// =============================================================================
// score_tag_fifo.v
// Carries K-position metadata from the scheduler load handshake to the
// delayed score result.  The MAC, reduction tree, and scale multiplier put
// several cycles between those two events, so using the controller's live
// block/row signals at score_valid is incorrect.
// =============================================================================
`timescale 1ns / 1ps

module score_tag_fifo #(
    parameter DEPTH = 4,
    parameter PTR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire clk,
    input  wire rst_n,

    input  wire push,
    input  wire push_block_first,
    input  wire push_row_first,
    input  wire push_row_last,
    input  wire push_key_valid,

    input  wire pop,
    output wire block_first,
    output wire row_first,
    output wire row_last,
    output wire key_valid,
    output wire empty,
    output wire full
);

    reg [PTR_WIDTH-1:0] wr_ptr, rd_ptr;
    reg [PTR_WIDTH:0] count;
    reg block_first_mem [0:DEPTH-1];
    reg row_first_mem   [0:DEPTH-1];
    reg row_last_mem    [0:DEPTH-1];
    reg key_valid_mem   [0:DEPTH-1];

    assign empty = (count == 0);
    assign full  = (count == DEPTH);
    assign block_first = !empty && block_first_mem[rd_ptr];
    assign row_first   = !empty && row_first_mem[rd_ptr];
    assign row_last    = !empty && row_last_mem[rd_ptr];
    assign key_valid   = !empty && key_valid_mem[rd_ptr];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            count  <= 0;
        end else begin
            case ({push && !full, pop && !empty})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase

            if (push && !full) begin
                block_first_mem[wr_ptr] <= push_block_first;
                row_first_mem[wr_ptr]   <= push_row_first;
                row_last_mem[wr_ptr]    <= push_row_last;
                key_valid_mem[wr_ptr]   <= push_key_valid;
                if (wr_ptr == DEPTH-1)
                    wr_ptr <= 0;
                else
                    wr_ptr <= wr_ptr + 1'b1;
            end

            if (pop && !empty) begin
                if (rd_ptr == DEPTH-1)
                    rd_ptr <= 0;
                else
                    rd_ptr <= rd_ptr + 1'b1;
            end
        end
    end

endmodule
