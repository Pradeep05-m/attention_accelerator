`timescale 1ns / 1ps
// Converts GROUP_SIZE simultaneous Q-head results into one lossless output
// stream.  The next group is held by the controller until all lane results
// have been captured here.
module gqa_output_collector #(
    parameter GROUP_SIZE = 4, DATA_WIDTH = 16, ACC_WIDTH = 128
) (
    input wire clk, input wire rst_n,
    input wire [GROUP_SIZE-1:0] lane_valid,
    input wire [GROUP_SIZE*ACC_WIDTH*DATA_WIDTH-1:0] lane_rows,
    output reg out_valid,
    output reg [ACC_WIDTH*DATA_WIDTH-1:0] out_row,
    input wire out_ready
);
    generate
        // The PYNQ-Z2 integration sets GROUP_SIZE=1.  In this case the
        // generic collector's saved[] row is redundant: out_row itself is
        // already the required one-entry skid buffer.  Eliminating saved[]
        // removes a 2,048-bit register bank and its reset/control set.
        if (GROUP_SIZE == 1) begin : SINGLE_LANE
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    out_valid <= 1'b0;
                    out_row   <= {ACC_WIDTH*DATA_WIDTH{1'b0}};
                end else if (!out_valid && lane_valid[0]) begin
                    out_row   <= lane_rows[ACC_WIDTH*DATA_WIDTH-1:0];
                    out_valid <= 1'b1;
                end else if (out_valid && out_ready) begin
                    out_valid <= 1'b0;
                end
            end
        end else begin : MULTI_LANE
            localparam IDX_BITS = $clog2(GROUP_SIZE);
            reg [ACC_WIDTH*DATA_WIDTH-1:0] saved [0:GROUP_SIZE-1];
            reg [IDX_BITS-1:0] emit_idx;
            reg pending;
            integer i;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    out_valid <= 0; out_row <= 0; emit_idx <= 0; pending <= 0;
                    for (i = 0; i < GROUP_SIZE; i = i + 1) saved[i] <= 0;
                end else begin
                    if (!out_valid && pending) begin
                        out_row <= saved[emit_idx];
                        out_valid <= 1'b1;
                    end else if (!out_valid && !pending && (&lane_valid)) begin
                        for (i = 0; i < GROUP_SIZE; i = i + 1)
                            saved[i] <= lane_rows[(i+1)*ACC_WIDTH*DATA_WIDTH-1 -: ACC_WIDTH*DATA_WIDTH];
                        emit_idx <= 0;
                        pending <= 1'b1;
                    end else if (out_valid && out_ready) begin
                        out_valid <= 1'b0;
                        if (emit_idx == GROUP_SIZE-1) begin
                            pending <= 1'b0;
                        end else begin
                            emit_idx <= emit_idx + 1'b1;
                        end
                    end
                end
            end
        end
    endgenerate
endmodule
