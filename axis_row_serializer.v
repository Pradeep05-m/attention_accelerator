`timescale 1ns / 1ps
// Converts one full 128-element attention row into N_PASSES practical AXI4-
// Stream beats.  With the default dimensions this is eight 256-bit beats,
// compatible with an AXI DMA width converter instead of a 2048-bit port.
module axis_row_serializer #(
    parameter HEAD_DIM = 128, TILE_DIM = 16, DATA_WIDTH = 16,
    parameter N_PASSES = HEAD_DIM / TILE_DIM,
    parameter IDX_BITS = (N_PASSES <= 1) ? 1 : $clog2(N_PASSES)
) (
    input wire clk, input wire rst_n,
    input wire row_valid,
    input wire [HEAD_DIM*DATA_WIDTH-1:0] row_data,
    output wire row_ready,
    output reg [TILE_DIM*DATA_WIDTH-1:0] m_axis_tdata,
    output reg m_axis_tvalid,
    input wire m_axis_tready,
    output reg m_axis_tlast
);
    reg [HEAD_DIM*DATA_WIDTH-1:0] row_reg;
    reg [IDX_BITS-1:0] beat_idx;
    reg active;
    assign row_ready = !active;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_reg <= 0; beat_idx <= 0; active <= 0;
            m_axis_tdata <= 0; m_axis_tvalid <= 0; m_axis_tlast <= 0;
        end else begin
            if (!active && row_valid) begin
                row_reg <= row_data;
                beat_idx <= 0;
                active <= 1'b1;
                m_axis_tdata <= row_data[TILE_DIM*DATA_WIDTH-1:0];
                m_axis_tvalid <= 1'b1;
                m_axis_tlast <= (N_PASSES == 1);
            end else if (active && m_axis_tvalid && m_axis_tready) begin
                if (beat_idx == N_PASSES-1) begin
                    active <= 1'b0;
                    m_axis_tvalid <= 1'b0;
                    m_axis_tlast <= 1'b0;
                end else begin
                    beat_idx <= beat_idx + 1'b1;
                    m_axis_tdata <= row_reg[(beat_idx+2)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH];
                    m_axis_tvalid <= 1'b1;
                    m_axis_tlast <= (beat_idx + 1'b1 == N_PASSES-1);
                end
            end
        end
    end
endmodule
