// =============================================================================
// AXI4-Stream output adapter.
//
// The accumulator produces a complete HEAD_DIM-wide BF16 row.  Exporting that
// as a 2048-bit AXIS port is not practical on PYNQ-Z2, so this module stores
// one row and emits TILE_DIM BF16 values per transfer (256 bits by default).
// TLAST marks the end of *every output row*, which is the packet convention
// expected by AXI DMA/PYNQ receive buffers.
// =============================================================================
`timescale 1ns / 1ps

module axi_stream_egress #(
    parameter HEAD_DIM   = 128,
    parameter TILE_DIM   = 16,
    parameter DATA_WIDTH = 16,
    parameter N_PASSES   = HEAD_DIM / TILE_DIM,
    parameter IDX_BITS   = (N_PASSES <= 1) ? 1 : $clog2(N_PASSES)
) (
    input  wire                             clk,
    input  wire                             rst_n,

    // One full normalized attention row from tile_scheduler.
    input  wire                             row_valid,
    input  wire [HEAD_DIM*DATA_WIDTH-1:0]   row_data,
    output wire                             row_ready,

    // Narrow AXI4-Stream master.  There is no implicit acknowledgement:
    // data and TLAST remain stable while TVALID is high and TREADY is low.
    output reg  [TILE_DIM*DATA_WIDTH-1:0]   m_axis_tdata,
    output reg                              m_axis_tvalid,
    input  wire                             m_axis_tready,
    output reg                              m_axis_tlast
);
    reg [HEAD_DIM*DATA_WIDTH-1:0] row_reg;
    reg [IDX_BITS-1:0]            beat_idx;
    reg                           active;

    assign row_ready = !active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_reg       <= {HEAD_DIM*DATA_WIDTH{1'b0}};
            beat_idx      <= {IDX_BITS{1'b0}};
            active        <= 1'b0;
            m_axis_tdata  <= {TILE_DIM*DATA_WIDTH{1'b0}};
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
        end else if (!active) begin
            if (row_valid) begin
                row_reg       <= row_data;
                beat_idx      <= {IDX_BITS{1'b0}};
                active        <= 1'b1;
                m_axis_tdata  <= row_data[TILE_DIM*DATA_WIDTH-1:0];
                m_axis_tvalid <= 1'b1;
                m_axis_tlast  <= (N_PASSES == 1);
            end
        end else if (m_axis_tvalid && m_axis_tready) begin
            if (beat_idx == N_PASSES-1) begin
                active        <= 1'b0;
                m_axis_tvalid <= 1'b0;
                m_axis_tlast  <= 1'b0;
            end else begin
                beat_idx      <= beat_idx + 1'b1;
                m_axis_tdata  <= row_reg[(beat_idx+2)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH];
                m_axis_tvalid <= 1'b1;
                m_axis_tlast  <= (beat_idx + 1'b1 == N_PASSES-1);
            end
        end
    end
endmodule
