// PYNQ-Z2 physical-interface wrapper for gqa_attention_wrapper.
//
// The accelerator itself uses 256-bit AXI-Stream beats.  Exporting two such
// streams as FPGA pins requires 388 I/Os, while xc7z020clg400 has 255.  This
// wrapper leaves the accelerator unmodified and serializes both streams to
// 8-bit AXI-Stream beats. The PYNQ configuration uses 128-bit internal
// beats (TILE_DIM=4), so each logical transfer takes eight
// external transfers.  AXI-Lite remains unchanged.
`timescale 1ns / 1ps

module gqa_attention_pynq_z2_top (
    input wire clk, input wire rst_n,
    input wire [5:0] s_axi_awaddr,
    input wire s_axi_awvalid, output wire s_axi_awready,
    input wire [31:0] s_axi_wdata, input wire [3:0] s_axi_wstrb,
    input wire s_axi_wvalid, output wire s_axi_wready,
    output wire [1:0] s_axi_bresp, output wire s_axi_bvalid, input wire s_axi_bready,
    input wire [5:0] s_axi_araddr,
    input wire s_axi_arvalid, output wire s_axi_arready,
    output wire [31:0] s_axi_rdata, output wire [1:0] s_axi_rresp,
    output wire s_axi_rvalid, input wire s_axi_rready,

    input wire [7:0] s_axis_tdata, input wire s_axis_tkeep,
    input wire s_axis_tvalid, output wire s_axis_tready, input wire s_axis_tlast,
    input wire [1:0] s_axis_tuser,
    output wire [7:0] m_axis_tdata, output wire m_axis_tkeep,
    output wire m_axis_tvalid, input wire m_axis_tready, output wire m_axis_tlast
);
    // Input serializer: collect 32 8-bit words before presenting the
    // original 256-bit beat to the unmodified core.
    reg [63:0] in_data;
    reg [7:0]  in_keep;
    reg [63:0] core_sdata;
    reg [7:0]  core_skeep;
    reg [2:0]  in_word;
    reg         core_svalid, core_slast;
    reg [1:0]   core_suser;
    wire        core_sready;

    // The final segment is accepted only when the core can accept the
    // completed wide beat.  This maintains AXI backpressure end-to-end.
    assign s_axis_tready = !core_svalid && ((in_word != 3'd7) || core_sready);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_data     <= 64'd0;
            in_keep     <= 8'd0;
            core_sdata  <= 64'd0;
            core_skeep  <= 8'd0;
            in_word     <= 3'd0;
            core_svalid <= 1'b0;
            core_slast  <= 1'b0;
            core_suser  <= 2'b00;
        end else begin
            if (core_svalid && core_sready)
                core_svalid <= 1'b0;
            if (s_axis_tvalid && s_axis_tready) begin
                in_data[in_word*8 +: 8] <= s_axis_tdata;
                in_keep[in_word]        <= s_axis_tkeep;
                if (in_word == 3'd7) begin
                    // Include the just-accepted final segment; nonblocking
                    // assignment means in_data itself updates after this edge.
                    core_sdata  <= {s_axis_tdata, in_data[55:0]};
                    core_skeep  <= {s_axis_tkeep, in_keep[6:0]};
                    core_svalid <= 1'b1;
                    core_slast  <= s_axis_tlast;
                    core_suser  <= s_axis_tuser;
                    in_word     <= 3'd0;
                end else begin
                    in_word <= in_word + 1'b1;
                end
            end
        end
    end

    // Output deserializer: hold every wide core beat until all eight narrow
    // AXI transfers have completed. TLAST is asserted only on segment 7.
    wire [63:0] core_mdata;
    wire [7:0]  core_mkeep;
    wire         core_mvalid, core_mready, core_mlast;
    reg [63:0]   out_data;
    reg [7:0]    out_keep;
    reg [2:0]    out_word;
    reg          out_active, out_last;

    assign core_mready    = !out_active;
    assign m_axis_tvalid  = out_active;
    assign m_axis_tdata   = out_data[out_word*8 +: 8];
    assign m_axis_tkeep   = out_keep[out_word];
    assign m_axis_tlast   = out_active && out_last && (out_word == 3'd7);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_data   <= 64'd0;
            out_keep   <= 8'd0;
            out_word   <= 3'd0;
            out_active <= 1'b0;
            out_last   <= 1'b0;
        end else if (!out_active) begin
            if (core_mvalid) begin
                out_data   <= core_mdata;
                out_keep   <= core_mkeep;
                out_last   <= core_mlast;
                out_word   <= 3'd0;
                out_active <= 1'b1;
            end
        end else if (m_axis_tready) begin
            if (out_word == 3'd7) begin
                out_active <= 1'b0;
                out_word   <= 3'd0;
            end else begin
                out_word <= out_word + 1'b1;
            end
        end
    end

    gqa_attention_wrapper #(
        .GROUP_SIZE(1),
        .TILE_DIM(4)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awprot(3'b000),
        .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arprot(3'b000),
        .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .s_axis_tdata(core_sdata), .s_axis_tkeep(core_skeep), .s_axis_tvalid(core_svalid),
        .s_axis_tready(core_sready), .s_axis_tlast(core_slast), .s_axis_tuser(core_suser),
        .m_axis_tdata(core_mdata), .m_axis_tkeep(core_mkeep), .m_axis_tvalid(core_mvalid),
        .m_axis_tready(core_mready), .m_axis_tlast(core_mlast)
    );
endmodule
