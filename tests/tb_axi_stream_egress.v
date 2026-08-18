`timescale 1ns / 1ps

// Protocol regression: verifies ordering, TLAST, and stability through
// AXI4-Stream backpressure for a deliberately small four-beat row.
module tb_axi_stream_egress;
    reg clk = 0, rst_n = 0;
    reg row_valid = 0;
    reg [63:0] row_data = 64'h4444_3333_2222_1111;
    wire row_ready;
    wire [15:0] tdata;
    wire tvalid, tlast;
    reg tready = 0;
    integer seen = 0;

    axi_stream_egress #(.HEAD_DIM(4), .TILE_DIM(1), .DATA_WIDTH(16)) dut (
        .clk(clk), .rst_n(rst_n), .row_valid(row_valid), .row_data(row_data), .row_ready(row_ready),
        .m_axis_tdata(tdata), .m_axis_tvalid(tvalid), .m_axis_tready(tready), .m_axis_tlast(tlast)
    );

    always #5 clk = ~clk;
    always @(posedge clk) begin
        if (tvalid && tready) begin
            case (seen)
                0: if (tdata !== 16'h1111 || tlast) $fatal(1, "bad beat 0");
                1: if (tdata !== 16'h2222 || tlast) $fatal(1, "bad beat 1");
                2: if (tdata !== 16'h3333 || tlast) $fatal(1, "bad beat 2");
                3: if (tdata !== 16'h4444 || !tlast) $fatal(1, "bad final beat");
            endcase
            seen = seen + 1;
        end
    end

    initial begin
        repeat (2) @(posedge clk);
        rst_n = 1;
        @(posedge clk); row_valid = 1;
        @(posedge clk); row_valid = 0;
        // Hold the first beat under backpressure before allowing transfers.
        repeat (3) @(posedge clk);
        if (!tvalid || tdata !== 16'h1111 || tlast) $fatal(1, "backpressure changed output");
        tready = 1;
        wait (seen == 4);
        @(posedge clk);
        if (tvalid || !row_ready) $fatal(1, "egress did not return idle");
        $display("PASS: axi_stream_egress");
        $finish;
    end
endmodule
