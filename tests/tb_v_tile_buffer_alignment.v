`timescale 1ns / 1ps

// Regression for the width-converting synchronous read path used by the
// time-multiplexed output accumulator.  Each slot and column has a unique
// value, so an off-by-one pass, stale BRAM word, or lane replication is
// caught independently of attention test vectors.
module tb_v_tile_buffer_alignment;
    localparam KV_BLOCK_LEN = 2;
    localparam HEAD_DIM = 8;
    localparam WRITE_TILE_DIM = 4;
    localparam READ_TILE_DIM = 1;
    localparam DATA_WIDTH = 16;

    reg clk = 0, rst_n = 0;
    reg row_valid = 0;
    reg [HEAD_DIM*DATA_WIDTH-1:0] row_in = 0;
    wire block_valid;
    reg [$clog2(HEAD_DIM/READ_TILE_DIM)-1:0] read_pass = 0;
    wire [KV_BLOCK_LEN*READ_TILE_DIM*DATA_WIDTH-1:0] read_data;
    integer slot, col;

    v_tile_buffer #(
        .KV_BLOCK_LEN(KV_BLOCK_LEN), .HEAD_DIM(HEAD_DIM),
        .DATA_WIDTH(DATA_WIDTH), .TILE_DIM(READ_TILE_DIM),
        .WRITE_TILE_DIM(WRITE_TILE_DIM)
    ) dut (
        .clk(clk), .rst_n(rst_n), .row_valid(row_valid), .block_first(1'b0),
        .row_in(row_in), .block_valid(block_valid), .read_pass(read_pass),
        .read_data(read_data)
    );

    always #5 clk = ~clk;

    task send_row(input integer which);
        integer k;
        begin
            for (k = 0; k < HEAD_DIM; k = k + 1)
                row_in[(k+1)*DATA_WIDTH-1 -: DATA_WIDTH] = which * 16'h100 + k;
            @(negedge clk); row_valid = 1'b1;
            @(negedge clk); row_valid = 1'b0;
            // The buffer writes one four-element word per clock.
            repeat (2) @(posedge clk);
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        send_row(0);
        send_row(1);
        wait (block_valid);

        // The read port is synchronous: issue a pass, then inspect its
        // registered data on the following cycle.
        for (col = 0; col < HEAD_DIM; col = col + 1) begin
            @(negedge clk); read_pass = col;
            @(posedge clk);
            @(posedge clk);
            for (slot = 0; slot < KV_BLOCK_LEN; slot = slot + 1)
                if (read_data[(slot+1)*DATA_WIDTH-1 -: DATA_WIDTH] !== slot * 16'h100 + col)
                    $fatal(1, "read mismatch slot=%0d col=%0d got=%h", slot, col,
                           read_data[(slot+1)*DATA_WIDTH-1 -: DATA_WIDTH]);
        end
        $display("PASS: v_tile_buffer width-conversion alignment");
        $finish;
    end
endmodule
