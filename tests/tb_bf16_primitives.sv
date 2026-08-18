`timescale 1ns/1ps
// Self-checking streaming conformance test.  Outputs are scored in-order,
// avoiding any assumption about the primitives' internal pipeline latency.
module tb_bf16_primitives #(
    parameter integer CASES = 65536,
    parameter integer MAX_ULP = 0
);
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg valid_in = 0;
    reg [15:0] a = 0, b = 0;
    wire add_valid, mul_valid;
    wire [15:0] add_result, mul_result;
    reg [15:0] a_mem [0:CASES-1];
    reg [15:0] b_mem [0:CASES-1];
    reg [15:0] add_golden [0:CASES-1];
    reg [15:0] mul_golden [0:CASES-1];
    integer sent = 0, add_seen = 0, mul_seen = 0, errors = 0;
    string mem_dir;

    bf16_adder u_add (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a(a), .b(b),
                      .valid_out(add_valid), .result(add_result));
    bf16_multiplier u_mul (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a(a), .b(b),
                           .valid_out(mul_valid), .result(mul_result));

    function integer ulp_distance(input [15:0] x, input [15:0] y);
        integer ox, oy;
        begin
            ox = x[15] ? (16'h8000 - {1'b0, x[14:0]}) : (16'h8000 + x);
            oy = y[15] ? (16'h8000 - {1'b0, y[14:0]}) : (16'h8000 + y);
            ulp_distance = (ox >= oy) ? ox - oy : oy - ox;
        end
    endfunction

    always @(posedge clk) begin
        if (rst_n && add_valid) begin
            if (ulp_distance(add_result, add_golden[add_seen]) > MAX_ULP) begin
                if (errors < 10) $display("ADD FAIL case=%0d rtl=%h golden=%h", add_seen, add_result, add_golden[add_seen]);
                errors = errors + 1;
            end
            add_seen = add_seen + 1;
        end
        if (rst_n && mul_valid) begin
            if (ulp_distance(mul_result, mul_golden[mul_seen]) > MAX_ULP) begin
                if (errors < 10) $display("MUL FAIL case=%0d rtl=%h golden=%h", mul_seen, mul_result, mul_golden[mul_seen]);
                errors = errors + 1;
            end
            mul_seen = mul_seen + 1;
        end
    end

    initial begin
        if (!$value$plusargs("MEM_DIR=%s", mem_dir)) mem_dir = "tests/data/bf16_primitives";
        $readmemh({mem_dir, "/a.mem"}, a_mem);
        $readmemh({mem_dir, "/b.mem"}, b_mem);
        $readmemh({mem_dir, "/add_golden.mem"}, add_golden);
        $readmemh({mem_dir, "/mul_golden.mem"}, mul_golden);
        repeat (3) @(posedge clk); rst_n = 1;
        while (sent < CASES) begin
            @(negedge clk); a = a_mem[sent]; b = b_mem[sent]; valid_in = 1; sent = sent + 1;
        end
        @(negedge clk); valid_in = 0;
        wait (add_seen == CASES && mul_seen == CASES);
        if (errors != 0) $fatal(1, "FAIL: %0d BF16 primitive mismatches", errors);
        $display("PASS: %0d BF16 add and multiply cases", CASES);
        $finish;
    end

    initial begin
        repeat (CASES + 1000) @(posedge clk);
        $fatal(1, "TIMEOUT: sent=%0d add_seen=%0d mul_seen=%0d", sent, add_seen, mul_seen);
    end
endmodule
