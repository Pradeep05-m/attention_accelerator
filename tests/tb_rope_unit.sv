`timescale 1ns/1ps
module tb_rope_unit #(parameter integer ROWS=512, parameter integer MAX_ULP=64);
 reg clk=0,rst_n=0,in_valid=0,in_first=0,in_last=0,out_ready=1; always #5 clk=~clk;
 reg [255:0] in_mem[0:ROWS*8-1], gold_mem[0:ROWS*8-1];
 reg [255:0] chunk_in; reg [15:0] pos=0;
 wire out_valid,out_first,out_last; wire [255:0] chunk_out; wire busy;
 integer row=0,pass=0,seen=0,errors=0; string dir;
 rope_unit u(.clk(clk),.rst_n(rst_n),.in_valid(in_valid),.in_pass_first(in_first),.in_pass_last(in_last),.pos(pos),.chunk_in(chunk_in),.out_valid(out_valid),.out_pass_first(out_first),.out_pass_last(out_last),.chunk_out(chunk_out),.out_ready(out_ready),.busy(busy));
 function integer ulp(input [15:0] a,input [15:0] b); integer x,y; begin x=a[15]?(16'h8000-{1'b0,a[14:0]}):(16'h8000+a); y=b[15]?(16'h8000-{1'b0,b[14:0]}):(16'h8000+b); ulp=(x>=y)?x-y:y-x; end endfunction
 integer i;
 always @(posedge clk) if(rst_n && out_valid) begin
   for(i=0;i<16;i=i+1) if(ulp(chunk_out[(i+1)*16-1-:16],gold_mem[seen][(i+1)*16-1-:16])>MAX_ULP) begin if(errors<100) $display("ROPE FAIL row=%0d pass=%0d elem=%0d rtl=%h golden=%h ulp=%0d",seen/8,seen%8,i,chunk_out[(i+1)*16-1-:16],gold_mem[seen][(i+1)*16-1-:16], ulp(chunk_out[(i+1)*16-1-:16],gold_mem[seen][(i+1)*16-1-:16])); errors=errors+1; end
   if(out_last) begin if(!out_first && seen%8!=7) $fatal(1,"bad TLAST"); seen=seen+1; end else seen=seen+1;
 end
 task send; input integer p; begin @(negedge clk); chunk_in=in_mem[row*8+p]; pos=row; in_first=(p==0); in_last=(p==7); in_valid=1; @(posedge clk); @(negedge clk); in_valid=0; in_first=0; in_last=0; end endtask
 initial begin
   if(!$value$plusargs("MEM_DIR=%s",dir)) dir="tests/data/rope";
   $readmemh({dir,"/in.mem"},in_mem); $readmemh({dir,"/golden.mem"},gold_mem);
   repeat(3) @(posedge clk); rst_n=1;
   for(row=0;row<ROWS;row=row+1) begin for(pass=0;pass<8;pass=pass+1) send(pass); wait(!busy); end
   wait(seen==ROWS*8); if(errors) $fatal(1,"FAIL: %0d RoPE element mismatches",errors); $display("PASS: %0d RoPE rows",ROWS); $finish;
 end
 initial begin repeat(ROWS*3000) @(posedge clk); $fatal(1,"TIMEOUT seen=%0d",seen); end
endmodule
