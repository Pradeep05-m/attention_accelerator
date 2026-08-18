`timescale 1ns/1ps
// =============================================================================
// tb_gqa_attention_wrapper_v2.sv
//
// End-to-end AXI-Lite + AXI-Stream regression for gqa_attention_wrapper, with
// built-in diagnostics: every gqa_controller FSM state transition and every
// major req/ack handshake pulse is logged with a timestamp automatically, and
// the watchdog prints a full diagnostic snapshot (not just a bare timeout)
// if the run stalls. This replaces needing separate "_debug" module variants
// -- everything is probed via hierarchical reference from the testbench, no
// DUT source modification required.
//
// Explicit port connections throughout (no ".*" wildcard) -- more portable
// and avoids a known issue where some simulators struggle to elaborate ".*"
// on designs with many DUT ports.
//
// Generate its .mem inputs with pyfiles/gen_gqa_wrapper_vectors.py.
// =============================================================================
module tb_gqa_attention_wrapper_v2 #(
    // Set to zero for bit-exact golden vectors. Non-zero is useful when the
    // golden model is FP32 while the DUT uses approximate BF16 math.
    parameter integer MAX_ULP    = 128, // Reduced ULP for tighter checking
    parameter integer MAX_CYCLES = 500000,
    // Keep the 16-key default regression, while permitting the PYNQ-Z2
    // eight-key hardware configuration to be checked against the same
    // vectors and golden result.
    parameter integer KV_BLOCK_LEN = 16,
    // Kept parameterized so the vector generator can create longer random
    // accuracy corpora (for example 30 tokens = 65,536 BF16 input elements).
    parameter integer SEQ_LEN = 16
);
    // -------------------------------------------------------------------
    // Fixed geometry (matches the .mem vectors -- change together with a
    // regenerated vector set, not independently)
    // -------------------------------------------------------------------
    localparam integer N_Q_HEADS  = 32, N_KV_HEADS = 8, GROUP_SIZE = 4;
    localparam integer HEAD_DIM   = 128, TILE_DIM = 16, DATA_WIDTH = 16;
    localparam integer N_PASSES   = HEAD_DIM / TILE_DIM;
    // K/V transport must fill the last flash-attention block.  Positions
    // beyond SEQ_LEN are zero-padded by the generator and masked in RTL.
    localparam integer KV_INPUT_LEN = ((SEQ_LEN + KV_BLOCK_LEN - 1) / KV_BLOCK_LEN) * KV_BLOCK_LEN;

    // -------------------------------------------------------------------
    // Clock / reset
    // -------------------------------------------------------------------
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------
    // DUT port-level signals
    // -------------------------------------------------------------------
    reg  [5:0]  awaddr  = 0; reg  [2:0] awprot = 0; reg awvalid = 0; wire awready;
    reg  [31:0] wdata   = 0; reg  [3:0] wstrb = 4'hf; reg wvalid = 0; wire wready;
    wire [1:0]  bresp;       wire bvalid;              reg  bready = 0;
    reg  [5:0]  araddr  = 0; reg  [2:0] arprot = 0; reg arvalid = 0; wire arready;
    wire [31:0] rdata;       wire [1:0] rresp;          wire rvalid; reg rready = 1;

    reg  [TILE_DIM*DATA_WIDTH-1:0] s_data = 0;
    reg  [TILE_DIM*DATA_WIDTH/8-1:0] s_keep = {TILE_DIM*DATA_WIDTH/8{1'b1}};
    reg s_valid = 0;
    wire s_ready; reg s_last = 0; reg [1:0] s_user = 0;

    wire [TILE_DIM*DATA_WIDTH-1:0] m_data; wire m_valid; reg m_ready = 1; wire m_last;
    bit output_backpressure;
    integer backpressure_cycle;
    initial output_backpressure = $test$plusargs("OUTPUT_BACKPRESSURE");

    // Deterministic sink stalls exercise AXI-stream hold/stability logic.
    // The normal regression remains permanently ready.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_ready <= 1'b1;
            backpressure_cycle <= 0;
        end else if (output_backpressure) begin
            backpressure_cycle <= backpressure_cycle + 1;
            m_ready <= (backpressure_cycle % 5) < 3;
        end else begin
            m_ready <= 1'b1;
        end
    end

    // -------------------------------------------------------------------
    // DUT instantiation -- explicit port connections (no ".*")
    // -------------------------------------------------------------------
    gqa_attention_wrapper #(
        .N_Q_HEADS(N_Q_HEADS), .N_KV_HEADS(N_KV_HEADS), .GROUP_SIZE(GROUP_SIZE),
        .HEAD_DIM(HEAD_DIM), .TILE_DIM(TILE_DIM), .KV_BLOCK_LEN(KV_BLOCK_LEN),
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(HEAD_DIM)
    ) dut (
        .clk(clk), .rst_n(rst_n),

        .s_axi_awaddr(awaddr), .s_axi_awprot(awprot), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata),   .s_axi_wstrb(wstrb),      .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp),   .s_axi_bvalid(bvalid),    .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arprot(arprot), .s_axi_arvalid(arvalid),  .s_axi_arready(arready),
        .s_axi_rdata(rdata),   .s_axi_rresp(rresp),      .s_axi_rvalid(rvalid), .s_axi_rready(rready),

        .s_axis_tdata(s_data), .s_axis_tkeep(s_keep), .s_axis_tvalid(s_valid), .s_axis_tready(s_ready),
        .s_axis_tlast(s_last), .s_axis_tuser(s_user),

        .m_axis_tdata(m_data), .m_axis_tvalid(m_valid), .m_axis_tready(m_ready), .m_axis_tlast(m_last)
    );

    // -------------------------------------------------------------------
    // Test vectors / golden reference
    // -------------------------------------------------------------------
    reg [TILE_DIM*DATA_WIDTH-1:0] q_mem      [0:N_Q_HEADS*N_PASSES-1];
    reg [TILE_DIM*DATA_WIDTH-1:0] k_mem      [0:N_KV_HEADS*KV_INPUT_LEN*N_PASSES-1];
    reg [TILE_DIM*DATA_WIDTH-1:0] v_mem      [0:N_KV_HEADS*KV_INPUT_LEN*N_PASSES-1];
    reg [HEAD_DIM*DATA_WIDTH-1:0] golden_mem [0:N_Q_HEADS-1];
    reg [HEAD_DIM*DATA_WIDTH-1:0] received_row = 0;
    integer out_beat = 0, rows_seen = 0, output_file, errors = 0;
    // These counters belong to concurrent processes.  They must not share
    // one module-scope variable: the output checker runs while the stimulus
    // loop is still streaming later GQA groups.
    integer element_idx;
    integer group_idx;
    string mem_dir, rtl_out_path;
    bit trace_enabled;

    initial trace_enabled = $test$plusargs("TRACE");

    function integer ulp_distance;
        input [15:0] a;
        input [15:0] b;
        integer oa, ob;
        begin
            oa = a[15] ? (16'h8000 - {1'b0,a[14:0]}) : (16'h8000 + a);
            ob = b[15] ? (16'h8000 - {1'b0,b[14:0]}) : (16'h8000 + b);
            ulp_distance = (oa >= ob) ? oa-ob : ob-oa;
        end
    endfunction

    // -------------------------------------------------------------------
    // AXI-Lite / AXI-Stream driver tasks
    // -------------------------------------------------------------------
    task axi_write;
        input [5:0] addr;
        input [31:0] data;
        begin
            @(negedge clk); awaddr = addr; wdata = data; awvalid = 1; wvalid = 1;
            while (!(awready && wready)) @(posedge clk);
            @(negedge clk); awvalid = 0; wvalid = 0;
            while (!bvalid) @(posedge clk);
            @(negedge clk); bready = 1;
            @(posedge clk); @(negedge clk); bready = 0;
        end
    endtask

    task axi_read;
        input  [5:0] addr;
        output [31:0] data;
        begin
            @(negedge clk); araddr = addr; arvalid = 1;
            while (!arready) @(posedge clk);
            @(negedge clk); arvalid = 0;
            while (!rvalid) @(posedge clk);
            data = rdata;
            @(negedge clk);
        end
    endtask

    task axis_send;
        input [TILE_DIM*DATA_WIDTH-1:0] data;
        input [1:0] user;
        input last;
        begin
            @(negedge clk); s_data = data; s_user = user; s_last = last; s_valid = 1;
            do @(posedge clk); while (!s_ready);
            @(negedge clk); s_valid = 0; s_last = 0;
        end
    endtask

    task send_group;
        input integer group_index;
        integer head, pos, pass, q_base, kv_base;
        begin
            q_base = group_index * GROUP_SIZE * N_PASSES;
            for (head = 0; head < GROUP_SIZE; head = head + 1)
                for (pass = 0; pass < N_PASSES; pass = pass + 1)
                    axis_send(q_mem[q_base + head*N_PASSES + pass], 2'b00,
                              pass == N_PASSES-1);
            kv_base = group_index * KV_INPUT_LEN * N_PASSES;
            for (pos = 0; pos < KV_INPUT_LEN; pos = pos + 1) begin
                for (pass = 0; pass < N_PASSES; pass = pass + 1)
                    axis_send(k_mem[kv_base + pos*N_PASSES + pass], 2'b01,
                              pass == N_PASSES-1);
                for (pass = 0; pass < N_PASSES; pass = pass + 1)
                    axis_send(v_mem[kv_base + pos*N_PASSES + pass], 2'b10,
                              pass == N_PASSES-1);
            end
        end
    endtask

    integer max_ulp;
    initial begin
        if (!$value$plusargs("MAX_ULP=%d", max_ulp)) max_ulp = MAX_ULP;
    end

    function automatic bit is_element_match(
        input [15:0] rtl_val,
        input [15:0] golden_val,
        input integer allowed_ulp
    );
        integer ulp_gap;
        begin
            ulp_gap = ulp_distance(rtl_val, golden_val);
            if (ulp_gap <= allowed_ulp) return 1'b1;
            // Only exempt true BF16 underflow-scale values.  A previous 0x7d
            // threshold accepted values as large as 0.0625, which hid genuine
            // sign and numerical errors in normal attention outputs.  Keep
            // this criterion aligned with pyfiles/check_gqa_output.py.
            if (rtl_val[14:7] <= 8'h3d && golden_val[14:7] <= 8'h3d) return 1'b1;
            return 1'b0;
        end
    endfunction

    // -------------------------------------------------------------------
    // Output capture + scoring
    // -------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && m_valid && m_ready) begin
            received_row[(out_beat+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH] = m_data;
            if (m_last) begin
                if (out_beat != N_PASSES-1) $fatal(1, "TLAST on beat %0d", out_beat);
                if (rows_seen >= N_Q_HEADS) $fatal(1, "more output rows than expected");
                $fdisplay(output_file, "%h", received_row);
                for (element_idx = 0; element_idx < HEAD_DIM; element_idx = element_idx + 1)
                    if (!is_element_match(received_row[(element_idx+1)*16-1 -: 16],
                                           golden_mem[rows_seen][(element_idx+1)*16-1 -: 16],
                                           max_ulp)) begin
                        if (errors < 10) $display("FAIL row=%0d element=%0d rtl=%h golden=%h", rows_seen, element_idx,
                            received_row[(element_idx+1)*16-1 -: 16], golden_mem[rows_seen][(element_idx+1)*16-1 -: 16]);
                        errors = errors + 1;
                    end
                $display("[%0t] OUTPUT: row %0d received (TLAST)", $time, rows_seen);
                rows_seen = rows_seen + 1;
                out_beat = 0;
                received_row = 0;
            end else begin
                if (out_beat == N_PASSES-1) $fatal(1, "missing TLAST");
                out_beat = out_beat + 1;
            end
        end
    end

    // =====================================================================
    // BUILT-IN DIAGNOSTICS -- probed via hierarchical reference into the
    // DUT, no source modification of gqa_attention_wrapper.v or its
    // submodules required.
    // =====================================================================

    // ---- gqa_controller FSM state-transition trace ----
    function automatic string state_name(input [3:0] s);
        case (s)
            4'd0: state_name = "S_IDLE";
            4'd1: state_name = "S_LOAD_Q";
            4'd2: state_name = "S_LOAD_KVPOS";
            4'd3: state_name = "S_MAC_PASSES";
            4'd4: state_name = "S_NEXT_KPOS";
            4'd5: state_name = "S_NEXT_BLOCK";
            4'd6: state_name = "S_NEXT_ROW";
            4'd7: state_name = "S_DONE";
            4'd8: state_name = "S_WAIT_DRAIN";
            4'd9: state_name = "S_WAIT_BLOCK";
            default: state_name = "S_UNKNOWN";
        endcase
    endfunction

    reg [3:0] state_prev;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_prev <= 4'd0;
        end else begin
            if (trace_enabled && dut.u_ctrl.state !== state_prev) begin
                $display("[%0t] FSM: %s -> %s", $time,
                          state_name(state_prev), state_name(dut.u_ctrl.state));
                state_prev <= dut.u_ctrl.state;
            end
        end
    end

    // ---- Handshake pulse trace ----
    always @(posedge clk) begin
        if (trace_enabled && rst_n) begin
            if (dut.q_row_req)   $display("[%0t] HS: q_row_req asserted",  $time);
            if (dut.q_row_ack)   $display("[%0t] HS: q_row_ack fired",     $time);
            if (dut.kv_pos_req)  $display("[%0t] HS: kv_pos_req asserted", $time);
            if (dut.kv_pos_ack)  $display("[%0t] HS: kv_pos_ack fired",    $time);
            if (dut.row_advance) $display("[%0t] HS: row_advance fired (row %0d complete)", $time, rows_seen);
            if (dut.config_error) $display("[%0t] *** CONFIG_ERROR asserted -- bad register programming ***", $time);
        end
    end

    // ---- Periodic liveness heartbeat (every 5000 cycles) so a stalled
    // run is visibly distinguishable from a simulator that's just slow ----
    always @(posedge clk) begin
        if (trace_enabled && rst_n && ($time % 50000 == 0)) begin
            $display("[%0t] HEARTBEAT: state=%s rows_seen=%0d q_row_req=%b q_row_ack=%b kv_pos_req=%b kv_pos_ack=%b mac_busy=%b",
                       $time, state_name(dut.u_ctrl.state), rows_seen,
                       dut.q_row_req, dut.q_row_ack, dut.kv_pos_req, dut.kv_pos_ack, dut.mac_busy);
        end
    end

    // -------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------
    initial begin : stimulus
        bit invalid_config_test;
        if (!$value$plusargs("MEM_DIR=%s", mem_dir)) mem_dir = ".";
        if (!$value$plusargs("RTL_OUT=%s", rtl_out_path)) rtl_out_path = {mem_dir, "/gqa_rtl_out.mem"};
        $readmemh({mem_dir, "/gqa_q.mem"}, q_mem);
        $readmemh({mem_dir, "/gqa_k.mem"}, k_mem);
        $readmemh({mem_dir, "/gqa_v.mem"}, v_mem);
        $readmemh({mem_dir, "/gqa_golden.mem"}, golden_mem);
        output_file = $fopen(rtl_out_path, "w");
        if (output_file == 0) $fatal(1, "cannot open %s", rtl_out_path);

        repeat (4) @(posedge clk); rst_n = 1;

        // AXI-Lite register map from axi_lite_regs.v.
        invalid_config_test = $test$plusargs("INVALID_CONFIG");
        axi_write(6'h08, invalid_config_test ? N_Q_HEADS-1 : N_Q_HEADS); axi_write(6'h0c, N_KV_HEADS);
        axi_write(6'h10, HEAD_DIM);  axi_write(6'h14, TILE_DIM);
        axi_write(6'h18, SEQ_LEN);   axi_write(6'h1c, (SEQ_LEN+KV_BLOCK_LEN-1)/KV_BLOCK_LEN);
        axi_write(6'h24, 32'h00002e8b); // BF16 1/sqrt(128)
        axi_write(6'h28, SEQ_LEN-1);    // all keys are causally visible
        if (trace_enabled) $display("[%0t] STIMULUS: config registers written, pulsing start", $time);
        axi_write(6'h00, 32'h00000001);
        if (invalid_config_test) begin
            repeat (5) @(posedge clk);
            if (!dut.config_error) $fatal(1, "FAIL: invalid configuration was not rejected");
            $fclose(output_file);
            $display("PASS: invalid configuration rejected");
            $finish;
        end

        for (group_idx = 0; group_idx < N_KV_HEADS; group_idx = group_idx + 1) begin
            if (trace_enabled) $display("[%0t] STIMULUS: streaming group %0d", $time, group_idx);
            send_group(group_idx);
        end
        if (trace_enabled) $display("[%0t] STIMULUS: all groups streamed, waiting for rows_seen==%0d", $time, N_Q_HEADS);

        wait (rows_seen == N_Q_HEADS);
        $fclose(output_file);
        if (errors != 0) $fatal(1, "FAIL: %0d elements exceeded %0d ULP", errors, MAX_ULP);
        $display("PASS: %0d GQA output rows matched golden within %0d ULP", rows_seen, MAX_ULP);
        $finish;
    end

    // -------------------------------------------------------------------
    // Watchdog -- prints a full diagnostic snapshot on timeout instead of
    // a bare "TIMEOUT" message, since the FSM/handshake trace above may
    // scroll past thousands of lines by the time a stall is reached.
    // -------------------------------------------------------------------
    initial begin
        repeat (MAX_CYCLES) @(posedge clk);
        $display("=====================================================");
        $display("TIMEOUT DIAGNOSTIC SNAPSHOT at %0t", $time);
        $display("  rows_seen        = %0d (expected %0d)", rows_seen, N_Q_HEADS);
        $display("  controller state = %s", state_name(dut.u_ctrl.state));
        $display("  config_error     = %b", dut.config_error);
        $display("  busy             = %b", dut.busy);
        $display("  q_row_req/ack    = %b / %b", dut.q_row_req, dut.q_row_ack);
        $display("  kv_pos_req/ack   = %b / %b", dut.kv_pos_req, dut.kv_pos_ack);
        $display("  mac_busy         = %b", dut.mac_busy);
        $display("  lane_acc_valid   = %b (row_done = AND of these)", dut.lane_acc_valid);
        $display("  lane0 score/probs = %b / %b (last=%b)",
                 dut.Q_LANE[0].block_score_valid, dut.Q_LANE[0].probs_valid,
                 dut.Q_LANE[0].probs_row_last);
        $display("  lane0 accumulator: stage1=%b stage2=%b final_pending=%b final_committed=%b inv_pending=%b norm=%b",
                 dut.Q_LANE[0].u_acc.stage1_busy, dut.Q_LANE[0].u_acc.stage2_busy,
                 dut.Q_LANE[0].u_acc.final_tile_pending, dut.Q_LANE[0].u_acc.final_acc_committed,
                 dut.Q_LANE[0].u_acc.inv_pending, dut.Q_LANE[0].u_acc.norm_busy);
        $display("  lane0 softmax: row_last_pending=%b l_new_valid=%b inv_l_valid=%b",
                 dut.Q_LANE[0].u_softmax.row_last_pending, dut.Q_LANE[0].u_softmax.l_new_valid,
                 dut.Q_LANE[0].inv_l_valid);
        $display("  ingest_q_wr_valid  = %b", dut.ingest_q_wr_valid);
        $display("  ingest_kv_wr_valid = %b", dut.ingest_kv_wr_valid);
        $display("  s_axis_tvalid/tready = %b / %b", s_valid, s_ready);
        $display("=====================================================");
        $fatal(1, "TIMEOUT after %0d cycles (rows seen=%0d)", MAX_CYCLES, rows_seen);
    end

endmodule
