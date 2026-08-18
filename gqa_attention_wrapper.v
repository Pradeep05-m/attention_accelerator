// =============================================================================
// gqa_attention_wrapper.v
// Top-level integration: axi_lite_regs (config/control) + axi_stream_ingest/
// egress (data I/O) + tile_scheduler (BRAM tile buffering) + gqa_controller
// (sequencing FSM, one K/V position at a time) + gqa_kv_broadcast (one
// KV-head's current K/V position fanned out to GROUP_SIZE Q-head lanes) +
// GROUP_SIZE parallel lanes of {bf16_dot_product_mac -> scale_unit ->
// score_tile_buffer -> flash_softmax} and {v_row_assembler -> v_tile_buffer
// -> output_accumulator}, one lane per Q-head sharing this KV-head.
//
// KV_BLOCK_LEN (K/V positions per flash-attention block) is intentionally
// a separate parameter from TILE_DIM (MAC lanes across HEAD_DIM) -- an
// earlier revision of this wrapper conflated the two, which is why
// score_tile_buffer/v_tile_buffer exist: they convert the one-scalar-per-
// K-position stream from the MAC/scale/V-assembly path into the
// KV_BLOCK_LEN-wide blocks flash_softmax/output_accumulator actually need.
//
// Given this is generated without the exact port widths of the 7 reused
// BF16 primitives, double check bit-widths/latencies against the real
// bf16_mac/bf16_adder/bf16_exp/reciprocal_unit source before synthesis --
// see bf16_dot_product_mac.v's MAC_LATENCY note.
//
// REV 2 wiring fixes (see tile_scheduler.v / axi_stream_ingest.v REV 2
// headers for the underlying bug this closes):
//   1. u_ctrl's mac_pass_ack/mac_busy inputs were previously left
//      UNCONNECTED -- the entire pass-pacing handshake gqa_controller.v
//      and bf16_dot_product_mac.v were built around never actually
//      reached the controller. Now tapped from Q_LANE[0]'s MAC instance
//      (representative of all GROUP_SIZE lanes -- see that instance's
//      port comments).
//   2. tile_scheduler.v now taps mac_pass_first/mac_pass_ack directly and
//      streams one TILE_DIM-wide pass window per K-position pass instead
//      of one static chunk per whole K-position/row.
//   3. u_bcast's load_pass_first/load_pass_last and u_mac's/u_v_asm's
//      pass_first/pass_last no longer come from gqa_controller's raw
//      (single-cycle, undelayed) mac_pass_first/mac_pass_last. Those
//      pulses would already be low by the time gqa_kv_broadcast's
//      registered valid_in/k_slice_out/v_slice_out actually appear one
//      cycle later, which would corrupt the accumulator reset on pass 0
//      and silently drop the final-pass completion strobe. pass_first/
//      pass_last now flow through tile_scheduler's pass_first_out/
//      pass_last_out -> gqa_kv_broadcast's already-delay-matched
//      slice_pass_first/slice_pass_last (previously computed but unused).
// =============================================================================
`timescale 1ns / 1ps

module gqa_attention_wrapper #(
    parameter N_Q_HEADS    = 32,
    parameter N_KV_HEADS   = 8,
    // Make a direct RTL synthesis safe for the PYNQ-Z2 too.  A throughput
    // experiment can override this to four, but four complete BF16 lanes
    // cannot be placed on an XC7Z020.
    parameter GROUP_SIZE   = 1,
    parameter HEAD_DIM     = 128,
    parameter TILE_DIM     = 16,   // MAC lanes across HEAD_DIM
    parameter KV_BLOCK_LEN = 16,   // K/V positions per flash-attention block
    parameter DATA_WIDTH   = 16,
    parameter ACC_WIDTH    = HEAD_DIM, // output_accumulator holds one full HEAD_DIM row per lane

    // This is deliberately independent of TILE_DIM.  TILE_DIM is fixed at
    // 16 by the 256-bit AXI DMA interface, whereas the weighted-V engine is
    // entirely internal.  One lane reuses the BF16 operators across the
    // 128 output elements and avoids instantiating a large 16x16 P*V array.
    // This is the fit-oriented PYNQ-Z2 default: a full-width value engine
    // exceeds the XC7Z020 fabric even after GQA head serialization.
    // It must divide ACC_WIDTH.
    parameter ACC_TILE_DIM = 1
) (
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:S_AXIS:M_AXIS, ASSOCIATED_RESET rst_n" *) input wire clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *) input wire rst_n,

    // AXI-Lite (S_AXI) -- config/control
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *) input wire [5:0]  s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *) input wire [2:0]  s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *) input wire        s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *) output wire        s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *) input wire [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *) input wire [3:0]  s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *) input wire        s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *) output wire        s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *) output wire [1:0]  s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *) output wire        s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *) input wire        s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *) input wire [5:0]  s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *) input wire [2:0]  s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *) input wire        s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *) output wire        s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *) output wire [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *) output wire [1:0]  s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *) output wire        s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *) input wire        s_axi_rready,

    // AXI-Stream slave -- Q/K/V ingest
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *) input wire [TILE_DIM*DATA_WIDTH-1:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *) input wire [TILE_DIM*DATA_WIDTH/8-1:0] s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *) input wire                            s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *) output wire                           s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *) input wire                            s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TUSER" *) input wire [1:0]                      s_axis_tuser,

    // AXI-Stream master -- output egress
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *) output wire [TILE_DIM*DATA_WIDTH-1:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *) output wire [TILE_DIM*DATA_WIDTH/8-1:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *) output wire                           m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *) input wire                            m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *) output wire                           m_axis_tlast
);

    genvar gi;

    // ---------------- Config/control ----------------
    wire        start_pulse, soft_reset, busy, done, error_flag, controller_busy;
    wire [31:0] n_q_heads_r, n_kv_heads_r, head_dim_r, tile_dim_r, seq_len_r, n_kv_tiles_r;
    wire [15:0] scale_factor_r, query_position_r; // BF16 scale and absolute Q RoPE position

    // AXI-Lite stays alive so software can recover after a soft reset; all
    // datapath state, stream buffers, and the controller are reset together.
    wire core_rst_n = rst_n & ~soft_reset;

    axi_lite_regs #(.DEFAULT_TILE_DIM(TILE_DIM)) u_regs (
        .s_axi_aclk    (clk),
        .s_axi_aresetn (rst_n),
        .s_axi_awaddr  (s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata   (s_axi_wdata),  .s_axi_wstrb(s_axi_wstrb),     .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp   (s_axi_bresp),  .s_axi_bvalid(s_axi_bvalid),   .s_axi_bready(s_axi_bready),
        .s_axi_araddr  (s_axi_araddr), .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata   (s_axi_rdata),  .s_axi_rresp(s_axi_rresp),     .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .start_pulse (start_pulse), .soft_reset(soft_reset),
        .busy_in (busy), .done_in(done), .error_in(error_flag),
        .n_q_heads(n_q_heads_r), .n_kv_heads(n_kv_heads_r),
        .head_dim(head_dim_r), .tile_dim(tile_dim_r),
        .seq_len(seq_len_r), .n_kv_tiles(n_kv_tiles_r),
        .scale_factor(scale_factor_r), .query_position(query_position_r)
    );

    // All accelerator stream beats are full 256-bit words.  TKEEP is carried
    // for AXI DMA interoperability but does not affect the fixed-width core.
    assign m_axis_tkeep = {TILE_DIM*DATA_WIDTH/8{1'b1}};

    // ---------------- Ingest / tile buffering ----------------
    wire ingest_q_wr_valid, ingest_q_ready;
    wire ingest_kv_wr_valid, ingest_kv_ready;
    wire [HEAD_DIM*DATA_WIDTH-1:0] ingest_q_data, ingest_k_data, ingest_v_data;

    axi_stream_ingest #(.TILE_DIM(TILE_DIM), .HEAD_DIM(HEAD_DIM), .DATA_WIDTH(DATA_WIDTH)) u_ingest (
        .clk(clk), .rst_n(core_rst_n),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast), .s_axis_tuser(s_axis_tuser),
        .ingest_q_wr_valid(ingest_q_wr_valid), .ingest_q_data(ingest_q_data), .ingest_q_ready(ingest_q_ready),
        .ingest_kv_wr_valid(ingest_kv_wr_valid),
        .ingest_k_data(ingest_k_data), .ingest_v_data(ingest_v_data), .ingest_kv_ready(ingest_kv_ready),
        .row_boundary()
    );

    // Declare controller outputs before any instance references them.  This
    // avoids an implicit-net warning from Verilog at u_sched.
    wire kv_tile_req, kv_tile_ack, q_row_req, q_row_ack, q_sched_ack;
    wire row_advance;
    wire mac_pass_first;
    wire block_first, block_last;
    wire kv_pos_req, kv_pos_ack;
    wire [15:0] kv_pos_idx;
    wire kv_row_last;
    wire score_block_first, score_row_first, score_row_last, score_key_valid;
    wire score_tag_empty, score_tag_full;
    wire score_valid_lane0;
    wire [TILE_DIM*DATA_WIDTH-1:0] sched_q_slice, sched_k_slice, sched_v_slice;
    wire [GROUP_SIZE*TILE_DIM*DATA_WIDTH-1:0] group_q_slices;
    wire [$clog2(HEAD_DIM/TILE_DIM)-1:0] sched_pass_idx;
    wire slice_valid, sched_pass_first, sched_pass_last;
    wire out_row_valid, out_row_store_ack;
    wire [ACC_WIDTH*DATA_WIDTH-1:0] out_row_in;
    wire [GROUP_SIZE-1:0] lane_acc_valid;
    wire [GROUP_SIZE-1:0] lane_tile_consumed;
    wire [GROUP_SIZE-1:0] lane_acc_busy;
    wire [GROUP_SIZE*ACC_WIDTH*DATA_WIDTH-1:0] lane_acc_rows;
    wire egress_valid, egress_ready;
    wire [ACC_WIDTH*DATA_WIDTH-1:0] egress_data;
    // While a Q/K row is being captured by RoPE, acknowledge each raw
    // scheduler pass.  The controller subsequently waits for rope_busy and
    // the MAC to drain before it loads the next K position.
    wire rope_capture_ack = slice_valid;
    wire mac_busy;

    gqa_output_collector #(.GROUP_SIZE(GROUP_SIZE), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) u_out_collect (
        .clk(clk), .rst_n(core_rst_n), .lane_valid(lane_acc_valid), .lane_rows(lane_acc_rows),
        .out_valid(out_row_valid), .out_row(out_row_in), .out_ready(out_row_store_ack)
    );

    tile_scheduler #(.TILE_DIM(TILE_DIM), .HEAD_DIM(HEAD_DIM), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) u_sched (
        .clk(clk), .rst_n(core_rst_n),
        .kv_tile_req(kv_tile_req), .kv_tile_ack(kv_tile_ack),
        // Q rows are owned by q_group_buffer; retain this legacy scheduler
        // bank only for K/V scheduling.
        .q_row_req(1'b0), .q_row_ack(q_sched_ack),
        .mac_pass_first(mac_pass_first), .mac_pass_ack(rope_capture_ack),
        .q_slice_out(sched_q_slice), .k_slice_out(sched_k_slice), .v_slice_out(sched_v_slice),
        .slice_valid(slice_valid), .pass_first_out(sched_pass_first), .pass_last_out(sched_pass_last), .pass_idx_out(sched_pass_idx),
        .out_row_valid(out_row_valid), .out_row_in(out_row_in), .out_row_store_ack(out_row_store_ack),
        .ingest_q_wr_valid(1'b0), .ingest_q_data({(HEAD_DIM*DATA_WIDTH){1'b0}}), .ingest_q_ready(),
        .ingest_kv_wr_valid(ingest_kv_wr_valid),
        .ingest_k_data(ingest_k_data), .ingest_v_data(ingest_v_data), .ingest_kv_ready(ingest_kv_ready),
        .egress_valid(egress_valid), .egress_data(egress_data), .egress_ready(egress_ready)
    );

    q_group_buffer #(.GROUP_SIZE(GROUP_SIZE), .HEAD_DIM(HEAD_DIM), .TILE_DIM(TILE_DIM), .DATA_WIDTH(DATA_WIDTH)) u_q_group (
        .clk(clk), .rst_n(core_rst_n),
        .q_wr_valid(ingest_q_wr_valid), .q_wr_data(ingest_q_data), .q_wr_ready(ingest_q_ready),
        .q_req(q_row_req), .q_ack(q_row_ack), .q_release(row_advance),
        .pass_idx(sched_pass_idx), .q_slices(group_q_slices)
    );

    // This is the deployed AXIS egress.  It serializes the internal 2048-bit
    // row into eight 256-bit DMA beats on PYNQ-Z2; axis_row_serializer is
    // retained only as a stand-alone compatibility/test module.
    axi_stream_egress #(.HEAD_DIM(ACC_WIDTH), .TILE_DIM(TILE_DIM), .DATA_WIDTH(DATA_WIDTH)) u_egress (
        .clk(clk), .rst_n(core_rst_n),
        .row_valid(egress_valid), .row_data(egress_data), .row_ready(egress_ready),
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast)
    );

    // ---------------- Sequencing FSM ----------------

    // This RTL is fixed to the Llama-3 8B GQA geometry.  Reject a bad
    // runtime programming sequence instead of silently producing a tensor
    // with a different layout.  A zero sequence or zero block count would
    // otherwise underflow controller comparisons and never complete.
    wire config_error = start_pulse &&
                        ((n_q_heads_r != N_Q_HEADS) ||
                         (n_kv_heads_r != N_KV_HEADS) ||
                         (head_dim_r != HEAD_DIM) ||
                         (tile_dim_r != TILE_DIM) ||
                         (seq_len_r == 0) ||
                         (n_kv_tiles_r != ((seq_len_r + KV_BLOCK_LEN - 1) / KV_BLOCK_LEN)) ||
                         (scale_factor_r == 16'h0000));

    gqa_controller #(
        .N_Q_HEADS(N_Q_HEADS), .N_KV_HEADS(N_KV_HEADS), .GROUP_SIZE(GROUP_SIZE),
        .HEAD_DIM(HEAD_DIM), .TILE_DIM(TILE_DIM), .KV_BLOCK_LEN(KV_BLOCK_LEN)
    ) u_ctrl (
        .clk(clk), .rst_n(core_rst_n),
        .start(start_pulse & ~config_error), .busy(controller_busy), .done(done),
        .seq_len(seq_len_r[15:0]), .n_kv_blocks(n_kv_tiles_r[15:0]),
        .kv_pos_req(kv_pos_req), .kv_pos_ack(kv_pos_ack),
        .block_first(block_first), .block_last(block_last),
        .q_row_req(q_row_req), .q_row_ack(q_row_ack),
        .q_head_idx(), .kv_head_idx(), .kv_pos_idx(kv_pos_idx), .kv_row_last(kv_row_last),
        .mac_pass_first(mac_pass_first), .mac_pass_last(),
        .mac_pass_ack(rope_capture_ack), .mac_busy(mac_busy),
        .block_consumed(lane_tile_consumed[0]), .row_done(&lane_acc_valid),
        .row_advance(row_advance)
    );

    // Keep the AXI-Lite busy bit asserted until the final value tile has
    // drained, and consume output_accumulator.busy rather than leaving the
    // port unconnected in synthesis.
    assign busy = controller_busy | (|lane_acc_busy);

    // kv_pos_req/ack reuse tile_scheduler's existing kv_tile_req/ack handshake
    // (one K/V position's HEAD_DIM vector per request, instead of one whole
    // multi-position tile as originally named).
    assign kv_tile_req = kv_pos_req;
    assign kv_pos_ack  = kv_tile_ack;

    // A score emerges after RoPE, the dot-product reduction and scaling;
    // by then the controller may already be requesting a later K position.
    // Preserve the per-position tags in order instead of sampling live FSM
    // signals at score_valid.  A depth of four is ample because the
    // controller does not issue the next K request until the current MAC
    // has drained; the FIFO also makes that ordering assumption explicit.
    score_tag_fifo #(.DEPTH(4)) u_score_tags (
        .clk(clk), .rst_n(core_rst_n),
        .push(kv_pos_ack),
        .push_block_first(block_first),
        .push_row_first(kv_pos_idx == 16'd0),
        .push_row_last(kv_row_last),
        .push_key_valid((kv_pos_idx < seq_len_r[15:0]) && (kv_pos_idx <= query_position_r)),
        .pop(score_valid_lane0),
        .block_first(score_block_first),
        .row_first(score_row_first),
        .row_last(score_row_last),
        .key_valid(score_key_valid),
        .empty(score_tag_empty), .full(score_tag_full)
    );

    assign error_flag = config_error;

    // ---------------- Rotary position embedding ----------------
    // Llama's rotate_half convention pairs element i with i+HEAD_DIM/2.
    // Consequently RoPE must capture a complete row and replay it; it cannot
    // be inserted as a combinational operation on a TILE_DIM-wide MAC pass.
    // Q uses the host-programmed query position. K uses the controller's
    // absolute cache position. V intentionally bypasses RoPE.
    wire rope_k_valid, rope_k_first, rope_k_last, rope_k_busy;
    wire [GROUP_SIZE-1:0] rope_q_valid, rope_q_first, rope_q_last, rope_q_busy;
    wire [GROUP_SIZE*TILE_DIM*DATA_WIDTH-1:0] rope_q_slices;
    wire [TILE_DIM*DATA_WIDTH-1:0] rope_k_slice;
    wire rope_replay_ready;

    rope_unit #(.HEAD_DIM(HEAD_DIM), .TILE_DIM(TILE_DIM), .DATA_WIDTH(DATA_WIDTH)) u_rope_k (
        .clk(clk), .rst_n(core_rst_n),
        .in_valid(slice_valid), .in_pass_first(sched_pass_first), .in_pass_last(sched_pass_last),
        .pos(kv_pos_idx), .chunk_in(sched_k_slice),
        .out_valid(rope_k_valid), .out_pass_first(rope_k_first), .out_pass_last(rope_k_last),
        .chunk_out(rope_k_slice), .out_ready(rope_replay_ready), .busy(rope_k_busy)
    );

    // ---------------- KV broadcast to GROUP_SIZE Q-head lanes ----------------
    wire [GROUP_SIZE-1:0] lane_consumer_ready;
    wire [TILE_DIM*DATA_WIDTH-1:0] bcast_k, bcast_v;
    wire bcast_pass_first, bcast_pass_last;
    wire [GROUP_SIZE-1:0] bcast_valid;

    gqa_kv_broadcast #(.GROUP_SIZE(GROUP_SIZE), .TILE_DIM(TILE_DIM), .DATA_WIDTH(DATA_WIDTH)) u_bcast (
        .clk(clk), .rst_n(core_rst_n),
        .load_valid(slice_valid),
        .k_slice_in(sched_k_slice), .v_slice_in(sched_v_slice),
        .load_pass_first(sched_pass_first), .load_pass_last(sched_pass_last),
        .consumer_ready(lane_consumer_ready),
        .k_slice_out(bcast_k), .v_slice_out(bcast_v),
        .slice_pass_first(bcast_pass_first), .slice_pass_last(bcast_pass_last),
        .consumer_valid(bcast_valid)
    );

    // ---------------- GROUP_SIZE parallel Q-head compute lanes ----------------
    generate
        for (gi = 0; gi < GROUP_SIZE; gi = gi + 1) begin : Q_LANE

            wire mac_valid_out, mac_result_valid_final;
            wire [DATA_WIDTH-1:0] mac_result;

            assign lane_consumer_ready[gi] = 1'b1; // simple always-ready; refine with real backpressure

            wire lane_pass_ack, lane_busy;

            bf16_dot_product_mac #(
                .HEAD_DIM(HEAD_DIM), .TILE_DIM(TILE_DIM), .DATA_WIDTH(DATA_WIDTH)
            ) u_mac (
                .clk(clk), .rst_n(core_rst_n),
                .valid_in(rope_q_valid[gi] & rope_k_valid & rope_replay_ready),
                .pass_first(rope_q_first[gi] & rope_k_first), .pass_last(rope_q_last[gi] & rope_k_last),
                .a_vec(rope_q_slices[(gi+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH]), // RoPE-applied Q slice
                .b_vec(rope_k_slice), // RoPE-applied K slice
                .valid_out(mac_valid_out),
                .result_valid_final(mac_result_valid_final),
                .result(mac_result),
                .pass_ack(lane_pass_ack),
                .busy(lane_busy)
            );

            // All GROUP_SIZE lanes share identical valid_in/pass_first/
            // pass_last cadence and fixed latency (see bf16_dot_product_
            // mac.v's port comments), so lane 0's pass_ack/busy is
            // representative.  The controller's pass pacing is driven by
            // RoPE row capture; mac_busy is still included in its drain wait.
            if (gi == 0) begin
                assign mac_busy     = lane_busy | (|rope_q_busy) | rope_k_busy;
                assign rope_replay_ready = !lane_busy | lane_pass_ack;
            end

            rope_unit #(.HEAD_DIM(HEAD_DIM), .TILE_DIM(TILE_DIM), .DATA_WIDTH(DATA_WIDTH)) u_rope_q (
                .clk(clk), .rst_n(core_rst_n),
                .in_valid(slice_valid), .in_pass_first(sched_pass_first), .in_pass_last(sched_pass_last),
                .pos(query_position_r),
                .chunk_in(group_q_slices[(gi+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH]),
                .out_valid(rope_q_valid[gi]), .out_pass_first(rope_q_first[gi]), .out_pass_last(rope_q_last[gi]),
                .chunk_out(rope_q_slices[(gi+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH]),
                .out_ready(rope_replay_ready), .busy(rope_q_busy[gi])
            );

            // Scale the raw Q.K dot product by 1/sqrt(HEAD_DIM) before it
            // reaches softmax, per standard scaled-dot-product attention.
            wire scaled_valid;
            wire [DATA_WIDTH-1:0] scaled_score;

            scale_unit u_scale (
                .clk             (clk),
                // Keep this pipeline in the datapath reset domain.  Using
                // the external reset here left a stale scaled_valid pulse
                // alive after an AXI-Lite soft reset.
                .rst_n           (core_rst_n),
                .valid_in        (mac_result_valid_final),
                .attention_score (mac_result),
                .scale_factor    (scale_factor_r),
                .valid_out       (scaled_valid),
                .scaled_score    (scaled_score)
            );

            if (gi == 0) begin
                assign score_valid_lane0 = scaled_valid;
            end

            // --- Score buffer: collect KV_BLOCK_LEN scalar scores (one per
            // K position) into the block flash_softmax actually consumes.
            wire block_score_valid;
            wire [KV_BLOCK_LEN*DATA_WIDTH-1:0] block_scores;
            wire [KV_BLOCK_LEN-1:0] block_score_mask;
            // Keep the delayed score-buffer row tags distinct from the
            // controller/FIFO tags above.  Reusing the same names here made
            // key_row_first/key_row_last feed the buffer's own outputs,
            // leaving the real row-boundary tags disconnected and preventing
            // output_accumulator from ever completing a row.
            wire score_buf_row_first, score_buf_row_last;

            score_tile_buffer #(.KV_BLOCK_LEN(KV_BLOCK_LEN), .DATA_WIDTH(DATA_WIDTH)) u_score_buf (
                .clk(clk), .rst_n(core_rst_n),
                .score_valid(scaled_valid),
                .key_block_first(score_block_first), .key_row_first(score_row_first), .key_row_last(score_row_last),
                .key_valid(score_key_valid),
                .score_in(scaled_score),
                .block_valid(block_score_valid), .row_first_tile(score_buf_row_first), .row_last_tile(score_buf_row_last),
                .block_out(block_scores), .block_valid_mask(block_score_mask)
            );

            // --- V path: assemble this K position's full HEAD_DIM V row
            // from the N_PASSES TILE_DIM chunks, then buffer KV_BLOCK_LEN
            // such rows for output_accumulator.
            wire v_row_valid;
            wire [HEAD_DIM*DATA_WIDTH-1:0] v_row;

            v_row_assembler #(.HEAD_DIM(HEAD_DIM), .TILE_DIM(TILE_DIM), .DATA_WIDTH(DATA_WIDTH)) u_v_asm (
                .clk(clk), .rst_n(core_rst_n),
                .chunk_valid(bcast_valid[gi]),
                .pass_first(bcast_pass_first), .pass_last(bcast_pass_last),
                .chunk_in(bcast_v),
                .row_valid(v_row_valid),
                .row_out(v_row)
            );

            wire block_v_valid;
            wire [$clog2(HEAD_DIM/ACC_TILE_DIM)-1:0] v_read_pass;
            wire [KV_BLOCK_LEN*ACC_TILE_DIM*DATA_WIDTH-1:0] v_slice;

            v_tile_buffer #(
                .KV_BLOCK_LEN(KV_BLOCK_LEN), .HEAD_DIM(HEAD_DIM), .DATA_WIDTH(DATA_WIDTH),
                .TILE_DIM(ACC_TILE_DIM), .WRITE_TILE_DIM(TILE_DIM)
            ) u_v_buf (
                .clk(clk), .rst_n(core_rst_n),
                .row_valid(v_row_valid),
                // Block assembly is defined by its own fill counter.  The
                // controller's one-cycle block_first pulse is not aligned
                // with this delayed V row, so do not sample it here.
                .block_first(1'b0),
                .row_in(v_row),
                .block_valid(block_v_valid),
                .read_pass(v_read_pass), .read_data(v_slice)
            );

            wire probs_valid, probs_row_first, probs_row_last, rescale_valid, inv_l_valid;
            wire [KV_BLOCK_LEN*DATA_WIDTH-1:0] probs_out;
            wire [DATA_WIDTH-1:0] rescale_alpha, inv_l;

            flash_softmax #(.DATA_WIDTH(DATA_WIDTH), .KV_BLOCK_LEN(KV_BLOCK_LEN)) u_softmax (
                .clk(clk), .rst_n(core_rst_n),
                .tile_valid(block_score_valid),
                .row_first_tile(score_buf_row_first),
                .row_last_tile(score_buf_row_last),
                .scores_in(block_scores),
                .score_valid_mask(block_score_mask),
                .probs_valid(probs_valid), .probs_out(probs_out), .probs_row_first(probs_row_first), .probs_row_last(probs_row_last),
                .rescale_valid(rescale_valid), .rescale_alpha(rescale_alpha),
                .inv_l_valid(inv_l_valid), .inv_l(inv_l)
            );

            wire acc_out_valid;
            wire [ACC_WIDTH*DATA_WIDTH-1:0] acc_out_row;
            assign lane_acc_valid[gi] = acc_out_valid;
            assign lane_acc_rows[(gi+1)*ACC_WIDTH*DATA_WIDTH-1 -: ACC_WIDTH*DATA_WIDTH] = acc_out_row;

            // NOTE: probs_valid/block_v_valid must arrive aligned (both mark
            // "this block's data is ready"). score_tile_buffer and
            // v_tile_buffer have matched fill logic so this should hold,
            // but verify the actual pipeline latencies of scale_unit vs.
            // v_row_assembler line up once real primitive timings are known.
            output_accumulator #(
                .DATA_WIDTH(DATA_WIDTH), .KV_BLOCK_LEN(KV_BLOCK_LEN),
                .ACC_WIDTH(ACC_WIDTH), .TILE_DIM(ACC_TILE_DIM)
            ) u_acc (
                .clk(clk), .rst_n(core_rst_n),
                .tile_valid(probs_valid),
                .row_first_tile(probs_row_first), .row_last_tile(probs_row_last),
                .probs_in(probs_out),
                .v_slice_in(v_slice), .v_pass_idx(v_read_pass),
                .rescale_valid(rescale_valid), .rescale_alpha(rescale_alpha),
                .inv_l_valid(inv_l_valid), .inv_l(inv_l),
                .out_valid(acc_out_valid), .out_row(acc_out_row),
                .busy(lane_acc_busy[gi]), .tile_consumed(lane_tile_consumed[gi])
            );

        end
    endgenerate

endmodule
