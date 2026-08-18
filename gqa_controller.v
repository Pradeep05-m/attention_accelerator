// =============================================================================
// gqa_controller.v
// Top-level FSM sequencing: load KV-head tile -> broadcast to group's
// GROUP_SIZE Q-heads -> dot-product passes -> scale -> flash-softmax update
// -> weighted-V accumulate -> advance to next KV-head/tile -> next Q-group.
// Drives control/valid strobes into gqa_kv_broadcast, bf16_dot_product_mac
// (x GROUP_SIZE instances, one per Q-head in gqa_attention_wrapper),
// flash_softmax, and output_accumulator. Actual datapath instances live in
// gqa_attention_wrapper.v; this module only issues control signals + status.
// =============================================================================
`timescale 1ns / 1ps

module gqa_controller #(
    parameter N_Q_HEADS   = 32,
    parameter N_KV_HEADS  = 8,
    parameter GROUP_SIZE  = N_Q_HEADS / N_KV_HEADS,
    parameter N_Q_GROUPS  = N_Q_HEADS / GROUP_SIZE,
    parameter HEAD_DIM    = 128,
    parameter TILE_DIM    = 16,       // MAC lanes across HEAD_DIM (unrelated to KV_BLOCK_LEN)
    parameter KV_BLOCK_LEN = 16,      // K/V positions per flash-attention block
    parameter N_PASSES    = HEAD_DIM / TILE_DIM,
    parameter SEQLEN_WIDTH = 16   // supports runtime-configurable sequence length
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,               // pulse from axi_lite_regs
    output reg         busy,
    output reg         done,

    input  wire [SEQLEN_WIDTH-1:0] seq_len,  // runtime sequence length (not hardcoded)
    input  wire [SEQLEN_WIDTH-1:0] n_kv_blocks, // number of KV_BLOCK_LEN-sized blocks per row

    // Handshake with tile_scheduler: request next K/V position load
    // (one full HEAD_DIM vector each, streamed as N_PASSES TILE_DIM chunks)
    output reg                       kv_pos_req,
    input  wire                      kv_pos_ack,
    output reg                       block_first,   // this K position is #0 of a new flash-attn block
    output reg                       block_last,    // this K position is the last of the current block

    // Handshake with tile_scheduler: request next Q row load
    output reg                       q_row_req,
    input  wire                      q_row_ack,
    output reg  [$clog2(N_Q_HEADS)-1:0]  q_head_idx,
    output reg  [$clog2(N_KV_HEADS)-1:0] kv_head_idx,
    // Absolute key position currently being loaded.  This is the RoPE
    // position for K if the accelerator receives an unrotated K cache.
    output wire [SEQLEN_WIDTH-1:0]         kv_pos_idx,
    output wire                              kv_row_last,

    // Dot-product pass sequencing (fed to bf16_dot_product_mac / v_row_assembler)
    output reg                        mac_pass_first,
    output reg                        mac_pass_last,

    // Backpressure from Q_LANE[0]'s bf16_dot_product_mac instance (all
    // GROUP_SIZE lanes run in lockstep off identical valid_in/pass_first/
    // pass_last cadence and share identical fixed latency, so lane 0's
    // pass_ack/busy is representative of every lane -- see
    // bf16_dot_product_mac.v's port comments).
    //   mac_pass_ack: pulses once the CURRENT pass's per-lane accumulate
    //                 has actually landed -- only safe to issue the next
    //                 pass after this, since the accumulator feedback
    //                 needs the MAC's full multi-cycle latency, not one
    //                 clock cycle.
    //   mac_busy:     high from pass_first until result_valid_final --
    //                 only safe to start the NEXT K-position's passes
    //                 once this has gone back low, since the MAC/adder-
    //                 tree instance is reused sequentially with no
    //                 internal queuing.
    input  wire                        mac_pass_ack,
    input  wire                        mac_busy,

    // Pulses after the value accumulator has read every V slice of the
    // current flash-attention block.  The next block must not overwrite the
    // single V tile buffer before this acknowledgement.
    input  wire                        block_consumed,

    // Asserted only after the final softmax/value result for this Q group
    // has been committed.  Prevents reusing per-lane accumulators early.
    input  wire                        row_done,

    // Row completion strobe to output_accumulator's normalize stage consumer
    output reg                        row_advance
);

    localparam S_IDLE       = 4'd0,
               S_LOAD_Q     = 4'd1,
               S_LOAD_KVPOS = 4'd2,
               S_MAC_PASSES = 4'd3,
               S_WAIT_DRAIN = 4'd8, // wait for mac_busy to clear before reusing the MAC for the next K-position
               S_NEXT_KPOS  = 4'd4, // advance to next K position within the current block
               S_NEXT_BLOCK = 4'd5, // advance to next flash-attention block
               S_NEXT_ROW   = 4'd6,
               S_DONE       = 4'd7,
               S_WAIT_BLOCK = 4'd9;

    reg [3:0] state;
    reg [$clog2(N_PASSES)-1:0]        pass_cnt;
    reg [SEQLEN_WIDTH-1:0]            block_cnt;   // which KV_BLOCK_LEN-sized block within the row
    reg [$clog2(KV_BLOCK_LEN):0]      kpos_cnt;    // which K position within the current block
    reg [$clog2(N_Q_HEADS)-1:0]  q_idx;
    reg [$clog2(N_KV_HEADS)-1:0] kv_idx;
    reg                          block_consumed_pending;

    assign kv_pos_idx = block_cnt * KV_BLOCK_LEN + kpos_cnt;
    assign kv_row_last = (block_cnt == n_kv_blocks - 1'b1) &&
                         (kpos_cnt == KV_BLOCK_LEN - 1'b1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            busy             <= 1'b0;
            done             <= 1'b0;
            kv_pos_req       <= 1'b0;
            block_first      <= 1'b0;
            block_last       <= 1'b0;
            q_row_req        <= 1'b0;
            mac_pass_first   <= 1'b0;
            mac_pass_last    <= 1'b0;
            row_advance      <= 1'b0;
            pass_cnt         <= {$clog2(N_PASSES){1'b0}};
            block_cnt        <= {SEQLEN_WIDTH{1'b0}};
            kpos_cnt         <= {($clog2(KV_BLOCK_LEN)+1){1'b0}};
            q_idx            <= {$clog2(N_Q_HEADS){1'b0}};
            kv_idx           <= {$clog2(N_KV_HEADS){1'b0}};
            block_consumed_pending <= 1'b0;
        end else begin
            if (block_consumed) block_consumed_pending <= 1'b1;

            // default de-asserts (single-cycle strobes)
            kv_pos_req     <= 1'b0;
            q_row_req      <= 1'b0;
            row_advance    <= 1'b0;
            done           <= 1'b0;
            block_first    <= 1'b0;
            block_last     <= 1'b0;
            mac_pass_first <= 1'b0;
            mac_pass_last  <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy      <= 1'b1;
                        q_idx     <= {$clog2(N_Q_HEADS){1'b0}};
                        kv_idx    <= {$clog2(N_KV_HEADS){1'b0}};
                        block_cnt <= {SEQLEN_WIDTH{1'b0}};
                        state     <= S_LOAD_Q;
                    end
                end

                S_LOAD_Q: begin
                    q_row_req  <= 1'b1;
                    // One scheduler iteration is a GQA group, not one
                    // individual Q head.  The wrapper supplies GROUP_SIZE
                    // Q rows to its parallel lanes from q_group_buffer.
                    q_head_idx  <= q_idx * GROUP_SIZE;
                    // When GROUP_SIZE is one (the PYNQ time-multiplexed
                    // configuration), four consecutive Q heads share each
                    // KV head.  Keep the same mapping as the parallel GQA
                    // configuration instead of advancing the KV head per Q.
                    kv_head_idx <= q_idx / (N_Q_HEADS / N_KV_HEADS);
                    if (q_row_ack) begin
                        kpos_cnt <= {($clog2(KV_BLOCK_LEN)+1){1'b0}};
                        state    <= S_LOAD_KVPOS;
                    end
                end

                // Request one K/V position's HEAD_DIM vector (streamed as
                // N_PASSES TILE_DIM chunks by gqa_kv_broadcast/v_row_assembler).
                S_LOAD_KVPOS: begin
                    kv_pos_req  <= 1'b1;
                    block_first <= (kpos_cnt == {($clog2(KV_BLOCK_LEN)+1){1'b0}});
                    block_last  <= (kpos_cnt == KV_BLOCK_LEN - 1'b1);
                    if (kv_pos_ack) begin
                        pass_cnt       <= {$clog2(N_PASSES){1'b0}};
                        mac_pass_first <= 1'b1; // issue pass 0 immediately on entry to S_MAC_PASSES
                        mac_pass_last  <= (N_PASSES == 1);
                        state          <= S_MAC_PASSES;
                    end
                end

                // Run the N_PASSES MAC/V-assembly passes for this one K
                // position, ONE PASS PER mac_pass_ack -- NOT one per clock
                // cycle. The MAC's accumulator feedback (lane_acc) needs
                // its full multi-cycle latency to land before the next
                // pass reads it; issuing every cycle would silently
                // accumulate against stale/zero values for most passes.
                // mac_pass_first/last also drive v_row_assembler so the K
                // and V HEAD_DIM vectors for this K position are built
                // from the same TILE_DIM-wide chunk stream -- they must
                // stay paced identically.
                S_MAC_PASSES: begin
                    if (mac_pass_ack) begin
                        if (pass_cnt == N_PASSES - 1'b1) begin
                            // Last pass's accumulate has landed; now wait
                            // for the adder-tree to fully drain before
                            // this MAC instance is safe to reuse.
                            state <= S_WAIT_DRAIN;
                        end else begin
                            pass_cnt       <= pass_cnt + 1'b1;
                            mac_pass_first <= 1'b0;
                            mac_pass_last  <= (pass_cnt + 1'b1 == N_PASSES - 1'b1);
                        end
                    end
                end

                // Do not start the next K-position's passes until this
                // MAC instance's adder-tree has fully drained (mac_busy
                // low) -- it's reused sequentially with no internal
                // queuing, so starting early would collide in-flight
                // pipeline stages from consecutive K-positions.
                S_WAIT_DRAIN: begin
                    if (!mac_busy) begin
                        state <= S_NEXT_KPOS;
                    end
                end

                // score_tile_buffer / v_tile_buffer accumulate this K
                // position's result internally; once KV_BLOCK_LEN positions
                // have been pushed, their own block_valid fires flash_softmax
                // and output_accumulator.
                S_NEXT_KPOS: begin
                    if (kpos_cnt == KV_BLOCK_LEN - 1'b1) begin
                        state <= S_NEXT_BLOCK;
                    end else begin
                        kpos_cnt <= kpos_cnt + 1'b1;
                        state    <= S_LOAD_KVPOS;
                    end
                end

                S_NEXT_BLOCK: begin
                    if (block_cnt == n_kv_blocks - 1'b1) begin
                        state <= S_NEXT_ROW;
                    end else begin
                        state <= S_WAIT_BLOCK;
                    end
                end

                // flash_softmax is delayed behind the dot-product path.
                // Wait for output_accumulator to consume this block's V
                // slices before allowing ingress to fill the same V bank
                // with the next block.
                S_WAIT_BLOCK: begin
                    if (block_consumed || block_consumed_pending) begin
                        block_consumed_pending <= 1'b0;
                        block_cnt <= block_cnt + 1'b1;
                        kpos_cnt  <= {($clog2(KV_BLOCK_LEN)+1){1'b0}};
                        state     <= S_LOAD_KVPOS;
                    end
                end

                S_NEXT_ROW: begin
                    // All score/value pipelines are variable-latency.  Do
                    // not release the Q group or overwrite its accumulators
                    // until the output accumulator reports completion.
                    if (row_done) begin
                        row_advance <= 1'b1;
                        block_cnt <= {SEQLEN_WIDTH{1'b0}};
                        if (q_idx == N_Q_GROUPS - 1'b1) begin
                            state <= S_DONE;
                        end else begin
                            q_idx <= q_idx + 1'b1;
                            kv_idx <= kv_idx + 1'b1;
                            state <= S_LOAD_Q;
                        end
                    end
                end

                S_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

`ifdef DEBUG
  // Debug waveform dump for controller signals
  initial begin
    $dumpfile("gqa_controller_debug.vcd");
    $dumpvars(0, gqa_controller);
    $dumpvars(0, kv_pos_req);
    $dumpvars(0, block_first);
    $dumpvars(0, block_last);
    $dumpvars(0, mac_pass_first);
    $dumpvars(0, mac_pass_last);
    $dumpvars(0, mac_pass_ack);
    $dumpvars(0, mac_busy);
    $dumpvars(0, row_advance);
  end
`endif
endmodule
