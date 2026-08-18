// =============================================================================
// tile_scheduler.v  (REV 2 -- per-pass chunk streaming)
//
// REV 1 held exactly one TILE_DIM-wide chunk per kv_tile_req/q_row_req
// transaction and exposed it as a level-held slice_valid. That does not
// match what the compute datapath actually needs: bf16_dot_product_mac.v
// and v_row_assembler.v both expect a FRESH TILE_DIM-wide chunk on every
// one of the N_PASSES MAC passes that make up one K-position's HEAD_DIM
// dot product, and Q needs the same per-pass chunk stream, re-swept from
// chunk 0 for every K-position in the row (not just once per row).
//
// REV 2 fixes this: each buffered entry (Q row / K+V position) now holds
// the FULL HEAD_DIM-wide vector, and a small pass selector -- driven
// directly by gqa_controller's mac_pass_first/mac_pass_ack (tapped
// straight from bf16_dot_product_mac.v's lane-0 pass_ack per the
// handoff notes) -- muxes out the correct TILE_DIM-wide window each
// pass and pulses slice_valid for exactly one cycle per pass.
//
// REV 2 also splits Q's buffer bank management from K/V's: Q is loaded
// once per Q-row and re-read across every K-position/pass in that row
// (freed only when the NEXT q_row_req arrives), while K/V is loaded
// once per K-position and freed only after that position's FINAL pass
// has been consumed (not on the initial ack). REV 1 incorrectly shared
// one read_ptr/buf_full array across Q, K, and V, so any kv_tile_ack
// would also flip the bank Q was being read from mid-row -- that no
// longer happens here.
// =============================================================================
`timescale 1ns / 1ps

module tile_scheduler #(
    parameter TILE_DIM   = 16,        // MAC-lane width consumed per pass
    parameter HEAD_DIM   = 128,       // full Q/K/V vector width per row/position
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 16,
    parameter N_PASSES   = HEAD_DIM / TILE_DIM,
    parameter PASS_BITS  = (N_PASSES <= 1) ? 1 : $clog2(N_PASSES)
) (
    input  wire clk,
    input  wire rst_n,

    // --- Requests from gqa_controller ---
    input  wire        kv_tile_req,
    output reg          kv_tile_ack,
    input  wire        q_row_req,
    output reg          q_row_ack,

    // --- Per-pass pacing, tapped from gqa_controller (which itself gets
    //     mac_pass_ack from bf16_dot_product_mac.v's lane 0) ---
    // mac_pass_first: pass 0 of the current K-position's sweep -- resets
    //                 the chunk selector back to window 0.
    // mac_pass_ack:   this pass's accumulate has landed -- advances the
    //                 chunk selector to the next window, and (when the
    //                 pass that just landed was the final one) frees the
    //                 K/V bank for reuse.
    input  wire         mac_pass_first,
    input  wire         mac_pass_ack,

    // --- Q/K/V pass-window outputs to compute datapath: one TILE_DIM-wide
    //     slice of the buffered HEAD_DIM row, selected by the internal
    //     pass counter. slice_valid/pass_first_out/pass_last_out are a
    //     matched one-cycle-pulse triple (see note at bottom of file on
    //     why gqa_kv_broadcast must use pass_first_out/pass_last_out --
    //     not gqa_controller's raw mac_pass_first/mac_pass_last -- to
    //     drive its own load_pass_first/load_pass_last). ---
    output wire [TILE_DIM*DATA_WIDTH-1:0] q_slice_out,
    output wire [TILE_DIM*DATA_WIDTH-1:0] k_slice_out,
    output wire [TILE_DIM*DATA_WIDTH-1:0] v_slice_out,
    output reg                            slice_valid,
    output reg                            pass_first_out,
    output reg                            pass_last_out,
    output wire [PASS_BITS-1:0]           pass_idx_out,

    // --- Output row store request from output_accumulator ---
    input  wire                          out_row_valid,
    input  wire [ACC_WIDTH*DATA_WIDTH-1:0] out_row_in,
    output wire                            out_row_store_ack,

    // --- Ingest side: independent Q-row and K/V-position completion
    //     events from axi_stream_ingest.v, each carrying one fully
    //     assembled HEAD_DIM-wide row (see that file for how the
    //     N_PASSES TILE_DIM-wide beats get assembled into these). ---
    input  wire                           ingest_q_wr_valid,
    input  wire [HEAD_DIM*DATA_WIDTH-1:0] ingest_q_data,
    output wire                           ingest_q_ready,

    input  wire                           ingest_kv_wr_valid,
    input  wire [HEAD_DIM*DATA_WIDTH-1:0] ingest_k_data,
    input  wire [HEAD_DIM*DATA_WIDTH-1:0] ingest_v_data,
    output wire                           ingest_kv_ready,

    // --- Egress side: drains stored output rows to axi_stream_egress.v ---
    output wire                           egress_valid,
    output wire [ACC_WIDTH*DATA_WIDTH-1:0] egress_data,
    input  wire                           egress_ready
);

    // -----------------------------------------------------------------
    // Shared pass selector: which TILE_DIM-wide window of the buffered
    // HEAD_DIM row(s) is currently on q/k/v_slice_out. Q, K, and V all
    // advance in lockstep off the same mac_pass_first/mac_pass_ack pair,
    // since one MAC pass consumes one chunk from each of them together.
    // -----------------------------------------------------------------
    reg [PASS_BITS-1:0] pass_sel;

    wire [PASS_BITS-1:0] pass_sel_next =
        mac_pass_first ? {PASS_BITS{1'b0}} :
        (mac_pass_ack && (pass_sel != N_PASSES-1)) ? pass_sel + 1'b1 :
        pass_sel;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pass_sel       <= {PASS_BITS{1'b0}};
            slice_valid    <= 1'b0;
            pass_first_out <= 1'b0;
            pass_last_out  <= 1'b0;
        end else begin
            pass_sel       <= pass_sel_next;
            // One-cycle pulse, registered so it lands on the SAME cycle
            // pass_sel_next (and therefore the muxed slice data below)
            // has become the new pass_sel -- this is what keeps
            // slice_valid aligned with q/k/v_slice_out.
            //
            // The mac_pass_ack that lands when pass_sel is ALREADY at
            // its last index is the sweep's FINAL ack -- gqa_controller
            // reacts to that one by moving to S_WAIT_DRAIN, not by
            // issuing another pass, so it must not produce another
            // slice_valid pulse here either (pass_sel_next == pass_sel,
            // unchanged -- re-pulsing would hand the broadcast/MAC path
            // a spurious extra valid_in for data that's about to be
            // freed below). Share the exact same guard pass_sel_next
            // uses so the two can never disagree.
            slice_valid    <= mac_pass_first || (mac_pass_ack && (pass_sel != N_PASSES-1));
            pass_first_out <= mac_pass_first;
            pass_last_out  <= (mac_pass_first || (mac_pass_ack && (pass_sel != N_PASSES-1)))
                               && (pass_sel_next == N_PASSES-1);
        end
    end

    // -----------------------------------------------------------------
    // Q row buffer: ping-pong, HEAD_DIM-wide, freed only on the NEXT
    // q_row_ack (i.e. once per Q-head row) -- it is read repeatedly,
    // once per pass, across every K-position in that row.
    // -----------------------------------------------------------------
    reg [HEAD_DIM*DATA_WIDTH-1:0] q_buf [0:1];
    reg       q_fill_ptr;
    reg       q_active_ptr;
    reg       q_buf_full [0:1];
    reg       q_active_valid;

    assign ingest_q_ready = !q_buf_full[q_fill_ptr];

    assign q_slice_out = q_buf[q_active_ptr][(pass_sel+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH];
    assign pass_idx_out = pass_sel;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_fill_ptr     <= 1'b0;
            q_active_ptr   <= 1'b0;
            q_buf_full[0]  <= 1'b0;
            q_buf_full[1]  <= 1'b0;
            q_active_valid <= 1'b0;
            q_row_ack      <= 1'b0;
        end else begin
            q_row_ack <= 1'b0;

            if (ingest_q_wr_valid && !q_buf_full[q_fill_ptr]) begin
                q_buf[q_fill_ptr]      <= ingest_q_data;
                q_buf_full[q_fill_ptr] <= 1'b1;
                q_fill_ptr             <= ~q_fill_ptr;
            end

            // A new row request retires whichever bank is currently
            // active (gqa_controller only re-issues q_row_req once
            // every K-position/block of the PREVIOUS row has finished,
            // so it is safe to free that bank right here) and promotes
            // the other bank if ingest has it ready.
            if (q_row_req) begin
                if (q_active_valid) begin
                    q_buf_full[q_active_ptr] <= 1'b0;
                end
                if (q_buf_full[~q_active_ptr]) begin
                    q_active_ptr   <= ~q_active_ptr;
                    q_active_valid <= 1'b1;
                    q_row_ack      <= 1'b1;
                end else if (!q_active_valid && q_buf_full[q_active_ptr]) begin
                    // Very first row: nothing claimed yet, bank 0 (or
                    // whichever ingest filled first) is already sitting
                    // at q_active_ptr's reset value -- claim it in place.
                    q_active_valid <= 1'b1;
                    q_row_ack      <= 1'b1;
                end
                // else: neither bank ready yet -- req stays asserted
                // (gqa_controller holds q_row_req level, not a pulse,
                // until it sees the ack) and this block re-evaluates
                // every cycle until ingest lands the row.
            end
        end
    end

    // -----------------------------------------------------------------
    // K/V position buffer: ping-pong, HEAD_DIM-wide, freed only after
    // this position's FINAL pass has been consumed (pass_sel_next
    // reaching N_PASSES-1 on a mac_pass_ack) -- not on the initial
    // kv_tile_ack, which only means pass 0's chunk is ready.
    // -----------------------------------------------------------------
    reg [HEAD_DIM*DATA_WIDTH-1:0] k_buf [0:1];
    reg [HEAD_DIM*DATA_WIDTH-1:0] v_buf [0:1];
    reg       kv_fill_ptr;
    reg       kv_read_ptr;
    reg       kv_buf_full [0:1];
    reg       kv_row_active;

    assign ingest_kv_ready = !kv_buf_full[kv_fill_ptr];

    assign k_slice_out = k_buf[kv_read_ptr][(pass_sel+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH];
    assign v_slice_out = v_buf[kv_read_ptr][(pass_sel+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kv_fill_ptr    <= 1'b0;
            kv_read_ptr    <= 1'b0;
            kv_buf_full[0] <= 1'b0;
            kv_buf_full[1] <= 1'b0;
            kv_row_active  <= 1'b0;
            kv_tile_ack    <= 1'b0;
        end else begin
            kv_tile_ack <= 1'b0;

            if (ingest_kv_wr_valid && !kv_buf_full[kv_fill_ptr]) begin
                k_buf[kv_fill_ptr]       <= ingest_k_data;
                v_buf[kv_fill_ptr]       <= ingest_v_data;
                kv_buf_full[kv_fill_ptr] <= 1'b1;
                kv_fill_ptr              <= ~kv_fill_ptr;
            end

            // Claim the next K-position's bank once requested and loaded.
            if (kv_tile_req && !kv_row_active && kv_buf_full[kv_read_ptr]) begin
                kv_row_active <= 1'b1;
                kv_tile_ack   <= 1'b1;
            end

            // Free the bank once its final pass has actually been
            // consumed -- lets ingest prefetch the NEXT K-position's
            // data into the other bank while this sweep is still
            // running its remaining passes + adder-tree drain.
            if (kv_row_active && mac_pass_ack && (pass_sel == N_PASSES-1)) begin
                kv_buf_full[kv_read_ptr] <= 1'b0;
                kv_read_ptr              <= ~kv_read_ptr;
                kv_row_active            <= 1'b0;
            end
        end
    end

    // -----------------------------------------------------------------
    // The collector already contains the required one-row skid buffer and
    // axi_stream_egress captures that row before serializing it.  A second
    // 2,048-bit FIFO here was redundant and consumes 2,048 FFs on PYNQ-Z2.
    // Directly couple the two handshakes instead.
    // -----------------------------------------------------------------
    assign egress_valid      = out_row_valid;
    assign egress_data       = out_row_in;
    assign out_row_store_ack = egress_ready;

    // -----------------------------------------------------------------
    // NOTE for gqa_attention_wrapper.v integration:
    // gqa_kv_broadcast.v registers whatever is on k_slice_in/v_slice_in
    // one cycle after load_valid, and separately registers
    // load_pass_first/load_pass_last into slice_pass_first/slice_pass_last
    // on that SAME edge. Wire load_pass_first/load_pass_last to THIS
    // module's pass_first_out/pass_last_out (not gqa_controller's raw
    // mac_pass_first/mac_pass_last) -- those are single-cycle pulses
    // that will have already gone low by the time gqa_kv_broadcast's
    // registered outputs (and therefore the MAC's valid_in) actually
    // appear, one cycle later. Downstream, bf16_dot_product_mac.v's and
    // v_row_assembler.v's pass_first/pass_last inputs should in turn
    // come from gqa_kv_broadcast's slice_pass_first/slice_pass_last
    // (already delay-matched to consumer_valid/bcast_valid), not from
    // gqa_controller directly.
    // -----------------------------------------------------------------

endmodule
