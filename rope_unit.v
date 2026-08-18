// =============================================================================
// rope_unit.v
// Applies Rotary Position Embeddings to a full HEAD_DIM row (a Q row or one
// K-cache position's vector) before it reaches bf16_dot_product_mac.
//
// IMPORTANT DEVIATION FROM "apply per TILE_DIM chunk as it streams":
// Llama (via HuggingFace) uses the HALF-SPLIT ("rotate_half") pairing
// convention: element i is paired with element i+HEAD_DIM/2, not with
// element i+1. Since HEAD_DIM/2 is normally several MAC passes away from
// element i in the TILE_DIM-wide chunk stream, the two elements of a pair
// do NOT arrive in the same (or even an adjacent) pass. A per-chunk-inline
// rotation is therefore not possible for this convention -- this module
// instead buffers the ENTIRE incoming row (N_PASSES chunks), rotates all
// HEAD_DIM/2 pairs once the full row has arrived, then REPLAYS the rotated
// row back out across N_PASSES chunks in the original pass order.
//
// LATENCY IMPACT (flagging per the handoff prompt's request): this adds
// roughly one full row's worth of chunk-arrival time (N_PASSES cycles)
// PLUS the pair-rotation compute time (see ROTATE state) BEFORE the first
// rotated chunk can even begin replaying -- this is not a small constant
// per-pass pipeline addition, it is closer to doubling the per-row latency
// contributed by this stage. gqa_controller's pass timing (mac_pass_first/
// mac_pass_last cadence) currently assumes the next stage is ready every
// cycle during S_MAC_PASSES; if rope_unit sits ahead of the MAC, either
// gqa_controller needs an explicit "rope_busy"/"rope_done" handshake before
// asserting mac_pass_first (analogous to the kv_pos_req/ack handshake it
// already has for load requests), or rope_unit's output must be double-
// buffered so pass N+1's row can be captured while pass N's rotated row is
// still replaying. Neither exists yet -- this module's `busy` output is
// provided for exactly this purpose, but nothing currently consumes it.
//
// PRECISION NOTE: pos is a runtime input (POS_WIDTH-bit integer), used to
// scale a fixed-point NCO phase-per-step value (see gen_rope_rom.py) via
// wraparound multiply -- no floating angle-reduction (mod 2*pi) needed.
// =============================================================================
`timescale 1ns / 1ps

module rope_unit #(
    parameter HEAD_DIM      = 128,
    parameter TILE_DIM      = 16,
    parameter DATA_WIDTH    = 16,
    parameter N_PASSES      = HEAD_DIM / TILE_DIM,
    parameter PHASE_WIDTH   = 32,
    // 16384 phase entries keep table-quantization error below the BF16
    // accuracy budget across the supported RoPE positions.
    parameter ROM_ADDR_BITS = 14,
    parameter POS_WIDTH     = 16
) (
    input  wire clk,
    input  wire rst_n,

    // Input row stream (from tile_scheduler, before gqa_kv_broadcast/MAC)
    input  wire                              in_valid,
    input  wire                              in_pass_first,
    input  wire                              in_pass_last,
    input  wire [POS_WIDTH-1:0]              pos,            // absolute seq position of this row
    input  wire [TILE_DIM*DATA_WIDTH-1:0]    chunk_in,

    // Rotated output row stream (to gqa_kv_broadcast/MAC), re-timed --
    // see LATENCY IMPACT note above. Consumer must NOT assume out_valid
    // lines up with in_valid on the same or a fixed-offset cycle; it only
    // begins once the whole input row has been captured and rotated.
    output reg                               out_valid,
    output reg                               out_pass_first,
    output reg                               out_pass_last,
    output reg [TILE_DIM*DATA_WIDTH-1:0]     chunk_out,
    input  wire                              out_ready,

    // Asserted from the first in_valid chunk of a row until the rotated
    // row has finished replaying. gqa_controller must gate the NEXT row's
    // mac_pass_first/loads on this being low (not currently wired up --
    // see LATENCY IMPACT note).
    output reg                               busy
);

    localparam N_PAIRS = HEAD_DIM / 2;

    // ---------------- Row capture ----------------
    reg [DATA_WIDTH-1:0] row_buf [0:HEAD_DIM-1];
    reg [$clog2(N_PASSES):0] in_pass_cnt;

    // ---------------- Phase / sincos ROMs ----------------
    reg [PHASE_WIDTH-1:0] phase_rom [0:N_PAIRS-1];
    reg [31:0]            sincos_rom[0:(1<<ROM_ADDR_BITS)-1]; // {cos[15:0], sin[15:0]}
`ifndef SYNTHESIS
    // XSim normally executes from a generated directory.  Supplying
    // +ROPE_ROM_DIR=<absolute-path> makes the ROM location unambiguous.
    reg [8*512-1:0] rope_rom_dir;
`endif

    initial begin
`ifndef SYNTHESIS
        if ($value$plusargs("ROPE_ROM_DIR=%s", rope_rom_dir)) begin
            $readmemh({rope_rom_dir, "/rope_phase_rom.mem"}, phase_rom);
            $readmemh({rope_rom_dir, "/rope_sincos_rom.mem"}, sincos_rom);
        end else begin
`endif
            $readmemh("rope_rom/rope_phase_rom.mem", phase_rom);
            $readmemh("rope_rom/rope_sincos_rom.mem", sincos_rom);
`ifndef SYNTHESIS
        end
`endif
    end

    // ---------------- FSM ----------------
    localparam S_IDLE    = 2'd0,
               S_CAPTURE = 2'd1,
               S_ROTATE  = 2'd2,
               S_REPLAY  = 2'd3;

    reg [1:0] state;
    reg [$clog2(N_PAIRS):0]  pair_idx;   // which pair (0..N_PAIRS-1) is being rotated
    reg [$clog2(N_PASSES):0] out_pass_cnt;
    // Replay needs TILE_DIM simultaneous reads and two independent writes
    // from the rotation pipeline, which is not a legal single-port BRAM
    // access pattern.  Declare the required register implementation
    // explicitly so synthesis does not emit a misleading failed-RAM warning.
    (* ram_style = "registers" *) reg [DATA_WIDTH-1:0] row_rot [0:HEAD_DIM-1];

    // Rotation math pipeline: x' = x*cos - y*sin, y' = x*sin + y*cos
    // Reuses bf16_multiplier/bf16_adder/bf16_subtractor as instructed.
    // Processes one pair at a time (see note in header comment about a
    // TILE_DIM/2-parallel version being a natural throughput optimization,
    // deferred here for simplicity).
    reg                   rot_valid_in;
    reg [DATA_WIDTH-1:0]  rot_x, rot_y, rot_cos, rot_sin;

    wire xrot_valid, yrot_valid;
    wire [DATA_WIDTH-1:0] xrot, yrot;
    wire xc_valid, ys_valid, xs_valid, yc_valid;
    wire [DATA_WIDTH-1:0] xc, ys, xs, yc;
    bf16_multiplier u_mul_xc (.clk(clk), .rst_n(rst_n), .valid_in(rot_valid_in), .a(rot_x), .b(rot_cos), .valid_out(xc_valid), .result(xc));
    bf16_multiplier u_mul_ys (.clk(clk), .rst_n(rst_n), .valid_in(rot_valid_in), .a(rot_y), .b(rot_sin), .valid_out(ys_valid), .result(ys));
    bf16_multiplier u_mul_xs (.clk(clk), .rst_n(rst_n), .valid_in(rot_valid_in), .a(rot_x), .b(rot_sin), .valid_out(xs_valid), .result(xs));
    bf16_multiplier u_mul_yc (.clk(clk), .rst_n(rst_n), .valid_in(rot_valid_in), .a(rot_y), .b(rot_cos), .valid_out(yc_valid), .result(yc));
    bf16_subtractor #(.LATENCY(5)) u_sub_xrot (.clk(clk), .rst_n(rst_n), .valid_in(xc_valid & ys_valid), .a(xc), .b(ys), .valid_out(xrot_valid), .result(xrot));
    bf16_adder #(.SPLIT_ARITH_STAGE(1)) u_add_yrot (.clk(clk), .rst_n(rst_n), .valid_in(xs_valid & yc_valid), .a(xs), .b(yc), .valid_out(yrot_valid), .result(yrot));

    // Track which pair's result is currently emerging from the rotation
    // pipeline (fixed latency = bf16_multiplier + bf16_adder/subtractor
    // stages, both branches matched in depth so xrot_valid/yrot_valid pulse
    // together).
    //
    // TAG_DEPTH = ROT_LATENCY + 1, not ROT_LATENCY: there's one extra cycle
    // of "flight time" between rot_valid_in being SCHEDULED (in this always
    // block, at the issue cycle) and bf16_multiplier's own always block
    // actually OBSERVING it high (one cycle later, since that read happens
    // in a separate module's always block triggered by the same clock,
    // which sees pre-edge values). The tag[0] write experiences the exact
    // same one-cycle registration delay, so a tag array sized ROT_LATENCY
    // arrives one cycle EARLY relative to xrot_valid/yrot_valid -- verified
    // by simulation (a first attempt at ROT_LATENCY-deep tags mis-tagged
    // every write by one pair index).
    localparam ROT_LATENCY = 2 /*mult*/ + 5 /*add/sub*/;
    localparam TAG_DEPTH   = ROT_LATENCY + 1;
    reg [$clog2(N_PAIRS):0] rot_pair_tag [0:TAG_DEPTH-1];
    integer ti;

    // Explicit wide product to avoid truncation: phase_rom[i] is
    // PHASE_WIDTH bits (32), pos is POS_WIDTH bits (16) -- the product
    // needs the full PHASE_WIDTH+POS_WIDTH bits before truncating down to
    // PHASE_WIDTH (the intended wraparound-mod-2^PHASE_WIDTH behavior),
    // otherwise Verilog's self-determined expression width could silently
    // truncate before the mod-wraparound is actually reached.
    // Register the phase multiply before the ROM lookup/rotation issue.
    // The original single-cycle path contained two cascaded DSP48s followed
    // by the wide sin/cos mux and missed the 100-MHz constraint by >2 ns.
    // This costs one cycle per row, not one cycle per pair, because the
    // phase stage and the rotation pipeline accept a new pair every cycle.
    wire [PHASE_WIDTH+POS_WIDTH-1:0] phase_product_comb = phase_rom[pair_idx] * pos;
    reg  [PHASE_WIDTH-1:0]           phase_wrapped_r;
    wire [ROM_ADDR_BITS-1:0]         rom_addr = phase_wrapped_r[PHASE_WIDTH-1 -: ROM_ADDR_BITS];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            out_valid      <= 1'b0;
            out_pass_first <= 1'b0;
            out_pass_last  <= 1'b0;
            pair_idx       <= 0;
            out_pass_cnt   <= 0;
            rot_valid_in   <= 1'b0;
            for (i = 0; i < TAG_DEPTH; i = i + 1) rot_pair_tag[i] <= 0;
        end else begin
            rot_valid_in <= 1'b0;

            // shift the pair-index tag alongside the rotation pipeline
            // (tag[0] itself is set directly in S_ROTATE below, alongside
            // rot_x/rot_y/rot_cos/rot_sin, using the SAME pair_idx snapshot
            // -- capturing it here instead, by re-reading pair_idx, used to
            // pick up its POST-increment value since both are nonblocking
            // updates within the same always block, causing an off-by-one
            // between the tag and the data it was meant to tag)
            for (ti = TAG_DEPTH-1; ti > 0; ti = ti - 1)
                rot_pair_tag[ti] <= rot_pair_tag[ti-1];

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    out_valid <= 1'b0;
                    if (in_valid && in_pass_first) begin
                        busy  <= 1'b1;
                        state <= S_CAPTURE;
                    end
                end

                S_CAPTURE: begin
                    // captured every cycle in_valid is high, see always-on
                    // capture block below; here we just track completion
                    if (in_valid && in_pass_last) begin
                        pair_idx <= 0;
                        state    <= S_ROTATE;
                    end
                end

                S_ROTATE: begin
                    // Issue one pair's rotation per cycle into the shared
                    // pipeline (pipeline itself allows back-to-back issue
                    // since all 4 multiplier instances + both add/sub
                    // instances are independently instantiated per pair
                    // slot -- NOT resource-shared across pairs, i.e. this
                    // uses 4 multipliers + 1 adder + 1 subtractor total,
                    // reused via streaming issue, not N_PAIRS copies).
                    // The registered phase result from the preceding pair
                    // drives the ROM and multiplier inputs this cycle.
                    // pair_idx itself is the phase-valid/tag state: after
                    // issuing pair n, it is n+1.  This avoids an extra
                    // x/y/tag/valid register bank in both Q and K RoPE
                    // units (a material slice saving on XC7Z020).
                    if ((pair_idx != 0) && (pair_idx <= N_PAIRS)) begin
                        rot_valid_in <= 1'b1;
                        rot_x   <= row_buf[pair_idx-1'b1];
                        rot_y   <= row_buf[pair_idx-1'b1 + N_PAIRS];
                        rot_cos <= sincos_rom[rom_addr][31:16];
                        rot_sin <= sincos_rom[rom_addr][15:0];
                        rot_pair_tag[0] <= pair_idx-1'b1;
                    end
                    if (pair_idx < N_PAIRS) begin
                        phase_wrapped_r <= phase_product_comb[PHASE_WIDTH-1:0];
                        pair_idx <= pair_idx + 1;
                    end else if (pair_idx == N_PAIRS) begin
                        // Last phase is issued above.  Advance to a drain
                        // sentinel so it is not issued a second time.
                        pair_idx <= pair_idx + 1'b1;
                    end
                    // once all pairs issued AND the pipeline has drained
                    // (last pair's result captured), move to replay
                    if (pair_idx == N_PAIRS + 1 && !rot_valid_in && rot_pair_tag[TAG_DEPTH-1] == N_PAIRS - 1) begin
                        out_pass_cnt <= 0;
                        state        <= S_REPLAY;
                    end
                end

                S_REPLAY: begin
                    // The MAC has a feedback dependency between chunks.
                    // Keep a replay chunk stable until it is accepted.
                    if (!out_valid) begin
                        out_valid      <= 1'b1;
                        out_pass_first <= (out_pass_cnt == 0);
                        out_pass_last  <= (out_pass_cnt == N_PASSES - 1);
                        chunk_out      <= pack_chunk(out_pass_cnt);
                    end else if (out_ready) begin
                        if (out_pass_cnt == N_PASSES - 1) begin
                            out_valid      <= 1'b0;
                            out_pass_first <= 1'b0;
                            out_pass_last  <= 1'b0;
                            state          <= S_IDLE;
                        end else begin
                            out_pass_cnt   <= out_pass_cnt + 1'b1;
                            out_valid      <= 1'b1;
                            out_pass_first <= 1'b0;
                            out_pass_last  <= (out_pass_cnt + 1'b1 == N_PASSES - 1);
                            chunk_out      <= pack_chunk(out_pass_cnt + 1'b1);
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Always-on capture of incoming chunks into row_buf (separate from the
    // main FSM's state-dependent logic above, since capture must happen on
    // the very first cycle before the FSM has even left S_IDLE). row_buf is
    // an unpacked array of DATA_WIDTH-wide elements, so a chunk (packed
    // TILE_DIM*DATA_WIDTH bits) must be unpacked element-by-element -- a
    // `+:` part-select range across array *indices* (as opposed to bits
    // within one element) is not valid Verilog.
    integer cap_k;
    always @(posedge clk) begin
        if (in_valid) begin
            for (cap_k = 0; cap_k < TILE_DIM; cap_k = cap_k + 1)
                // in_pass_cnt is reset on the same edge as the first beat.
                // Use the first-beat tag directly here: nonblocking updates
                // would otherwise write the first chunk at the preceding
                // row's terminal counter (leaving row_buf[0:15] stale).
                row_buf[(in_pass_first ? 0 : in_pass_cnt)*TILE_DIM + cap_k] <=
                    chunk_in[(cap_k+1)*DATA_WIDTH-1 -: DATA_WIDTH];
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) in_pass_cnt <= 0;
        else if (in_valid) in_pass_cnt <= in_pass_first ? 1 : (in_pass_cnt + 1);
    end

    // Capture rotated pair results as they emerge from the pipeline
    always @(posedge clk) begin
        if (xrot_valid) begin
            row_rot[rot_pair_tag[TAG_DEPTH-1]]          <= xrot;
        end
        if (yrot_valid) begin
            row_rot[rot_pair_tag[TAG_DEPTH-1] + N_PAIRS] <= yrot;
        end
    end

    // Helper: pack TILE_DIM elements starting at pass*TILE_DIM from row_rot
    // into one chunk_out word. (Function used only combinationally at the
    // point of assignment above.)
    function [TILE_DIM*DATA_WIDTH-1:0] pack_chunk(input [$clog2(N_PASSES):0] p);
        integer k;
        begin
            for (k = 0; k < TILE_DIM; k = k + 1)
                pack_chunk[(k+1)*DATA_WIDTH-1 -: DATA_WIDTH] = row_rot[p*TILE_DIM + k];
        end
    endfunction

endmodule
