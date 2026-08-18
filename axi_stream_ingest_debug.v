// =============================================================================
// axi_stream_ingest.v  (REV 2 -- assembles full HEAD_DIM rows)
//
// REV 1 treated one TILE_DIM-wide beat of each of Q, K, and V as a
// complete "tile" and fired a single ingest_wr_valid once all three had
// arrived. That only works if HEAD_DIM == TILE_DIM; it does not assemble
// the N_PASSES TILE_DIM-wide beats that make up one real HEAD_DIM-wide
// Q row or K/V position.
//
// REV 2 collects N_PASSES beats per plane into a HEAD_DIM-wide staging
// register before handing anything to tile_scheduler.v, and fires Q and
// K/V completion independently, since they arrive on different cadences:
// one Q-plane sweep (N_PASSES beats) per Q-row, but one K-plane AND
// V-plane sweep (N_PASSES beats each) per K-position -- many more K/V
// completions than Q completions over a full run.
//
// ASSUMPTION (flag for confirmation against the actual DMA descriptor
// packing): beats for a given plane arrive in strict pass order
// (pass 0 first ... pass N_PASSES-1 last) before that plane's next
// sweep begins, and a given K-position's K beats and V beats don't
// interleave with a DIFFERENT K-position's beats. TUSER[1:0] decode
// (00=Q, 01=K, 10=V) is unchanged from REV 1.
// =============================================================================
`timescale 1ns / 1ps

module axi_stream_ingest #(
    parameter TILE_DIM    = 16,
    parameter HEAD_DIM    = 128,
    parameter DATA_WIDTH  = 16,
    parameter N_PASSES    = HEAD_DIM / TILE_DIM,
    parameter TDATA_WIDTH = TILE_DIM * DATA_WIDTH,
    parameter ROW_WIDTH   = HEAD_DIM * DATA_WIDTH
) (
    input  wire clk,
    input  wire rst_n,

    // AXI4-Stream slave (from DMA / PS side) -- still one TILE_DIM-wide
    // beat per transfer; N_PASSES beats now make up one assembled row.
    input  wire [TDATA_WIDTH-1:0] s_axis_tdata,
    input  wire                    s_axis_tvalid,
    output wire                    s_axis_tready,
    input  wire                    s_axis_tlast,
    input  wire [1:0]              s_axis_tuser,   // 00=Q, 01=K, 10=V

    // Q row: fires once per completed N_PASSES-beat Q sweep.
    output reg                     ingest_q_wr_valid,
    output reg  [ROW_WIDTH-1:0]    ingest_q_data,
    input  wire                    ingest_q_ready,

    // K/V position: fires once BOTH the K-plane and V-plane sweeps for
    // the current position have each completed N_PASSES beats.
    output reg                     ingest_kv_wr_valid,
    output reg  [ROW_WIDTH-1:0]    ingest_k_data,
    output reg  [ROW_WIDTH-1:0]    ingest_v_data,
    input  wire                    ingest_kv_ready,

    output reg                     row_boundary   // pulses on TLAST
);

    localparam PASS_BITS = (N_PASSES <= 1) ? 1 : $clog2(N_PASSES);

    reg [ROW_WIDTH-1:0] q_stage, k_stage, v_stage;
    reg [PASS_BITS-1:0] q_cnt, k_cnt, v_cnt;
    reg                 k_done, v_done;
    reg                 q_last_pending;

    // Whole-beat backpressure: only accept a beat if whichever plane it
    // could complete has somewhere to land. Q and K/V destinations are
    // independent, so gate on the one relevant to each incoming beat's
    // TUSER rather than ANDing both together (that would stall K/V
    // ingest just because a Q row hasn't been claimed yet, or vice
    // versa).
    reg ready_mux;
    always @(*) begin
        case (s_axis_tuser)
            2'b00:   ready_mux = ingest_q_ready;
            2'b01,
            2'b10:   ready_mux = ingest_kv_ready;
            default: ready_mux = 1'b1;
        endcase
    end
    assign s_axis_tready = ready_mux;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_cnt              <= {PASS_BITS{1'b0}};
            k_cnt              <= {PASS_BITS{1'b0}};
            v_cnt              <= {PASS_BITS{1'b0}};
            k_done             <= 1'b0;
            v_done             <= 1'b0;
            q_last_pending      <= 1'b0;
            ingest_q_wr_valid  <= 1'b0;
            ingest_kv_wr_valid <= 1'b0;
            row_boundary       <= 1'b0;
        end else begin
            ingest_q_wr_valid  <= 1'b0;
            ingest_kv_wr_valid <= 1'b0;
            row_boundary       <= 1'b0;

            // Flush whatever completed on the PREVIOUS cycle -- the
            // staging register that plane's final beat landed in has
            // now settled and is safe to read out.
            if (q_last_pending) begin
                ingest_q_data     <= q_stage;
                ingest_q_wr_valid <= 1'b1;
                $display("[%0t] DEBUG-INGEST: firing ingest_q_wr_valid (full row -> q_group_buffer)", $time);
            end
            q_last_pending <= 1'b0;

            if (k_done && v_done) begin
                ingest_k_data      <= k_stage;
                ingest_v_data      <= v_stage;
                ingest_kv_wr_valid <= 1'b1;
                k_done             <= 1'b0;
                v_done             <= 1'b0;
            end

            if (s_axis_tvalid && s_axis_tready) begin
                case (s_axis_tuser)
                    2'b00: begin // Q
                        q_stage[(q_cnt+1)*TDATA_WIDTH-1 -: TDATA_WIDTH] <= s_axis_tdata;
                        $display("[%0t] DEBUG-INGEST: Q beat accepted, q_cnt=%0d", $time, q_cnt);
                        if (q_cnt == N_PASSES-1) begin
                            q_cnt          <= {PASS_BITS{1'b0}};
                            q_last_pending <= 1'b1;
                            $display("[%0t] DEBUG-INGEST: Q row COMPLETE (last beat), q_last_pending will fire next cycle", $time);
                        end else begin
                            q_cnt <= q_cnt + 1'b1;
                        end
                    end
                    2'b01: begin // K
                        k_stage[(k_cnt+1)*TDATA_WIDTH-1 -: TDATA_WIDTH] <= s_axis_tdata;
                        if (k_cnt == N_PASSES-1) begin
                            k_cnt  <= {PASS_BITS{1'b0}};
                            k_done <= 1'b1;
                        end else begin
                            k_cnt <= k_cnt + 1'b1;
                        end
                    end
                    2'b10: begin // V
                        v_stage[(v_cnt+1)*TDATA_WIDTH-1 -: TDATA_WIDTH] <= s_axis_tdata;
                        if (v_cnt == N_PASSES-1) begin
                            v_cnt  <= {PASS_BITS{1'b0}};
                            v_done <= 1'b1;
                        end else begin
                            v_cnt <= v_cnt + 1'b1;
                        end
                    end
                    default: ;
                endcase

                if (s_axis_tlast)
                    row_boundary <= 1'b1;
            end
        end
    end

endmodule