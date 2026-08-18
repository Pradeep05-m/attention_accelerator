// =============================================================================
// axi_lite_regs.v
// Standard AXI4-Lite control/status register bank.
//
// Register map (word-addressed, 32-bit AXI-Lite data width):
//   0x00  CTRL      [0]=start (self-clearing pulse), [1]=soft_reset
//   0x04  STATUS    [0]=busy, [1]=done (sticky, W1C), [2]=error
//   0x08  N_Q_HEADS
//   0x0C  N_KV_HEADS
//   0x10  HEAD_DIM
//   0x14  TILE_DIM
//   0x18  SEQ_LEN        (runtime-configurable, NOT hardcoded)
//   0x1C  N_KV_TILES     (= ceil(SEQ_LEN / KV_BLOCK_LEN), can be host-computed
//                          or computed here; exposed as writable for host
//                          simplicity)
//   0x20  ERROR_FLAGS    sticky, W1C
//   0x24  SCALE_FACTOR   [15:0] = BF16 encoding of 1/sqrt(HEAD_DIM), host-
//                          computed and written once at config time (sqrt
//                          is not computed on-chip; this avoids needing a
//                          runtime sqrt/rsqrt unit for a value that is
//                          static for a given HEAD_DIM).
//   0x28  QUERY_POSITION Absolute position of the query token.  This is
//                          consumed by the Q RoPE engine; K receives its
//                          position from the controller's KV traversal.
// =============================================================================
`timescale 1ns / 1ps

module axi_lite_regs #(
    parameter C_S_AXI_ADDR_WIDTH = 6,
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter DEFAULT_TILE_DIM   = 16
) (
    input  wire                              s_axi_aclk,
    input  wire                              s_axi_aresetn,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  wire                              s_axi_awvalid,
    output reg                               s_axi_awready,

    input  wire [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                              s_axi_wvalid,
    output reg                               s_axi_wready,

    output reg  [1:0]                        s_axi_bresp,
    output reg                               s_axi_bvalid,
    input  wire                              s_axi_bready,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    input  wire                              s_axi_arvalid,
    output reg                               s_axi_arready,

    output reg  [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    output reg  [1:0]                        s_axi_rresp,
    output reg                               s_axi_rvalid,
    input  wire                              s_axi_rready,

    // --- Core-facing outputs ---
    output reg                               start_pulse,
    output reg                               soft_reset,
    input  wire                              busy_in,
    input  wire                              done_in,     // pulse from gqa_controller
    input  wire                              error_in,
    output reg  [31:0]                       n_q_heads,
    output reg  [31:0]                       n_kv_heads,
    output reg  [31:0]                       head_dim,
    output reg  [31:0]                       tile_dim,
    output reg  [31:0]                       seq_len,
    output reg  [31:0]                       n_kv_tiles,
    output reg  [15:0]                       scale_factor,
    output reg  [15:0]                       query_position
);

    localparam ADDR_CTRL       = 6'h00,
               ADDR_STATUS     = 6'h04,
               ADDR_NQ         = 6'h08,
               ADDR_NKV        = 6'h0C,
               ADDR_HEADDIM    = 6'h10,
               ADDR_TILEDIM    = 6'h14,
               ADDR_SEQLEN     = 6'h18,
               ADDR_NKVTILES   = 6'h1C,
               ADDR_ERRFLAGS   = 6'h20,
               ADDR_SCALE      = 6'h24,
               ADDR_QUERY_POS  = 6'h28;

    reg        done_sticky;
    reg [31:0] error_flags;

    wire [C_S_AXI_ADDR_WIDTH-1:0] waddr = s_axi_awaddr;
    wire [C_S_AXI_ADDR_WIDTH-1:0] raddr = s_axi_araddr;

    // ---------------- Write channel ----------------
    // AXI4-Lite permits AW and W to arrive in different cycles.  Keep one
    // entry for each channel instead of requiring an illegal simultaneous
    // handshake.
    reg        aw_pending, w_pending;
    reg [C_S_AXI_ADDR_WIDTH-1:0] awaddr_saved;
    reg [C_S_AXI_DATA_WIDTH-1:0] wdata_saved;
    reg [(C_S_AXI_DATA_WIDTH/8)-1:0] wstrb_saved;
    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            s_axi_bresp   <= 2'b00;
            start_pulse   <= 1'b0;
            soft_reset    <= 1'b0;
            n_q_heads     <= 32'd32;
            n_kv_heads    <= 32'd8;
            head_dim      <= 32'd128;
            tile_dim      <= DEFAULT_TILE_DIM;
            seq_len       <= 32'd0;
            n_kv_tiles    <= 32'd0;
            // BF16 round-to-nearest encoding of 1/sqrt(128), the fixed
            // Llama-3 head dimension.  Software may still override this.
            scale_factor  <= 16'h3DB5;
            query_position <= 16'd0;
            done_sticky   <= 1'b0;
            error_flags   <= 32'd0;
            aw_pending    <= 1'b0;
            w_pending     <= 1'b0;
            awaddr_saved  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
            wdata_saved   <= {C_S_AXI_DATA_WIDTH{1'b0}};
            wstrb_saved   <= {(C_S_AXI_DATA_WIDTH/8){1'b0}};
        end else begin
            start_pulse <= 1'b0;
            soft_reset  <= 1'b0;

            s_axi_awready <= !aw_pending && !s_axi_bvalid;
            s_axi_wready  <= !w_pending  && !s_axi_bvalid;

            if (s_axi_awvalid && s_axi_awready) begin
                awaddr_saved <= s_axi_awaddr;
                aw_pending   <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                wdata_saved <= s_axi_wdata;
                wstrb_saved <= s_axi_wstrb;
                w_pending   <= 1'b1;
            end

            // Execute a complete write exactly once.  The one-cycle delay
            // after the second channel handshake is intentional: it keeps
            // this logic simple and fully AXI compliant.
            if (aw_pending && w_pending && !s_axi_bvalid) begin
                case (awaddr_saved)
                    ADDR_CTRL: begin
                        if (wstrb_saved[0]) begin
                            start_pulse <= wdata_saved[0];
                            soft_reset  <= wdata_saved[1];
                        end
                    end
                    ADDR_STATUS: begin
                        // W1C for done/error bits
                        if (wdata_saved[1]) done_sticky <= 1'b0;
                        if (wdata_saved[2]) error_flags <= 32'd0;
                    end
                    ADDR_NQ:       n_q_heads   <= wdata_saved;
                    ADDR_NKV:      n_kv_heads  <= wdata_saved;
                    ADDR_HEADDIM:  head_dim    <= wdata_saved;
                    ADDR_TILEDIM:  tile_dim    <= wdata_saved;
                    ADDR_SEQLEN:   seq_len     <= wdata_saved;
                    ADDR_NKVTILES: n_kv_tiles  <= wdata_saved;
                    ADDR_ERRFLAGS: if (wdata_saved != 0) error_flags <= 32'd0;
                    ADDR_SCALE:    scale_factor <= wdata_saved[15:0];
                    ADDR_QUERY_POS: query_position <= wdata_saved[15:0];
                    default: ;
                endcase
                aw_pending   <= 1'b0;
                w_pending    <= 1'b0;
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (done_in)  done_sticky <= 1'b1;
            if (error_in) error_flags <= error_flags | 32'h1;
        end
    end

    // ---------------- Read channel ----------------
    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_rdata   <= 32'd0;
        end else begin
            // Do not acknowledge a second read address while its first
            // response is still outstanding.  The previous toggle-style
            // ready logic could reassert ARREADY with RVALID high and
            // overwrite RDATA when the master applied backpressure.
            s_axi_arready <= !s_axi_rvalid;

            if (s_axi_arready && s_axi_arvalid) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= 2'b00;
                case (raddr)
                    ADDR_CTRL:     s_axi_rdata <= 32'd0;
                    ADDR_STATUS:   s_axi_rdata <= {29'd0, error_flags[0], done_sticky, busy_in};
                    ADDR_NQ:       s_axi_rdata <= n_q_heads;
                    ADDR_NKV:      s_axi_rdata <= n_kv_heads;
                    ADDR_HEADDIM:  s_axi_rdata <= head_dim;
                    ADDR_TILEDIM:  s_axi_rdata <= tile_dim;
                    ADDR_SEQLEN:   s_axi_rdata <= seq_len;
                    ADDR_NKVTILES: s_axi_rdata <= n_kv_tiles;
                    ADDR_ERRFLAGS: s_axi_rdata <= error_flags;
                    ADDR_SCALE:    s_axi_rdata <= {16'd0, scale_factor};
                    ADDR_QUERY_POS:s_axi_rdata <= {16'd0, query_position};
                    default:       s_axi_rdata <= 32'd0;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
