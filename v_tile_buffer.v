// =============================================================================
// v_tile_buffer.v
//
// Stores one flash-attention V block in block RAM.  There is one independent
// RAM bank per KV row, which gives output_accumulator the KV_BLOCK_LEN values
// for a selected output-dimension pass in parallel.  A full V row arrives as
// a wide bus, so it is latched then written to its RAM bank at the input
// stream's existing chunk cadence.  The read port is synchronous (one clock
// of latency) and can expose a smaller accumulator window from that word.
// =============================================================================
`timescale 1ns / 1ps

module v_tile_buffer #(
    parameter KV_BLOCK_LEN = 16,
    parameter HEAD_DIM     = 128,
    parameter DATA_WIDTH   = 16,
    parameter TILE_DIM     = 1,
    parameter WRITE_TILE_DIM = TILE_DIM,
    parameter N_PASSES     = HEAD_DIM / TILE_DIM
) (
    input  wire clk,
    input  wire rst_n,
    input  wire                                  row_valid,
    input  wire                                  block_first,
    input  wire [HEAD_DIM*DATA_WIDTH-1:0]        row_in,
    output reg                                   block_valid,
    input  wire [$clog2(N_PASSES)-1:0]           read_pass,
    output wire [KV_BLOCK_LEN*TILE_DIM*DATA_WIDTH-1:0] read_data
);
    localparam WORD_WIDTH = WRITE_TILE_DIM * DATA_WIDTH;
    localparam WRITE_PASSES = HEAD_DIM / WRITE_TILE_DIM;
    localparam READS_PER_WORD = WRITE_TILE_DIM / TILE_DIM;
    localparam SLOT_BITS  = (KV_BLOCK_LEN <= 1) ? 1 : $clog2(KV_BLOCK_LEN);
    localparam WRITE_PASS_BITS = (WRITE_PASSES <= 1) ? 1 : $clog2(WRITE_PASSES);
    localparam READ_OFFSET_BITS = (READS_PER_WORD <= 1) ? 1 : $clog2(READS_PER_WORD);

    reg [SLOT_BITS-1:0] fill_slot, write_slot;
    reg [WRITE_PASS_BITS-1:0] write_pass;
    reg                  write_active;
    reg [HEAD_DIM*DATA_WIDTH-1:0] write_row;

    // The full-row producer supplies a new row every HEAD_DIM/WRITE_TILE_DIM
    // cycles.  This buffer writes one input-width BRAM word per cycle, so it
    // finishes exactly in that interval without dropping a row.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fill_slot    <= {SLOT_BITS{1'b0}};
            write_slot   <= {SLOT_BITS{1'b0}};
            write_pass   <= {WRITE_PASS_BITS{1'b0}};
            write_active <= 1'b0;
            block_valid  <= 1'b0;
        end else begin
            block_valid <= 1'b0;

            if (write_active) begin
                if (write_pass == WRITE_PASSES-1) begin
                    write_active <= 1'b0;
                    write_pass   <= {WRITE_PASS_BITS{1'b0}};
                    if (write_slot == KV_BLOCK_LEN-1) begin
                        fill_slot   <= {SLOT_BITS{1'b0}};
                        block_valid <= 1'b1;
                    end else begin
                        fill_slot <= write_slot + 1'b1;
                    end
                end else begin
                    write_pass <= write_pass + 1'b1;
                end
            end

            // A row is expected only while the previous row's last word is
            // being written (or when idle), matching the V-row assembler's
            // WRITE_PASSES-wide input cadence.
            if (row_valid && (!write_active || write_pass == WRITE_PASSES-1)) begin
                write_row    <= row_in;
                // On the terminal write edge, fill_slot still holds the
                // *previous* bank because all assignments are nonblocking.
                // Derive the next bank directly from write_slot to avoid
                // overwriting the row that just completed.
                write_slot   <= block_first ? {SLOT_BITS{1'b0}} :
                                (write_active ?
                                 ((write_slot == KV_BLOCK_LEN-1) ? {SLOT_BITS{1'b0}} : write_slot + 1'b1) :
                                 fill_slot);
                write_pass   <= {WRITE_PASS_BITS{1'b0}};
                write_active <= 1'b1;
                if (block_first)
                    fill_slot <= {SLOT_BITS{1'b0}};
            end
        end
    end

    generate
        genvar g;
        for (g = 0; g < KV_BLOCK_LEN; g = g + 1) begin : V_BANK
            // A separate bank per V row provides the required 16-way read
            // bandwidth while mapping each 32-Kbit block to BRAM rather than
            // to flip-flops.  No reset is applied to BRAM contents.
            (* ram_style = "block" *) reg [WORD_WIDTH-1:0] mem [0:WRITE_PASSES-1];
            reg [WORD_WIDTH-1:0] read_word;
            reg [READ_OFFSET_BITS:0] read_offset;

            always @(posedge clk) begin
                if (write_active && write_slot == g)
                    mem[write_pass] <= write_row[(write_pass+1)*WORD_WIDTH-1 -: WORD_WIDTH];
                read_word   <= mem[(read_pass*TILE_DIM) / WRITE_TILE_DIM];
                read_offset <= ((read_pass*TILE_DIM) % WRITE_TILE_DIM) / TILE_DIM;
            end

            assign read_data[(g+1)*TILE_DIM*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH] =
                read_word[((read_offset+1)*TILE_DIM)*DATA_WIDTH-1 -: TILE_DIM*DATA_WIDTH];
        end
    endgenerate
endmodule
