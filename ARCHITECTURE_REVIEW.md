# PYNQ-Z2 architecture review

The top-level integration is structurally complete: `gqa_attention_wrapper`
instantiates `axi_stream_egress`, and its output is eight 256-bit AXI4-Stream
beats per 128-element BF16 output row. `TLAST` is asserted only on beat eight,
and the output remains stable while `TREADY` is low.

The functional dataflow is appropriate for Llama-3 GQA:

```
four Q heads -> shared KV head -> Q.K score -> online softmax -> weighted V
```

RoPE uses Llama's half-split `rotate_half` pairing and must buffer/replay a
full 128-element row. This is correct, but it is a throughput cost.

## Fit risk

This revision is not yet a safe PYNQ-Z2 bitstream candidate. The current
`output_accumulator` is fully unrolled: every Q lane has `128 * 16 = 2048`
probability-by-V BF16 multipliers, and the wrapper has four Q lanes. That is
at least 8192 such multiplier instances before the dot-product, RoPE, softmax,
and normalization hardware. A Zynq-7020 has only 220 DSP48E1s and limited LUT
fabric, so it will not meet the competition's resource objective.

## Required resource-reduced implementation direction

Use a microcoded/tiled weighted-V engine for one Q lane at a time (or one
small PE array) and reuse the BF16 multiplier/adder across output dimensions
and KV entries. Keep the existing online-softmax state (`m`, `l`, and output
accumulator) but store the 128-element partial output in BRAM. This exchanges
latency for a design that can plausibly fit the board. Synthesis reports from
`scripts/synth_pynq_z2.tcl` should be the acceptance gate before bitstream
generation.
