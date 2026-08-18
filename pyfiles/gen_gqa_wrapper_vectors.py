#!/usr/bin/env python3
"""Generate AXIS stimulus and a floating-point GQA golden result.

Default ``smoke`` mode uses zero Q/K and structured V.  This makes every
causally-visible score equal, so each output is the mean of V rows 0..query
position.  It is an end-to-end, reproducible check of ingest, GQA grouping,
softmax, accumulation and egress without depending on random test luck.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List

from gqa_reference import bf16_to_f32, f32_to_bf16, scale_for_head_dim, write_mem


def chunks(row: List[int], tile_dim: int):
    for start in range(0, len(row), tile_dim):
        yield row[start:start + tile_dim]


def rope_half_split(row: List[float], position: int, theta: float = 500000.0) -> List[float]:
    """Llama rotate_half RoPE, matching rope_unit's i/i+HEAD_DIM/2 pairing."""
    half = len(row) // 2
    out = list(row)
    for i in range(half):
        angle = position * theta ** (-2.0 * i / len(row))
        c, s = math.cos(angle), math.sin(angle)
        x, y = row[i], row[i + half]
        out[i] = x * c - y * s
        out[i + half] = x * s + y * c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("tests/data"))
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--target-input-vectors", type=int, default=None,
                    help="Derive SEQ_LEN to generate exactly this many BF16 input "
                         "elements across Q, K, and V.  For the default geometry, "
                         "65536 selects SEQ_LEN=30.")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--tile-dim", type=int, default=16)
    ap.add_argument("--kv-block-len", type=int, default=16,
                    help="K/V positions per hardware block. Use 4 for the "
                         "fit-oriented PYNQ-Z2 bitstream; the RTL default is 16.")
    ap.add_argument("--q-heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--query-position", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--mode", choices=("smoke", "random"), default="smoke")
    args = ap.parse_args()
    if args.q_heads % args.kv_heads or args.head_dim % args.tile_dim:
        ap.error("q-heads must be divisible by kv-heads and head-dim by tile-dim")
    if args.kv_block_len <= 0:
        ap.error("kv-block-len must be positive")
    if args.seq_len is not None and args.target_input_vectors is not None:
        ap.error("--seq-len and --target-input-vectors are mutually exclusive")
    if args.target_input_vectors is not None:
        q_elements = args.q_heads * args.head_dim
        elements_per_position = 2 * args.kv_heads * args.head_dim  # K and V
        remainder = args.target_input_vectors - q_elements
        if remainder <= 0 or remainder % elements_per_position:
            ap.error("target-input-vectors must equal q-heads*head-dim + "
                     "N*(2*kv-heads*head-dim), for a positive integer N")
        args.seq_len = remainder // elements_per_position
    if args.seq_len is None:
        args.seq_len = 16
    if args.seq_len <= 0:
        ap.error("seq-len must be positive")
    group = args.q_heads // args.kv_heads
    qpos = args.seq_len - 1 if args.query_position is None else args.query_position
    if not 0 <= qpos < args.seq_len:
        ap.error("query-position must be in [0, seq-len)")
    rng = random.Random(args.seed)
    # The score/value blocks are fixed width.  Keep the logical sequence
    # length in the config/golden model, but emit zero K/V rows through the
    # end of the final block; hardware masks these positions.
    kv_input_len = ((args.seq_len + args.kv_block_len - 1) // args.kv_block_len) * args.kv_block_len

    # q[q_head][d], k/v[kv_head][position][d], all held as Python floats.
    if args.mode == "smoke":
        q = [[0.0] * args.head_dim for _ in range(args.q_heads)]
        k = [[[0.0] * args.head_dim for _ in range(kv_input_len)]
             for _ in range(args.kv_heads)]
        # Values stay in a small normal BF16 range and differ in every axis.
        v = [[[(0.03125 * (1 + h) + 0.0078125 * p + 0.001953125 * (d % 17))
               if p < args.seq_len else 0.0
               for d in range(args.head_dim)] for p in range(kv_input_len)]
             for h in range(args.kv_heads)]
    else:
        q = [[rng.uniform(-0.25, 0.25) for _ in range(args.head_dim)]
             for _ in range(args.q_heads)]
        k = [[[(rng.uniform(-0.25, 0.25) if p < args.seq_len else 0.0)
              for _ in range(args.head_dim)] for p in range(kv_input_len)]
             for _ in range(args.kv_heads)]
        v = [[[(rng.uniform(-0.25, 0.25) if p < args.seq_len else 0.0)
              for _ in range(args.head_dim)] for p in range(kv_input_len)]
             for _ in range(args.kv_heads)]

    q_bits = [[f32_to_bf16(x) for x in row] for row in q]
    k_bits = [[[f32_to_bf16(x) for x in row] for row in head] for head in k]
    v_bits = [[[f32_to_bf16(x) for x in row] for row in head] for head in v]

    # AXIS order used by tb_gqa_attention_wrapper.sv: one group of Q rows,
    # then all K rows and V rows for its associated KV head.
    q_beats = [beat for row in q_bits for beat in chunks(row, args.tile_dim)]
    # The AXIS testbench transports complete KV blocks.  Emit the padded
    # positions too (they are zero-filled above and masked by the logical
    # sequence length in RTL), otherwise a non-multiple-of-16 sequence makes
    # $readmemh leave the final transport rows uninitialized.
    k_beats = [beat for h in range(args.kv_heads) for p in range(kv_input_len)
               for beat in chunks(k_bits[h][p], args.tile_dim)]
    v_beats = [beat for h in range(args.kv_heads) for p in range(kv_input_len)
               for beat in chunks(v_bits[h][p], args.tile_dim)]

    # Reference uses BF16-quantized inputs, ideal FP32 RoPE/softmax, and the
    # Llama half-split rotation convention.  RTL uses approximate BF16 units,
    # hence the ULP tolerance in the testbench/checker.
    q_ref = [rope_half_split([bf16_to_f32(x) for x in row], qpos) for row in q_bits]
    k_ref = [[rope_half_split([bf16_to_f32(x) for x in row], p)
              for p, row in enumerate(head[:args.seq_len])] for head in k_bits]
    v_ref = [[[bf16_to_f32(x) for x in row] for row in head[:args.seq_len]] for head in v_bits]
    golden = []
    scale = 1.0 / math.sqrt(args.head_dim)
    for qh in range(args.q_heads):
        kh = qh // group
        scores = [sum(q_ref[qh][d] * k_ref[kh][p][d] for d in range(args.head_dim)) * scale
                  for p in range(qpos + 1)]
        max_score = max(scores)
        weights = [math.exp(score - max_score) for score in scores]
        normalizer = sum(weights)
        golden.append([f32_to_bf16(sum(weights[p] * v_ref[kh][p][d]
                                         for p in range(qpos + 1)) / normalizer)
                       for d in range(args.head_dim)])

    write_mem(args.out_dir / "gqa_q.mem", q_beats)
    write_mem(args.out_dir / "gqa_k.mem", k_beats)
    write_mem(args.out_dir / "gqa_v.mem", v_beats)
    write_mem(args.out_dir / "gqa_golden.mem", golden)
    (args.out_dir / "gqa_config.txt").write_text(
        f"SEQ_LEN={args.seq_len}\nHEAD_DIM={args.head_dim}\nTILE_DIM={args.tile_dim}\n"
        f"N_Q_HEADS={args.q_heads}\nN_KV_HEADS={args.kv_heads}\nGROUP_SIZE={group}\n"
        f"KV_BLOCK_LEN={args.kv_block_len}\n"
        f"QUERY_POSITION={qpos}\nKV_INPUT_SEQ_LEN={kv_input_len}\nSCALE_BF16=0x{scale_for_head_dim(args.head_dim):04x}\n"
        f"MODE={args.mode}\nSEED={args.seed}\n"
        f"VALID_INPUT_BF16_VECTORS={len(q_bits) * args.head_dim + 2 * args.kv_heads * args.seq_len * args.head_dim}\n"
        f"TRANSPORT_BF16_VECTORS={len(q_bits) * args.head_dim + 2 * args.kv_heads * kv_input_len * args.head_dim}\n")
    print(f"Wrote {args.q_heads * args.head_dim + 2 * args.kv_heads * args.seq_len * args.head_dim} "
          f"valid BF16 input-vector elements ({kv_input_len - args.seq_len} masked K/V pad rows) "
          f"and golden output to {args.out_dir}")


if __name__ == "__main__":
    main()
