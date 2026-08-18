#!/usr/bin/env python3
"""Generate 512 RoPE rows (65,536 BF16 elements) and Python golden output."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from gen_gqa_wrapper_vectors import rope_half_split, chunks
from gqa_reference import bf16_to_f32, f32_to_bf16, write_mem

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("tests/data/rope"))
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260805)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [[rng.uniform(-0.25, 0.25) for _ in range(128)] for _ in range(args.rows)]
    inp = [[f32_to_bf16(x) for x in row] for row in rows]
    golden = [[f32_to_bf16(x) for x in rope_half_split([bf16_to_f32(v) for v in row], pos)]
              for pos, row in enumerate(inp)]
    write_mem(args.out_dir / "in.mem", [beat for row in inp for beat in chunks(row, 16)])
    write_mem(args.out_dir / "golden.mem", [beat for row in golden for beat in chunks(row, 16)])
    print(f"Wrote {args.rows * 128} RoPE BF16 elements to {args.out_dir}")
if __name__ == "__main__": main()
