#!/usr/bin/env python3
"""Generate reproducible BF16 add/multiply conformance vectors."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from gqa_reference import bf16_to_f32, f32_to_bf16


def write_words(path: Path, words: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{word:04x}" for word in words) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("tests/data/bf16_primitives"))
    ap.add_argument("--cases", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=20260805)
    args = ap.parse_args()
    if args.cases <= 0:
        ap.error("--cases must be positive")

    rng = random.Random(args.seed)
    # Keep operands finite and normal; special values need a separate IEEE
    # policy test and must not obscure arithmetic-conformance failures.
    a = [f32_to_bf16(rng.uniform(-0.25, 0.25)) for _ in range(args.cases)]
    b = [f32_to_bf16(rng.uniform(-0.25, 0.25)) for _ in range(args.cases)]
    add = [f32_to_bf16(bf16_to_f32(x) + bf16_to_f32(y)) for x, y in zip(a, b)]
    mul = [f32_to_bf16(bf16_to_f32(x) * bf16_to_f32(y)) for x, y in zip(a, b)]
    write_words(args.out_dir / "a.mem", a)
    write_words(args.out_dir / "b.mem", b)
    write_words(args.out_dir / "add_golden.mem", add)
    write_words(args.out_dir / "mul_golden.mem", mul)
    (args.out_dir / "config.txt").write_text(f"CASES={args.cases}\nSEED={args.seed}\n")
    print(f"Wrote {args.cases} BF16 add/multiply cases to {args.out_dir}")


if __name__ == "__main__":
    main()
