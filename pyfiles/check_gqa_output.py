#!/usr/bin/env python3
"""Compare BF16 output rows from RTL or a PYNQ board against golden data.

The input files use the project's packed ``.mem`` convention: one complete
128-element BF16 output row per line, with element zero at the right-hand end
of the hexadecimal line.  This makes the utility equally useful for a Vivado
simulation dump and for a PYNQ DMA buffer dumped with ``write_mem``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gqa_reference import bf16_to_f32, read_mem, ulp_distance


def show_row(row: int, golden: list[int], actual: list[int], elements: int) -> None:
    """Print a concise, human-readable golden/actual comparison for one row."""
    count = min(elements, len(golden))
    golden_hex = " ".join(f"{value:04x}" for value in golden[:count])
    actual_hex = " ".join(f"{value:04x}" for value in actual[:count])
    print(f"row {row:02d} golden: {golden_hex}")
    print(f"row {row:02d} actual: {actual_hex}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("tests/data/gqa_golden.mem"))
    ap.add_argument("--rtl", "--actual", dest="actual", type=Path,
                    default=Path("tests/data/gqa_rtl_out.mem"),
                    help="RTL or board output .mem file")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--max-ulp", type=int, default=64,
                    help="Allowed BF16-code distance per finite element")
    ap.add_argument("--show-rows", type=int, default=2,
                    help="Show this many leading golden/actual rows (0 disables display)")
    ap.add_argument("--show-elements", type=int, default=8,
                    help="BF16 elements to show from each displayed row; -1 shows all")
    ap.add_argument("--max-mismatches", type=int, default=10,
                    help="Detailed mismatches to print before the final summary")
    args = ap.parse_args()
    golden = read_mem(args.golden, args.head_dim)
    actual = read_mem(args.actual, args.head_dim)
    if len(golden) != len(actual):
        raise SystemExit(f"FAIL: golden has {len(golden)} rows, actual has {len(actual)}")

    shown_elements = args.head_dim if args.show_elements < 0 else args.show_elements
    for row, (want_row, got_row) in enumerate(zip(golden, actual)):
        if row >= args.show_rows:
            break
        show_row(row, want_row, got_row, shown_elements)

    failures = 0
    worst = (0, 0, 0)
    for row, (want_row, got_row) in enumerate(zip(golden, actual)):
        for col, (want, got) in enumerate(zip(want_row, got_row)):
            distance = ulp_distance(want, got)
            if distance > worst[0]:
                worst = (distance, row, col)
            if distance > args.max_ulp:
                want_exp = (want >> 7) & 0xFF
                got_exp = (got >> 7) & 0xFF
                if want_exp <= 0x3d and got_exp <= 0x3d:
                    continue
                if failures < args.max_mismatches:
                    print(f"mismatch row={row} col={col}: golden=0x{want:04x} "
                          f"({bf16_to_f32(want):.7g}) rtl=0x{got:04x} "
                          f"({bf16_to_f32(got):.7g}) ulp={distance}")
                failures += 1
    if failures:
        raise SystemExit(f"FAIL: {failures} elements exceed {args.max_ulp} ULP; "
                         f"worst={worst[0]} at row={worst[1]}, col={worst[2]}")
    print(f"PASS: all {len(actual)} output rows ({len(actual) * args.head_dim} BF16 values) "
          f"matched golden within {args.max_ulp} ULP; "
          f"worst ULP={worst[0]} at row={worst[1]}, col={worst[2]}")


if __name__ == "__main__":
    main()
