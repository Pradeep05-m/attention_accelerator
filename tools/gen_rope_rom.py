#!/usr/bin/env python3
"""Generate the synthesizable RoPE ROM contents used by rope_unit.v.

The hardware represents phase as turns in unsigned Q0.32.  The phase ROM
contains one phase increment per Llama rotate_half pair; the sine/cosine ROM
contains 1024 uniformly-spaced samples, stored as {cos_bf16, sin_bf16}.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def f32_to_bf16_bits(value: float) -> int:
    """IEEE round-to-nearest-even float32 -> BF16 bit conversion."""
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="rope_rom", type=Path)
    parser.add_argument("--head-dim", default=128, type=int)
    parser.add_argument("--rope-theta", default=500000.0, type=float)
    parser.add_argument("--rom-addr-bits", default=10, type=int)
    args = parser.parse_args()

    if args.head_dim <= 0 or args.head_dim % 2:
        parser.error("--head-dim must be a positive even integer")
    if args.rom_addr_bits < 1:
        parser.error("--rom-addr-bits must be at least 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = args.head_dim // 2
    entries = 1 << args.rom_addr_bits

    # Llama's rotate_half maps element i to i + head_dim/2.  Its i-th pair
    # uses inv_freq[i] = theta^(-i/(head_dim/2)).
    phase_lines = []
    for i in range(pairs):
        inv_freq = args.rope_theta ** (-float(i) / pairs)
        phase_step = int(round(inv_freq * (1 << 32) / (2.0 * math.pi))) & 0xFFFFFFFF
        phase_lines.append(f"{phase_step:08x}\n")

    sincos_lines = []
    for i in range(entries):
        angle = 2.0 * math.pi * i / entries
        cos_bf16 = f32_to_bf16_bits(math.cos(angle))
        sin_bf16 = f32_to_bf16_bits(math.sin(angle))
        sincos_lines.append(f"{cos_bf16:04x}{sin_bf16:04x}\n")

    (args.out_dir / "rope_phase_rom.mem").write_text("".join(phase_lines), encoding="ascii")
    (args.out_dir / "rope_sincos_rom.mem").write_text("".join(sincos_lines), encoding="ascii")


if __name__ == "__main__":
    main()
