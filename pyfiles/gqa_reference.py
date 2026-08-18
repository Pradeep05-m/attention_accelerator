"""Small, dependency-free helpers for GQA BF16 vector files.

The memory-file convention used by the RTL is little-endian by element:
element 0 occupies bits [15:0] of a packed AXI word.  Text in a .mem line is
therefore written from the highest element down to element 0.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Iterable, List


def f32_to_bf16(value: float) -> int:
    """Round a Python float to BF16, round-to-nearest-even."""
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF


def bf16_to_f32(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", (bits & 0xFFFF) << 16))[0]


def pack_bf16(values: Iterable[int]) -> str:
    values = list(values)
    return "".join(f"{value & 0xFFFF:04x}" for value in reversed(values))


def unpack_bf16(line: str, elements: int) -> List[int]:
    text = line.strip().replace("_", "")
    if len(text) != elements * 4:
        raise ValueError(f"expected {elements * 4} hex digits, got {len(text)}")
    return [int(text[-4 * (i + 1):len(text) - 4 * i if i else None], 16)
            for i in range(elements)]


def write_mem(path: Path, rows: Iterable[Iterable[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pack_bf16(row) for row in rows) + "\n")


def read_mem(path: Path, elements: int) -> List[List[int]]:
    return [unpack_bf16(line, elements) for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("//")]


def ulp_distance(a: int, b: int) -> int:
    """Monotonic BF16-code distance; suitable for finite numeric outputs."""
    def ordered(x: int) -> int:
        return 0x8000 - (x & 0x7FFF) if x & 0x8000 else 0x8000 + x
    return abs(ordered(a) - ordered(b))


def scale_for_head_dim(head_dim: int) -> int:
    return f32_to_bf16(1.0 / math.sqrt(head_dim))
