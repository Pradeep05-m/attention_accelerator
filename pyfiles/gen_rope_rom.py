"""
gen_rope_rom.py
================
Generates the two ROM files rope_unit.v loads via $readmemh:
    rope_rom/rope_phase_rom.mem   (N_PAIRS=64 entries,   32-bit hex)
    rope_rom/rope_sincos_rom.mem  (2**ROM_ADDR_BITS=1024 entries, 32-bit hex)

These files do not exist yet in the project -- without them $readmemh will
leave both ROMs as X in simulation and every RoPE-rotated value (i.e.
everything downstream of rope_unit.v) will be X. Run this once and drop the
rope_rom/ folder next to the simulation sources (same relative path
$readmemh uses).

Derivation (reverse-engineered from rope_unit.v's actual address logic --
see that file's comments):

    phase_wrapped = (phase_rom[pair_idx] * pos) mod 2**PHASE_WIDTH
    rom_addr      = top ROM_ADDR_BITS bits of phase_wrapped

phase_wrapped / 2**PHASE_WIDTH is therefore the fractional part of the total
rotation (in turns, not radians) after `pos` steps, and rom_addr is that
fraction quantized to 2**ROM_ADDR_BITS points around one full turn. So:

    phase_rom[i] = round(2**PHASE_WIDTH * freq_i / (2*pi)) mod 2**PHASE_WIDTH

where freq_i is the standard RoPE angular frequency for pair i:

    freq_i = theta ** (-2*i / HEAD_DIM),   i = 0 .. HEAD_DIM/2 - 1

Llama 3 8B's rope_theta = 500000.0 (confirmed from the model's published
config.json: hidden_size=4096, num_attention_heads=32 -> head_dim=128,
num_key_value_heads=8, matching this project's parameters exactly).

sincos_rom[addr] = {cos(2*pi*addr/2**ROM_ADDR_BITS)[BF16], sin(...)[BF16]},
packed as rope_unit.v's comment specifies: cos in bits [31:16], sin in [15:0].
"""
import os
import math
from pathlib import Path

from gqa_reference import f32_to_bf16

HEAD_DIM      = 128
N_PAIRS       = HEAD_DIM // 2     # 64
PHASE_WIDTH   = 32
ROM_ADDR_BITS = 14                # 16384-point sin/cos table
ROPE_THETA    = 500000.0          # Llama 3 8B (confirmed via config.json)

OUT_DIR = "rope_rom"


def build_phase_rom():
    freq = [ROPE_THETA ** (-(2.0 * i) / HEAD_DIM) for i in range(N_PAIRS)]
    phase = [round((value / (2.0 * math.pi)) * (2 ** PHASE_WIDTH)) % (2 ** PHASE_WIDTH)
             for value in freq]
    return phase, freq


def build_sincos_rom():
    n = 1 << ROM_ADDR_BITS
    return [(f32_to_bf16(math.cos(2.0 * math.pi * addr / n)) << 16) |
            f32_to_bf16(math.sin(2.0 * math.pi * addr / n))
            for addr in range(n)]


def write_mem(path, values, hex_digits):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v):0{hex_digits}x}\n")


def main():
    phase_rom, freq = build_phase_rom()
    sincos_rom = build_sincos_rom()

    write_mem(os.path.join(OUT_DIR, "rope_phase_rom.mem"), phase_rom, 8)
    write_mem(os.path.join(OUT_DIR, "rope_sincos_rom.mem"), sincos_rom, 8)

    print(f"wrote {N_PAIRS} entries to {OUT_DIR}/rope_phase_rom.mem")
    print(f"wrote {1<<ROM_ADDR_BITS} entries to {OUT_DIR}/rope_sincos_rom.mem")
    print()
    print("sanity check -- freq_i (radians/step) for a few pairs:")
    for i in (0, 1, 2, 8, 16, 32, 63):
        print(f"  pair {i:2d}: freq={freq[i]:.8e} rad/step  phase_rom[{i}]=0x{int(phase_rom[i]):08x}")


if __name__ == "__main__":
    main()
