#!/usr/bin/env python3
"""Run reproducible end-to-end GQA wrapper regressions in Vivado/XSim."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


CASES = (
    ("smoke_len1", "smoke", 1, None, False),
    ("random_len15", "random", 15, 20260815, False),
    ("random_len16", "random", 16, 20260816, False),
    ("random_len17_bp", "random", 17, 20260817, True),
    ("random_len30", "random", 30, 20260805, False),
    ("random_len32_bp", "random", 32, 20260832, True),
)


def execute(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, env=env, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("build/gqa_regression"))
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    vivado = os.environ.get("VIVADO") or shutil.which("vivado") or "/home/prad/2025.2/Vivado/bin/vivado"
    if not args.generate_only and not Path(vivado).is_file():
        raise SystemExit(f"Vivado not found: {vivado}. Set VIVADO=/path/to/vivado or use --generate-only.")

    failures: list[str] = []
    for name, mode, seq_len, seed, backpressure in CASES:
        case_dir, vectors = out_dir / name, out_dir / name / "vectors"
        vectors.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(root / "pyfiles/gen_gqa_wrapper_vectors.py"),
                   "--out-dir", str(vectors), "--mode", mode, "--seq-len", str(seq_len)]
        if seed is not None:
            command += ["--seed", str(seed)]
        result = execute(command, os.environ.copy())
        print(f"[{name}] vector generation\n{result.stdout}", end="")
        if result.returncode:
            failures.append(name)
        elif not args.generate_only:
            env = os.environ.copy()
            env.update({"VECTORS_DIR": str(vectors), "SIM_BUILD_DIR": str(case_dir / "sim")})
            if backpressure:
                env["OUTPUT_BACKPRESSURE"] = "1"
            result = execute([vivado, "-mode", "batch", "-source", "scripts/run_gqa_wrapper_sim.tcl"], env)
            (case_dir / "vivado.log").write_text(result.stdout)
            passed = result.returncode == 0 and "PASS:" in result.stdout
            print(f"[{name}] {'PASS' if passed else 'FAIL'} (log: {case_dir / 'vivado.log'})")
            if not passed:
                failures.append(name)
        if failures and not args.keep_going:
            break

    if not args.generate_only and not failures:
        name, case_dir = "invalid_config", out_dir / "invalid_config"
        env = os.environ.copy()
        env.update({"VECTORS_DIR": str(out_dir / "smoke_len1" / "vectors"),
                    "SIM_BUILD_DIR": str(case_dir / "sim"), "INVALID_CONFIG": "1"})
        result = execute([vivado, "-mode", "batch", "-source", "scripts/run_gqa_wrapper_sim.tcl"], env)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "vivado.log").write_text(result.stdout)
        passed = result.returncode == 0 and "PASS: invalid configuration rejected" in result.stdout
        print(f"[{name}] {'PASS' if passed else 'FAIL'} (log: {case_dir / 'vivado.log'})")
        if not passed:
            failures.append(name)
    if failures:
        print("GQA REGRESSION FAILED: " + ", ".join(failures))
        return 1
    if args.generate_only:
        print(f"GQA REGRESSION VECTORS GENERATED: {len(CASES)} datapath cases")
    else:
        print(f"GQA REGRESSION PASSED: {len(CASES)} datapath cases plus invalid-config rejection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
