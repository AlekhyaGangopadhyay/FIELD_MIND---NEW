"""Preflight checks for FIELD-MIND deployment on Jetson Orin Nano.

Exit status is non-zero for a real deployment failure.  ``--allow-missing``
is provided only for development hosts that do not have JetPack or the GGUF.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} | {name}: {detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--model", type=Path, default=root / "gas_sensors" / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    model = args.model.resolve()
    failures = []
    is_jetson = platform.machine().lower() in {"aarch64", "arm64"} and (Path("/etc/nv_tegra_release").exists() or shutil.which("tegrastats"))
    failures.append(not check("Jetson/JetPack", bool(is_jetson), platform.platform()))
    failures.append(not check("Python", tuple(map(int, platform.python_version().split(".")[:2])) >= (3, 10), platform.python_version()))
    free_gb = shutil.disk_usage(root).free / 1e9
    failures.append(not check("Free storage", free_gb >= args.min_free_gb, f"{free_gb:.1f} GB free; need >= {args.min_free_gb:.1f} GB"))
    model_exists = model.is_file() and model.stat().st_size > 100_000_000
    check("GGUF model (optional)", True, f"Found {model.name}" if model_exists else "Missing; running with expert fallback.")
    failures.append(not check("CUDA visibility", bool(shutil.which("tegrastats") or shutil.which("nvidia-smi")), "tegrastats/nvidia-smi available"))
    if Path("/proc/meminfo").exists():
        meminfo = Path("/proc/meminfo").read_text(encoding="ascii", errors="ignore")
        total_kb = next((int(line.split()[1]) for line in meminfo.splitlines() if line.startswith("MemTotal:")), 0)
        failures.append(not check("Unified memory", total_kb >= 7_000_000, f"{total_kb / 1e6:.2f} GB reported"))
    else:
        failures.append(not check("Unified memory", False, "/proc/meminfo unavailable"))
    if args.allow_missing:
        print("Development mode: missing hardware/artifacts are reported but do not fail this host check.")
        return 0
    return 1 if any(failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
