"""Convert merged HuggingFace weights to a Jetson-ready GGUF artifact."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--merged-dir", type=Path, default=root / "gas_sensors" / "models" / "qwen2_5_mining" / "merged_fp16")
    parser.add_argument("--llama-cpp", type=Path, required=True, help="Path to a checked-out llama.cpp directory.")
    parser.add_argument("--output", type=Path, default=root / "gas_sensors" / "models" / "Qwen2.5-7B-Instruct-mining-f16.gguf")
    parser.add_argument("--quantized-output", type=Path, default=root / "gas_sensors" / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--quantization", default="Q4_K_M")
    args = parser.parse_args()
    merged = args.merged_dir.resolve()
    llama = args.llama_cpp.resolve()
    if not merged.is_dir():
        parser.error(f"Merged model directory not found: {merged}")
    converter = llama / "convert_hf_to_gguf.py"
    quantizer = llama / "llama-quantize.exe"
    if not quantizer.exists():
        quantizer = llama / "build" / "bin" / "llama-quantize"
    if not converter.exists():
        parser.error(f"Missing converter: {converter}")
    if not quantizer.exists():
        parser.error("Missing llama-quantize binary; build llama.cpp before conversion.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.quantized_output.parent.mkdir(parents=True, exist_ok=True)
    run_checked(["python", str(converter), str(merged), "--outfile", str(args.output), "--outtype", "f16"], cwd=llama)
    run_checked([str(quantizer), str(args.output), str(args.quantized_output), args.quantization], cwd=llama)
    if args.quantized_output.stat().st_size < 100_000_000:
        raise RuntimeError("Quantized GGUF is unexpectedly small; refusing to publish a likely-invalid artifact.")
    print(f"Created {args.quantized_output} ({args.quantized_output.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
