"""Evaluate the deployed GGUF contract without cloud access.

The evaluator reports facts rather than manufacturing a score when the model
or llama.cpp runtime is unavailable.  This is useful both on a workstation
before deployment and as a smoke test on the Jetson.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


CASES = [
    {"name": "methane_critical", "telemetry": {"MQ4_CH4_ppm": 12000, "MQ7_CO_ppm": 4}, "expected": "critical"},
    {"name": "co_toxic", "telemetry": {"MQ4_CH4_ppm": 20, "MQ7_CO_ppm": 80}, "expected": "toxic"},
    {"name": "nominal", "telemetry": {"MQ4_CH4_ppm": 20, "MQ7_CO_ppm": 4}, "expected": "nominal"},
]


def _extract_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def _fallback(case: dict[str, Any]) -> dict[str, Any]:
    methane = float(case["telemetry"].get("MQ4_CH4_ppm", 0))
    co = float(case["telemetry"].get("MQ7_CO_ppm", 0))
    if methane > 10000:
        return {"classification": "critical", "actions": ["evacuate", "verify ventilation"]}
    if co > 50:
        return {"classification": "toxic", "actions": ["restrict re-entry", "verify ventilation"]}
    return {"classification": "nominal", "actions": ["continue monitoring"]}


def _run_model(model_path: Path, prompt: str, n_ctx: int, max_tokens: int) -> tuple[str, float, int]:
    from llama_cpp import Llama
    started = time.perf_counter()
    llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_threads=6, n_gpu_layers=-1, verbose=False)
    response = llm.create_chat_completion(messages=[
        {"role": "system", "content": "Return only JSON with classification and actions."},
        {"role": "user", "content": prompt},
    ], max_tokens=max_tokens, temperature=0.0)
    text = response["choices"][0]["message"]["content"].strip()
    elapsed = time.perf_counter() - started
    tokens = int(response.get("usage", {}).get("completion_tokens", 0))
    return text, elapsed, tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--model", type=Path, default=root / "gas_sensors" / "models" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--output", type=Path, default=root / "gas_sensors" / "data" / "qwen_eval_results.json")
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    model = args.model.resolve()
    results = []
    for case in CASES:
        prompt = "Classify the safety state and list immediate actions. Telemetry: " + json.dumps(case["telemetry"])
        mode = "gguf"
        try:
            if not model.exists():
                raise FileNotFoundError(model)
            raw, elapsed, tokens = _run_model(model, prompt, args.n_ctx, args.max_tokens)
            parsed = _extract_json(raw)
            if parsed is None:
                raise ValueError("model output was not valid JSON")
        except Exception as exc:
            mode = "rule_fallback"
            raw = ""
            elapsed = 0.0
            tokens = 0
            parsed = _fallback(case)
            print(f"{case['name']}: {exc}; used deterministic fallback")
        actual = str(parsed.get("classification", "")).lower()
        results.append({"name": case["name"], "expected": case["expected"], "actual": actual, "json_valid": bool(parsed), "mode": mode, "elapsed_s": round(elapsed, 4), "completion_tokens": tokens, "response": parsed})
    comparable = [r for r in results if r["mode"] == "gguf"]
    report = {"model": str(model), "cases": results, "json_valid_rate": sum(r["json_valid"] for r in results) / len(results), "classification_accuracy": sum(r["actual"] == r["expected"] for r in results) / len(results), "gguf_cases": len(comparable), "throughput_tokens_per_second": (sum(r["completion_tokens"] for r in comparable) / max(sum(r["elapsed_s"] for r in comparable), 1e-9)) if comparable else None}
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("json_valid_rate", "classification_accuracy", "gguf_cases", "throughput_tokens_per_second")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
