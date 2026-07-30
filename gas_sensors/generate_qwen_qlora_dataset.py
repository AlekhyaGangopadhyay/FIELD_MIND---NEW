"""Build a deterministic, audit-friendly QLoRA instruction dataset.

The generator deliberately uses only files already present in FIELD-MIND.  It
does not download data or call a cloud service, which makes it safe to run in
an offline mine environment (and easy to reproduce on a development machine).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are FIELD-MIND, an offline underground-mine safety assistant. "
    "Use the supplied telemetry and safety context. Never invent a sensor "
    "reading or override an immediate evacuation rule. If evidence is "
    "insufficient, say so and request verification."
)


def _read_text(path: Path, limit: int = 1800) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _telemetry_rows(root: Path, limit: int = 250) -> list[dict[str, Any]]:
    paths = [
        root / "gas_sensors" / "data" / "FIELDMIND_physics_dataset.csv",
        root / "gas_sensors" / "data" / "mine_part2_bands.csv",
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                for row in csv.DictReader(handle):
                    clean = {k: v for k, v in row.items() if k and v not in (None, "")}
                    if clean:
                        rows.append(clean)
                    if len(rows) >= limit:
                        return rows
        except (OSError, csv.Error):
            continue
    return rows


def _source_context(root: Path) -> str:
    docs = sorted((root / "faiss_rag" / "knowledge_base").glob("*.md"))
    excerpts = [f"[{path.name}]\n{_read_text(path)}" for path in docs[:6]]
    graph = _load_json(root / "expedition_knowledge_graph" / "data" / "mine_graph.json")
    if graph:
        excerpts.append("[mine_graph.json]\n" + json.dumps(graph, ensure_ascii=False)[:1800])
    return "\n\n".join(excerpts)[:10000]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _answer(row: dict[str, Any], task: str) -> str:
    methane = _num(row.get("MQ4_CH4_ppm"))
    carbon_monoxide = _num(row.get("MQ7_CO_ppm"))
    temperature = _num(row.get("Temp_C", row.get("temp")))
    hazard = _num(row.get("Hazard_Alert"))
    if task == "classify":
        status = "hazard" if hazard == 1 or methane > 10000 or carbon_monoxide > 50 else "no confirmed hazard"
        return json.dumps({"status": status, "evidence": {"methane_ppm": methane, "co_ppm": carbon_monoxide}}, ensure_ascii=False)
    if task == "prioritize":
        actions = ["continue monitoring"]
        if methane > 10000:
            actions = ["raise critical gas alarm", "evacuate affected area", "isolate ignition sources", "verify ventilation"]
        elif carbon_monoxide > 50:
            actions = ["raise toxic-gas alarm", "restrict re-entry", "verify ventilation", "confirm with calibrated instrument"]
        elif temperature > 35:
            actions = ["raise heat-stress alert", "increase cooling/ventilation", "check personnel welfare"]
        return json.dumps({"priority": "critical" if len(actions) > 1 else "routine", "actions": actions}, ensure_ascii=False)
    return json.dumps({"answer": "Use the supplied evidence, apply the site safety procedure, and verify with a calibrated sensor before re-entry."}, ensure_ascii=False)


def build_examples(root: Path, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = _telemetry_rows(root)
    if not rows:
        rows = [{"MQ4_CH4_ppm": 0, "MQ7_CO_ppm": 0, "Temp_C": 22, "Hazard_Alert": 0}]
    context = _source_context(root)
    tasks = ("classify", "prioritize", "explain")
    examples: list[dict[str, Any]] = []
    for index in range(count):
        row = dict(rows[index % len(rows)])
        # Keep generation deterministic while varying phrasing for generalization.
        task = tasks[index % len(tasks)]
        wording = rng.choice((
            "Assess this telemetry for an underground mine.",
            "Give a conservative safety assessment of this sensor frame.",
            "What should the local safety agent do with this reading?",
        ))
        user = f"{wording}\nTask: {task}\nTelemetry: {json.dumps(row, ensure_ascii=False, sort_keys=True)}\nSafety context:\n{context}"
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": _answer(row, task)},
            ],
            "metadata": {"source": "FIELD-MIND local files", "task": task, "example_id": index},
        })
    return examples


def write_jsonl(examples: Iterable[dict[str, Any]], output: Path, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}. Use --overwrite.")
    fd, temp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    root = args.root.resolve()
    output = (args.output or root / "gas_sensors" / "data" / "qwen_mining_instructions.jsonl").resolve()
    examples = build_examples(root, args.count, args.seed)
    write_jsonl(examples, output, args.overwrite)
    print(f"Wrote {len(examples)} instruction pairs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
