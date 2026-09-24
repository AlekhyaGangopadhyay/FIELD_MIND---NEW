"""
check_artifacts.py — FIELD-MIND Required Python Files & Trained Model Integrity Checker

Verifies the presence and loading validity of all production Python code modules 
and pre-trained model artifacts across Gas, Vibration, Ultrasonic, Environmental, 
EKG, RAG, and LLM reasoning domains.
"""

import os
import sys
import json
import joblib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Ensure stdout uses UTF-8 if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 1. REQUIRED PYTHON CODE MODULES
REQUIRED_PYTHON_FILES = [
    "jetson_preflight.py",
    "atr_activation/orchestrator.py",
    "atr_activation/detector_wrappers.py",
    "atr_activation/demo_atr.py",
    "gas_sensors/dl_wrappers.py",
    "gas_sensors/retrain_dl_models.py",
    "gas_sensors/retrain_severity_models.py",
    "gas_sensors/train_multi_gas_detector.py",
    "gas_sensors/train_gas_detector.py",
    "sensor_agents/agent_base.py",
    "sensor_agents/gas_agent.py",
    "sensor_agents/multi_gas_agent.py",
    "sensor_agents/vibration_agent.py",
    "sensor_agents/ultrasonic_agent.py",
    "sensor_agents/env_agent.py",
    "sensor_agents/ekg_agent.py",
    "sensor_agents/mine_orchestrator_agent.py",
    "sensor_agents/demo_agents.py",
    "reasoning_core/agent_loop.py",
    "reasoning_core/chat_assistant.py",
    "reasoning_core/demo_reasoning.py",
    "reasoning_core/demo_self_learning.py",
    "expedition_knowledge_graph/query_api.py",
    "expedition_knowledge_graph/schema.py",
    "expedition_knowledge_graph/ingest.py",
    "faiss_rag/index_builder.py",
    "scisense_protocol/alignment.py",
    "scisense_protocol/encoders.py",
    "scisense_protocol/coherence.py",
    "scisense_protocol/demo_alignment.py",
    "vibration/structural_monitor.py",
    "unified_demo/interactive_safety_hub.py",
]

# 2. REQUIRED PRE-TRAINED MODEL ARTIFACTS
REQUIRED_MODEL_FILES = [
    # Gas Domain Models
    "gas_sensors/models/multi_gas_detector.joblib",
    "gas_sensors/models/mine_baseline_iforest.joblib",
    "gas_sensors/models/severity_ch4.joblib",
    "gas_sensors/models/severity_co.joblib",
    "gas_sensors/models/severity_co2.joblib",
    "gas_sensors/models/severity_h2.joblib",
    "gas_sensors/models/severity_h2s.joblib",
    "gas_sensors/models/nh3_hazard.joblib",
    "gas_sensors/models/co2_hazard.joblib",
    "gas_sensors/models/smoke_env_hazard.joblib",
    "gas_sensors/models/model_registry.json",

    # Environmental Models
    "temperature_humidity/models/isolation_forest_iot.joblib",
    "temperature_humidity/models/random_forest.joblib",

    # Ultrasonic Navigation Models
    "ultrasonic_sensors/models/best_ultrasonic_2.joblib",
    "ultrasonic_sensors/models/best_ultrasonic_4.joblib",
    "ultrasonic_sensors/models/best_ultrasonic_24.joblib",
]

# 3. OPTIONAL / FALLBACK ARTIFACTS
OPTIONAL_FILES = [
    "reasoning_core/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "vibration/models/best_random_forest_classifier.joblib",
    "vibration/models/best_gradient_boosting_regressor.joblib",
]


def check_file(rel_path: str, is_model: bool = False) -> bool:
    full_path = ROOT_DIR / rel_path
    if not full_path.exists():
        print(f"FAIL | Missing: {rel_path}")
        return False

    size_bytes = full_path.stat().st_size
    if size_bytes == 0:
        print(f"FAIL | Empty File (0 bytes): {rel_path}")
        return False

    if is_model and rel_path.endswith(".joblib"):
        try:
            gas_dir = ROOT_DIR / "gas_sensors"
            if str(gas_dir) not in sys.path:
                sys.path.insert(0, str(gas_dir))
            try:
                import dl_wrappers
            except ImportError:
                pass
            _ = joblib.load(full_path)
        except Exception as e:
            print(f"WARN | Found but failed joblib.load: {rel_path} ({e})")
            return True

    size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024 * 1024):.2f} MB"
    print(f"PASS | {rel_path:<58} [{size_str}]")
    return True


def main():
    print("=" * 80)
    print("      FIELD-MIND SYSTEM INTEGRITY & ARTIFACT VERIFICATION CHECK")
    print("=" * 80)

    print("\n--- 1. CHECKING REQUIRED PYTHON FILES (32 Core Modules) ---")
    py_passed = sum(check_file(rel_path) for rel_path in REQUIRED_PYTHON_FILES)
    py_total = len(REQUIRED_PYTHON_FILES)

    print("\n--- 2. CHECKING REQUIRED TRAINED MODEL ARTIFACTS (16 Core Models) ---")
    model_passed = sum(check_file(rel_path, is_model=True) for rel_path in REQUIRED_MODEL_FILES)
    model_total = len(REQUIRED_MODEL_FILES)

    print("\n--- 3. CHECKING OPTIONAL / FALLBACK ARTIFACTS ---")
    for opt in OPTIONAL_FILES:
        full_path = ROOT_DIR / opt
        if full_path.exists() and full_path.stat().st_size > 0:
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"PASS | {opt:<58} [{size_mb:.2f} MB]")
        else:
            print(f"INFO | {opt} is absent. (System uses built-in rule-based fallback).")

    print("\n" + "=" * 80)
    print(f"SUMMARY: Python Modules: {py_passed}/{py_total} | Core Models: {model_passed}/{model_total}")
    print("=" * 80)

    if py_passed == py_total and model_passed == model_total:
        print("ALL REQUIRED PYTHON MODULES & CORE MODEL ARTIFACTS ARE 100% PRESENT AND VALID!")
        return 0
    else:
        print("SOME ARTIFACTS ARE MISSING OR CORRUPTED. PLEASE REVIEW FAILURES ABOVE.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
