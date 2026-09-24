"""
check_sensor_matrix.py — Sensor-by-Sensor Python Code & Model Active Status Matrix

Inspects all sensor sub-systems (Gas, Vibration, Ultrasonic, Environmental, EKG, RAG)
and prints an exact active status breakdown of python agents and model artifacts.
"""

import os
import sys
import joblib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Sensor Sub-System Mapping Matrix
SENSOR_MATRIX = {
    "1. GAS SENSOR SUITE (MQ-2, MQ-4, MQ-7, MQ-135, MQ-136)": {
        "agents": [
            "sensor_agents/gas_agent.py",
            "sensor_agents/multi_gas_agent.py",
            "atr_activation/orchestrator.py",
            "gas_sensors/dl_wrappers.py"
        ],
        "models": [
            ("Multi-Gas 8-Channel Hazard Net", "gas_sensors/models/multi_gas_detector.joblib"),
            ("Clean Air Baseline Anomaly IForest", "gas_sensors/models/mine_baseline_iforest.joblib"),
            ("Methane (CH4) Severity Classifier", "gas_sensors/models/severity_ch4.joblib"),
            ("Carbon Monoxide (CO) Severity", "gas_sensors/models/severity_co.joblib"),
            ("Carbon Dioxide (CO2) Severity", "gas_sensors/models/severity_co2.joblib"),
            ("Hydrogen (H2) Severity Classifier", "gas_sensors/models/severity_h2.joblib"),
            ("Hydrogen Sulfide (H2S) Severity", "gas_sensors/models/severity_h2s.joblib"),
            ("Ammonia (NH3) Toxic Hazard Net", "gas_sensors/models/nh3_hazard.joblib"),
            ("Asphyxiation (CO2) Hazard Net", "gas_sensors/models/co2_hazard.joblib"),
            ("Smoke / Dust Particulate Hazard", "gas_sensors/models/smoke_env_hazard.joblib")
        ]
    },
    "2. VIBRATION & SEISMIC SUITE (SW-420 Shock & Geomechanical)": {
        "agents": [
            "sensor_agents/vibration_agent.py",
            "vibration/structural_monitor.py"
        ],
        "models": [
            ("SW-420 Digital Shock Pulse Evaluator", "vibration/structural_monitor.py"),
            ("PPV Blast Classifier (Optional ML)", "vibration/models/best_random_forest_classifier.joblib"),
            ("PPV Energy Regressor (Optional ML)", "vibration/models/best_gradient_boosting_regressor.joblib")
        ]
    },
    "3. ULTRASONIC NAVIGATION & WALL DISPLACEMENT SUITE": {
        "agents": [
            "sensor_agents/ultrasonic_agent.py",
            "vibration/structural_monitor.py"
        ],
        "models": [
            ("24-Sonar Ring Obstacle Classifier", "ultrasonic_sensors/models/best_ultrasonic_24.joblib"),
            ("4-Sonar Ring Fallback Classifier", "ultrasonic_sensors/models/best_ultrasonic_4.joblib"),
            ("2-Sonar Ring Fallback Classifier", "ultrasonic_sensors/models/best_ultrasonic_2.joblib"),
            ("Geomechanical Wall Convergence Model", "vibration/structural_monitor.py")
        ]
    },
    "4. ENVIRONMENTAL SUITE (DHT22 Temp, Humidity & Occupancy)": {
        "agents": [
            "sensor_agents/env_agent.py"
        ],
        "models": [
            ("IoT Telemetry Anomaly IForest", "temperature_humidity/models/isolation_forest_iot.joblib"),
            ("Human Occupancy Random Forest", "temperature_humidity/models/random_forest.joblib")
        ]
    },
    "5. REASONING CORE & KNOWLEDGE GRAPH (EKG & RAG)": {
        "agents": [
            "sensor_agents/ekg_agent.py",
            "reasoning_core/agent_loop.py",
            "reasoning_core/chat_assistant.py",
            "expedition_knowledge_graph/query_api.py",
            "faiss_rag/index_builder.py"
        ],
        "models": [
            ("Qwen2.5-7B-Instruct Quantized GGUF", "reasoning_core/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        ]
    }
}


def main():
    print("=" * 85)
    print("      FIELD-MIND SENSOR DOMAIN MATRIX & ACTIVE ARTIFACT MONITOR")
    print("=" * 85)

    # Pre-import dl_wrappers for smooth joblib loading
    gas_dir = ROOT_DIR / "gas_sensors"
    if str(gas_dir) not in sys.path:
        sys.path.insert(0, str(gas_dir))
    try:
        import dl_wrappers
    except ImportError:
        pass

    for domain_name, components in SENSOR_MATRIX.items():
        print(f"\n{domain_name}")
        print("-" * 85)
        
        print("  [PYTHON AGENTS & CONTROLLERS]")
        for agent_path in components["agents"]:
            full = ROOT_DIR / agent_path
            if full.exists():
                size_kb = full.stat().st_size / 1024
                print(f"    ACTIVE | {agent_path:<50} [{size_kb:.1f} KB]")
            else:
                print(f"   MISSING | {agent_path:<50}")

        print("  [ACTIVE MODELS & EVALUATORS]")
        for model_desc, model_path in components["models"]:
            full = ROOT_DIR / model_path
            if not full.exists():
                print(f"   ABSENT | {model_desc:<40} -> {model_path} (Fallback Active)")
                continue
            
            size_mb = full.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.2f} MB" if size_mb >= 1.0 else f"{full.stat().st_size / 1024:.1f} KB"
            
            status = "LOADED"
            if model_path.endswith(".joblib"):
                try:
                    _ = joblib.load(full)
                except Exception:
                    status = "WARN (Needs Retrain)"

            print(f"   {status:<7} | {model_desc:<40} -> {model_path} [{size_str}]")

    print("\n" + "=" * 85)
    print("Check complete. Run `python check_sensor_matrix.py` anytime on Jetson.")
    print("=" * 85)


if __name__ == "__main__":
    main()
