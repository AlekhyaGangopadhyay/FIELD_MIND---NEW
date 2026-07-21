"""
interactive_safety_hub.py — Unified FIELD-MIND Interactive ML Model Safety Hub
=============================================================================
Interactive terminal application that:
  1. Requests raw features for Gas, Env, Vibration, and Navigation domains.
  2. Runs inputs through the actual pre-trained ML models via Tier1Monitor.
  3. Displays model hazard classifications (Methane, LPG, Smoke, PPV, Anomaly).
  4. Triggers conversational LangGraph reasoning using MineSafetyChatAssistant
     which retrieves safety rules from FAISS RAG and logs to EKG.

Keep in: unified_demo/
Run: py -X utf8 unified_demo/interactive_safety_hub.py
"""

import os
import sys
import time
import warnings
import numpy as np
from typing import Any, Dict

# Ignore user warnings for clean console outputs
warnings.simplefilter("ignore", category=UserWarning)

# Setup workspace root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from atr_activation.detector_wrappers import Tier1Monitor
from reasoning_core.chat_assistant import MineSafetyChatAssistant


def get_float_input(prompt: str, default: float) -> float:
    """Prompt user for a float value with safe parsing and default fallback."""
    val_str = input(f"{prompt} [default: {default}]: ").strip()
    if not val_str:
        return default
    try:
        return float(val_str)
    except ValueError:
        print(f"  ⚠ Invalid numeric input. Falling back to default: {default}")
        return default


def main():
    print("\n" + "█" * 80)
    print("  FIELD-MIND — Unified Interactive ML Model Safety Hub")
    print("  Actual ML Inference + FAISS RAG Grounding + EKG Graph Memory")
    print("█" * 80 + "\n")

    # 1. Initialize resources
    print("[Setup] Loading pre-trained ML models via Tier1Monitor...")
    try:
        monitor = Tier1Monitor(root_dir=WORKSPACE_ROOT)
        print("  ✓ ML models loaded successfully.")
    except Exception as e:
        print(f"  ✗ Failed to load ML models: {e}")
        sys.exit(1)

    print("\n[Setup] Loading safety retrieval resources...")
    try:
        assistant = MineSafetyChatAssistant(workspace_root=WORKSPACE_ROOT)
        if assistant.rag_retriever:
            print("[Setup] Loading the local FAISS embedding model (first run may take several seconds)...")
            try:
                warmup_ms = assistant.rag_retriever.warmup()
                print(f"[Setup] FAISS embedder ready in {warmup_ms:.0f} ms.")
            except Exception as warmup_error:
                print(f"[Setup] FAISS warm-up skipped: {warmup_error}")
        print("  ✓ Conversational resources loaded successfully.")
    except Exception as e:
        print(f"  ✗ Failed to initialize chat assistant: {e}")
        sys.exit(1)

    print("\n" + "─" * 80)
    print("📋 STEP 1: Enter Current Mine Telemetry Inputs")
    print("─" * 80)

    # Segment
    segment_id = input("Enter Tunnel Segment ID [default: TUNNEL_A1]: ").strip().upper()
    if not segment_id:
        segment_id = "TUNNEL_A1"
    elif segment_id.isdigit():
        segment_id = f"TUNNEL_{segment_id}"

    # Gas concentrations
    print("\n--- [Gas Sensor Features] ---")
    ch4 = get_float_input("  Methane (CH4) level in ppm (Typical: 0-20000)", default=400.0)
    co = get_float_input("  Carbon Monoxide (CO) in ppm (Typical: 0-100)", default=12.0)
    lpg = get_float_input("  LPG/CNG level in ppm (Typical: 0-2000)", default=80.0)
    nox = get_float_input("  Nitrogen Oxides (NOx) in ppm (Typical: 0-50)", default=2.1)
    benzene = get_float_input("  Benzene in ppm (Typical: 0-30)", default=0.5)
    dust = get_float_input("  PM2.5 Dust load in ug/m3 (Typical: 0-500)", default=35.0)

    # Environmental
    print("\n--- [Environmental Features] ---")
    temp = get_float_input("  Ambient Temperature in °C (Typical: 15-40)", default=22.0)
    humidity = get_float_input("  Relative Humidity % (Typical: 20-95)", default=55.0)

    # Vibration
    print("\n--- [Blast Vibration Features] ---")
    max_charge = get_float_input("  Maximum charge weight per blast delay in kg", default=45.0)
    num_holes = get_float_input("  Number of blast holes", default=18.0)
    distance = get_float_input("  Seismic geophone distance from blast in m", default=350.0)

    # Robot Navigation Proximity
    print("\n--- [Robot Navigation Features] ---")
    min_dist = get_float_input("  Robotic platform minimum ultrasonic distance in m", default=2.5)

    # ───────────────────────────────────────────────────────────────────────
    # Feature Construction & ML Model Evaluation
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("⚙️ STEP 2: Running Pre-trained ML Models (Tier 1 Inference)")
    print("─" * 80)

    # Gas feature engineering
    # Methane: 128 features (MQ4 reading replicated with slight variations)
    mq4_feats = [ch4 * (1.0 + 0.01 * np.sin(i)) for i in range(128)]
    # Smoke: 36 features
    smoke_feats = [dust * (1.0 + 0.02 * np.cos(i)) for i in range(36)]
    # Air quality: 7 features
    aq_feats = [ch4, co, lpg, nox, benzene, temp, humidity]

    gas_inputs = {
        "mq4_features": mq4_feats,
        "smoke_features": smoke_feats,
        "MQ2_LPG_ppm": lpg,
        "MQ4_CH4_ppm": ch4,
        "MQ7_CO_ppm": co,
        "MQ135_NOx_ppm": nox,
        "MQ3_Benzene_ppm": benzene,
        "PM25_Dust_ugm3": dust,
        "Temp_C": temp,
        "Humidity_pct": humidity,
        "air_quality_features": aq_feats
    }

    # Env feature engineering: 9 features for Isolation Forest
    temp_hum_product = temp * humidity
    temp_hum_ratio = temp / max(1.0, humidity)
    # Humidex calculation
    e = 6.105 * np.exp(17.27 * humidity / (237.7 + humidity))
    humidex = temp + 0.33 * e - 4.0

    env_inputs = {
        "temp": temp,
        "humidity": humidity,
        "temp_hum_product": temp_hum_product,
        "temp_hum_ratio": temp_hum_ratio,
        "humidex": humidex,
        "temp_roll_mean_5": temp,
        "temp_roll_std_5": 0.05,
        "humidity_roll_mean_5": humidity,
        "humidity_roll_std_5": 0.2,
        "occupancy_features": [temp, humidity, temp_hum_product, temp_hum_ratio, humidex] + [0.0] * 18  # RF Occupancy
    }

    # Vibration feature engineering: 14 classifier features & 17 regressor features
    # USBM scaled distance: distance / sqrt(max_charge)
    sd_usbm = distance / np.sqrt(max(1.0, max_charge))
    # Langefors scaled distance: distance / max_charge^(2/3)
    sd_lk = distance / (max_charge ** (2.0 / 3.0))

    vib_inputs = {
        "offset": 0.005,
        "max_charge": max_charge,
        "total_charge": max_charge * 1.5,
        "num_holes": num_holes,
        "detonator_code": 102.0,
        "trid_12": 1,
        "trid_13": 0,
        "trid_14": 0,
        "gx": 100.0,
        "gy": 200.0,
        "gelev": -150.0,
        "sx": 105.0,
        "sy": 202.0,
        "selev": -148.0,
        "scaled_distance_usbm": sd_usbm,
        "scaled_distance_langefors": sd_lk,
        "elevation_diff": 2.0
    }

    # Ultrasonic navigation: 24 sensors populated with default and minimum
    ultra_inputs = {f"US{i}": 3.0 for i in range(1, 25)}
    # Replicate minimum on front sectors
    for i in (1, 2, 12, 13, 24):
        ultra_inputs[f"US{i}"] = min_dist

    # Run predictions
    gas_results = monitor.evaluate_gas(gas_inputs)
    env_results = monitor.evaluate_env(env_inputs)
    vib_results = monitor.evaluate_vibration(vib_inputs)
    ultra_results = monitor.evaluate_ultrasonic(ultra_inputs, config_type=24)

    # Print model predictions
    print("\n[Gas ML Model Predictions]")
    print(f"  • Methane Hazard Classifier : {gas_results.get('methane_hazard', 0)}")
    print(f"  • LPG/CNG Hazard Classifier : {gas_results.get('lpg_hazard', 0)}")
    print(f"  • Smoke/Fire Alarm          : {gas_results.get('smoke_alarm', 0)}")
    print(f"  • CO/NOx Hazard Classifier  : {gas_results.get('co_nox_hazard', 0)}")
    print(f"  • Predicted Air Quality Score: {gas_results.get('air_quality_score', 100.0):.2f}")

    print("\n[Environmental ML Model Predictions]")
    print(f"  • Isolation Forest Anomaly  : {env_results.get('anomaly_detected', 0)}")
    print(f"  • Occupancy State           : {env_results.get('occupancy_state', 0)} (1=Occupied, 0=Empty)")

    print("\n[Blast Vibration ML Model Predictions]")
    print(f"  • PPV Hazard Classifier     : {vib_results.get('vibration_hazard', 0)}")
    print(f"  • Predicted PPV Regressor   : {vib_results.get('predicted_ppv', 0.0):.4f} mm/s")

    print("\n[Robot Navigation ML Model Predictions]")
    print(f"  • Predicted Steering command: {ultra_results.get('steering_decision', 'Move-Forward')}")
    print(f"  • Collision Risk Warning    : {ultra_results.get('sharp_turn_required', 0)}")

    # ───────────────────────────────────────────────────────────────────────
    # Run Chat Assistant Conversational Reasoning
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("💬 STEP 3: Chat Assistant Safety Analysis & Advice (Layer 3)")
    print("─" * 80)

    # Assemble anomalies based on model outputs
    active_anomalies = {}
    if gas_results.get("methane_hazard") == 1 or ch4 > 5000:
        active_anomalies["MQ4_CH4_ppm"] = ch4
    if gas_results.get("lpg_hazard") == 1:
        active_anomalies["MQ2_LPG_ppm"] = lpg
    if gas_results.get("co_nox_hazard") == 1 or co > 25:
        active_anomalies["MQ7_CO_ppm"] = co
    if gas_results.get("smoke_alarm") == 1:
        active_anomalies["smoke_alarm"] = 1
    if env_results.get("anomaly_detected") == 1 or temp > 28.0:
        active_anomalies["temp"] = temp
        active_anomalies["humidity"] = humidity
        if env_results.get("anomaly_detected") == 1:
            active_anomalies["environment_model_anomaly"] = 1
    if vib_results.get("vibration_hazard") == 1 or vib_results.get("predicted_ppv", 0.0) > 1.0:
        active_anomalies["predicted_ppv"] = vib_results.get("predicted_ppv", 0.0)
    if ultra_results.get("sharp_turn_required") == 1 or min_dist < 0.5:
        active_anomalies["min_distance"] = min_dist

    default_query = "Summarize the safety conditions of this tunnel segment and suggest what safety measures apply."
    if active_anomalies:
        default_query = f"Explain the hazard cause of the alerts at {segment_id} and provide an emergency evacuation or response action plan."

    user_query = input(f"Enter your question for FIELD-MIND\n[default: \"{default_query}\"]:\n").strip()
    if not user_query:
        user_query = default_query

    print("\n  🔍 Analyzing data inputs, EKG records, and FAISS safety regulations...")
    t0 = time.time()
    all_readings = {
        "MQ4_CH4_ppm": ch4,
        "MQ7_CO_ppm": co,
        "MQ2_LPG_ppm": lpg,
        "MQ135_NOx_ppm": nox,
        "temp": temp,
        "humidity": humidity,
        "predicted_ppv": vib_results.get("predicted_ppv", 0.0),
        "min_distance": min_dist,
    }
    model_predictions = {
        **gas_results,
        **env_results,
        **vib_results,
        **ultra_results,
    }
    response = assistant.chat(
        user_query,
        segment_id,
        active_anomalies,
        model_predictions=model_predictions,
        sensor_readings=all_readings,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("🤖 FIELD-MIND RESPONSE")
    print("=" * 80)
    print(response)
    print("=" * 80)
    print(f"  ⏱  Inference time: {elapsed:.1f} seconds")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
