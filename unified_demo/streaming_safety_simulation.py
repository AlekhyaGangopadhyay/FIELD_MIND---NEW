"""Three-node, multi-timestamp FIELD-MIND stress-test demo.

Run from the project root:

    py -X utf8 unified_demo/streaming_safety_simulation.py

The simulation generates 3 nodes in one tunnel, samples every 2 seconds for
10 timestamps, runs the real Tier-1 models for every sample, saves the full
history, and sends a cross-node trend summary to the safety assistant.
Use ``--fast`` to skip the two-second wall-clock pacing while stress-testing.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from atr_activation.detector_wrappers import Tier1Monitor
from faiss_rag import SafetyProtocolEvaluator
from reasoning_core.chat_assistant import MineSafetyChatAssistant


DEFAULT_HISTORY_PATH = os.path.join(SCRIPT_DIR, "data", "simulation_history.json")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate_readings(rng: random.Random, node_index: int, tick: int) -> Dict[str, float]:
    """Generate one plausible sample with gradual, node-specific hazard drift."""
    node_scale = (0.96, 1.0, 1.04)[node_index % 3]
    gas_rise = max(0, tick - 4)
    heat_rise = max(0, tick - 5)
    clearance_rise = max(0, tick - 5)

    ch4 = rng.gauss(400 * node_scale, 28)
    co = rng.gauss(12 * node_scale, 1.5)
    lpg = rng.gauss(80 * node_scale, 6)
    nox = rng.gauss(2.1, 0.15)
    benzene = rng.gauss(0.5, 0.05)
    dust = rng.gauss(35 * node_scale, 2.5)
    temp = rng.gauss(22 + 0.1 * node_index, 0.15)
    humidity = rng.gauss(55 + 0.5 * node_index, 0.8)
    max_charge = rng.gauss(45, 1.5)
    num_holes = rng.gauss(18, 0.7)
    distance = rng.gauss(350, 5)
    min_distance = rng.gauss(2.5, 0.08)

    # Node 1: slow gas accumulation. The final samples cross warning ranges.
    if node_index == 0:
        ch4 += 180 * tick + 900 * gas_rise
        co += 0.7 * tick + 4.0 * gas_rise
        dust += 1.2 * tick

    # Node 2: thermal and blast-load drift.
    elif node_index == 1:
        temp += 0.25 * tick + 1.2 * heat_rise
        humidity += 0.8 * tick + 2.0 * heat_rise
        max_charge += 2.0 * tick + 7.0 * heat_rise
        distance -= 5.0 * tick + 24.0 * heat_rise

    # Node 3: gradually closing robot clearance, creating a late collision risk.
    else:
        min_distance -= 0.12 * tick + 0.45 * clearance_rise
        max_charge += 0.4 * tick

    return {
        "MQ4_CH4_ppm": round(max(0.0, ch4), 3),
        "MQ7_CO_ppm": round(max(0.0, co), 3),
        "MQ2_LPG_ppm": round(max(0.0, lpg), 3),
        "MQ135_NOx_ppm": round(max(0.0, nox), 3),
        "MQ3_Benzene_ppm": round(max(0.0, benzene), 3),
        "PM25_Dust_ugm3": round(max(0.0, dust), 3),
        "temp": round(temp, 3),
        "humidity": round(_clamp(humidity, 5.0, 98.0), 3),
        "max_charge": round(max(1.0, max_charge), 3),
        "num_holes": round(max(1.0, num_holes), 3),
        "distance": round(max(30.0, distance), 3),
        "min_distance": round(_clamp(min_distance, 0.18, 5.0), 3),
    }


def build_model_inputs(readings: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Build the exact feature shapes expected by the existing Tier1Monitor."""
    ch4 = readings["MQ4_CH4_ppm"]
    co = readings["MQ7_CO_ppm"]
    lpg = readings["MQ2_LPG_ppm"]
    nox = readings["MQ135_NOx_ppm"]
    benzene = readings["MQ3_Benzene_ppm"]
    dust = readings["PM25_Dust_ugm3"]
    temp = readings["temp"]
    humidity = readings["humidity"]
    max_charge = readings["max_charge"]
    num_holes = readings["num_holes"]
    distance = readings["distance"]

    mq4_features = [ch4 * (1.0 + 0.01 * np.sin(i)) for i in range(128)]
    smoke_features = [dust * (1.0 + 0.02 * np.cos(i)) for i in range(36)]
    air_quality_features = [ch4, co, lpg, nox, benzene, temp, humidity]

    temp_hum_product = temp * humidity
    temp_hum_ratio = temp / max(1.0, humidity)
    e = 6.105 * np.exp(17.27 * humidity / (237.7 + humidity))
    humidex = temp + 0.33 * e - 4.0

    gas_inputs = {
        "mq4_features": mq4_features,
        "smoke_features": smoke_features,
        "MQ2_LPG_ppm": lpg,
        "MQ4_CH4_ppm": ch4,
        "MQ7_CO_ppm": co,
        "MQ135_NOx_ppm": nox,
        "MQ3_Benzene_ppm": benzene,
        "PM25_Dust_ugm3": dust,
        "Temp_C": temp,
        "Humidity_pct": humidity,
        "air_quality_features": air_quality_features,
    }
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
        "occupancy_features": [temp, humidity, temp_hum_product, temp_hum_ratio, humidex] + [0.0] * 18,
    }
    scaled_distance_usbm = distance / np.sqrt(max(1.0, max_charge))
    scaled_distance_langefors = distance / (max_charge ** (2.0 / 3.0))
    vibration_inputs = {
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
        "scaled_distance_usbm": scaled_distance_usbm,
        "scaled_distance_langefors": scaled_distance_langefors,
        "elevation_diff": 2.0,
    }
    ultrasonic_inputs = {f"US{i}": 3.0 for i in range(1, 25)}
    for i in (1, 2, 12, 13, 24):
        ultrasonic_inputs[f"US{i}"] = readings["min_distance"]

    return {
        "gas": gas_inputs,
        "env": env_inputs,
        "vibration": vibration_inputs,
        "ultrasonic": ultrasonic_inputs,
    }


def evaluate_sample(monitor: Tier1Monitor, readings: Dict[str, float]) -> Dict[str, Any]:
    features = build_model_inputs(readings)
    gas = monitor.evaluate_gas(features["gas"])
    env = monitor.evaluate_env(features["env"])
    vibration = monitor.evaluate_vibration(features["vibration"])
    ultrasonic = monitor.evaluate_ultrasonic(features["ultrasonic"], config_type=24)
    predictions = {**gas, **env, **vibration, **ultrasonic}
    evaluated_readings = dict(readings)
    evaluated_readings["predicted_ppv"] = float(predictions.get("predicted_ppv", 0.0))
    return {"readings": evaluated_readings, "predictions": predictions}


def _direction(values: Iterable[float]) -> str:
    values = list(values)
    if len(values) < 2:
        return "stable"
    delta = values[-1] - values[0]
    # Do not call small random sensor noise a trend. A 10% deadband keeps the
    # baseline nodes stable while still exposing the deliberately injected
    # ramps and drops.
    tolerance = max(0.05, abs(values[0]) * 0.10)
    if delta > tolerance:
        return "rising"
    if delta < -tolerance:
        return "falling"
    return "stable"


def build_trend_summary(history: List[Dict[str, Any]]) -> str:
    by_node: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_timestamp: Dict[str, List[str]] = defaultdict(list)
    for item in history:
        by_node[item["node_id"]].append(item)
        by_timestamp[item["timestamp"]].append(item["status"])

    lines = [
        f"10-sample trend window across {len(by_node)} nodes; samples are 2 seconds apart.",
        "Each node reports the same sensor set. Rising/falling describes first-to-last movement.",
    ]
    for node_id, rows in sorted(by_node.items()):
        def vals(key):
            return [float(row["readings"].get(key, 0.0)) for row in rows]

        lines.append(
            f"{node_id}: CH4 {vals('MQ4_CH4_ppm')[0]:.0f}->{vals('MQ4_CH4_ppm')[-1]:.0f} ppm ({_direction(vals('MQ4_CH4_ppm'))}); "
            f"CO {vals('MQ7_CO_ppm')[0]:.1f}->{vals('MQ7_CO_ppm')[-1]:.1f} ppm ({_direction(vals('MQ7_CO_ppm'))}); "
            f"PPV {vals('predicted_ppv')[0]:.2f}->{vals('predicted_ppv')[-1]:.2f} mm/s ({_direction(vals('predicted_ppv'))}); "
            f"clearance {vals('min_distance')[0]:.2f}->{vals('min_distance')[-1]:.2f} m ({_direction(vals('min_distance'))})."
        )
        alarms = [
            key for key in ("methane_hazard", "co_nox_hazard", "smoke_alarm", "anomaly_detected", "vibration_hazard", "sharp_turn_required")
            if sum(int(bool(row["predictions"].get(key, 0))) for row in rows) > 0
        ]
        lines.append(f"  Model alarm streams observed: {', '.join(alarms) if alarms else 'none' }.")

    lines.append("Latest node snapshots (values below are node-specific; do not merge them into one physical location):")
    for node_id, rows in sorted(by_node.items()):
        latest = rows[-1]
        lines.append(
            f"  {node_id}: status={latest['status']}; "
            f"CH4={latest['readings']['MQ4_CH4_ppm']:.0f} ppm; "
            f"CO={latest['readings']['MQ7_CO_ppm']:.1f} ppm; "
            f"temp={latest['readings']['temp']:.1f} C; "
            f"PPV={latest['readings']['predicted_ppv']:.2f} mm/s; "
            f"clearance={latest['readings']['min_distance']:.2f} m."
        )

    lines.append("Timestamp status progression:")
    for timestamp, statuses in sorted(by_timestamp.items()):
        lines.append(f"  {timestamp}: {', '.join(statuses)}")

    rising_nodes = [
        node for node, rows in by_node.items()
        if _direction([row["readings"]["MQ4_CH4_ppm"] for row in rows]) == "rising"
        or _direction([row["readings"]["predicted_ppv"] for row in rows]) == "rising"
    ]
    falling_clearance = [
        node for node, rows in by_node.items()
        if _direction([row["readings"]["min_distance"] for row in rows]) == "falling"
    ]
    if rising_nodes:
        lines.append(f"Interpretation: gradual gas/PPV escalation is visible at {', '.join(sorted(rising_nodes))}.")
    if falling_clearance:
        lines.append(f"Interpretation: robot clearance is shrinking at {', '.join(sorted(falling_clearance))}; late-window readings deserve priority.")
    if not rising_nodes and not falling_clearance:
        lines.append("Interpretation: no strong monotonic hazard trend was detected in this window.")
    return "\n".join(lines)


def aggregate_latest(history: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create worst-case current readings/predictions for the final reasoning turn."""
    latest_timestamp = max(item["timestamp"] for item in history)
    rows = [item for item in history if item["timestamp"] == latest_timestamp]
    severity_rank = {"SAFE": 0, "WARNING": 1, "REVIEW_MODEL_DISAGREEMENT": 2, "CRITICAL": 3}
    focus_row = max(rows, key=lambda row: severity_rank.get(row["status"], 0))
    readings = {
        "MQ4_CH4_ppm": max(row["readings"]["MQ4_CH4_ppm"] for row in rows),
        "MQ7_CO_ppm": max(row["readings"]["MQ7_CO_ppm"] for row in rows),
        "MQ2_LPG_ppm": max(row["readings"]["MQ2_LPG_ppm"] for row in rows),
        "MQ135_NOx_ppm": max(row["readings"]["MQ135_NOx_ppm"] for row in rows),
        "temp": max(row["readings"]["temp"] for row in rows),
        "humidity": max(row["readings"]["humidity"] for row in rows),
        "predicted_ppv": max(row["readings"]["predicted_ppv"] for row in rows),
        "min_distance": min(row["readings"]["min_distance"] for row in rows),
    }
    predictions: Dict[str, Any] = {}
    flag_keys = ("methane_hazard", "lpg_hazard", "smoke_alarm", "co_nox_hazard", "anomaly_detected", "vibration_hazard", "sharp_turn_required")
    for key in flag_keys:
        predictions[key] = int(any(bool(row["predictions"].get(key, 0)) for row in rows))
    predictions["occupancy_state"] = int(any(bool(row["predictions"].get("occupancy_state", 0)) for row in rows))
    predictions["air_quality_score"] = min(float(row["predictions"].get("air_quality_score", 1.0)) for row in rows)
    predictions["predicted_ppv"] = readings["predicted_ppv"]
    predictions["steering_decision"] = (
        "Sharp-Right-Turn" if predictions["sharp_turn_required"] else "Trend review required"
    )
    return {
        "readings": readings,
        "predictions": predictions,
        "focus_node_id": focus_row["node_id"],
        "focus": {
            "readings": focus_row["readings"],
            "predictions": focus_row["predictions"],
            "status": focus_row["status"],
        },
    }


def active_anomalies(readings: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
    active = {}
    if predictions.get("methane_hazard") or readings.get("MQ4_CH4_ppm", 0) > 1000:
        active["MQ4_CH4_ppm"] = readings["MQ4_CH4_ppm"]
    if predictions.get("co_nox_hazard") or readings.get("MQ7_CO_ppm", 0) > 25:
        active["MQ7_CO_ppm"] = readings["MQ7_CO_ppm"]
    if predictions.get("smoke_alarm"):
        active["smoke_alarm"] = 1
    if predictions.get("anomaly_detected") or readings.get("temp", 0) > 28:
        active["environment_model_anomaly"] = int(bool(predictions.get("anomaly_detected")))
    if predictions.get("vibration_hazard") or readings.get("predicted_ppv", 0) > 1:
        active["predicted_ppv"] = readings["predicted_ppv"]
    if predictions.get("sharp_turn_required") or readings.get("min_distance", 5) < 1:
        active["min_distance"] = readings["min_distance"]
    return active


def run_simulation(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    tunnel_id = args.tunnel_id.upper()
    print("\nFIELD-MIND MULTI-NODE STREAMING SAFETY SIMULATION")
    print(f"Tunnel: {tunnel_id} | Nodes: {args.nodes} | Samples: {args.timestamps} | Interval: {args.interval:.1f}s")
    print(f"Mode: {'fast' if args.fast else 'real-time paced'} | Seed: {args.seed}\n")

    print("[Setup] Loading Tier-1 models...")
    monitor = Tier1Monitor(root_dir=WORKSPACE_ROOT)
    print("[Setup] Loading safety reasoning resources...")
    assistant = MineSafetyChatAssistant(workspace_root=WORKSPACE_ROOT)
    if assistant.rag_retriever:
        try:
            warmup_ms = assistant.rag_retriever.warmup()
            print(f"[Setup] FAISS embedder ready in {warmup_ms:.0f} ms.")
        except Exception as exc:
            print(f"[Setup] FAISS warm-up skipped: {exc}")

    start = datetime.now(timezone.utc)
    history: List[Dict[str, Any]] = []
    evaluator = SafetyProtocolEvaluator()

    for tick in range(args.timestamps):
        loop_started = time.perf_counter()
        timestamp = (start + timedelta(seconds=tick * args.interval)).isoformat()
        for node_index in range(args.nodes):
            node_id = f"{tunnel_id}_NODE_{node_index + 1}"
            readings = generate_readings(rng, node_index, tick)
            result = evaluate_sample(monitor, readings)
            check_readings = result["readings"]
            check = evaluator.assess(check_readings, result["predictions"])
            history.append({
                "timestamp": timestamp,
                "tunnel_id": tunnel_id,
                "node_id": node_id,
                "readings": check_readings,
                "predictions": result["predictions"],
                "status": check.overall_status,
            })
            print(
                f"[t={tick:02d} +{tick * args.interval:>4.0f}s] {node_id:<22} "
                f"CH4={check_readings['MQ4_CH4_ppm']:>7.0f} "
                f"CO={check_readings['MQ7_CO_ppm']:>5.1f} "
                f"PPV={check_readings['predicted_ppv']:>6.2f} "
                f"clearance={check_readings['min_distance']:>4.2f}m "
                f"status={check.overall_status}"
            )

        if not args.fast and tick < args.timestamps - 1:
            remaining = args.interval - (time.perf_counter() - loop_started)
            if remaining > 0:
                time.sleep(remaining)

    trend_context = build_trend_summary(history)
    final_state = aggregate_latest(history)
    focus_state = final_state["focus"]
    current_active = active_anomalies(focus_state["readings"], focus_state["predictions"])
    history_path = Path(args.history_out)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 80)
    print("MULTI-NODE TREND SUMMARY")
    print("=" * 80)
    print(trend_context)
    print(f"\nSaved {len(history)} samples to: {history_path}")
    print("\n[Layer 3] Sending the complete trend and latest worst-case state to the assistant...")

    query = (
        f"Analyze the complete {args.timestamps}-timestamp trend across {args.nodes} nodes in {tunnel_id}. "
        "Explain gradual rises, falling clearance, persistence, cross-node differences, and the safest prioritized response. "
        f"The latest highest-priority node is {final_state['focus_node_id']}; assess it numerically, "
        "but keep the other nodes separate and do not merge readings from different physical locations. "
        "Do not treat a single model alarm as confirmed danger without checking the numeric readings."
    )
    response = assistant.chat(
        query,
        tunnel_id,
        current_active,
        model_predictions=focus_state["predictions"],
        sensor_readings=focus_state["readings"],
        trend_context=trend_context,
    )
    print("\n" + "=" * 80)
    print("FIELD-MIND MULTI-NODE RESPONSE")
    print("=" * 80)
    print(response)
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="FIELD-MIND 3-node streaming safety simulation")
    parser.add_argument("--nodes", type=int, default=3, help="Nodes in the same tunnel (default: 3)")
    parser.add_argument("--timestamps", type=int, default=10, help="Samples per node (default: 10)")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between samples (default: 2)")
    parser.add_argument("--tunnel-id", default="TUNNEL_SIM_01", help="Shared tunnel identifier")
    parser.add_argument("--seed", type=int, default=69, help="Deterministic random seed")
    parser.add_argument("--fast", action="store_true", help="Skip real-time waiting")
    parser.add_argument("--history-out", default=DEFAULT_HISTORY_PATH, help="JSON path for the time-series history")
    args = parser.parse_args()
    if args.nodes < 1 or args.timestamps < 1 or args.interval <= 0:
        parser.error("--nodes and --timestamps must be positive; --interval must be greater than zero")
    run_simulation(args)


if __name__ == "__main__":
    main()
