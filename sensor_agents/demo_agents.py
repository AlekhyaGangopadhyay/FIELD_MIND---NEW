"""
demo_agents.py — FIELD-MIND AI Agent System End-to-End Demo
============================================================
Demonstrates all 5 sensor AI agents running simultaneously:
  • GasSensorAgent
  • EnvSensorAgent
  • VibrationSensorAgent
  • UltrasonicSensorAgent
  • EKGAgent (knowledge graph memory)
  • MineOrchestratorAgent (global fusion)

Each agent runs through its full Observe → Reason → Act → Learn cycle.
The demo uses ORIGINAL DATASET rows for both simulation inputs AND for
experience replay labels, so learning is always grounded in real data.

Run from project root:
    python sensor_agents/demo_agents.py

Or for a longer run with more learning cycles:
    python sensor_agents/demo_agents.py --ticks 1000 --verbose
"""

import os
import sys
import time
import random
import argparse
import numpy as np

# ── Project path setup ────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# ── Import agent modules ───────────────────────────────────────────────────
from sensor_agents.agent_bus           import AgentBus, MessageType, Severity
from sensor_agents.gas_agent           import GasSensorAgent
from sensor_agents.env_agent           import EnvSensorAgent
from sensor_agents.vibration_agent     import VibrationSensorAgent
from sensor_agents.ultrasonic_agent    import UltrasonicSensorAgent
from sensor_agents.ekg_agent           import EKGAgent
from sensor_agents.mine_orchestrator_agent import MineOrchestratorAgent


# ═══════════════════════════════════════════════════════════════════════════
# Dataset-Driven Sensor Simulators
# Each simulator reads directly from the original CSV datasets to produce
# realistic sensor readings (not pure random noise).
# ═══════════════════════════════════════════════════════════════════════════

import pandas as pd

def load_dataset_safe(path: str, nrows: int = 5000) -> pd.DataFrame:
    """Load a CSV with graceful fallback if file is missing."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, nrows=nrows)
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            print(f"  [Demo] Warning: Could not load {path}: {e}")
    return pd.DataFrame()


class DatasetSensorSimulator:
    """
    Streams real rows from an original dataset to feed into agents.
    Cycles through the dataset infinitely.
    """
    def __init__(self, df: pd.DataFrame, mode: str):
        self.df   = df
        self.mode = mode
        self.idx  = 0

    def next_reading(self, inject_hazard: bool = False) -> dict:
        if self.df.empty:
            return self._fallback(inject_hazard)

        row = self.df.iloc[self.idx % len(self.df)].to_dict()
        self.idx += 1

        if inject_hazard:
            row = self._inject_hazard(row)

        return row

    def _inject_hazard(self, row: dict) -> dict:
        """Artificially spike readings to simulate a real hazard."""
        if self.mode == "gas":
            row["MQ2_LPG_ppm"]  = row.get("MQ2_LPG_ppm", 100.0) * 12.0
            row["MQ4_CH4_ppm"]  = row.get("MQ4_CH4_ppm", 100.0) * 15.0
            row["MQ7_CO_ppm"]   = row.get("MQ7_CO_ppm",  50.0)  * 20.0
        elif self.mode == "env":
            row["temp"]     = 42.0    # dangerously high temperature
            row["humidity"] = 92.0    # dangerously high humidity
        elif self.mode == "vibration":
            row["ppv"] = 8.5          # high PPV well above 1.0 threshold
            row["max_charge"] = row.get("max_charge", 50.0) * 5.0
        elif self.mode == "ultrasonic":
            # Very close obstacle on all sensors
            for i in range(1, 25):
                row[f"US{i}"] = 0.1   # 10 cm — imminent collision
        return row

    def _fallback(self, inject_hazard: bool) -> dict:
        """Generate synthetic fallback readings if dataset unavailable."""
        rng = random.random
        if self.mode == "gas":
            hazard_mult = 10.0 if inject_hazard else 1.0
            return {
                "MQ2_LPG_ppm"     : rng() * 500.0 * hazard_mult,
                "MQ4_CH4_ppm"     : rng() * 400.0 * hazard_mult,
                "MQ7_CO_ppm"      : rng() * 200.0 * hazard_mult,
                "MQ135_NOx_ppm"   : rng() * 50.0,
                "MQ3_Benzene_ppm" : rng() * 30.0,
                "PM25_Dust_ugm3"  : rng() * 100.0,
                "Temp_C"          : 20.0 + rng() * 10.0,
                "Humidity_pct"    : 40.0 + rng() * 30.0,
                "MG811_CO2_ppm"   : 400.0 + rng() * 200.0,
            }
        elif self.mode == "env":
            t = (42.0 if inject_hazard else 20.0 + rng() * 8.0)
            h = (92.0 if inject_hazard else 40.0 + rng() * 20.0)
            return {"temp": t, "humidity": h}
        elif self.mode == "vibration":
            ppv = (8.5 if inject_hazard else rng() * 2.0)
            return {k: rng() * 10.0 for k in [
                "offset", "max_charge", "total_charge", "num_holes", "detonator_code",
                "trid_12", "trid_13", "trid_14",
                "gx", "gy", "gelev", "sx", "sy", "selev",
                "scaled_distance_usbm", "scaled_distance_langefors", "elevation_diff",
                "ppv",
            ]}
        elif self.mode == "ultrasonic":
            v = (0.1 if inject_hazard else 1.5 + rng() * 2.0)
            return {f"US{i}": v for i in range(1, 25)}
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Pretty-print a step result
# ═══════════════════════════════════════════════════════════════════════════

def print_step_summary(tick: int, results: dict, global_state: dict) -> None:
    """Print a compact multi-agent step summary."""
    print(f"\n{'═'*70}")
    print(f"  TICK {tick:>4}  |  Global State: {global_state['device_state']}  |  "
          f"Hazard Score: {global_state['global_score']:.3f}")
    print(f"{'─'*70}")
    for agent_name, res in results.items():
        conf = res.get("confidence", 0.0)
        act  = res.get("action", "—")
        buf  = res.get("replay_buffer_size", 0)
        rfits= res.get("metrics", {}).get("refit_count", 0)
        print(
            f"  {agent_name:<28} conf={conf:.3f}  action={act:<22}"
            f"  buf={buf:<4}  refits={rfits}"
        )
    if global_state["active_sources"]:
        print(f"{'─'*70}")
        print(f"  Active alert sources: {global_state['active_sources']}")
    print(f"{'═'*70}")


# ═══════════════════════════════════════════════════════════════════════════
# Main Demo Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_demo(n_ticks: int = 300, verbose_agents: bool = False) -> None:
    print("\n" + "█"*70)
    print("  FIELD-MIND — AI SENSOR AGENT SYSTEM DEMO")
    print("  Each sensor runs as an autonomous agent: Observe→Reason→Act→Learn")
    print("█"*70 + "\n")

    # ── 1. Load original datasets for simulation ───────────────────────────
    print("[Setup] Loading original datasets for sensor simulation...")
    gas_df  = load_dataset_safe(
        os.path.join(WORKSPACE_ROOT, "gas_sensors", "data", "FIELDMIND_physics_dataset.csv"),
        nrows=5000
    )
    env_df  = load_dataset_safe(
        os.path.join(WORKSPACE_ROOT, "temperature_humidity", "data", "data_clean", "iot_telemetry_clean.csv"),
        nrows=5000
    )
    vib_df  = load_dataset_safe(
        os.path.join(WORKSPACE_ROOT, "vibration", "data", "vibration_features.csv"),
        nrows=5000
    )
    ultra_df = load_dataset_safe(
        os.path.join(WORKSPACE_ROOT, "ultrasonic_sensors", "data", "sensor_readings_24.csv"),
        nrows=5000
    )

    # Pre-process vibration (one-hot trid)
    if not vib_df.empty and "trid" in vib_df.columns:
        vib_df = pd.get_dummies(vib_df, columns=["trid"], prefix="trid")
        for col in ["trid_12", "trid_13", "trid_14"]:
            if col not in vib_df.columns:
                vib_df[col] = 0
            vib_df[col] = vib_df[col].astype(int)

    gas_sim   = DatasetSensorSimulator(gas_df,   "gas")
    env_sim   = DatasetSensorSimulator(env_df,   "env")
    vib_sim   = DatasetSensorSimulator(vib_df,   "vibration")
    ultra_sim = DatasetSensorSimulator(ultra_df, "ultrasonic")

    print(f"  Gas dataset    : {len(gas_df)} rows")
    print(f"  Env dataset    : {len(env_df)} rows")
    print(f"  Vibration data : {len(vib_df)} rows")
    print(f"  Ultrasonic data: {len(ultra_df)} rows")

    # ── 2. Create shared AgentBus ──────────────────────────────────────────
    print("\n[Setup] Initializing AgentBus...")
    bus = AgentBus(history_limit=1000)

    # ── 3. Initialize all agents ───────────────────────────────────────────
    print("\n[Setup] Initializing Sensor AI Agents...\n")
    gas_agent   = GasSensorAgent(WORKSPACE_ROOT,   bus, verbose=verbose_agents)
    env_agent   = EnvSensorAgent(WORKSPACE_ROOT,   bus, verbose=verbose_agents)
    vib_agent   = VibrationSensorAgent(WORKSPACE_ROOT, bus, verbose=verbose_agents)
    ultra_agent = UltrasonicSensorAgent(WORKSPACE_ROOT, bus, verbose=verbose_agents)
    ekg_agent   = EKGAgent(WORKSPACE_ROOT, bus, verbose=False)
    orchestrator = MineOrchestratorAgent(bus, verbose=True)

    print("\n[Setup] All agents initialized. Starting streaming loop...\n")
    time.sleep(0.5)

    # ── 4. Define hazard injection schedule ───────────────────────────────
    # Hazard events at specific tick windows to demonstrate ALERT + LEARN
    hazard_schedule = {
        range(30,  45):  {"gas": True},                        # Gas spike episode
        range(80,  95):  {"env": True},                        # Env anomaly
        range(140, 160): {"vibration": True},                  # Blast vibration
        range(200, 215): {"ultrasonic": True},                 # Collision risk
        range(250, 270): {"gas": True, "vibration": True},     # Multi-hazard
    }

    def should_inject(tick: int) -> dict:
        flags = {"gas": False, "env": False, "vibration": False, "ultrasonic": False}
        for tick_range, inject_flags in hazard_schedule.items():
            if tick in tick_range:
                flags.update(inject_flags)
        return flags

    # ── 5. Main streaming loop ─────────────────────────────────────────────
    print_interval = 25   # print summary every N ticks

    try:
        for tick in range(1, n_ticks + 1):
            ts      = time.time()
            inject  = should_inject(tick)

            # ── Simulate sensor readings from original datasets ────────
            gas_raw   = gas_sim.next_reading(inject_hazard=inject["gas"])
            env_raw   = env_sim.next_reading(inject_hazard=inject["env"])
            vib_raw   = vib_sim.next_reading(inject_hazard=inject["vibration"])
            ultra_raw = ultra_sim.next_reading(inject_hazard=inject["ultrasonic"])

            # ── Run each agent's full cycle ────────────────────────────
            gas_result   = gas_agent.step(gas_raw,     timestamp=ts)
            env_result   = env_agent.step(env_raw,     timestamp=ts)
            vib_result   = vib_agent.step(vib_raw,     timestamp=ts)
            ultra_result = ultra_agent.step(ultra_raw, timestamp=ts)

            # ── Orchestrator tick (global state fusion) ────────────────
            global_state = orchestrator.tick(timestamp=ts)

            # ── Print summary at intervals ─────────────────────────────
            if tick % print_interval == 0 or any(inject.values()):
                results = {
                    "GasSensorAgent"       : gas_result,
                    "EnvSensorAgent"       : env_result,
                    "VibrationSensorAgent" : vib_result,
                    "UltrasonicSensorAgent": ultra_result,
                }
                print_step_summary(tick, results, global_state)

            # Small sleep to prevent 100% CPU in demo
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Demo] Interrupted by user.")

    # ── 6. Final Reports ───────────────────────────────────────────────────
    print("\n\n" + "█"*70)
    print("  DEMO COMPLETE — Final Agent Status Reports")
    print("█"*70 + "\n")

    for agent in [gas_agent, env_agent, vib_agent, ultra_agent]:
        print(agent.status_report())
        print()
    print(ekg_agent.status_report())
    print()
    print(orchestrator.status_report())
    print()

    # ── Bus summary ────────────────────────────────────────────────────────
    print(bus.summary())

    # ── Save EKG graph ─────────────────────────────────────────────────────
    ekg_agent.save_graph()

    print("\n[Demo] Run complete.")
    print(f"  Total ticks      : {n_ticks}")
    print(f"  Hazard episodes  : {len(hazard_schedule)}")
    print(f"  Gas refits       : {gas_agent._metrics['refit_count']}")
    print(f"  Env refits       : {env_agent._metrics['refit_count']}")
    print(f"  Vibration refits : {vib_agent._metrics['refit_count']}")
    print(f"  Ultrasonic refits: {ultra_agent._metrics['refit_count']}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIELD-MIND AI Agent System Demo")
    parser.add_argument(
        "--ticks",
        type=int,
        default=300,
        help="Number of simulation ticks to run (default: 300)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable per-tick verbose output for each agent"
    )
    args = parser.parse_args()

    run_demo(n_ticks=args.ticks, verbose_agents=args.verbose)
