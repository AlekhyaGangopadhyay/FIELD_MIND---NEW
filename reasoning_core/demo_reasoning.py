"""
demo_reasoning.py — Scientific Reasoning Core Demo Runner
=========================================================
Executes and demonstrates the LangGraph reasoning workflow on typical underground 
mining hazards. Simulates context retrieval from EKG and RAG (FAISS) to formulate 
hypotheses and actionable suggestions.

Usage:
  py -X utf8 reasoning_core/demo_reasoning.py
"""

import os
import sys

# Setup workspace root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from reasoning_core.agent_loop import ScientificReasoningCore


def print_workflow_result(scenario_name: str, result: dict) -> None:
    print("\n" + "=" * 80)
    print(f"  SCENARIO: {scenario_name}")
    print("=" * 80)
    
    print("\n[Execution Trace]")
    for step in result.get("trace", []):
        print(f"  → {step}")
        
    print("\n[Hypothesis]")
    print(f"  {result.get('hypothesis')}")
    
    print("\n[Actionable Suggestions]")
    for sug in result.get("suggestions", []):
        print(f"  ✔ {sug}")
        
    print("\n[Final Status]")
    print(f"  {result.get('status')}")
    print("=" * 80 + "\n")


def run_demo() -> None:
    print("\n" + "█" * 80)
    print("  FIELD-MIND — Scientific Reasoning Core (LangGraph Agent) Demo")
    print("  Offline on-device diagnostic reasoning and context integration")
    print("█" * 80 + "\n")

    # Initialize core
    core = ScientificReasoningCore(workspace_root=WORKSPACE_ROOT)

    # ───────────────────────────────────────────────────────────────────────
    # Scenario 1: Critical Gas buildup in Tunnel Segment 1
    # ───────────────────────────────────────────────────────────────────────
    gas_anomalies = {
        "MQ4_CH4_ppm": 12500.0,    # 1.25% - above evacuation limit
        "MQ7_CO_ppm": 35.0,
        "Temp_C": 28.5
    }
    res_gas = core.reason(anomalies=gas_anomalies, segment_id="TUNNEL_A1")
    print_workflow_result("Critical Methane Build-up in Heading A1", res_gas)

    # ───────────────────────────────────────────────────────────────────────
    # Scenario 2: Blast PPV ground vibration exceedance
    # ───────────────────────────────────────────────────────────────────────
    vib_anomalies = {
        "predicted_ppv": 15.8,     # 15.8 mm/s - unsafe for residential structures
        "max_charge": 120.0,
        "num_holes": 24
    }
    res_vib = core.reason(anomalies=vib_anomalies, segment_id="TUNNEL_A3")
    print_workflow_result("Excessive Blast Ground Vibration", res_vib)

    # ───────────────────────────────────────────────────────────────────────
    # Scenario 3: Autonomous robot proximity collision risk
    # ───────────────────────────────────────────────────────────────────────
    nav_anomalies = {
        "min_distance": 0.18,      # 18 cm - critical collision risk
        "steering_decision": "Sharp-Right-Turn",
        "US12": 0.18
    }
    res_nav = core.reason(anomalies=nav_anomalies, segment_id="TUNNEL_B1")
    print_workflow_result("Robot Evasive Manoeuvre & Obstacle Collision Risk", res_nav)


if __name__ == "__main__":
    run_demo()
