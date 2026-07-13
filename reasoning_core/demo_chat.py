"""
demo_chat.py — Conversational Mine Safety Assistant Demo
=========================================================
Runs simulated conversation scenarios demonstrating the chat assistant
communicating with the operator, analyzing data inputs, and safety measures.

Usage:
  py -X utf8 reasoning_core/demo_chat.py
"""

import os
import sys

# Setup workspace root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from reasoning_core.chat_assistant import MineSafetyChatAssistant


def run_chat_scenario(
    assistant: MineSafetyChatAssistant,
    scenario_title: str,
    user_query: str,
    segment_id: str,
    active_anomalies: dict
) -> None:
    print("\n" + "=" * 80)
    print(f"🎬 SCENARIO: {scenario_title}")
    print(f"📍 Location : {segment_id}")
    print(f"📊 Inputs   : {active_anomalies}")
    print(f"💬 User     : \"{user_query}\"")
    print("=" * 80)
    
    print("\n[FIELD-MIND Response]")
    response = assistant.chat(user_query, segment_id, active_anomalies)
    print(response)
    print("=" * 80 + "\n")


def main():
    print("\n" + "█" * 80)
    print("  FIELD-MIND — Conversational Safety Assistant Demo")
    print("  Communicating safety measures and data analysis offline")
    print("█" * 80 + "\n")

    # Initialize assistant
    assistant = MineSafetyChatAssistant(workspace_root=WORKSPACE_ROOT)

    # ───────────────────────────────────────────────────────────────────────
    # Scenario 1: Gas safety questions
    # ───────────────────────────────────────────────────────────────────────
    query_1 = "We just saw gas levels spiking in Tunnel A1. Analyze these inputs and tell us what safety measures apply."
    anomalies_1 = {
        "MQ4_CH4_ppm": 12500.0,    # 1.25% - dangerous methane
        "MQ7_CO_ppm": 35.0
    }
    run_chat_scenario(
        assistant,
        "Methane Spike Hazard Analysis",
        query_1,
        "TUNNEL_A1",
        anomalies_1
    )

    # ───────────────────────────────────────────────────────────────────────
    # Scenario 2: Blast vibration questions
    # ───────────────────────────────────────────────────────────────────────
    query_2 = "What structural safety measures should we follow for the vibration reading of 15.8 mm/s in heading A3?"
    anomalies_2 = {
        "predicted_ppv": 15.8,     # 15.8 mm/s - exceeding safe residential threshold
        "max_charge": 120.0
    }
    run_chat_scenario(
        assistant,
        "Structural Vibration Ground Limits",
        query_2,
        "TUNNEL_A3",
        anomalies_2
    )

    # ───────────────────────────────────────────────────────────────────────
    # Scenario 3: Robotic navigation safety
    # ───────────────────────────────────────────────────────────────────────
    query_3 = "The inspection robot in tunnel B1 is showing an obstacle. What is the collision avoidance protocol?"
    anomalies_3 = {
        "min_distance": 0.18,      # 18 cm
        "steering_decision": "Sharp-Right-Turn"
    }
    run_chat_scenario(
        assistant,
        "Robotic Obstacle Evasion",
        query_3,
        "TUNNEL_B1",
        anomalies_3
    )


if __name__ == "__main__":
    main()
