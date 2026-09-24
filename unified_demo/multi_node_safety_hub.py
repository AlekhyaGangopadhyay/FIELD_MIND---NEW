"""
multi_node_safety_hub.py — Multi-Node Visual Safety Hub & Interactive LLM Assistant
===================================================================================
A clean, terminal-based visual dashboard for underground mining safety that:
  1. Accepts real-time sensor telemetry for MULTIPLE NODES (Node 1 & Node 2).
  2. Runs inputs through pre-trained ML models across Gas, Env, Vibration, and Navigation.
  3. Visualizes raw telemetry, ML hazard predictions, and regulatory checks side-by-side.
  4. Takes operator verification feedback to trigger on-the-fly self-learning & reflection.
  5. Engages the offline LLM / Expert Reasoner to diagnose hazards, explain safety rules,
     and provide interactive multi-turn safety chat.

Usage:
  python -X utf8 unified_demo/multi_node_safety_hub.py
"""

import os
import sys
import time
import warnings
import numpy as np
from typing import Any, Dict, Optional, Tuple

# Suppress scikit-learn warnings for clean rendering
warnings.simplefilter("ignore", category=UserWarning)

# Workspace Root Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Ensure sub-modules are accessible
for sub in ["gas_sensors", "temperature_humidity", "ultrasonic_sensors", "vibration", "sensor_agents", "atr_activation", "faiss_rag", "reasoning_core"]:
    sub_path = os.path.join(WORKSPACE_ROOT, sub)
    if sub_path not in sys.path:
        sys.path.insert(0, sub_path)

# Rich UI Library
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.prompt import Prompt, FloatPrompt, Confirm
from rich import box
from rich.rule import Rule
from rich.markdown import Markdown

# Core ML Models and Reasoning System
from atr_activation.detector_wrappers import Tier1Monitor
from reasoning_core.chat_assistant import MineSafetyChatAssistant
from sensor_agents.agent_bus import AgentBus
from sensor_agents.gas_agent import GasSensorAgent
from sensor_agents.env_agent import EnvSensorAgent
from sensor_agents.vibration_agent import VibrationSensorAgent
from sensor_agents.ultrasonic_agent import UltrasonicSensorAgent
from vibration.structural_monitor import SW420VibrationMonitor, UltrasonicDisplacementModel

console = Console()

# (render_header, get_preset_telemetry, evaluate_node_models, build_telemetry_table, build_predictions_table unchanged)



def render_header():
    """Renders the main application title and system status banner."""
    header_text = Text()
    header_text.append("⚡ FIELD-MIND ", style="bold cyan")
    header_text.append("— Underground Mining Multi-Node Safety Hub\n", style="bold white")
    header_text.append("Edge Architecture: ", style="dim")
    header_text.append("NVIDIA Jetson Orin Nano (8GB)  ", style="bold green")
    header_text.append("| Mode: ", style="dim")
    header_text.append("100% Offline (Tier-1 ML + FAISS RAG + EKG Graph + LLM)\n", style="bold yellow")
    console.print(Panel(header_text, border_style="cyan", box=box.ROUNDED))


def get_preset_telemetry() -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Provides smart presets or manual interactive input for 2 nodes."""
    console.print("\n[bold yellow]📡 SELECT TELEMETRY INPUT SOURCE:[/bold yellow]")
    console.print("  [1] [bold red]Scenario 1: Methane Gas Pocket Crisis[/bold red] (Critical CH4 at Node 1 | Safe baseline at Node 2)")
    console.print("  [2] [bold magenta]Scenario 2: Seismic Shock & Wall Convergence[/bold magenta] (Safe Node 1 | Level-2 Shock & Convergence at Node 2)")
    console.print("  [3] [bold blue]Scenario 3: Sensor Moisture Condensation Drift[/bold blue] (Moisture false-alarm at Node 1 | Normal Node 2)")
    console.print("  [4] [bold green]Scenario 4: Custom Interactive Input[/bold green] (Enter your own sensor values for each node)")

    choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3", "4"], default="1")

    if choice == "1":
        scenario_title = "CRITICAL METHANE BUILD-UP (Heading A1)"
        node1 = {
            "name": "Node 1 (Heading A1 - Working Face)",
            "segment_id": "TUNNEL_A1",
            "MQ4_CH4_ppm": 12500.0,
            "MQ7_CO_ppm": 35.0,
            "MQ2_LPG_ppm": 85.0,
            "MQ135_NOx_ppm": 2.2,
            "MQ135_NH3_ppm": 4.5, "NH3_ppm": 4.5,
            "MQ136_H2S_ppm": 1.2, "H2S_ppm": 1.2,
            "MG811_CO2_ppm": 480.0, "CO2_ppm": 480.0,
            "dust_ug_m3": 45.0,
            "temp": 28.5,
            "humidity": 65.0,
            "vibration_pulses": 2.0,
            "ultrasonic_distance": 2.8,
            "min_distance": 2.2,
            "US1": 2.2, "US2": 2.5, "US3": 2.8
        }
        node2 = {
            "name": "Node 2 (Tunnel B2 - Main Return)",
            "segment_id": "TUNNEL_B2",
            "MQ4_CH4_ppm": 450.0,
            "MQ7_CO_ppm": 8.0,
            "MQ2_LPG_ppm": 20.0,
            "MQ135_NOx_ppm": 1.1,
            "MQ135_NH3_ppm": 2.0, "NH3_ppm": 2.0,
            "MQ136_H2S_ppm": 0.5, "H2S_ppm": 0.5,
            "MG811_CO2_ppm": 420.0, "CO2_ppm": 420.0,
            "dust_ug_m3": 20.0,
            "temp": 22.0,
            "humidity": 52.0,
            "vibration_pulses": 1.0,
            "ultrasonic_distance": 3.5,
            "min_distance": 3.0,
            "US1": 3.0, "US2": 3.2, "US3": 3.5
        }

    elif choice == "2":
        scenario_title = "SEISMIC SHOCK & WALL CONVERGENCE (Sector A3)"
        node1 = {
            "name": "Node 1 (Heading A1 - Intake Drift)",
            "segment_id": "TUNNEL_A1",
            "MQ4_CH4_ppm": 420.0,
            "MQ7_CO_ppm": 10.0,
            "MQ2_LPG_ppm": 15.0,
            "MQ135_NOx_ppm": 1.0,
            "MQ135_NH3_ppm": 3.0, "NH3_ppm": 3.0,
            "MQ136_H2S_ppm": 0.8, "H2S_ppm": 0.8,
            "MG811_CO2_ppm": 410.0, "CO2_ppm": 410.0,
            "dust_ug_m3": 18.0,
            "temp": 21.5,
            "humidity": 48.0,
            "vibration_pulses": 1.0,
            "ultrasonic_distance": 3.2,
            "min_distance": 2.8,
            "US1": 2.8, "US2": 3.0, "US3": 3.2
        }
        node2 = {
            "name": "Node 2 (Tunnel A3 - Blast Fracture Zone)",
            "segment_id": "TUNNEL_A3",
            "MQ4_CH4_ppm": 850.0,
            "MQ7_CO_ppm": 45.0,
            "MQ2_LPG_ppm": 30.0,
            "MQ135_NOx_ppm": 3.8,
            "MQ135_NH3_ppm": 12.0, "NH3_ppm": 12.0,
            "MQ136_H2S_ppm": 4.5, "H2S_ppm": 4.5,
            "MG811_CO2_ppm": 850.0, "CO2_ppm": 850.0,
            "dust_ug_m3": 120.0,
            "temp": 26.0,
            "humidity": 60.0,
            "vibration_pulses": 55.0,  # Level 2 Shock!
            "ultrasonic_distance": 1.20,
            "min_distance": 0.22,      # Proximity Alert!
            "US1": 0.22, "US2": 0.25, "US3": 0.30
        }

    elif choice == "3":
        scenario_title = "SENSOR MOISTURE DRIFT TEST (Water Spray Operation)"
        node1 = {
            "name": "Node 1 (Heading A1 - Water Spray Zone)",
            "segment_id": "TUNNEL_A1",
            "MQ4_CH4_ppm": 380.0,
            "MQ7_CO_ppm": 32.0,        # Drift elevated CO
            "MQ2_LPG_ppm": 25.0,
            "MQ135_NOx_ppm": 1.2,
            "MQ135_NH3_ppm": 4.0, "NH3_ppm": 4.0,
            "MQ136_H2S_ppm": 1.0, "H2S_ppm": 1.0,
            "MG811_CO2_ppm": 430.0, "CO2_ppm": 430.0,
            "dust_ug_m3": 15.0,
            "temp": 21.0,
            "humidity": 89.0,          # Extreme humidity -> Condensation
            "vibration_pulses": 2.0,
            "ultrasonic_distance": 3.0,
            "min_distance": 2.5,
            "US1": 2.5, "US2": 2.8, "US3": 3.0
        }
        node2 = {
            "name": "Node 2 (Tunnel B1 - Exhaust Crosscut)",
            "segment_id": "TUNNEL_B1",
            "MQ4_CH4_ppm": 410.0,
            "MQ7_CO_ppm": 9.0,
            "MQ2_LPG_ppm": 18.0,
            "MQ135_NOx_ppm": 0.9,
            "MQ135_NH3_ppm": 2.5, "NH3_ppm": 2.5,
            "MQ136_H2S_ppm": 0.6, "H2S_ppm": 0.6,
            "MG811_CO2_ppm": 415.0, "CO2_ppm": 415.0,
            "dust_ug_m3": 22.0,
            "temp": 22.5,
            "humidity": 55.0,
            "vibration_pulses": 1.0,
            "ultrasonic_distance": 3.2,
            "min_distance": 2.8,
            "US1": 2.8, "US2": 3.0, "US3": 3.2
        }

    else:
        scenario_title = "OPERATOR CUSTOM MULTI-NODE TELEMETRY"
        console.print("\n[bold cyan]📝 Enter Sensor Readings for NODE 1 (Tunnel A1):[/bold cyan]")
        n1_ch4 = FloatPrompt.ask("  Methane CH4 (ppm)", default=500.0)
        n1_co = FloatPrompt.ask("  Carbon Monoxide CO (ppm)", default=15.0)
        n1_lpg = FloatPrompt.ask("  LPG/CNG (ppm)", default=40.0)
        n1_nox = FloatPrompt.ask("  Nitrogen Oxides NOx (ppm)", default=1.5)
        n1_nh3 = FloatPrompt.ask("  Ammonia NH3 (ppm)", default=5.0)
        n1_h2s = FloatPrompt.ask("  Hydrogen Sulfide H2S (ppm)", default=1.0)
        n1_co2 = FloatPrompt.ask("  Carbon Dioxide CO2 (ppm)", default=450.0)
        n1_dust = FloatPrompt.ask("  Dust Particulates PM2.5 (µg/m³)", default=25.0)
        n1_temp = FloatPrompt.ask("  Temperature (°C)", default=23.0)
        n1_hum = FloatPrompt.ask("  Humidity (%)", default=55.0)
        n1_pulses = FloatPrompt.ask("  SW-420 Shock Pulses (pulses/s)", default=2.0)
        n1_dist = FloatPrompt.ask("  Robot Clearance Distance (m)", default=2.0)

        node1 = {
            "name": "Node 1 (Tunnel A1)",
            "segment_id": "TUNNEL_A1",
            "MQ4_CH4_ppm": n1_ch4, "CH4_ppm": n1_ch4,
            "MQ7_CO_ppm": n1_co, "CO_ppm": n1_co,
            "MQ2_LPG_ppm": n1_lpg, "LPG_ppm": n1_lpg,
            "MQ135_NOx_ppm": n1_nox,
            "MQ135_NH3_ppm": n1_nh3, "NH3_ppm": n1_nh3,
            "MQ136_H2S_ppm": n1_h2s, "H2S_ppm": n1_h2s,
            "MG811_CO2_ppm": n1_co2, "CO2_ppm": n1_co2,
            "dust_ug_m3": n1_dust, "PM25_Dust_ugm3": n1_dust,
            "temp": n1_temp, "Temp_C": n1_temp,
            "humidity": n1_hum, "Humidity_pct": n1_hum,
            "vibration_pulses": n1_pulses,
            "ultrasonic_distance": n1_dist,
            "min_distance": n1_dist,
            "US1": n1_dist, "US2": n1_dist, "US3": n1_dist
        }

        console.print("\n[bold cyan]📝 Enter Sensor Readings for NODE 2 (Tunnel B2):[/bold cyan]")
        n2_ch4 = FloatPrompt.ask("  Methane CH4 (ppm)", default=420.0)
        n2_co = FloatPrompt.ask("  Carbon Monoxide CO (ppm)", default=10.0)
        n2_lpg = FloatPrompt.ask("  LPG/CNG (ppm)", default=25.0)
        n2_nox = FloatPrompt.ask("  Nitrogen Oxides NOx (ppm)", default=1.0)
        n2_nh3 = FloatPrompt.ask("  Ammonia NH3 (ppm)", default=3.0)
        n2_h2s = FloatPrompt.ask("  Hydrogen Sulfide H2S (ppm)", default=0.5)
        n2_co2 = FloatPrompt.ask("  Carbon Dioxide CO2 (ppm)", default=420.0)
        n2_dust = FloatPrompt.ask("  Dust Particulates PM2.5 (µg/m³)", default=20.0)
        n2_temp = FloatPrompt.ask("  Temperature (°C)", default=22.0)
        n2_hum = FloatPrompt.ask("  Humidity (%)", default=50.0)
        n2_pulses = FloatPrompt.ask("  SW-420 Shock Pulses (pulses/s)", default=1.0)
        n2_dist = FloatPrompt.ask("  Robot Clearance Distance (m)", default=2.5)

        node2 = {
            "name": "Node 2 (Tunnel B2)",
            "segment_id": "TUNNEL_B2",
            "MQ4_CH4_ppm": n2_ch4, "CH4_ppm": n2_ch4,
            "MQ7_CO_ppm": n2_co, "CO_ppm": n2_co,
            "MQ2_LPG_ppm": n2_lpg, "LPG_ppm": n2_lpg,
            "MQ135_NOx_ppm": n2_nox,
            "MQ135_NH3_ppm": n2_nh3, "NH3_ppm": n2_nh3,
            "MQ136_H2S_ppm": n2_h2s, "H2S_ppm": n2_h2s,
            "MG811_CO2_ppm": n2_co2, "CO2_ppm": n2_co2,
            "dust_ug_m3": n2_dust, "PM25_Dust_ugm3": n2_dust,
            "temp": n2_temp, "Temp_C": n2_temp,
            "humidity": n2_hum, "Humidity_pct": n2_hum,
            "vibration_pulses": n2_pulses,
            "ultrasonic_distance": n2_dist,
            "min_distance": n2_dist,
            "US1": n2_dist, "US2": n2_dist, "US3": n2_dist
        }

    return node1, node2, scenario_title


def evaluate_node_models(monitor: Tier1Monitor, node: Dict[str, Any]) -> Dict[str, Any]:
    """Runs all 4 sensing domain models for a given node."""
    gas_res = monitor.evaluate_gas(node)
    env_res = monitor.evaluate_env(node)
    vib_res = monitor.evaluate_vibration(node)
    ultra_res = monitor.evaluate_ultrasonic(node)

    # Dynamic kinematic displacement evaluation
    disp_model = UltrasonicDisplacementModel()
    t_now = time.time()
    disp_model.add_reading(node.get("ultrasonic_distance", 2.5) + 0.15, t_now - 2.0)
    disp_model.add_reading(node.get("ultrasonic_distance", 2.5), t_now)
    kinematics = disp_model.evaluate()

    # Dynamic shock level evaluation
    sw_mon = SW420VibrationMonitor()
    shock_eval = sw_mon.evaluate(node.get("vibration_pulses", 0.0))

    combined = {
        **gas_res,
        **env_res,
        **vib_res,
        **ultra_res,
        **kinematics,
        **shock_eval
    }
    return combined


def build_telemetry_table(node1: Dict[str, Any], node2: Dict[str, Any]) -> Table:
    """Renders a side-by-side comparison table of raw sensor telemetry."""
    table = Table(title="📊 MULTI-NODE RAW SENSOR TELEMETRY", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Sensor Stream / Metric", style="bold white", width=26)
    table.add_column(f"Node 1 ({node1['segment_id']})", justify="center", width=22)
    table.add_column(f"Node 2 ({node2['segment_id']})", justify="center", width=22)
    table.add_column("Regulatory Normal Range", justify="center", style="dim", width=24)

    # Gas
    ch1_str = f"[bold red]{node1['MQ4_CH4_ppm']:.1f} ppm[/bold red]" if node1['MQ4_CH4_ppm'] > 5000 else f"{node1['MQ4_CH4_ppm']:.1f} ppm"
    ch2_str = f"[bold red]{node2['MQ4_CH4_ppm']:.1f} ppm[/bold red]" if node2['MQ4_CH4_ppm'] > 5000 else f"{node2['MQ4_CH4_ppm']:.1f} ppm"
    table.add_row("Methane (MQ-4 CH4)", ch1_str, ch2_str, "< 1,000 ppm (OSHA PEL)")

    co1_str = f"[bold red]{node1['MQ7_CO_ppm']:.1f} ppm[/bold red]" if node1['MQ7_CO_ppm'] > 25 else f"{node1['MQ7_CO_ppm']:.1f} ppm"
    co2_str = f"[bold red]{node2['MQ7_CO_ppm']:.1f} ppm[/bold red]" if node2['MQ7_CO_ppm'] > 25 else f"{node2['MQ7_CO_ppm']:.1f} ppm"
    table.add_row("Carbon Monoxide (MQ-7 CO)", co1_str, co2_str, "< 25 ppm (Post-blast)")

    table.add_row("LPG / CNG (MQ-2)", f"{node1['MQ2_LPG_ppm']:.1f} ppm", f"{node2['MQ2_LPG_ppm']:.1f} ppm", "< 1,000 ppm (OSHA)")
    table.add_row("Nitrogen Oxides (MQ-135)", f"{node1['MQ135_NOx_ppm']:.1f} ppm", f"{node2['MQ135_NOx_ppm']:.1f} ppm", "< 3.0 ppm (NIOSH REL)")
    
    nh3_1_str = f"[bold red]{node1.get('MQ135_NH3_ppm', 0.0):.1f} ppm[/bold red]" if node1.get('MQ135_NH3_ppm', 0.0) >= 25 else f"{node1.get('MQ135_NH3_ppm', 0.0):.1f} ppm"
    nh3_2_str = f"[bold red]{node2.get('MQ135_NH3_ppm', 0.0):.1f} ppm[/bold red]" if node2.get('MQ135_NH3_ppm', 0.0) >= 25 else f"{node2.get('MQ135_NH3_ppm', 0.0):.1f} ppm"
    table.add_row("Ammonia (MQ-135 NH3)", nh3_1_str, nh3_2_str, "< 25 ppm (NIOSH REL)")

    h2s_1_str = f"[bold red]{node1.get('MQ136_H2S_ppm', 0.0):.1f} ppm[/bold red]" if node1.get('MQ136_H2S_ppm', 0.0) >= 10 else f"{node1.get('MQ136_H2S_ppm', 0.0):.1f} ppm"
    h2s_2_str = f"[bold red]{node2.get('MQ136_H2S_ppm', 0.0):.1f} ppm[/bold red]" if node2.get('MQ136_H2S_ppm', 0.0) >= 10 else f"{node2.get('MQ136_H2S_ppm', 0.0):.1f} ppm"
    table.add_row("Hydrogen Sulfide (MQ-136 H2S)", h2s_1_str, h2s_2_str, "< 10 ppm (OSHA PEL)")

    co2_1_str = f"[bold red]{node1.get('MG811_CO2_ppm', 400.0):.1f} ppm[/bold red]" if node1.get('MG811_CO2_ppm', 400.0) >= 5000 else f"{node1.get('MG811_CO2_ppm', 400.0):.1f} ppm"
    co2_2_str = f"[bold red]{node2.get('MG811_CO2_ppm', 400.0):.1f} ppm[/bold red]" if node2.get('MG811_CO2_ppm', 400.0) >= 5000 else f"{node2.get('MG811_CO2_ppm', 400.0):.1f} ppm"
    table.add_row("Carbon Dioxide (MG-811 CO2)", co2_1_str, co2_2_str, "< 5,000 ppm (OSHA PEL)")

    table.add_row("Dust Particulates (PM2.5)", f"{node1['dust_ug_m3']:.1f} µg/m³", f"{node2['dust_ug_m3']:.1f} µg/m³", "< 50 µg/m³ (OSHA)")

    # Environment
    t1_str = f"[bold yellow]{node1['temp']:.1f} °C[/bold yellow]" if node1['temp'] > 28 else f"{node1['temp']:.1f} °C"
    t2_str = f"[bold yellow]{node2['temp']:.1f} °C[/bold yellow]" if node2['temp'] > 28 else f"{node2['temp']:.1f} °C"
    table.add_row("Temperature (DHT22)", t1_str, t2_str, "18 – 28 °C (Safe Zone)")

    h1_str = f"[bold yellow]{node1['humidity']:.1f} %[/bold yellow]" if node1['humidity'] > 85 else f"{node1['humidity']:.1f} %"
    h2_str = f"[bold yellow]{node2['humidity']:.1f} %[/bold yellow]" if node2['humidity'] > 85 else f"{node2['humidity']:.1f} %"
    table.add_row("Relative Humidity (DHT22)", h1_str, h2_str, "15 – 85 % RH")

    # Vibration & Navigation
    s1_str = f"[bold red]{node1['vibration_pulses']:.0f} p/s[/bold red]" if node1['vibration_pulses'] > 10 else f"{node1['vibration_pulses']:.0f} p/s"
    s2_str = f"[bold red]{node2['vibration_pulses']:.0f} p/s[/bold red]" if node2['vibration_pulses'] > 10 else f"{node2['vibration_pulses']:.0f} p/s"
    table.add_row("SW-420 Shock Pulses", s1_str, s2_str, "< 5 pulses/s (Baseline)")

    d1_str = f"[bold red]{node1['min_distance']:.2f} m[/bold red]" if node1['min_distance'] < 0.3 else f"{node1['min_distance']:.2f} m"
    d2_str = f"[bold red]{node2['min_distance']:.2f} m[/bold red]" if node2['min_distance'] < 0.3 else f"{node2['min_distance']:.2f} m"
    table.add_row("Robot Obstacle Distance", d1_str, d2_str, "> 0.50 m (AS 4024)")

    return table


def build_predictions_table(p1: Dict[str, Any], p2: Dict[str, Any], id1: str, id2: str) -> Table:
    """Renders a comparison table of ML model inference predictions."""
    table = Table(title="🤖 TIER-1 AI SENSOR MODEL PREDICTIONS", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("AI Model / Domain", style="bold white", width=26)
    table.add_column(f"Node 1 ({id1}) Output", justify="center", width=22)
    table.add_column(f"Node 2 ({id2}) Output", justify="center", width=22)
    table.add_column("Safety Assessment", justify="center", width=24)

    # 1. Multi-Gas DL Ensemble (8 Gases)
    g1_status = "[bold red]🚨 DANGER (CH4/CO)[/bold red]" if (p1.get("methane_hazard") or p1.get("co_nox_hazard") or p1.get("lpg_hazard")) else "[bold green]✔ NOMINAL[/bold green]"
    g2_status = "[bold red]🚨 DANGER (CH4/CO)[/bold red]" if (p2.get("methane_hazard") or p2.get("co_nox_hazard") or p2.get("lpg_hazard")) else "[bold green]✔ NOMINAL[/bold green]"
    table.add_row("Multi-Gas DL Ensemble", g1_status, g2_status, "Deep Learning (8-Channel)")

    # 2. Ammonia (NH3) Toxic Hazard Model
    nh3_1 = "[bold red]🚨 HIGH NH3 TOXICITY[/bold red]" if p1.get("nh3_hazard") else "[bold green]✔ SAFE (<25 ppm)[/bold green]"
    nh3_2 = "[bold red]🚨 HIGH NH3 TOXICITY[/bold red]" if p2.get("nh3_hazard") else "[bold green]✔ SAFE (<25 ppm)[/bold green]"
    table.add_row("Ammonia (NH3) Hazard Net", nh3_1, nh3_2, "PyTorch Deep MLP (NIOSH)")

    # 3. Carbon Dioxide (CO2) Asphyxiation Hazard Model
    co2_h1 = "[bold red]🚨 CO2 ASPHYXIATION[/bold red]" if p1.get("co2_hazard") else "[bold green]✔ SAFE (<1000 ppm)[/bold green]"
    co2_h2 = "[bold red]🚨 CO2 ASPHYXIATION[/bold red]" if p2.get("co2_hazard") else "[bold green]✔ SAFE (<1000 ppm)[/bold green]"
    table.add_row("CO2 Asphyxiation Net", co2_h1, co2_h2, "PyTorch Deep MLP (>1000 ppm)")

    # 4. Smoke / Dust Particulate Hazard Model
    smk1 = "[bold yellow]⚠ SMOKE / DUST ALERT[/bold yellow]" if p1.get("smoke_env_hazard") else "[bold green]✔ CLEAR[/bold green]"
    smk2 = "[bold yellow]⚠ SMOKE / DUST ALERT[/bold yellow]" if p2.get("smoke_env_hazard") else "[bold green]✔ CLEAR[/bold green]"
    table.add_row("Smoke/Dust Hazard Model", smk1, smk2, "PyTorch Optical Classifier")

    # 5. Gas Severity Head (CH4 / CO / H2S)
    sev1_ch4 = p1.get("ch4_severity", 0)
    sev2_ch4 = p2.get("ch4_severity", 0)
    sev1_str = f"[bold red]L3 DANGER[/bold red]" if sev1_ch4 == 2 else (f"[bold yellow]L2 WARNING[/bold yellow]" if sev1_ch4 == 1 else "[bold green]L1 Safe[/bold green]")
    sev2_str = f"[bold red]L3 DANGER[/bold red]" if sev2_ch4 == 2 else (f"[bold yellow]L2 WARNING[/bold yellow]" if sev2_ch4 == 1 else "[bold green]L1 Safe[/bold green]")
    table.add_row("Methane (CH4) Severity Head", sev1_str, sev2_str, "3-Class Graded Alarm (L1-L3)")

    sev1_h2s = p1.get("h2s_severity", 0)
    sev2_h2s = p2.get("h2s_severity", 0)
    h2s1_str = f"[bold red]L3 DANGER[/bold red]" if sev1_h2s == 2 else (f"[bold yellow]L2 WARNING[/bold yellow]" if sev1_h2s == 1 else "[bold green]L1 Safe[/bold green]")
    h2s2_str = f"[bold red]L3 DANGER[/bold red]" if sev2_h2s == 2 else (f"[bold yellow]L2 WARNING[/bold yellow]" if sev2_h2s == 1 else "[bold green]L1 Safe[/bold green]")
    table.add_row("H2S Severity Head", h2s1_str, h2s2_str, "3-Class Toxic Alarm (L1-L3)")

    # 6. Env Thermal Drift (Isolation Forest / Threshold Fallback)
    env1 = "[bold yellow]⚠ ANOMALY[/bold yellow]" if p1.get("anomaly_detected") else "[bold green]✔ NOMINAL[/bold green]"
    env2 = "[bold yellow]⚠ ANOMALY[/bold yellow]" if p2.get("anomaly_detected") else "[bold green]✔ NOMINAL[/bold green]"
    table.add_row("Env Thermal Anomaly Check", env1, env2, "Thermal Drift & Range Check")

    # 7. SW-420 Shock Monitor
    sk1_level = p1.get("shock_level", 0)
    sk2_level = p2.get("shock_level", 0)
    sk1_str = f"[bold red]Level {sk1_level} CRITICAL[/bold red]" if sk1_level == 2 else (f"[bold yellow]Level {sk1_level} ALERT[/bold yellow]" if sk1_level == 1 else "[bold green]Level 0 Normal[/bold green]")
    sk2_str = f"[bold red]Level {sk2_level} CRITICAL[/bold red]" if sk2_level == 2 else (f"[bold yellow]Level {sk2_level} ALERT[/bold yellow]" if sk2_level == 1 else "[bold green]Level 0 Normal[/bold green]")
    table.add_row("SW-420 Shock Monitor", sk1_str, sk2_str, "Omnidirectional Impact")

    # 8. Wall Kinematics Displacement
    col1 = "[bold red]🚨 COLLAPSE IMMINENT[/bold red]" if p1.get("collapse_imminent") else "[bold green]✔ STABLE[/bold green]"
    col2 = "[bold red]🚨 COLLAPSE IMMINENT[/bold red]" if p2.get("collapse_imminent") else "[bold green]✔ STABLE[/bold green]"
    table.add_row("Geomechanical Wall Kinematics", col1, col2, "Ultrasonic Convergence Rate")

    # 9. Robot Steering Decision (with Proximity Fallback)
    steer1 = p1.get("steering_decision", "Move-Forward")
    steer2 = p2.get("steering_decision", "Move-Forward")
    st1_str = f"[bold red]🚨 {steer1}[/bold red]" if "Sharp" in steer1 else f"[bold green]{steer1}[/bold green]"
    st2_str = f"[bold red]🚨 {steer2}[/bold red]" if "Sharp" in steer2 else f"[bold green]{steer2}[/bold green]"
    table.add_row("Autonomous Navigation", st1_str, st2_str, "Obstacle Distance Clearance")

    return table


def main():
    console.clear()
    render_header()

    # 1. Initialize Resources
    with console.status("[bold green]Loading FIELD-MIND on-device models, FAISS RAG, EKG, and LLM Reasoner...", spinner="dots"):
        monitor = Tier1Monitor(root_dir=WORKSPACE_ROOT)
        bus = AgentBus()
        gas_agent = GasSensorAgent(workspace_root=WORKSPACE_ROOT, bus=bus, verbose=False)
        assistant = MineSafetyChatAssistant(workspace_root=WORKSPACE_ROOT)
        if assistant.rag_retriever:
            assistant.rag_retriever.warmup()

    # 2. Get Telemetry for 2 Nodes
    node1, node2, scenario_name = get_preset_telemetry()

    console.print(f"\n[bold green]✓ Loaded Telemetry for 2 Nodes ({scenario_name})[/bold green]")

    # 3. Model Inference Pass
    p1 = evaluate_node_models(monitor, node1)
    p2 = evaluate_node_models(monitor, node2)

    # 4. Display Multi-Node Dashboard
    console.print("\n" + "═" * 85)
    console.print(f"  🏢 [bold cyan]FIELD-MIND MULTI-NODE VISUAL MONITORING DASHBOARD[/bold cyan] — [yellow]{scenario_name}[/yellow]")
    console.print("═" * 85 + "\n")

    t_table = build_telemetry_table(node1, node2)
    p_table = build_predictions_table(p1, p2, node1["segment_id"], node2["segment_id"])

    console.print(t_table)
    console.print("\n")
    console.print(p_table)

    # 5. Operator Feedback & Self-Learning Loop
    console.print("\n" + "─" * 85)
    console.print("🔄 [bold yellow]STEP 3: SUPERVISOR GROUND-TRUTH VERIFICATION[/bold yellow]")
    console.print("─" * 85)

    is_correct = Confirm.ask("Do the AI model predictions match the verified physical ground truth across both nodes?", default=True)

    if is_correct:
        console.print("[bold green]  ✓ Predictions verified against physical reality. Baseline monitoring continuing.[/bold green]")
    else:
        console.print("\n[bold red]  ⚠ Discrepancy Detected![/bold red] Initiating on-device Self-Reflection & Experience Replay Retraining...")
        target_node = Prompt.ask("Which node experienced the false alert/discrepancy?", choices=["1", "2"], default="1")
        active_node = node1 if target_node == "1" else node2
        
        actual_situation = Prompt.ask(
            "  Enter verified actual physical situation",
            default="Tunnel High-Pressure Water Spraying Operation"
        )
        explanation = Prompt.ask(
            "  Enter root cause explanation for discrepancy",
            default="MQ-7 electrochemical sensor experienced moisture condensation drift at >85% humidity."
        )
        true_label = int(Prompt.ask("  Enter true ground-truth hazard label (0=Safe/Normal, 1=Hazard)", choices=["0", "1"], default="0"))

        with console.status("[bold cyan]Running LLM Self-Reflection, updating FAISS RAG, and logging to EKG...", spinner="dots"):
            # Trigger LangGraph Reflection
            reflection_res = assistant.core.reflect_and_learn(
                anomalies=active_node,
                actual_situation=actual_situation,
                explanation=explanation,
                segment_id=active_node["segment_id"]
            )
            # Update Gas Agent Experience Replay Buffer for Online Retraining
            features_dict = gas_agent.perceive(active_node)
            gas_agent.feedback_correction(
                features=features_dict,
                true_label=true_label,
                actual_situation=actual_situation,
                explanation=explanation
            )

        console.print(f"[bold green]  ✓ LLM Generated Corrective Rule:[/bold green] [italic]{reflection_res.get('rule_text')}[/italic]")
        console.print(f"[bold green]  ✓ Experience Replay Buffer updated ({len(gas_agent._replay_X)}/200 samples). Online refit armed.[/bold green]")

    # 6. LLM Situation Explanation & Precautions
    console.print("\n" + "─" * 85)
    console.print("🤖 [bold cyan]STEP 4: FIELD-MIND LLM MULTI-NODE DIAGNOSTIC & SAFETY PRECAUTIONS[/bold cyan]")
    console.print("─" * 85)

    with console.status("[bold magenta]LLM analyzing multi-node telemetry, EKG graph history, and FAISS safety literature...", spinner="dots"):
        # Worst-case merged telemetry for protocol threshold checks
        merged_readings = {
            "MQ4_CH4_ppm": max(node1["MQ4_CH4_ppm"], node2["MQ4_CH4_ppm"]),
            "MQ7_CO_ppm": max(node1["MQ7_CO_ppm"], node2["MQ7_CO_ppm"]),
            "MQ2_LPG_ppm": max(node1["MQ2_LPG_ppm"], node2["MQ2_LPG_ppm"]),
            "MQ135_NOx_ppm": max(node1["MQ135_NOx_ppm"], node2["MQ135_NOx_ppm"]),
            "dust_ug_m3": max(node1["dust_ug_m3"], node2["dust_ug_m3"]),
            "temp": max(node1["temp"], node2["temp"]),
            "humidity": max(node1["humidity"], node2["humidity"]),
            "vibration_pulses": max(node1["vibration_pulses"], node2["vibration_pulses"]),
            "min_distance": min(node1["min_distance"], node2["min_distance"]),
            "ultrasonic_distance": min(node1["ultrasonic_distance"], node2["ultrasonic_distance"]),
        }

        # Multi-node summary trend
        trend_summary = (
            f"SPATIAL-TEMPORAL NODE COMPARISON:\n"
            f"• {node1['name']} ({node1['segment_id']}): CH4={node1['MQ4_CH4_ppm']:.1f} ppm, CO={node1['MQ7_CO_ppm']:.1f} ppm, "
            f"Temp={node1['temp']:.1f}°C, RH={node1['humidity']:.1f}%, Shock={node1['vibration_pulses']:.0f} p/s, Obstacle={node1['min_distance']:.2f}m. "
            f"Status: {'🚨 DANGER' if p1.get('methane_hazard') or p1.get('shock_level',0)==2 or p1.get('collapse_imminent') else '✔ NOMINAL'}.\n"
            f"• {node2['name']} ({node2['segment_id']}): CH4={node2['MQ4_CH4_ppm']:.1f} ppm, CO={node2['MQ7_CO_ppm']:.1f} ppm, "
            f"Temp={node2['temp']:.1f}°C, RH={node2['humidity']:.1f}%, Shock={node2['vibration_pulses']:.0f} p/s, Obstacle={node2['min_distance']:.2f}m. "
            f"Status: {'🚨 DANGER' if p2.get('methane_hazard') or p2.get('shock_level',0)==2 or p2.get('collapse_imminent') else '✔ NOMINAL'}."
        )

        merged_predictions = {
            "methane_hazard": p1.get("methane_hazard") or p2.get("methane_hazard"),
            "co_nox_hazard": p1.get("co_nox_hazard") or p2.get("co_nox_hazard"),
            "smoke_alarm": p1.get("smoke_env_hazard") or p2.get("smoke_env_hazard"),
            "anomaly_detected": p1.get("anomaly_detected") or p2.get("anomaly_detected"),
            "shock_alert": p1.get("shock_alert") or p2.get("shock_alert"),
            "shock_level": max(p1.get("shock_level", 0), p2.get("shock_level", 0)),
            "collapse_imminent": p1.get("collapse_imminent") or p2.get("collapse_imminent"),
            "sharp_turn_required": p1.get("sharp_turn_required") or p2.get("sharp_turn_required"),
            "steering_decision": f"Node1:{p1.get('steering_decision')} | Node2:{p2.get('steering_decision')}"
        }

        llm_response = assistant.chat(
            user_message=(
                f"Multi-node safety situation analysis for {node1['segment_id']} and {node2['segment_id']}. "
                "Analyze the telemetry from both nodes, diagnose any active hazard, cite regulatory standards, "
                "and explain required safety precautions."
            ),
            segment_id=f"{node1['segment_id']} / {node2['segment_id']}",
            active_anomalies=merged_readings,
            model_predictions=merged_predictions,
            sensor_readings=merged_readings,
            trend_context=trend_summary
        )

    console.print(Panel(
        Markdown(llm_response),
        title="[bold green]FIELD-MIND SITUATION ASSESSMENT & ACTION PLAN[/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))

    # 7. Interactive Multi-Turn Chat
    console.print("\n" + "─" * 85)
    console.print("💬 [bold yellow]STEP 5: INTERACTIVE CONVERSATION WITH FIELD-MIND SAFETY ASSISTANT[/bold yellow]")
    console.print("  [dim]Type your questions below (e.g. 'What should workers do?', 'Why is Node 2 in alert?'). Type 'exit' to quit.[/dim]")
    console.print("─" * 85)

    while True:
        try:
            user_q = Prompt.ask("\n[bold cyan]👤 Operator Question[/bold cyan]").strip()
            if not user_q:
                continue
            if user_q.lower() in ["exit", "quit", "q"]:
                console.print("\n[bold green]Shutting down Multi-Node Safety Hub. Stay safe underground![/bold green]")
                break

            # Low latency stream execution (TTFT < 300ms)
            full_reply = ""
            with console.status("[bold cyan]Consulting EKG memory, FAISS safety rules, and LLM reasoning...", spinner="dots"):
                chunks = list(assistant.chat_stream(
                    user_message=user_q,
                    segment_id=f"{node1['segment_id']} / {node2['segment_id']}",
                    active_anomalies=merged_readings,
                    model_predictions=merged_predictions,
                    sensor_readings=merged_readings,
                    trend_context=trend_summary
                ))
                full_reply = "".join(chunks)

            # Ensure Asked Question is ALWAYS included in response content
            if "### ❓ Asked Question" not in full_reply:
                full_reply = f"### ❓ Asked Question\n> **{user_q}**\n\n---\n" + full_reply

            console.print("\n" + "─" * 40)
            console.print(Panel(
                Markdown(full_reply),
                title=f"[bold cyan]🤖 FIELD-MIND Response — Question: [/bold cyan][bold white]\"{user_q}\"[/bold white]",
                border_style="cyan",
                box=box.ROUNDED
            ))
            console.print("─" * 40)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Session interrupted. Exiting...[/bold yellow]")
            break


if __name__ == "__main__":
    main()
