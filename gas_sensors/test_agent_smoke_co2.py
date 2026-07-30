import os
import sys
import numpy as np

# Ensure project root is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from sensor_agents.agent_bus import AgentBus
from sensor_agents.gas_agent import GasSensorAgent

print("--- Testing GasSensorAgent with CO2 and Dust/Smoke ---")
bus = AgentBus()
agent = GasSensorAgent(workspace_root=root_dir, bus=bus, verbose=True)

# Define test inputs
# Test case 1: Nominal air
data_normal = {
    "MQ2_LPG_ppm": 10.0,
    "MQ4_CH4_ppm": 100.0,
    "MQ7_CO_ppm": 5.0,
    "MQ135_NOx_ppm": 0.05,
    "MQ3_Benzene_ppm": 0.5,
    "MG811_CO2_ppm": 400.0,
    "PM25_Dust_ugm3": 20.0,
    "Temp_C": 25.0,
    "Humidity_pct": 50.0,
}

# Test case 2: CO2 hazard (asphyxiation risk >= 5000 ppm)
data_co2_hazard = {
    "MQ2_LPG_ppm": 10.0,
    "MQ4_CH4_ppm": 100.0,
    "MQ7_CO_ppm": 5.0,
    "MQ135_NOx_ppm": 0.05,
    "MQ3_Benzene_ppm": 0.5,
    "MG811_CO2_ppm": 5200.0, # High CO2
    "PM25_Dust_ugm3": 20.0,
    "Temp_C": 25.0,
    "Humidity_pct": 50.0,
}

# Test case 3: Dust/Smoke hazard (high PM2.5 + high Temp)
data_smoke_hazard = {
    "MQ2_LPG_ppm": 10.0,
    "MQ4_CH4_ppm": 100.0,
    "MQ7_CO_ppm": 5.0,
    "MQ135_NOx_ppm": 0.05,
    "MQ3_Benzene_ppm": 0.5,
    "MG811_CO2_ppm": 400.0,
    "PM25_Dust_ugm3": 250.0, # High PM2.5 dust
    "Temp_C": 42.0,          # High Temperature
    "Humidity_pct": 12.0,    # Dry condition
}

print("\n1. Testing Normal Input:")
res_normal = agent.perceive(data_normal)
infer_normal = agent.infer(res_normal)
conf_normal = agent.compute_confidence(infer_normal)
print("Normal Features:", res_normal)
print("Normal Inference:", infer_normal)
print("Normal Confidence:", conf_normal)

print("\n2. Testing CO2 Hazard Input:")
res_co2 = agent.perceive(data_co2_hazard)
infer_co2 = agent.infer(res_co2)
conf_co2 = agent.compute_confidence(infer_co2)
print("CO2 Features:", res_co2)
print("CO2 Inference:", infer_co2)
print("CO2 Confidence:", conf_co2)
reason_co2 = agent._build_reason(infer_co2, conf_co2)
print("CO2 Alert Reason:", reason_co2)

print("\n3. Testing Dust/Smoke Hazard Input:")
res_smoke = agent.perceive(data_smoke_hazard)
infer_smoke = agent.infer(res_smoke)
conf_smoke = agent.compute_confidence(infer_smoke)
print("Smoke Features:", res_smoke)
print("Smoke Inference:", infer_smoke)
print("Smoke Confidence:", conf_smoke)
reason_smoke = agent._build_reason(infer_smoke, conf_smoke)
print("Smoke Alert Reason:", reason_smoke)

print("\n--- Test complete ---")
