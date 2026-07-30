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

print("--- Testing GasSensorAgent with H2S and NH3 ---")
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
    "Sensor3[ppm]": 0.0,  # H2S
    "NH3": 0.0,            # NH3
}

# Test case 2: NH3 hazard
data_nh3_hazard = {
    "MQ2_LPG_ppm": 10.0,
    "MQ4_CH4_ppm": 100.0,
    "MQ7_CO_ppm": 5.0,
    "MQ135_NOx_ppm": 0.05,
    "MQ3_Benzene_ppm": 0.5,
    "Sensor3[ppm]": 1.0,  # low H2S
    "NH3": 28.0,           # High NH3 >= 25
}

# Test case 3: H2S severity L2 warning
data_h2s_l2 = {
    "MQ2_LPG_ppm": 10.0,
    "MQ4_CH4_ppm": 100.0,
    "MQ7_CO_ppm": 5.0,
    "MQ135_NOx_ppm": 0.05,
    "MQ3_Benzene_ppm": 0.5,
    "Sensor3[ppm]": 15.0, # H2S = 15 ppm (L2)
    "NH3": 10.0,
}

print("\n1. Testing Normal Input:")
res_normal = agent.perceive(data_normal)
infer_normal = agent.infer(res_normal)
conf_normal = agent.compute_confidence(infer_normal)
print("Normal Features:", res_normal)
print("Normal Inference:", infer_normal)
print("Normal Confidence:", conf_normal)

print("\n2. Testing NH3 Hazard Input:")
res_nh3 = agent.perceive(data_nh3_hazard)
infer_nh3 = agent.infer(res_nh3)
conf_nh3 = agent.compute_confidence(infer_nh3)
print("NH3 Features:", res_nh3)
print("NH3 Inference:", infer_nh3)
print("NH3 Confidence:", conf_nh3)
reason_nh3 = agent._build_reason(infer_nh3, conf_nh3)
print("NH3 Alert Reason:", reason_nh3)

print("\n3. Testing H2S L2 Severity Input:")
res_h2s = agent.perceive(data_h2s_l2)
infer_h2s = agent.infer(res_h2s)
conf_h2s = agent.compute_confidence(infer_h2s)
print("H2S Features:", res_h2s)
print("H2S Inference:", infer_h2s)
print("H2S Confidence:", conf_h2s)
reason_h2s = agent._build_reason(infer_h2s, conf_h2s)
print("H2S Alert Reason:", reason_h2s)

print("\n--- Test complete ---")
