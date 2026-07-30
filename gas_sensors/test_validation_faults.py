import os
import sys
import numpy as np

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from sensor_agents.agent_bus import AgentBus
from sensor_agents.gas_agent import GasSensorAgent
from sensor_agents.input_validator import validate_sensor, SensorHealthReport

def main():
    print("=" * 60)
    print("TEST SUITE: SENSOR INPUT VALIDATION & FAULT DETECTION")
    print("=" * 60)
    
    bus = AgentBus()
    agent = GasSensorAgent(workspace_root=root_dir, bus=bus, verbose=False)
    
    # -------------------------------------------------------------------
    # Test 1: Negative and NaN Inputs (Clamping & Range Flags)
    # -------------------------------------------------------------------
    print("\n1. Testing Negative and NaN Inputs...")
    rep_neg = validate_sensor("MQ4_CH4_ppm", -50.0, history=[])
    assert not rep_neg.is_valid, "Negative input should be marked invalid"
    assert rep_neg.clamped_value == 0.0, f"Expected clamped_value 0.0, got {rep_neg.clamped_value}"
    print("  [PASS] Negative input correctly clamped to 0.0 ppm and flagged.")

    rep_nan = validate_sensor("Temp_C", float("nan"), history=[])
    assert not rep_nan.is_valid, "NaN input should be marked invalid"
    assert rep_nan.clamped_value == -40.0, f"Expected clamped_value -40.0, got {rep_nan.clamped_value}"
    assert rep_nan.status == "OUT_OF_RANGE", f"Expected status OUT_OF_RANGE, got {rep_nan.status}"
    print("  [PASS] NaN input correctly clamped to min limit (-40.0 C) and flagged.")

    # -------------------------------------------------------------------
    # Test 2: Stuck Sensor Detection (10 identical consecutive values)
    # -------------------------------------------------------------------
    print("\n2. Testing Stuck Sensor Detection...")
    stuck_history = [25.0] * 10
    rep_stuck = validate_sensor("Temp_C", 25.0, history=stuck_history)
    assert rep_stuck.is_stuck, "10 identical values should trigger is_stuck"
    assert rep_stuck.status == "STUCK", f"Expected status STUCK, got {rep_stuck.status}"
    print("  [PASS] Stuck sensor detected after 10 identical consecutive ticks.")

    # -------------------------------------------------------------------
    # Test 3: Dead Sensor Detection (Analog reading at or below dead threshold)
    # -------------------------------------------------------------------
    print("\n3. Testing Dead Sensor Detection...")
    rep_dead = validate_sensor("MQ4_CH4_ppm", 0.0, history=[])
    assert rep_dead.is_dead, "0.0 ppm CH4 should trigger is_dead (dead threshold 5.0)"
    assert rep_dead.status == "DEAD", f"Expected status DEAD, got {rep_dead.status}"
    print("  [PASS] Dead analog sensor detected (0.0 ppm <= 5.0 ppm threshold).")

    # -------------------------------------------------------------------
    # Test 4: Out-of-Bounds Input (Exceeding max datasheet limit)
    # -------------------------------------------------------------------
    print("\n4. Testing Out-of-Bounds Input...")
    rep_oob = validate_sensor("MQ4_CH4_ppm", 75000.0, history=[])
    assert not rep_oob.is_valid, "Exceeding max range should be marked invalid"
    assert rep_oob.clamped_value == 50000.0, f"Expected 50000.0 max clamp, got {rep_oob.clamped_value}"
    assert rep_oob.status == "OUT_OF_RANGE", f"Expected status OUT_OF_RANGE, got {rep_oob.status}"
    print("  [PASS] Out-of-bounds reading (75000 ppm) clamped to max limit (50000 ppm).")

    # -------------------------------------------------------------------
    # Test 5: End-to-End GasSensorAgent Perception with Faults & MQ-4 Multiclass
    # -------------------------------------------------------------------
    print("\n5. Testing End-to-End GasSensorAgent perceive(), infer(), and fault alerts...")
    
    # Generate 128-dimensional mock features
    mock_128_feats = [float(np.sin(i * 0.1)) for i in range(128)]
    
    faulty_raw_data = {
        "MQ2_LPG_ppm": 10.0,
        "MQ4_CH4_ppm": -10.0,  # Faulty negative CH4 input
        "MQ7_CO_ppm": 5.0,
        "MQ135_NOx_ppm": 0.05,
        "MQ3_Benzene_ppm": 0.5,
        "MG811_CO2_ppm": 400.0,
        "PM25_Dust_ugm3": 20.0,
        "Temp_C": 25.0,
        "Humidity_pct": 50.0,
        "mq4_features": mock_128_feats,
    }
    
    perceived = agent.perceive(faulty_raw_data)
    assert perceived["MQ4_CH4_ppm"] == 0.0, f"Perceived MQ4_CH4_ppm should be clamped to 0.0, got {perceived['MQ4_CH4_ppm']}"
    assert "mq4_features" in perceived, "128-dimensional mq4_features should be retained"
    
    inferred = agent.infer(perceived)
    assert "mq4_class" in inferred, "Inference result must contain mq4_class"
    
    conf = agent.compute_confidence(inferred)
    assert inferred.get("sensor_fault") == 1.0, "Sensor fault flag should be 1.0 in inference result"
    
    reason = agent._build_reason(inferred, conf)
    assert "SENSOR FAULT DETECTED" in reason, f"Reason should contain fault alert, got: {reason}"
    print(f"  [PASS] End-to-End validation succeeded!")
    print(f"  Inferred dictionary: {inferred}")
    print(f"  Alert confidence: {conf:.2f}")
    print(f"  Alert Reason String: {reason}")
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
