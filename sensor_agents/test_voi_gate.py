"""
test_voi_gate.py — Unit Tests for VoI Decision-Theoretic Escalation Gate
"""

import sys
import os
import unittest

# Adjust system path to import from the workspace
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensor_agents.agent_bus import AgentBus, AgentMessage, MessageType, Severity
from sensor_agents.mine_orchestrator_agent import MineOrchestratorAgent

class TestVoIGate(unittest.TestCase):
    def setUp(self):
        self.bus = AgentBus()
        self.orchestrator = MineOrchestratorAgent(self.bus, verbose=False)

    def test_voi_math_low_score(self):
        # Low risk: S = 0.01
        score = 0.01
        voi = self.orchestrator.compute_voi(score)
        
        # Expected utility of immediate: max(-0.01 * 100, -15) = max(-1, -15) = -1
        # Expected utility with reasoning: -0.01 * 15 = -0.15
        # Expected VoI: -0.15 - (-1) = 0.85
        self.assertAlmostEqual(voi, 0.85, places=4)
        
        # Since VoI (0.85) <= c_reason (1.5), we shouldn't wake the LLM.
        # Immediate action is max(-1, -15) = -1 (choose base action, i.e., IDLE)
        self.assertLessEqual(voi, self.orchestrator.c_reason)

    def test_voi_math_medium_score(self):
        # Medium risk: S = 0.40 (uncertainty zone)
        score = 0.40
        voi = self.orchestrator.compute_voi(score)
        
        # Expected utility of immediate: max(-40, -15) = -15 (choose evacuation)
        # Expected utility with reasoning: -40 * 0.15 * 100 = -6.0
        # Expected VoI: -6 - (-15) = 9.0
        self.assertAlmostEqual(voi, 9.0, places=4)
        
        # VoI (9.0) > c_reason (1.5) -> wake LLM (ACTIVE_REASONING)
        self.assertGreater(voi, self.orchestrator.c_reason)

    def test_voi_math_high_score(self):
        # High risk: S = 0.95 (danger is certain)
        score = 0.95
        voi = self.orchestrator.compute_voi(score)
        
        # Expected utility of immediate: max(-95, -15) = -15 (choose evacuation)
        # Expected utility with reasoning: -0.95 * 15 = -14.25
        # Expected VoI: -14.25 - (-15) = 0.75
        self.assertAlmostEqual(voi, 0.75, places=4)
        
        # VoI (0.75) <= c_reason (1.5) -> do not wake LLM.
        # Immediate action is max(-95, -15) = -15 -> choose evacuation (EMERGENCY)
        self.assertLessEqual(voi, self.orchestrator.c_reason)

    def test_state_transition_triggers_correctly(self):
        # Let's verify the orchestrator's state transitions under different score inputs:
        
        # Scenario A: S = 0.01 (IDLE)
        # Injects low confidence sensor alerts to get score = 0.01
        self.orchestrator.use_voi = True
        self.orchestrator.device_state = "IDLE"
        self.orchestrator._low_score_streak = 10
        self.orchestrator._evaluate_global_state(100.0)
        self.assertEqual(self.orchestrator.device_state, "IDLE")

        # Scenario B: S = 0.40 (ACTIVE_REASONING)
        # Mocking an alert message on the bus to trigger evaluation
        alert = AgentMessage(
            source="GasSensorAgent",
            msg_type=MessageType.ALERT,
            severity=Severity.HIGH,
            payload={"confidence": 0.95}, # weight 0.35 * HIGH (1.0) * conf (0.95) = 0.3325
            reason="High gas presence",
            timestamp=101.0
        )
        self.orchestrator._active_alerts["GasSensorAgent"] = alert
        self.orchestrator._evaluate_global_state(101.0)
        self.assertEqual(self.orchestrator.device_state, "ACTIVE_REASONING")

        # Scenario C: S = 0.95 (EMERGENCY)
        # Mocking multiple alerts to trigger score > 0.95
        self.orchestrator._active_alerts["VibrationSensorAgent"] = AgentMessage(
            source="VibrationSensorAgent",
            msg_type=MessageType.ALERT,
            severity=Severity.CRITICAL, # weight 0.30 * 1.25 = 0.375
            payload={"confidence": 1.0},
            reason="Blast vibration",
            timestamp=102.0
        )
        self.orchestrator._active_alerts["EnvSensorAgent"] = AgentMessage(
            source="EnvSensorAgent",
            msg_type=MessageType.ALERT,
            severity=Severity.CRITICAL, # weight 0.20 * 1.25 = 0.25
            payload={"confidence": 1.0},
            reason="Thermal anomaly",
            timestamp=102.0
        )
        # Total score: 0.3325 (Gas) + 0.375 (Vib) + 0.25 (Env) = 0.9575
        self.orchestrator._evaluate_global_state(102.0)
        self.assertEqual(self.orchestrator.device_state, "EMERGENCY")

if __name__ == "__main__":
    unittest.main()
