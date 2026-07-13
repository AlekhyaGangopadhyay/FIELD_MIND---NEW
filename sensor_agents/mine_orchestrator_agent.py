"""
mine_orchestrator_agent.py — Global Mine Hazard Orchestrator
=============================================================
The MineOrchestratorAgent is the top-level coordinator that:

  1. Subscribes to ALERT messages from all 4 sensor agents
  2. Fuses multi-agent hazard evidence into a global threat score
  3. Manages global device state: IDLE ↔ ACTIVE_REASONING
  4. Publishes CLEAR messages when hazard conditions subside
  5. Maintains a global hazard timeline for situational awareness

This replaces / extends the existing ATROrchestrator with a
fully agentic multi-source fusion approach.

Global Hazard Score
-------------------
Each agent's ALERT contributes a weighted score:
  GasSensorAgent        → weight 0.35  (gas is most critical in mining)
  VibrationSensorAgent  → weight 0.30
  EnvSensorAgent        → weight 0.20
  UltrasonicSensorAgent → weight 0.15

Thresholds:
  score >= 0.30 → ACTIVE_REASONING (any single critical agent alert)
  score >= 0.60 → EMERGENCY (multi-agent corroboration)
  score <  0.10 for N consecutive ticks → IDLE (hazard cleared)
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional

from .agent_bus import AgentBus, AgentMessage, MessageType, Severity


# Source weights for global hazard score fusion
_SOURCE_WEIGHTS: Dict[str, float] = {
    "GasSensorAgent"       : 0.35,
    "VibrationSensorAgent" : 0.30,
    "EnvSensorAgent"       : 0.20,
    "UltrasonicSensorAgent": 0.15,
}

_SEVERITY_MULTIPLIERS: Dict[Severity, float] = {
    Severity.LOW     : 0.5,
    Severity.MEDIUM  : 0.75,
    Severity.HIGH    : 1.0,
    Severity.CRITICAL: 1.25,
}

_ACTIVE_THRESHOLD   = 0.30    # engage reasoning
_EMERGENCY_THRESHOLD= 0.60    # multi-agent emergency
_CLEAR_TICKS        = 5       # consecutive low-score ticks before → IDLE


class MineOrchestratorAgent:
    """
    Top-level multi-agent fusion and state management orchestrator.

    Parameters
    ----------
    bus     : AgentBus
    verbose : bool
    """

    def __init__(self, bus: AgentBus, verbose: bool = True):
        self.bus         = bus
        self.verbose     = verbose
        self.agent_name  = "MineOrchestratorAgent"

        # Device state
        self.device_state: str = "IDLE"   # "IDLE" | "ACTIVE_REASONING" | "EMERGENCY"

        # Live alert window: store latest alert per source
        self._active_alerts: Dict[str, AgentMessage] = {}

        # History for display / analysis
        self._alert_history : deque = deque(maxlen=500)
        self._score_history : deque = deque(maxlen=200)
        self._event_log     : List[Dict[str, Any]] = []

        # Consecutive ticks below clear threshold
        self._low_score_streak = 0

        # Subscribe to all sensor agent alerts
        for source in _SOURCE_WEIGHTS:
            self.bus.subscribe(source, self._on_sensor_alert)

        print(f"  [MineOrchestratorAgent] Initialized. Monitoring {len(_SOURCE_WEIGHTS)} sensor agents.")

    # -----------------------------------------------------------------------
    # Message Handling
    # -----------------------------------------------------------------------

    def _on_sensor_alert(self, msg: AgentMessage) -> None:
        """Receive an ALERT from a sensor agent and update global state."""
        if msg.msg_type == MessageType.ALERT:
            self._active_alerts[msg.source] = msg
            self._alert_history.append(msg)
            self._evaluate_global_state(msg.timestamp)

        elif msg.msg_type == MessageType.CLEAR:
            self._active_alerts.pop(msg.source, None)
            self._evaluate_global_state(msg.timestamp)

    # -----------------------------------------------------------------------
    # Global Hazard Fusion
    # -----------------------------------------------------------------------

    def compute_global_hazard_score(self) -> float:
        """
        Compute a fused global hazard score from all active agent alerts.
        Returns a float in [0, 1].
        """
        score = 0.0
        for source, weight in _SOURCE_WEIGHTS.items():
            alert = self._active_alerts.get(source)
            if alert is not None:
                sev_mult = _SEVERITY_MULTIPLIERS.get(alert.severity, 1.0)
                conf     = float(alert.payload.get("confidence", 0.5))
                score   += weight * sev_mult * conf
        return min(1.0, score)

    def _evaluate_global_state(self, timestamp: float) -> None:
        """Evaluate global hazard state and trigger transitions."""
        score = self.compute_global_hazard_score()
        self._score_history.append({"timestamp": timestamp, "score": score})

        if score < _ACTIVE_THRESHOLD:
            self._low_score_streak += 1
        else:
            self._low_score_streak = 0

        prev_state = self.device_state

        # ── State Transition Logic ─────────────────────────────────────
        if score >= _EMERGENCY_THRESHOLD:
            new_state = "EMERGENCY"
        elif score >= _ACTIVE_THRESHOLD:
            new_state = "ACTIVE_REASONING"
        elif self._low_score_streak >= _CLEAR_TICKS:
            new_state = "IDLE"
        else:
            new_state = self.device_state   # No change

        if new_state != prev_state:
            self._transition_state(prev_state, new_state, score, timestamp)

    def _transition_state(
        self,
        prev_state: str,
        new_state: str,
        score: float,
        timestamp: float,
    ) -> None:
        """Execute a state transition, logging and publishing."""
        self.device_state = new_state

        # Build transition message
        active_sources = list(self._active_alerts.keys())
        reasons = [a.reason for a in self._active_alerts.values()]

        event = {
            "timestamp"    : timestamp,
            "prev_state"   : prev_state,
            "new_state"    : new_state,
            "global_score" : round(score, 4),
            "active_sources": active_sources,
        }
        self._event_log.append(event)

        # ── Console output ─────────────────────────────────────────────
        border = "!" if new_state in ("EMERGENCY", "ACTIVE_REASONING") else "."
        print(f"\n{border * 70}")
        print(f"[MineOrchestrator] STATE: {prev_state} → {new_state}  (score={score:.3f})")
        if reasons:
            for r in reasons:
                print(f"  ↳ {r}")

        if new_state == "EMERGENCY":
            print("  ⚠ MULTI-AGENT EMERGENCY — Multiple sensor domains flagging hazards!")
        elif new_state == "ACTIVE_REASONING":
            print("  → Activating reasoning core. Loading SciSense projection layers...")
        elif new_state == "IDLE":
            print("  ✓ All hazards cleared. Returning to low-power monitoring mode.")
        print(f"{border * 70}\n")

        # Publish system-level message
        msg_type = MessageType.ALERT if new_state in ("EMERGENCY", "ACTIVE_REASONING") else MessageType.CLEAR
        sev      = Severity.CRITICAL if new_state == "EMERGENCY" else (
                   Severity.HIGH     if new_state == "ACTIVE_REASONING" else Severity.LOW)
        self.bus.publish(AgentMessage(
            source    = self.agent_name,
            msg_type  = msg_type,
            severity  = sev,
            payload   = {"global_score": score, "new_state": new_state, "active_sources": active_sources},
            reason    = f"State transition: {prev_state} → {new_state}",
            timestamp = timestamp,
        ))

    # -----------------------------------------------------------------------
    # Tick (called every streaming step, even without new alerts)
    # -----------------------------------------------------------------------

    def tick(self, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Called each cycle by the demo runner. Re-evaluates global score even
        if no new alerts arrived (alert expiry / score decay).

        Returns
        -------
        dict  Global state snapshot
        """
        if timestamp is None:
            timestamp = time.time()

        score = self.compute_global_hazard_score()

        # Natural alert expiry: remove alerts older than 30 seconds
        expired = [
            src for src, msg in self._active_alerts.items()
            if (timestamp - msg.timestamp) > 30.0
        ]
        for src in expired:
            del self._active_alerts[src]
            if self.verbose:
                print(f"  [MineOrchestrator] Alert from {src} expired (>30s old).")

        return {
            "timestamp"      : timestamp,
            "device_state"   : self.device_state,
            "global_score"   : score,
            "active_sources" : list(self._active_alerts.keys()),
            "alert_count"    : len(self._alert_history),
        }

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def status_report(self) -> str:
        score = self.compute_global_hazard_score()
        scores = [s["score"] for s in self._score_history]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        lines = [
            "╔══ MineOrchestratorAgent Status Report ══╗",
            f"  Device state      : {self.device_state}",
            f"  Global hazard score: {score:.4f}",
            f"  Avg score (history): {avg_score:.4f}",
            f"  Active alert sources: {list(self._active_alerts.keys())}",
            f"  Total alerts seen  : {len(self._alert_history)}",
            f"  State transitions  : {len(self._event_log)}",
        ]
        if self._event_log:
            last = self._event_log[-1]
            lines.append(f"  Last transition    : {last['prev_state']} → {last['new_state']} (score={last['global_score']})")
        return "\n".join(lines)

    def get_event_log(self) -> List[Dict[str, Any]]:
        """Return the full state transition event log."""
        return list(self._event_log)
