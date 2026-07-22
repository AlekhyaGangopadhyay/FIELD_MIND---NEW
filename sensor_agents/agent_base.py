"""
agent_base.py — Abstract Base Class for all FIELD-MIND Sensor Agents
=====================================================================
Every sensor agent inherits from SensorAgentBase, which provides:

  1. Observe  — receives raw sensor data dict each tick
  2. Reason   — runs domain ML inference + confidence scoring
  3. Act      — emits AgentMessage via AgentBus
  4. Learn    — collects experience, refits model when buffer is full

Experience Replay Buffer
------------------------
Instead of partial_fit() (which doesn't work for RF/GBM), we use
an experience replay buffer:

  - On each tick, the raw feature vector + a derived label are stored.
  - When the buffer reaches REPLAY_BUFFER_SIZE samples, the agent calls
    _refit_model() which trains a fresh copy of the model on the
    accumulated data and replaces the live model.
  - Buffer is then cleared and the cycle starts again.

Ground-Truth Labels for Learning
---------------------------------
Labels come from the ORIGINAL dataset preloaded at agent init time.
The agent uses a dataset_sampler that iterates through the original
CSV, providing (features, label) pairs as replay seeds, supplemented
by any confirmed anomaly signals broadcast on the AgentBus.
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .agent_bus import AgentBus, AgentMessage, MessageType, Severity


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPLAY_BUFFER_SIZE   = 200   # Refit when this many new samples are collected
MEMORY_WINDOW        = 30    # Short-term rolling window (ticks) for trend analysis
MIN_REFIT_SAMPLES    = 50    # Do not refit unless at least this many samples


# ---------------------------------------------------------------------------
# Abstract Base Agent
# ---------------------------------------------------------------------------

class SensorAgentBase(ABC):
    """
    Abstract base for all FIELD-MIND sensor AI agents.

    Subclasses must implement:
      - perceive(raw_data)   -> dict   Extract & normalise features
      - infer(features)      -> dict   Run domain ML models
      - compute_confidence(inference_result) -> float
      - derive_label(raw_data, inference_result) -> Optional[int]
        Returns 1 (hazard/anomaly), 0 (normal), or None (uncertain)
      - _build_fresh_model() -> sklearn estimator
        Returns a NEW untrained model instance for refitting
      - _refit_model(X, y)   Fits the fresh model on buffer data

    Parameters
    ----------
    agent_name       : str        Human-readable name for logging/bus
    bus              : AgentBus   Shared event bus instance
    primary_model    : object     Pre-trained sklearn model (from joblib)
    dataset_path     : str        Path to original CSV dataset for replay seeding
    replay_buffer_size : int      How many samples before auto-refit
    memory_window    : int        Rolling window size for trend memory
    verbose          : bool       Print per-tick reasoning trace
    """

    def __init__(
        self,
        agent_name     : str,
        bus            : AgentBus,
        primary_model  : Any,
        dataset_path   : Optional[str]   = None,
        replay_buffer_size: int          = REPLAY_BUFFER_SIZE,
        memory_window  : int             = MEMORY_WINDOW,
        verbose        : bool            = True,
    ):
        self.agent_name   = agent_name
        self.bus          = bus
        self.model        = primary_model          # live model (may be updated by refit)
        self.dataset_path = dataset_path
        self.replay_buffer_size = replay_buffer_size
        self.memory_window = memory_window
        self.verbose      = verbose

        # --- Short-term memory (rolling window of inference outputs) ---
        self._memory: deque = deque(maxlen=memory_window)

        # --- Experience replay buffer: list of (feature_vector, label) ---
        self._replay_X: List[np.ndarray] = []
        self._replay_y: List[int]        = []

        # --- Dataset replay sampler (iterator over original CSV rows) ---
        self._dataset_iter    = None
        self._dataset_columns : List[str] = []
        if dataset_path:
            self._init_dataset_sampler(dataset_path)

        # --- Metrics tracking ---
        self._metrics: Dict[str, Any] = {
            "tick_count"       : 0,
            "alert_count"      : 0,
            "refit_count"      : 0,
            "confidence_history": deque(maxlen=200),
            "accuracy_history" : deque(maxlen=50),   # accuracy after each refit
            "last_refit_tick"  : None,
        }

        # --- Consecutive hazard counter for alert threshold ---
        self._consecutive_hazards = 0
        self._alert_threshold     = 2   # emit ALERT only after N consecutive hazards

        # Register to hear CLEAR messages from orchestrator (to reset hazard counter)
        self.bus.subscribe("MineOrchestratorAgent", self._on_orchestrator_message)

    # -----------------------------------------------------------------------
    # Abstract Interface — must be implemented by each sensor agent
    # -----------------------------------------------------------------------

    @abstractmethod
    def perceive(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalise a feature dict from raw sensor input."""
        ...

    @abstractmethod
    def infer(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Run all domain ML models and return a combined result dict."""
        ...

    @abstractmethod
    def compute_confidence(self, inference_result: Dict[str, Any]) -> float:
        """Return a float in [0, 1] representing hazard confidence."""
        ...

    @abstractmethod
    def derive_label(
        self,
        raw_data: Dict[str, Any],
        inference_result: Dict[str, Any],
    ) -> Optional[int]:
        """
        Derive the ground-truth label for this observation.
        Returns: 1 = hazard/anomaly, 0 = normal, None = unknown/skip
        """
        ...

    @abstractmethod
    def _build_fresh_model(self):
        """Return a NEW untrained sklearn estimator of the same type as self.model."""
        ...

    @abstractmethod
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert the features dict to a 1-D numpy array for replay buffer."""
        ...

    # -----------------------------------------------------------------------
    # Core Agent Loop
    # -----------------------------------------------------------------------

    def step(self, raw_data: Dict[str, Any], timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute ONE full agent cycle: Observe → Reason → Act → Learn.

        Parameters
        ----------
        raw_data  : dict   Raw sensor readings for this tick
        timestamp : float  Unix timestamp (defaults to now)

        Returns
        -------
        dict  Complete result: features, inference, confidence, action, metrics
        """
        if timestamp is None:
            timestamp = time.time()

        self._metrics["tick_count"] += 1
        tick = self._metrics["tick_count"]

        # ── 1. OBSERVE ────────────────────────────────────────────────────
        features = self.perceive(raw_data)

        # ── 2. REASON ────────────────────────────────────────────────────
        inference   = self.infer(features)
        confidence  = self.compute_confidence(inference)
        self._metrics["confidence_history"].append(confidence)

        # Push to short-term memory
        memory_entry = {
            "tick"       : tick,
            "timestamp"  : timestamp,
            "confidence" : confidence,
            **inference
        }
        self._memory.append(memory_entry)

        # ── 3. ACT ──────────────────────────────────────────────────────
        action = self._decide_action(inference, confidence, timestamp)

        # ── 4. LEARN ─────────────────────────────────────────────────────
        label = self.derive_label(raw_data, inference)
        if label is not None:
            feat_vec = self._features_to_vector(features)
            self._add_to_replay(feat_vec, label)

        # ── Return combined result ────────────────────────────────────────
        result = {
            "agent"      : self.agent_name,
            "tick"       : tick,
            "timestamp"  : timestamp,
            "features"   : features,
            "inference"  : inference,
            "confidence" : confidence,
            "action"     : action,
            "replay_buffer_size": len(self._replay_X),
            "metrics"    : self._get_metrics_snapshot(),
        }

        if self.verbose:
            self._print_tick(result)

        return result

    # -----------------------------------------------------------------------
    # Action Decision
    # -----------------------------------------------------------------------

    def _decide_action(
        self,
        inference  : Dict[str, Any],
        confidence : float,
        timestamp  : float,
    ) -> str:
        """
        Decide what action to take based on inference + rolling memory trend.
        Emits an AgentMessage to the bus and returns the action string.
        """
        is_hazard = confidence >= 0.5

        if is_hazard:
            self._consecutive_hazards += 1
        else:
            self._consecutive_hazards = 0

        # Only alert after N consecutive hazard readings (reduces false alarms)
        if self._consecutive_hazards >= self._alert_threshold:
            severity = self._map_severity(confidence)
            reason   = self._build_reason(inference, confidence)
            msg = AgentMessage(
                source   = self.agent_name,
                msg_type = MessageType.ALERT,
                severity = severity,
                payload  = {**inference, "confidence": confidence},
                reason   = reason,
                timestamp= timestamp,
            )
            self.bus.publish(msg)
            self._metrics["alert_count"] += 1
            return f"ALERT:{severity.name}"

        elif self._metrics["tick_count"] % 10 == 0:
            # Publish a periodic INFO heartbeat every 10 ticks
            msg = AgentMessage(
                source   = self.agent_name,
                msg_type = MessageType.INFO,
                severity = Severity.LOW,
                payload  = {**inference, "confidence": confidence},
                reason   = f"Tick {self._metrics['tick_count']} — nominal",
                timestamp= timestamp,
            )
            self.bus.publish(msg)
            return "INFO:NOMINAL"

        return "IDLE"

    # -----------------------------------------------------------------------
    # Experience Replay & Learning
    # -----------------------------------------------------------------------

    def _add_to_replay(self, feature_vector: np.ndarray, label: int) -> None:
        """Add a labeled observation to the replay buffer. Trigger refit if full."""
        self._replay_X.append(feature_vector)
        self._replay_y.append(label)

        if len(self._replay_X) >= self.replay_buffer_size:
            self._trigger_refit()

    def _trigger_refit(self) -> None:
        """
        Refit the agent's primary model on all accumulated replay data.
        After refit, clear the buffer and publish a LEARNING_UPDATE message.
        """
        if len(self._replay_X) < MIN_REFIT_SAMPLES:
            return

        X = np.array(self._replay_X)
        y = np.array(self._replay_y)

        # Validate label variety (no refit on single-class data)
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            if self.verbose:
                print(f"  [{self.agent_name}] Refit skipped — only class {unique_labels} in buffer.")
            # Keep accumulating; don't clear
            return

        try:
            new_model = self._build_fresh_model()
            new_model.fit(X, y)

            # Evaluate on the buffer itself (in-sample; real accuracy tracked externally)
            in_sample_acc = np.mean(new_model.predict(X) == y)
            self._metrics["accuracy_history"].append(in_sample_acc)
            self._metrics["refit_count"] += 1
            self._metrics["last_refit_tick"] = self._metrics["tick_count"]

            # Atomically swap model
            self.model = new_model

            print(
                f"\n  [*] [{self.agent_name}] LEARNING UPDATE -- "
                f"Refit #{self._metrics['refit_count']} on {len(X)} samples | "
                f"In-sample accuracy: {in_sample_acc:.3f}"
            )

            # Clear buffer
            self._replay_X.clear()
            self._replay_y.clear()

            # Broadcast learning update
            self.bus.publish(AgentMessage(
                source   = self.agent_name,
                msg_type = MessageType.LEARNING_UPDATE,
                severity = Severity.LOW,
                payload  = {
                    "refit_count"   : self._metrics["refit_count"],
                    "samples_used"  : len(X),
                    "in_sample_acc" : in_sample_acc,
                },
                reason   = f"Model refitted on {len(X)} replay samples.",
            ))

        except Exception as e:
            print(f"  [{self.agent_name}] Refit failed: {e}")

    # -----------------------------------------------------------------------
    # Dataset Replay Sampler
    # -----------------------------------------------------------------------

    def _init_dataset_sampler(self, path: str) -> None:
        """Load the dataset into memory and create an infinite cyclic iterator."""
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            self._dataset_df      = df
            self._dataset_index   = 0
            self._dataset_columns = df.columns.tolist()
            if self.verbose:
                print(f"  [{self.agent_name}] Dataset loaded: {path} ({len(df)} rows)")
        except Exception as e:
            print(f"  [{self.agent_name}] Dataset load failed: {e}")
            self._dataset_df = None

    def get_dataset_row(self) -> Optional[Dict[str, Any]]:
        """
        Return the next row from the original dataset as a dict.
        Cycles back to the beginning when exhausted.
        """
        if self._dataset_df is None or len(self._dataset_df) == 0:
            return None
        row = self._dataset_df.iloc[self._dataset_index].to_dict()
        self._dataset_index = (self._dataset_index + 1) % len(self._dataset_df)
        return row

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _map_severity(self, confidence: float) -> Severity:
        if confidence >= 0.9:
            return Severity.CRITICAL
        elif confidence >= 0.75:
            return Severity.HIGH
        elif confidence >= 0.6:
            return Severity.MEDIUM
        return Severity.LOW

    def _build_reason(self, inference: Dict[str, Any], confidence: float) -> str:
        """Build a human-readable reason string from inference results."""
        parts = [f"conf={confidence:.2f}"]
        for k, v in inference.items():
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v}")
        return " | ".join(parts)

    def _on_orchestrator_message(self, msg: AgentMessage) -> None:
        """Handle incoming messages from the MineOrchestratorAgent."""
        if msg.msg_type == MessageType.CLEAR:
            self._consecutive_hazards = 0

    def _get_metrics_snapshot(self) -> Dict[str, Any]:
        """Return a lightweight snapshot of current metrics."""
        conf_hist = list(self._metrics["confidence_history"])
        acc_hist  = list(self._metrics["accuracy_history"])
        return {
            "tick_count"       : self._metrics["tick_count"],
            "alert_count"      : self._metrics["alert_count"],
            "refit_count"      : self._metrics["refit_count"],
            "avg_confidence"   : float(np.mean(conf_hist)) if conf_hist else 0.0,
            "last_accuracy"    : float(acc_hist[-1])        if acc_hist  else None,
            "replay_buffer_len": len(self._replay_X),
        }

    def _print_tick(self, result: Dict[str, Any]) -> None:
        """Print a concise per-tick trace line."""
        tick = result["tick"]
        conf = result["confidence"]
        act  = result["action"]
        buf  = result["replay_buffer_size"]
        print(
            f"  [{self.agent_name}] tick={tick:>5} | conf={conf:.3f} | "
            f"action={act:<20} | replay_buf={buf}/{self.replay_buffer_size}"
        )

    def status_report(self) -> str:
        """Return a formatted multi-line status string for this agent."""
        snap = self._get_metrics_snapshot()
        lines = [
            f"╔══ {self.agent_name} Status Report ══╗",
            f"  Ticks processed   : {snap['tick_count']}",
            f"  Alerts emitted    : {snap['alert_count']}",
            f"  Model refits      : {snap['refit_count']}",
            f"  Avg confidence    : {snap['avg_confidence']:.4f}",
            f"  Last model accuracy: {snap['last_accuracy']}",
            f"  Replay buffer     : {snap['replay_buffer_len']}/{self.replay_buffer_size}",
        ]
        return "\n".join(lines)
