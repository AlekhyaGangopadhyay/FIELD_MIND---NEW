"""
ultrasonic_agent.py — Ultrasonic Sensor AI Agent
=================================================
Wraps the 24-sensor robot navigation classifier from
ultrasonic_sensors/models/ into an autonomous agent that observes
proximity sensor readings, classifies navigation decisions, detects
collision risk, and learns from sensor_readings_24.csv.

Ground-Truth Labels:
  Derived from sensor_readings_24.csv ('Class' column).
  Label mapping: 'Sharp-Right-Turn' → collision_risk=1, others → 0.
  Agent learns which combinations of sensor readings predict
  sharp/unsafe turns requiring evasive action.

Self-Learning:
  Primary learnable model = RandomForestClassifier on 24 US features.
  When replay buffer (size=200) fills, a fresh RF is fitted on accumulated
  (feature_vector, collision_label) pairs from the original dataset.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestClassifier

from .agent_base import SensorAgentBase
from .agent_bus import AgentBus


_COLLISION_CLASS = "Sharp-Right-Turn"
_N_SENSORS       = 24


class UltrasonicSensorAgent(SensorAgentBase):
    """
    Autonomous Ultrasonic Sensor AI Agent (Robot Navigation).

    Parameters
    ----------
    workspace_root : str
    bus            : AgentBus
    verbose        : bool
    """

    def __init__(self, workspace_root: str, bus: AgentBus, verbose: bool = True):
        self.workspace_root = workspace_root

        ultra_model_dir = os.path.join(workspace_root, "ultrasonic_sensors", "models")

        # Load the 24-sensor model (best model dict)
        self._nav_model_data = self._load_model(
            ultra_model_dir, "best_ultrasonic_24.joblib", "Ultrasonic-24"
        )
        self._classes_map: Dict[int, str] = {}
        nav_model = self._extract_model_and_classes()

        dataset_path = os.path.join(
            workspace_root, "ultrasonic_sensors", "data", "sensor_readings_24.csv"
        )

        super().__init__(
            agent_name   = "UltrasonicSensorAgent",
            bus          = bus,
            primary_model= nav_model,
            dataset_path = dataset_path if os.path.exists(dataset_path) else None,
            replay_buffer_size = 200,
            memory_window      = 30,
            verbose            = verbose,
        )

        # Build label encoder from dataset
        self._label_to_int: Dict[str, int] = {}
        self._int_to_label: Dict[int, str] = {}
        self._preprocess_dataset()
        self._seed_replay_from_dataset(n_seed=100)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _load_model(self, model_dir: str, fname: str, name: str):
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            try:
                m = joblib.load(path)
                print(f"  [UltrasonicSensorAgent] ✓ Loaded model: {name}")
                return m
            except Exception as e:
                print(f"  [UltrasonicSensorAgent] ✗ Failed to load {name}: {e}")
        else:
            print(f"  [UltrasonicSensorAgent] ⚠ Model not found: {fname}")
        return None

    def _extract_model_and_classes(self):
        """Unpack the model dict (model + classes_map) saved during training."""
        data = self._nav_model_data
        if data is None:
            return RandomForestClassifier(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
            )
        if isinstance(data, dict) and "model" in data:
            self._classes_map = data.get("classes", {})
            return data["model"]
        self._classes_map = {}
        return data

    def _preprocess_dataset(self) -> None:
        """Strip column names and build label encoder from dataset."""
        if self._dataset_df is None:
            return
        try:
            df = self._dataset_df
            df.columns = df.columns.str.strip()
            if "Class" in df.columns:
                df["Class"] = df["Class"].str.strip()
                classes = sorted(df["Class"].unique())
                self._label_to_int = {c: i for i, c in enumerate(classes)}
                self._int_to_label = {i: c for c, i in self._label_to_int.items()}
                self._dataset_df = df
        except Exception as e:
            print(f"  [UltrasonicSensorAgent] Dataset preprocessing error: {e}")

    def _seed_replay_from_dataset(self, n_seed: int = 100) -> None:
        if self._dataset_df is None:
            return
        seeded = 0
        for _ in range(min(n_seed, len(self._dataset_df))):
            row = self.get_dataset_row()
            if row is None:
                break
            feat_vec = self._row_to_sensor_vector(row)
            raw_class = str(row.get("Class", "")).strip()
            label = int(raw_class == _COLLISION_CLASS)
            self._replay_X.append(feat_vec)
            self._replay_y.append(label)
            seeded += 1
        print(f"  [UltrasonicSensorAgent] Replay buffer seeded with {seeded} dataset rows.")

    def _row_to_sensor_vector(self, row: Dict[str, Any]) -> np.ndarray:
        """Extract 24 US sensor readings from a dataset row dict."""
        df = self._dataset_df
        if df is not None:
            sensor_cols = [c for c in df.columns if c != "Class"]
        else:
            sensor_cols = [f"US{i}" for i in range(1, 25)]
        return np.array([float(row.get(c, 0.0)) for c in sensor_cols[:_N_SENSORS]])

    # -----------------------------------------------------------------------
    # Abstract Interface Implementation
    # -----------------------------------------------------------------------

    def perceive(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        for i in range(1, _N_SENSORS + 1):
            key = f"US{i}"
            features[key] = float(raw_data.get(key, 0.0))

        # Derived safety features
        all_readings = [features[f"US{i}"] for i in range(1, _N_SENSORS + 1)]
        features["min_distance"]  = float(min(all_readings))
        features["mean_distance"] = float(np.mean(all_readings))
        return features

    def infer(self, features: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        sensor_vec = np.array(
            [features[f"US{i}"] for i in range(1, _N_SENSORS + 1)]
        ).reshape(1, -1)

        # Navigation classification
        try:
            pred_encoded = int(self.model.predict(sensor_vec)[0])
            pred_label   = self._classes_map.get(pred_encoded, str(pred_encoded))
            result["steering_decision"]  = pred_label
            result["sharp_turn_required"]= int(pred_label == _COLLISION_CLASS)
        except Exception:
            result["steering_decision"]   = "unknown"
            result["sharp_turn_required"] = 0

        # Rule-based proximity check
        min_dist = features.get("min_distance", float("inf"))
        result["proximity_alert"] = int(min_dist < 0.3)   # < 30 cm
        result["min_distance"]    = float(min_dist)

        return result

    def compute_confidence(self, inference_result: Dict[str, Any]) -> float:
        conf = 0.0

        if inference_result.get("sharp_turn_required") == 1:
            conf += 0.60
        if inference_result.get("proximity_alert") == 1:
            conf += 0.30

        # Trend check: consecutive collision alerts
        recent = list(self._memory)[-3:]
        if len(recent) == 3 and all(e.get("sharp_turn_required", 0) == 1 for e in recent):
            conf = min(1.0, conf + 0.15)

        return min(1.0, conf)

    def derive_label(
        self,
        raw_data: Dict[str, Any],
        inference_result: Dict[str, Any],
    ) -> Optional[int]:
        row = self.get_dataset_row()
        if row is not None:
            raw_class = str(row.get("Class", "")).strip()
            return int(raw_class == _COLLISION_CLASS)
        return inference_result.get("sharp_turn_required", 0)

    def _build_fresh_model(self):
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )

    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        return np.array([features[f"US{i}"] for i in range(1, _N_SENSORS + 1)])

    def _build_reason(self, inference: Dict[str, Any], confidence: float) -> str:
        parts = []
        decision = inference.get("steering_decision", "unknown")
        if inference.get("sharp_turn_required") == 1:
            parts.append(f"Collision avoidance required: {decision}")
        if inference.get("proximity_alert") == 1:
            parts.append(f"Proximity alert — min distance={inference.get('min_distance', 0):.2f}m")
        parts.append(f"confidence={confidence:.2f}")
        return " | ".join(parts)
