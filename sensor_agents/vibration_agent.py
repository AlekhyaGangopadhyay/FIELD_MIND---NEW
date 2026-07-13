"""
vibration_agent.py — Vibration Sensor AI Agent
===============================================
Wraps the PPV classifier (RandomForest) and PPV regressor (GradientBoosting)
from vibration/models/ into an autonomous agent.

Ground-Truth Labels:
  Derived from vibration/data/vibration_features.csv using the same rule
  used in training: vibration_hazard = 1 if PPV > 1.0 mm/s

Self-Learning:
  Primary learnable model = RandomForestClassifier on 14 vibration features.
  When replay buffer (size=200) fills, a fresh RF is trained on accumulated
  (feature, hazard_label) pairs from the original dataset.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from .agent_base import SensorAgentBase
from .agent_bus import AgentBus


# 14-feature set used by the classifier model
_CLASSIFIER_FEATURES = [
    "offset", "max_charge", "total_charge", "num_holes", "detonator_code",
    "trid_12", "trid_13", "trid_14",
    "gx", "gy", "gelev", "sx", "sy", "selev",
]

# 17-feature set used by the regressor
_REGRESSOR_FEATURES = _CLASSIFIER_FEATURES + [
    "scaled_distance_usbm", "scaled_distance_langefors", "elevation_diff"
]

_PPV_HAZARD_THRESHOLD = 1.0   # mm/s (same as training script)


class VibrationSensorAgent(SensorAgentBase):
    """
    Autonomous Vibration Sensor AI Agent (Blast PPV).

    Parameters
    ----------
    workspace_root : str
    bus            : AgentBus
    verbose        : bool
    """

    def __init__(self, workspace_root: str, bus: AgentBus, verbose: bool = True):
        self.workspace_root = workspace_root

        vib_model_dir = os.path.join(workspace_root, "vibration", "models")
        self._clf_model = self._load_model(
            vib_model_dir, "best_random_forest_classifier.joblib", "RF-Classifier"
        )
        self._reg_model = self._load_model(
            vib_model_dir, "best_gradient_boosting_regressor.joblib", "GB-Regressor"
        )

        primary = self._clf_model if self._clf_model is not None else RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        )

        dataset_path = os.path.join(
            workspace_root, "vibration", "data", "vibration_features.csv"
        )

        super().__init__(
            agent_name   = "VibrationSensorAgent",
            bus          = bus,
            primary_model= primary,
            dataset_path = dataset_path if os.path.exists(dataset_path) else None,
            replay_buffer_size = 200,
            memory_window      = 30,
            verbose            = verbose,
        )

        # Preprocess dataset (one-hot encode trid column) after loading
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
                print(f"  [VibrationSensorAgent] ✓ Loaded model: {name}")
                return m
            except Exception as e:
                print(f"  [VibrationSensorAgent] ✗ Failed to load {name}: {e}")
        else:
            print(f"  [VibrationSensorAgent] ⚠ Model not found: {fname}")
        return None

    def _preprocess_dataset(self) -> None:
        """One-hot encode trid column in loaded dataset (matching training script)."""
        if self._dataset_df is None:
            return
        try:
            df = self._dataset_df
            if "trid" in df.columns:
                df = pd.get_dummies(df, columns=["trid"], prefix="trid")
                for col in ["trid_12", "trid_13", "trid_14"]:
                    if col not in df.columns:
                        df[col] = 0
                    df[col] = df[col].astype(int)
                self._dataset_df = df
        except Exception as e:
            print(f"  [VibrationSensorAgent] Dataset preprocessing error: {e}")

    def _seed_replay_from_dataset(self, n_seed: int = 100) -> None:
        if self._dataset_df is None:
            return
        seeded = 0
        for _ in range(min(n_seed, len(self._dataset_df))):
            row = self.get_dataset_row()
            if row is None:
                break
            feat_vec = self._row_to_clf_vector(row)
            ppv   = float(row.get("ppv", 0.0))
            label = int(ppv > _PPV_HAZARD_THRESHOLD)
            self._replay_X.append(feat_vec)
            self._replay_y.append(label)
            seeded += 1
        print(f"  [VibrationSensorAgent] Replay buffer seeded with {seeded} dataset rows.")

    def _row_to_clf_vector(self, row: Dict[str, Any]) -> np.ndarray:
        return np.array([float(row.get(f, 0.0)) for f in _CLASSIFIER_FEATURES])

    # -----------------------------------------------------------------------
    # Abstract Interface Implementation
    # -----------------------------------------------------------------------

    def perceive(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        features = {f: float(raw_data.get(f, 0.0)) for f in _REGRESSOR_FEATURES}
        return features

    def infer(self, features: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        clf_vec = np.array([features[f] for f in _CLASSIFIER_FEATURES]).reshape(1, -1)
        reg_vec = np.array([features[f] for f in _REGRESSOR_FEATURES]).reshape(1, -1)

        # 1. Hazard classification (primary learnable model)
        try:
            result["vibration_hazard"] = int(self.model.predict(clf_vec)[0])
        except Exception:
            result["vibration_hazard"] = 0

        # 2. PPV regression (frozen GB model)
        if self._reg_model is not None:
            try:
                n_feat = self._reg_model.n_features_in_ if hasattr(self._reg_model, "n_features_in_") else 17
                if reg_vec.shape[1] != n_feat:
                    pad = np.zeros((1, n_feat))
                    pad[0, :reg_vec.shape[1]] = reg_vec[0, :n_feat]
                    reg_vec = pad
                log_ppv = float(self._reg_model.predict(reg_vec)[0])
                result["predicted_ppv"] = float(np.exp(log_ppv))
            except Exception:
                result["predicted_ppv"] = 0.0
        else:
            result["predicted_ppv"] = 0.0

        # 3. Rule-based check
        ppv = result["predicted_ppv"]
        result["ppv_exceeds_threshold"] = int(ppv > _PPV_HAZARD_THRESHOLD)

        return result

    def compute_confidence(self, inference_result: Dict[str, Any]) -> float:
        conf = 0.0

        if inference_result.get("vibration_hazard") == 1:
            conf += 0.55
        if inference_result.get("ppv_exceeds_threshold") == 1:
            conf += 0.30

        # Severity scaling by PPV magnitude
        ppv = inference_result.get("predicted_ppv", 0.0)
        if ppv > 10.0:
            conf = min(1.0, conf + 0.20)
        elif ppv > 5.0:
            conf = min(1.0, conf + 0.10)

        # Trend check
        recent = list(self._memory)[-3:]
        if len(recent) == 3 and all(e.get("vibration_hazard", 0) == 1 for e in recent):
            conf = min(1.0, conf + 0.15)

        return min(1.0, conf)

    def derive_label(
        self,
        raw_data: Dict[str, Any],
        inference_result: Dict[str, Any],
    ) -> Optional[int]:
        row = self.get_dataset_row()
        if row is not None and "ppv" in row:
            return int(float(row["ppv"]) > _PPV_HAZARD_THRESHOLD)
        if row is not None and "vibration_hazard" in row:
            return int(row["vibration_hazard"])
        return inference_result.get("vibration_hazard", 0)

    def _build_fresh_model(self):
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )

    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        return np.array([features[f] for f in _CLASSIFIER_FEATURES])

    def _build_reason(self, inference: Dict[str, Any], confidence: float) -> str:
        ppv = inference.get("predicted_ppv", 0.0)
        parts = []
        if inference.get("vibration_hazard") == 1:
            parts.append(f"Blast vibration hazard detected (PPV={ppv:.2f} mm/s)")
        if ppv > _PPV_HAZARD_THRESHOLD:
            parts.append(f"PPV exceeds threshold ({_PPV_HAZARD_THRESHOLD} mm/s)")
        parts.append(f"confidence={confidence:.2f}")
        return " | ".join(parts)
