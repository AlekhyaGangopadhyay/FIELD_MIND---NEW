"""
env_agent.py — Environmental Sensor AI Agent
============================================
Wraps the Isolation Forest (anomaly detection) and Random Forest
(occupancy classification) from temperature_humidity/models/ into an
autonomous agent that observes temp/humidity readings, reasons about
anomalies, and learns from the original IoT Telemetry dataset.

Ground-Truth Labels:
  Derived from iot_telemetry_clean.csv using the same rule applied
  during training:
    anomaly = 1 if (temp > 28.0) or (temp < 5.0)
                 or (humidity > 85.0) or (humidity < 15.0)

Self-Learning:
  Primary learnable model = IsolationForest on 9 env features.
  When replay buffer (size=200) fills, a fresh IsolationForest is
  fitted on the accumulated (unlabelled) data + contamination=0.05.
  The occupancy RF remains frozen (it uses a supervised label not
  present in real-time streaming).
"""

import os
import joblib
import numpy as np
from typing import Any, Dict, Optional

from sklearn.ensemble import IsolationForest, RandomForestClassifier

from .agent_base import SensorAgentBase
from .agent_bus import AgentBus


_ENV_FEATURES = [
    "temp", "humidity", "temp_hum_product", "temp_hum_ratio", "humidex",
    "temp_roll_mean_5", "temp_roll_std_5", "humidity_roll_mean_5", "humidity_roll_std_5",
]

_ANOMALY_TEMP_HIGH    = 28.0
_ANOMALY_TEMP_LOW     = 5.0
_ANOMALY_HUM_HIGH     = 85.0
_ANOMALY_HUM_LOW      = 15.0


class EnvSensorAgent(SensorAgentBase):
    """
    Autonomous Environmental Sensor AI Agent (Temperature & Humidity).

    Parameters
    ----------
    workspace_root : str
    bus            : AgentBus
    verbose        : bool
    """

    def __init__(self, workspace_root: str, bus: AgentBus, verbose: bool = True):
        self.workspace_root = workspace_root

        env_model_dir = os.path.join(workspace_root, "temperature_humidity", "models")
        self._iforest   = self._load_model(env_model_dir, "isolation_forest_iot.joblib", "IsolationForest")
        self._rf_occ    = self._load_model(env_model_dir, "random_forest.joblib",       "RF-Occupancy")

        # Primary learnable model = IsolationForest
        primary = self._iforest if self._iforest is not None else IsolationForest(
            n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1
        )

        dataset_path = os.path.join(
            workspace_root, "temperature_humidity", "data", "data_clean", "iot_telemetry_clean.csv"
        )

        super().__init__(
            agent_name   = "EnvSensorAgent",
            bus          = bus,
            primary_model= primary,
            dataset_path = dataset_path if os.path.exists(dataset_path) else None,
            replay_buffer_size = 200,
            memory_window      = 30,
            verbose            = verbose,
        )

        # Rolling history for rolling statistics calculation
        self._temp_history: list     = []
        self._humidity_history: list = []

        self._seed_replay_from_dataset(n_seed=100)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _load_model(self, model_dir: str, fname: str, name: str):
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            try:
                m = joblib.load(path)
                print(f"  [EnvSensorAgent] ✓ Loaded model: {name}")
                return m
            except Exception as e:
                print(f"  [EnvSensorAgent] ✗ Failed to load {name}: {e}")
        else:
            print(f"  [EnvSensorAgent] ⚠ Model not found: {fname}")
        return None

    def _seed_replay_from_dataset(self, n_seed: int = 100) -> None:
        if self._dataset_df is None:
            return
        seeded = 0
        for _ in range(min(n_seed, len(self._dataset_df))):
            row = self.get_dataset_row()
            if row is None:
                break
            feat_vec = self._row_to_env_vector(row)
            # Label: 1 = anomaly (Isolation Forest uses -1/1, we convert: -1 → 1, 1 → 0)
            temp = float(row.get("temp", 20.0))
            hum  = float(row.get("humidity", 50.0))
            label = int(
                (temp > _ANOMALY_TEMP_HIGH) or (temp < _ANOMALY_TEMP_LOW)
                or (hum > _ANOMALY_HUM_HIGH) or (hum < _ANOMALY_HUM_LOW)
            )
            self._replay_X.append(feat_vec)
            self._replay_y.append(label)
            seeded += 1
        print(f"  [EnvSensorAgent] Replay buffer seeded with {seeded} dataset rows.")

    def _row_to_env_vector(self, row: Dict[str, Any]) -> np.ndarray:
        temp = float(row.get("temp", 20.0))
        hum  = float(row.get("humidity", 50.0))
        return np.array([
            temp, hum,
            temp * hum,
            temp / (hum + 1e-6),
            temp + 0.33 * (6.105 * np.exp(17.27 * hum / (237.7 + hum))) - 4.0,
            float(row.get("temp_roll_mean_5", temp)),
            float(row.get("temp_roll_std_5",  0.0)),
            float(row.get("humidity_roll_mean_5", hum)),
            float(row.get("humidity_roll_std_5",  0.0)),
        ])

    # -----------------------------------------------------------------------
    # Abstract Interface Implementation
    # -----------------------------------------------------------------------

    def perceive(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        temp = float(raw_data.get("temp", raw_data.get("temperature", 20.0)))
        hum  = float(raw_data.get("humidity", 50.0))

        self._temp_history.append(temp)
        self._humidity_history.append(hum)

        # Keep rolling windows
        win = 5
        t_win = self._temp_history[-win:]
        h_win = self._humidity_history[-win:]

        features = {
            "temp"                 : temp,
            "humidity"             : hum,
            "temp_hum_product"     : temp * hum,
            "temp_hum_ratio"       : temp / (hum + 1e-6),
            "humidex"              : temp + 0.33 * (6.105 * np.exp(17.27 * hum / (237.7 + hum))) - 4.0,
            "temp_roll_mean_5"     : float(np.mean(t_win)),
            "temp_roll_std_5"      : float(np.std(t_win)),
            "humidity_roll_mean_5" : float(np.mean(h_win)),
            "humidity_roll_std_5"  : float(np.std(h_win)),
        }
        return features

    def infer(self, features: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        # 1. Isolation Forest anomaly detection
        f_vec = np.array([features[k] for k in _ENV_FEATURES]).reshape(1, -1)
        try:
            pred = self.model.predict(f_vec)[0]
            result["anomaly_detected"] = int(pred == -1)
            # Decision score: more negative = more anomalous
            score = float(self.model.decision_function(f_vec)[0])
            result["anomaly_score"] = score
        except Exception:
            result["anomaly_detected"] = 0
            result["anomaly_score"]    = 0.0

        # 2. Rule-based supplemental checks (always available)
        temp = features["temp"]
        hum  = features["humidity"]
        result["temp_out_of_range"] = int(temp > _ANOMALY_TEMP_HIGH or temp < _ANOMALY_TEMP_LOW)
        result["hum_out_of_range"]  = int(hum  > _ANOMALY_HUM_HIGH  or hum  < _ANOMALY_HUM_LOW)

        # 3. Occupancy classification (frozen RF)
        if self._rf_occ is not None:
            try:
                # RF occupancy was trained on 23 features; use only temp/hum features
                # available and pad with zeros for compatibility
                n_feat = self._rf_occ.n_features_in_ if hasattr(self._rf_occ, "n_features_in_") else 23
                occ_vec = np.zeros((1, n_feat))
                occ_vec[0, 0] = features["temp"]
                occ_vec[0, 1] = features["humidity"]
                result["occupancy_pred"] = int(self._rf_occ.predict(occ_vec)[0])
            except Exception:
                result["occupancy_pred"] = -1

        return result

    def compute_confidence(self, inference_result: Dict[str, Any]) -> float:
        conf = 0.0

        if inference_result.get("anomaly_detected") == 1:
            conf += 0.50
        if inference_result.get("temp_out_of_range") == 1:
            conf += 0.25
        if inference_result.get("hum_out_of_range") == 1:
            conf += 0.20

        # Boost from anomaly score (more negative = higher confidence)
        score = inference_result.get("anomaly_score", 0.0)
        if score < -0.1:
            conf += min(0.15, abs(score) * 0.3)

        # Trend check: if last 3 steps all had anomaly
        recent = list(self._memory)[-3:]
        if len(recent) == 3 and all(e.get("anomaly_detected", 0) == 1 for e in recent):
            conf = min(1.0, conf + 0.10)

        return min(1.0, conf)

    def derive_label(
        self,
        raw_data: Dict[str, Any],
        inference_result: Dict[str, Any],
    ) -> Optional[int]:
        """Use original dataset row labels for replay."""
        row = self.get_dataset_row()
        if row is not None:
            temp = float(row.get("temp", 20.0))
            hum  = float(row.get("humidity", 50.0))
            return int(
                (temp > _ANOMALY_TEMP_HIGH) or (temp < _ANOMALY_TEMP_LOW)
                or (hum > _ANOMALY_HUM_HIGH) or (hum < _ANOMALY_HUM_LOW)
            )
        return inference_result.get("anomaly_detected", 0)

    def _build_fresh_model(self):
        """
        IsolationForest does not use labels — treat replay buffer as
        unsupervised data (ignore y) and refit with contamination=0.05.
        """
        return IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )

    def _refit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Override: IsolationForest is unsupervised — fit without labels."""
        new_model = self._build_fresh_model()
        new_model.fit(X)          # unsupervised: ignore y
        self.model = new_model

    def _trigger_refit(self) -> None:
        """Override for unsupervised refit (no label-variety check needed)."""
        from .agent_bus import AgentMessage, MessageType, Severity
        import time

        X = np.array(self._replay_X)
        if len(X) < 50:
            return
        try:
            self._refit_model(X, None)
            self._metrics["refit_count"]     += 1
            self._metrics["last_refit_tick"]  = self._metrics["tick_count"]
            self._replay_X.clear()
            self._replay_y.clear()
            print(
                f"\n  ★ [EnvSensorAgent] LEARNING UPDATE — "
                f"IsolationForest refitted on {len(X)} samples "
                f"(refit #{self._metrics['refit_count']})"
            )
            self.bus.publish(AgentMessage(
                source   = self.agent_name,
                msg_type = MessageType.LEARNING_UPDATE,
                severity = Severity.LOW,
                payload  = {"refit_count": self._metrics["refit_count"], "samples_used": len(X)},
                reason   = f"IsolationForest refitted on {len(X)} samples.",
                timestamp= time.time(),
            ))
        except Exception as e:
            print(f"  [EnvSensorAgent] Refit failed: {e}")

    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        return np.array([features[k] for k in _ENV_FEATURES])

    def _build_reason(self, inference: Dict[str, Any], confidence: float) -> str:
        parts = []
        if inference.get("anomaly_detected") == 1:
            parts.append(f"IsolationForest anomaly (score={inference.get('anomaly_score', 0):.3f})")
        if inference.get("temp_out_of_range") == 1:
            parts.append("Temperature out of safe range")
        if inference.get("hum_out_of_range") == 1:
            parts.append("Humidity out of safe range")
        parts.append(f"confidence={confidence:.2f}")
        return " | ".join(parts)
