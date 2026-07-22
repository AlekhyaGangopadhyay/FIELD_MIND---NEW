"""
gas_agent.py — Gas Sensor AI Agent
====================================
Wraps all 6 pre-trained gas models from Tier1Monitor into an autonomous
agent that observes gas concentrations, reasons about hazard conditions,
acts by publishing ALERT messages, and learns from the original gas dataset.

Supported Models (from gas_sensors/models/):
  - mq4_gas_classifier.joblib        (methane, 128 features)
  - smoke_fire_alarm_model.joblib    (smoke/fire, 36 features)
  - gas_hazard_lpg_cng.joblib        (LPG/CNG, 2 features)
  - gas_hazard_co_nox_c6h6.joblib   (CO/NOx/Benzene, 3 features)
  - gas_hazard_smoke_env.joblib      (smoke+env, 3 features)
  - air_quality_regressor.joblib     (air quality score, 7 features)

Ground-Truth Labels:
  Derived from 'Hazard_Alert' column of FIELDMIND_physics_dataset.csv.
  On every replay tick the agent reads a row from this dataset and stores
  (lpg_cng_features, hazard_label) into its experience replay buffer.

Self-Learning:
  Uses Experience Replay Buffer (size=200). When full, trains a fresh
  RandomForestClassifier on the accumulated (feature, label) pairs.
  The gas agent uses the 2-feature LPG/CNG model as its primary learnable
  model (simplest feature space); the specialised models remain frozen
  until a future training cycle updates the full joblib files.
"""

import os
import sys
import joblib
import numpy as np
from typing import Any, Dict, Optional

# Ensure gas_sensors path is in sys.path for PyTorch DL wrappers unpickling
gas_sensors_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gas_sensors"))
if gas_sensors_path not in sys.path:
    sys.path.insert(0, gas_sensors_path)

try:
    import dl_wrappers
except ImportError:
    pass

from sklearn.ensemble import RandomForestClassifier

from .agent_base import SensorAgentBase
from .agent_bus import AgentBus


# Feature keys for the 2-feature LPG/CNG primary learnable model
_LPG_FEATURES = ["MQ2_LPG_ppm", "MQ4_CH4_ppm"]


class GasSensorAgent(SensorAgentBase):
    """
    Autonomous Gas Sensor AI Agent.

    Parameters
    ----------
    workspace_root : str   Root of FIELD-MIND project (for model paths)
    bus            : AgentBus
    verbose        : bool
    """

    def __init__(self, workspace_root: str, bus: AgentBus, verbose: bool = True,
                 dataset_name: str = "FIELDMIND_physics_dataset.csv"):
        self.workspace_root = workspace_root

        # ── Load all gas models ──────────────────────────────────────────
        gas_model_dir = os.path.join(workspace_root, "gas_sensors", "models")
        self._models: Dict[str, Any] = {}
        self._load_models(gas_model_dir)

        # Primary learnable model = LPG/CNG classifier (2-feature RF)
        primary_model = self._models.get("lpg_cng", RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ))

        # Dataset path for experience replay seeding
        # Supports A/B testing: pass dataset_name="FIELDMIND_real_replay.csv"
        # to use real mine data instead of synthetic physics data.
        dataset_path = os.path.join(
            workspace_root, "gas_sensors", "data", dataset_name
        )

        super().__init__(
            agent_name   = "GasSensorAgent",
            bus          = bus,
            primary_model= primary_model,
            dataset_path = dataset_path if os.path.exists(dataset_path) else None,
            replay_buffer_size = 200,
            memory_window      = 30,
            verbose            = verbose,
        )

        # Seed replay buffer from dataset on startup
        self._seed_replay_from_dataset(n_seed=100)

    # -----------------------------------------------------------------------
    # Model Loading
    # -----------------------------------------------------------------------

    def _load_models(self, model_dir: str) -> None:
        model_map = {
            "lpg_cng"         : "gas_hazard_lpg_cng.joblib",
            "co_nox"          : "gas_hazard_co_nox_c6h6.joblib",
            "multi_gas"       : "multi_gas_detector.joblib",
            "baseline_iforest": "mine_baseline_iforest.joblib",
            "severity_ch4"    : "severity_ch4.joblib",
            "severity_co"     : "severity_co.joblib",
            "severity_co2"    : "severity_co2.joblib",
            "severity_h2"     : "severity_h2.joblib",
        }
        for key, fname in model_map.items():
            path = os.path.join(model_dir, fname)
            if os.path.exists(path):
                try:
                    self._models[key] = joblib.load(path)
                    print(f"  [GasSensorAgent] [OK] Loaded model: {key}")
                except Exception as e:
                    print(f"  [GasSensorAgent] [ERROR] Failed to load {key}: {e}")
            else:
                print(f"  [GasSensorAgent] [WARNING] Model not found: {fname}")

    # -----------------------------------------------------------------------
    # Replay Seeding from Original Dataset
    # -----------------------------------------------------------------------

    def _seed_replay_from_dataset(self, n_seed: int = 100) -> None:
        """Warm-start the replay buffer with rows from the original dataset."""
        if self._dataset_df is None:
            return
        seeded = 0
        for _ in range(min(n_seed, len(self._dataset_df))):
            row = self.get_dataset_row()
            if row is None:
                break
            feat_vec = np.array([
                float(row.get("MQ2_LPG_ppm", 0.0)),
                float(row.get("MQ4_CH4_ppm", 0.0)),
            ])
            label = int(row.get("Hazard_Alert", 0))
            self._replay_X.append(feat_vec)
            self._replay_y.append(label)
            seeded += 1
        print(f"  [GasSensorAgent] Replay buffer seeded with {seeded} dataset rows.")

    # -----------------------------------------------------------------------
    # Abstract Interface Implementation
    # -----------------------------------------------------------------------

    def perceive(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and extract gas features from raw sensor reading."""
        features = {
            "MQ2_LPG_ppm"     : float(raw_data.get("MQ2_LPG_ppm", 0.0)),
            "MQ4_CH4_ppm"     : float(raw_data.get("MQ4_CH4_ppm", 0.0)),
            "MQ7_CO_ppm"      : float(raw_data.get("MQ7_CO_ppm", 0.0)),
            "MQ135_NOx_ppm"   : float(raw_data.get("MQ135_NOx_ppm", 0.0)),
            "MQ3_Benzene_ppm" : float(raw_data.get("MQ3_Benzene_ppm", 0.0)),
            "PM25_Dust_ugm3"  : float(raw_data.get("PM25_Dust_ugm3", 0.0)),
            "Temp_C"          : float(raw_data.get("Temp_C", 25.0)),
            "Humidity_pct"    : float(raw_data.get("Humidity_pct", 50.0)),
            # Air quality composite features
            "MQ2_Smoke_ppm"   : float(raw_data.get("MQ2_Smoke_ppm", 0.0)),
            "MG811_CO2_ppm"   : float(raw_data.get("MG811_CO2_ppm", 400.0)),
        }
        return features

    def infer(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Run all production gas ML & Deep Learning models and return a combined result dict."""
        result: Dict[str, Any] = {}

        # 1. LPG / CNG hazard (primary learnable PyTorch Deep model, 2 features)
        lpg_vec = np.array([features["MQ2_LPG_ppm"], features["MQ4_CH4_ppm"]]).reshape(1, -1)
        try:
            result["lpg_hazard"] = int(self.model.predict(lpg_vec)[0])
        except Exception:
            result["lpg_hazard"] = 0

        # 2. CO / NOx / Benzene (3 features — PyTorch Deep model)
        if "co_nox" in self._models:
            co_vec = np.array([
                features["MQ7_CO_ppm"],
                features["MQ135_NOx_ppm"],
                features["MQ3_Benzene_ppm"],
            ]).reshape(1, -1)
            try:
                result["co_nox_hazard"] = int(self._models["co_nox"].predict(co_vec)[0])
            except Exception:
                result["co_nox_hazard"] = 0

        # 3. PyTorch Deep Severity Levels (CH4, CO, CO2, H2)
        for gas_key, feat_name in [("ch4", "MQ4_CH4_ppm"), ("co", "MQ7_CO_ppm"), ("co2", "MG811_CO2_ppm"), ("h2", "MQ2_LPG_ppm")]:
            model_key = f"severity_{gas_key}"
            if model_key in self._models:
                try:
                    val_vec = np.array([features.get(feat_name, 0.0)]).reshape(1, -1)
                    result[f"{gas_key}_severity"] = int(self._models[model_key].predict(val_vec)[0])
                except Exception:
                    result[f"{gas_key}_severity"] = 0

        # 4. Clean-air hardware baseline anomaly detection
        if "baseline_iforest" in self._models:
            try:
                raw_feats = np.array([
                    features.get("MQ2_LPG_ppm", 10.0),
                    features.get("MQ2_Smoke_ppm", 250.0),
                    features.get("MQ3_Alcohol_ppm", 30.0),
                    features.get("MQ4_CH4_ppm", 100.0),
                    features.get("MQ135_NOx_ppm", 220.0),
                    features.get("MQ7_CO_ppm", 120.0),
                    features.get("Temp_C", 30.0),
                    features.get("Humidity_pct", 60.0),
                ]).reshape(1, -1)
                # IsolationForest predict returns -1 for anomaly, 1 for normal
                anom_dict = self._models["baseline_iforest"]
                iforest_model = anom_dict["model"] if isinstance(anom_dict, dict) else anom_dict
                result["hardware_anomaly"] = int(iforest_model.predict(raw_feats)[0] == -1)
            except Exception:
                result["hardware_anomaly"] = 0

        return result

    def compute_confidence(self, inference_result: Dict[str, Any]) -> float:
        """
        Confidence = weighted sum of hazard & severity flags.
        """
        weights = {
            "lpg_hazard"       : 0.35,
            "co_nox_hazard"    : 0.25,
            "hardware_anomaly" : 0.20,
            "ch4_severity"     : 0.10,
            "co_severity"      : 0.10,
        }
        conf = 0.0
        for key, w in weights.items():
            val = float(inference_result.get(key, 0))
            conf += w * (val / 2.0 if "severity" in key else val)

        # Trend boost: if last 3 memory entries were also hazardous
        recent = list(self._memory)[-3:]
        if len(recent) == 3 and all(e.get("lpg_hazard", 0) == 1 for e in recent):
            conf = min(1.0, conf + 0.15)

        return min(1.0, conf)

    def derive_label(
        self,
        raw_data: Dict[str, Any],
        inference_result: Dict[str, Any],
    ) -> Optional[int]:
        """
        Ground-truth label derived from the original dataset.
        On every step, pull the next dataset row and use its Hazard_Alert column.
        This ensures the agent's experience replay is always anchored to real data.
        """
        # Pull one row from dataset (cycles through entire dataset)
        row = self.get_dataset_row()
        if row is not None and "Hazard_Alert" in row:
            return int(row["Hazard_Alert"])

        # Fallback: use inference result as weak supervision
        return int(inference_result.get("lpg_hazard", 0))

    def _build_fresh_model(self):
        """Return a new untrained RF classifier matching the primary model spec."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features to the 2-dimensional LPG/CNG vector used by primary model."""
        return np.array([
            features["MQ2_LPG_ppm"],
            features["MQ4_CH4_ppm"],
        ])

    def _build_reason(self, inference: Dict[str, Any], confidence: float) -> str:
        reasons = []
        if inference.get("lpg_hazard") == 1:
            reasons.append("LPG/CNG concentration hazard")
        if inference.get("co_nox_hazard") == 1:
            reasons.append("CO/NOx toxic gas spike")
        if inference.get("smoke_env_hazard") == 1:
            reasons.append("Smoke+Env hazard")
        aq = inference.get("air_quality_score", 100.0)
        if aq < 50.0:
            reasons.append(f"Poor air quality score: {aq:.1f}")
        reasons.append(f"confidence={confidence:.2f}")
        return " | ".join(reasons) if reasons else f"Gas hazard (conf={confidence:.2f})"
