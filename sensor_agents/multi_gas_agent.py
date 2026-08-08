import os
import sys
import numpy as np
from typing import Any, Dict, Optional, List

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
from .agent_bus import AgentBus, AgentMessage, MessageType, Severity

_MULTIGAS_FEATURES = ['CH4_ppm', 'CO_ppm', 'CO2_ppm', 'H2_ppm', 'H2S_ppm', 'NH3_ppm', 'LPG_ppm', 'CNG_ppm']

class MultiGasDetectorAgent(SensorAgentBase):
    """
    Autonomous Multi-Gas Detector Sensor AI Agent (Option B Standalone).
    Loads the trained PyTorch multi-gas detector model and predicts active gases.
    """

    def __init__(self, workspace_root: str, bus: AgentBus, verbose: bool = True,
                 dataset_name: str = "multi_gas_detector_real_v2.csv"):
        self.workspace_root = workspace_root
        self.gas_names = ["Methane", "CO", "CO2", "H2", "H2S", "NH3", "LPG", "CNG"]
        self.feature_names = _MULTIGAS_FEATURES

        # Load the PyTorch multilabel classifier model
        model_path = os.path.join(workspace_root, "gas_sensors", "models", "multi_gas_detector.joblib")
        if os.path.exists(model_path):
            try:
                import joblib
                primary_model = joblib.load(model_path)
                if verbose:
                    print(f"  [MultiGasDetectorAgent] [OK] Loaded model: multi_gas_detector")
            except Exception as e:
                print(f"  [MultiGasDetectorAgent] [ERROR] Failed to load multi_gas_detector: {e}")
                primary_model = None
        else:
            print(f"  [MultiGasDetectorAgent] [WARNING] Model not found: multi_gas_detector.joblib")
            primary_model = None

        if primary_model is None:
            # Fallback dummy model
            primary_model = RandomForestClassifier(n_estimators=10, random_state=42)

        dataset_path = os.path.join(
            workspace_root, "gas_sensors", "data", dataset_name
        )

        super().__init__(
            agent_name   = "MultiGasDetectorAgent",
            bus          = bus,
            primary_model= primary_model,
            dataset_path = dataset_path if os.path.exists(dataset_path) else None,
            replay_buffer_size = 200,
            memory_window      = 30,
            verbose            = verbose,
        )

    def perceive(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw sensor inputs into the 8 expected features."""
        return {
            "CH4_ppm": float(raw_data.get("CH4_ppm", raw_data.get("MQ4_CH4_ppm", 0.0))),
            "CO_ppm": float(raw_data.get("CO_ppm", raw_data.get("MQ7_CO_ppm", 0.0))),
            "CO2_ppm": float(raw_data.get("CO2_ppm", raw_data.get("MG811_CO2_ppm", 400.0))),
            # LPG and H2 are different channels; keep H2 at zero when no H2
            # sensor is present instead of creating a false positive.
            "H2_ppm": float(raw_data.get("H2_ppm", 0.0)),
            "H2S_ppm": float(raw_data.get("H2S_ppm", raw_data.get("MQ136_H2S_ppm", raw_data.get("Sensor3[ppm]", 0.0)))),
            "NH3_ppm": float(raw_data.get("NH3_ppm", raw_data.get("MQ135_NH3_ppm", raw_data.get("NH3", 0.0)))),
            "LPG_ppm": float(raw_data.get("LPG_ppm", raw_data.get("MQ2_LPG_ppm", 0.0))),
            "CNG_ppm": float(raw_data.get("CNG_ppm", 0.0)),
        }

    def infer(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Run multi-gas detector DL model and return predictions & probabilities."""
        result: Dict[str, Any] = {}
        
        # Prepare 1x8 input vector
        X_vec = np.array([features[f] for f in self.feature_names]).reshape(1, -1)
        
        try:
            # Live model predictions
            y_pred = self.model.predict(X_vec)
            y_pred_proba = self.model.predict_proba(X_vec)
            
            # Extract lists of length 8
            preds = np.asarray(y_pred[0]).reshape(-1)
            probs = np.asarray(y_pred_proba[0]).reshape(-1)
            if len(preds) != len(self.gas_names) or len(probs) != len(self.gas_names):
                raise ValueError("multi-gas model returned an unexpected output width")
        except Exception as e:
            # Fallback if model behaves differently (e.g. dummy RandomForest)
            try:
                preds = self.model.predict(X_vec)[0]
                if isinstance(preds, (int, np.integer)):
                    preds = [preds] + [0]*7
                probs = [0.9 if p else 0.1 for p in preds]
            except Exception:
                preds = [0] * 8
                probs = [0.0] * 8

        # Map predictions to dictionary
        for name, pred, prob in zip(self.gas_names, preds, probs):
            result[f"{name}_active"] = int(pred)
            result[f"{name}_prob"] = float(prob)

        return result

    def compute_confidence(self, inference_result: Dict[str, Any]) -> float:
        """Return the maximum probability of any predicted active gas."""
        active_probs = [
            inference_result[f"{gas}_prob"]
            for gas in self.gas_names
            if inference_result.get(f"{gas}_active", 0) == 1
        ]
        if active_probs:
            return max(active_probs)
        return 0.0

    def derive_label(self, raw_data: Dict[str, Any], inference_result: Dict[str, Any]) -> Optional[int]:
        """Derive label (1 if any target is active in CSV or prediction, else 0)."""
        row = self.get_dataset_row()
        if row is not None:
            # Check if any target_* is active in original CSV
            for gas in self.gas_names:
                if int(row.get(f"target_{gas}", 0)) == 1:
                    return 1
            return 0
        # Fallback to inference
        for gas in self.gas_names:
            if inference_result.get(f"{gas}_active", 0) == 1:
                return 1
        return 0

    def _build_fresh_model(self):
        # The production model is multi-label.  A binary RandomForest trained
        # from the base class's single aggregate label would silently corrupt
        # its eight-output contract, so this agent keeps the deployed model
        # frozen until a dedicated multi-label retraining job is run.
        return RandomForestClassifier(n_estimators=10, random_state=42)

    def _trigger_refit(self) -> None:
        """Discard aggregate replay labels without replacing the 8-head model."""
        if len(self._replay_X) < self.replay_buffer_size:
            return
        self._replay_X.clear()
        self._replay_y.clear()

    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        return np.array([features[f] for f in self.feature_names])

    def _build_reason(self, inference: Dict[str, Any], confidence: float) -> str:
        active_gases = [gas for gas in self.gas_names if inference.get(f"{gas}_active", 0) == 1]
        if active_gases:
            return f"MULTIGAS_ALERT: Active gases detected: {', '.join(active_gases)} | confidence={confidence:.2f}"
        return f"Nominal | confidence={confidence:.2f}"

    def _decide_action(self, inference: Dict[str, Any], confidence: float, timestamp: float) -> str:
        active_gases = [gas for gas in self.gas_names if inference.get(f"{gas}_active", 0) == 1]
        
        is_hazard = confidence >= 0.5
        if is_hazard:
            self._consecutive_hazards += 1
        else:
            self._consecutive_hazards = 0

        if self._consecutive_hazards >= self._alert_threshold:
            severity = self._map_severity(confidence)
            reason = self._build_reason(inference, confidence)
            
            # Broadcast the ALERT message
            msg = AgentMessage(
                source=self.agent_name,
                msg_type=MessageType.MULTIGAS_ALERT,
                severity=severity,
                payload={
                    **inference,
                    "confidence": confidence,
                    "active_gases": active_gases
                },
                reason=reason,
                timestamp=timestamp,
            )
            self.bus.publish(msg)
            self._metrics["alert_count"] += 1
            return f"ALERT:{severity.name}"
            
        elif self._metrics["tick_count"] % 10 == 0:
            msg = AgentMessage(
                source=self.agent_name,
                msg_type=MessageType.INFO,
                severity=Severity.LOW,
                payload={
                    **inference,
                    "confidence": confidence,
                    "active_gases": active_gases
                },
                reason=f"Tick {self._metrics['tick_count']} — nominal",
                timestamp=timestamp,
            )
            self.bus.publish(msg)
            return "INFO:NOMINAL"
            
        return "IDLE"
