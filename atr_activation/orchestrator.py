import os
import sys
import torch
import numpy as np

# Set up system paths so we can import from scisense_protocol
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

try:
    from .detector_wrappers import Tier1Monitor
except ImportError:  # Support ``python atr_activation/demo_atr.py``.
    from detector_wrappers import Tier1Monitor

from scisense_protocol.encoders import GasEncoder, EnvironmentalEncoder, VibrationEncoder, UltrasonicEncoder
from scisense_protocol.coherence import SciSenseCoherenceTracker, normalize_modal_vector

class ATROrchestrator:
    """
    Anomaly-Triggered Reasoning (Layer 2A) Orchestrator.
    Monitors streaming sensor features, evaluates them against pre-trained local classifiers (Tier 1),
    and triggers SciSense projection + simulated LLM boot (Tier 2) upon hazard detection.
    """
    def __init__(self, workspace_root):
        self.workspace_root = workspace_root
        self.monitor = Tier1Monitor(workspace_root)
        self.coherence_tracker = SciSenseCoherenceTracker()
        
        # Load SciSense PyTorch projection encoders
        self.gas_encoder = GasEncoder()
        self.env_encoder = EnvironmentalEncoder()
        self.vib_encoder = VibrationEncoder()
        self.ultra_encoder = UltrasonicEncoder()
        
        self.gas_encoder.eval()
        self.env_encoder.eval()
        self.vib_encoder.eval()
        self.ultra_encoder.eval()
        
        # Memory states: "IDLE" (low-power, LLM unloaded) or "ACTIVE_REASONING" (LLM loaded)
        self.device_state = "IDLE"
        self.sliding_history = {
            'gas': [],
            'env': [],
            'vibration': [],
            'ultrasonic': []
        }
        self.history_limit = 50 # Keep last 50 updates for alignment window

    @staticmethod
    def _value(features, *keys, default=0.0):
        """Read the first finite numeric alias from a feature dictionary."""
        if not isinstance(features, dict):
            return float(default)
        for key in keys:
            try:
                value = float(features.get(key, default))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
        return float(default)

    def _project_modalities(self, gas_feat, env_feat, vib_feat, ultra_feat):
        """Project the current frame into the four SciSense spaces."""
        gas_values = [
            self._value(gas_feat, "MQ4_CH4_ppm", "CH4_ppm", "methane"),
            self._value(gas_feat, "MQ7_CO_ppm", "CO_ppm", "co"),
            self._value(gas_feat, "MQ2_LPG_ppm", "LPG_ppm", "lpg"),
            self._value(gas_feat, "MQ2_Smoke_ppm", "smoke"),
            self._value(gas_feat, "MQ135_NOx_ppm", "NOx_ppm", "nox"),
            self._value(gas_feat, "MG811_CO2_ppm", "CO2_ppm", "co2"),
        ]
        env_values = [
            self._value(env_feat, "temp", "temperature", "Temperature[C]", "Temperature"),
            self._value(env_feat, "humidity", "Humidity[%]", "Humidity"),
            self._value(env_feat, "pressure", "Pressure[hPa]", "Pressure"),
            self._value(env_feat, "occupancy", "occupancy_state", "Occupancy"),
        ]
        vib_values = [
            self._value(vib_feat, f"feature_{i}") for i in range(15)
        ]
        if not any(vib_values):
            vib_values = [
                self._value(vib_feat, "offset"),
                self._value(vib_feat, "max_charge"),
                self._value(vib_feat, "total_charge"),
                self._value(vib_feat, "num_holes"),
                self._value(vib_feat, "detonator_code"),
                self._value(vib_feat, "trid_12"),
                self._value(vib_feat, "trid_13"),
                self._value(vib_feat, "trid_14"),
                self._value(vib_feat, "gx"),
                self._value(vib_feat, "gy"),
                self._value(vib_feat, "gelev"),
                self._value(vib_feat, "sx"),
                self._value(vib_feat, "sy"),
                self._value(vib_feat, "selev"),
                self._value(vib_feat, "elevation_diff"),
            ]
        ultra_values = [
            self._value(ultra_feat, f"US{i}") for i in range(1, 25)
        ]
        if not any(ultra_values):
            ultra_values[:2] = [
                self._value(ultra_feat, "SD_front"),
                self._value(ultra_feat, "SD_left"),
            ]

        tensors = {
            "gas": torch.tensor(normalize_modal_vector(gas_values, [10000, 50, 1000, 100, 5, 5000])).unsqueeze(0),
            "env": torch.tensor(normalize_modal_vector(env_values, [40, 100, 1100, 1])).unsqueeze(0),
            "vibration": torch.tensor(normalize_modal_vector(vib_values, [1, 100, 1000, 100, 1000, 1, 1, 1, 1000, 1000, 1000, 1000, 1000, 1000, 100])).unsqueeze(0),
            "ultrasonic": torch.tensor(normalize_modal_vector(ultra_values, [5] * 24)).unsqueeze(0),
        }
        with torch.no_grad():
            return {
                "gas": self.gas_encoder(tensors["gas"]),
                "env": self.env_encoder(tensors["env"]),
                "vibration": self.vib_encoder(tensors["vibration"]),
                "ultrasonic": self.ultra_encoder(tensors["ultrasonic"]),
            }
        
    def add_to_history(self, stream_name, timestamp, features):
        """Appends raw features to the sliding temporal history log (keeping only numeric scalars)."""
        log_entry = {'timestamp': timestamp}
        for k, v in features.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                log_entry[k] = float(v)
                
        self.sliding_history[stream_name].append(log_entry)
        # Limit sliding history size to conserve memory
        if len(self.sliding_history[stream_name]) > self.history_limit:
            self.sliding_history[stream_name].pop(0)

    def process_stream_frame(self, gas_feat, env_feat, vib_feat, ultra_feat, timestamp):
        """
        Processes a single synchronized streaming frame across all modalities.
        
        Returns:
        - dict containing evaluation results, trigger status, and projection embeddings (if activated).
        """
        # 1. Update sliding histories
        self.add_to_history('gas', timestamp, gas_feat)
        self.add_to_history('env', timestamp, env_feat)
        self.add_to_history('vibration', timestamp, vib_feat)
        self.add_to_history('ultrasonic', timestamp, ultra_feat)
        
        # 2. Run Tier 1 monitors (Continuous inference on pre-trained models)
        gas_res = self.monitor.evaluate_gas(gas_feat)
        env_res = self.monitor.evaluate_env(env_feat)
        vib_res = self.monitor.evaluate_vibration(vib_feat)
        ultra_res = self.monitor.evaluate_ultrasonic(ultra_feat, config_type=24)
        
        # 3. Evaluate Trigger Conditions
        is_triggered = False
        trigger_reasons = []
        
        # Gas hazards
        if gas_res.get('methane_hazard') == 1:
            is_triggered = True
            trigger_reasons.append("Methane Gas build-up detected!")
        if gas_res.get('smoke_alarm') == 1:
            is_triggered = True
            trigger_reasons.append("Smoke / Fire Alarm triggered!")
        if gas_res.get('lpg_hazard') == 1:
            is_triggered = True
            trigger_reasons.append("LPG / CNG concentration hazard!")
        if gas_res.get('co_nox_hazard') == 1:
            is_triggered = True
            trigger_reasons.append("CO / NOx toxic gas spike!")
        if gas_res.get('co2_hazard') == 1:
            is_triggered = True
            trigger_reasons.append("Carbon dioxide concentration hazard!")
        if gas_res.get('h2s_hazard') == 1:
            is_triggered = True
            trigger_reasons.append("Hydrogen sulfide concentration hazard!")
        if gas_res.get('smoke_env_hazard') == 1:
            is_triggered = True
            trigger_reasons.append("Smoke / environmental gas hazard!")
            
        # Environmental anomaly
        if env_res.get('anomaly_detected') == 1:
            is_triggered = True
            trigger_reasons.append("Environmental temperature/humidity anomaly detected (Isolation Forest)!")
            
        # Vibration & Structural Stability hazard
        if vib_res.get('collapse_imminent') == 1:
            is_triggered = True
            trigger_reasons.append("CRITICAL: Structural collapse imminent (Wall displacement accelerating)!")
        if vib_res.get('shock_alert') == 1:
            is_triggered = True
            trigger_reasons.append(f"Vibration shock detected! SW-420 pulse count: {vib_res.get('vibration_pulses', 0):.0f} (Level {vib_res.get('shock_level')})")
        if vib_res.get('vibration_hazard') == 1:
            is_triggered = True
            trigger_reasons.append(f"High-amplitude blast vibration hazard! PPV: {vib_res.get('predicted_ppv', 0.0):.2f} mm/s")
            
        # Robot safety command
        if ultra_res.get('sharp_turn_required') == 1:
            is_triggered = True
            trigger_reasons.append("Collision alert! Navigation system required sharp steering evasive action!")

        # SciSense is load-bearing: project every frame and compare the
        # resulting cross-modal relationships with the learned normal state.
        embeddings = self._project_modalities(gas_feat, env_feat, vib_feat, ultra_feat)
        coherence = self.coherence_tracker.update(
            embeddings,
            # Tier-1 hazard frames are not safe baseline observations.
            update_baseline=not is_triggered,
        )
        if coherence["is_anomaly"]:
            is_triggered = True
            trigger_reasons.append(
                "Cross-modal coherence residual anomaly detected "
                f"(R_t={coherence['residual']:.4f} > "
                f"{coherence['threshold']:.4f})!"
            )

        # 4. Handle State Transitions
        output = {
            'timestamp': timestamp,
            'gas_eval': gas_res,
            'env_eval': env_res,
            'vibration_eval': vib_res,
            'ultrasonic_eval': ultra_res,
            'triggered': is_triggered,
            'trigger_reasons': trigger_reasons,
            'device_state': self.device_state,
            'coherence_residual': coherence['residual'],
            'coherence_threshold': coherence['threshold'],
            'coherence_anomaly': coherence['is_anomaly'],
            'coherence_baseline_ready': coherence['baseline_ready'],
            'coherence_baseline_updates': coherence['baseline_updates'],
            'coherence_residual_mean': coherence['residual_mean'],
            'coherence_residual_std': coherence['residual_std'],
            'coherence_warming_up': coherence['sample_count'] < self.coherence_tracker.min_history,
            'coherence_similarity_matrix': coherence['similarity_matrix'].tolist(),
            'coherence_baseline_matrix': (
                None if coherence['baseline_matrix'] is None
                else coherence['baseline_matrix'].tolist()
            ),
            # Kept on every tick so callers can inspect the representation
            # that produced R_t, including safe baseline frames.
            'aligned_embeddings': embeddings,
        }
        
        if is_triggered:
            if self.device_state == "IDLE":
                print("\n" + "!" * 80)
                print("[ATR TRIGGER] Significant anomaly/hazard detected by Tier 1 monitors!")
                for r in trigger_reasons:
                    print(f"  Reason: {r}")
                print("-" * 80)
                print("[ATR STATE TRANSITION] IDLE -> ACTIVE_REASONING")
                print("  -> Swapping device memory context (NVIDIA Jetson Orin Nano 8GB)...")
                print("  -> Loading Qwen2.5-7B-Instruct-Q4_K_M quantized reasoning engine (~4.35 GB RAM)...")
                print("  -> Activating SciSense projection embedding layers...")
                print("!" * 80 + "\n")
                self.device_state = "ACTIVE_REASONING"
                output['device_state'] = self.device_state
                
            print(
                f"[SciSense] Coherence residual R_t={coherence['residual']:.4f} "
                f"(threshold={coherence['threshold']:.4f})"
            )
            
        else:
            if self.device_state == "ACTIVE_REASONING":
                # If no anomalies persist for a sliding window, we transition back to conserve memory
                # In this demo, we swap back immediately if the frame is clean, simulating cache timeouts
                print("\n" + "." * 80)
                print("[ATR STATE TRANSITION] ACTIVE_REASONING -> IDLE")
                print("  -> Anomaly cleared. Suspending reasoning core weights to swap out of memory.")
                print("  -> Restoring low-power background monitoring mode.")
                print("." * 80 + "\n")
                self.device_state = "IDLE"
                output['device_state'] = self.device_state
                
        return output
