import os
import sys
import torch
import numpy as np
from detector_wrappers import Tier1Monitor

# Set up system paths so we can import from scisense_protocol
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from scisense_protocol.alignment import TemporalAligner
from scisense_protocol.encoders import GasEncoder, EnvironmentalEncoder, VibrationEncoder, UltrasonicEncoder

class ATROrchestrator:
    """
    Anomaly-Triggered Reasoning (Layer 2A) Orchestrator.
    Monitors streaming sensor features, evaluates them against pre-trained local classifiers (Tier 1),
    and triggers SciSense projection + simulated LLM boot (Tier 2) upon hazard detection.
    """
    def __init__(self, workspace_root):
        self.workspace_root = workspace_root
        self.monitor = Tier1Monitor(workspace_root)
        self.aligner = TemporalAligner(time_window_seconds=1.0)
        
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
            
        # Environmental anomaly
        if env_res.get('anomaly_detected') == 1:
            is_triggered = True
            trigger_reasons.append("Environmental temperature/humidity anomaly detected (Isolation Forest)!")
            
        # Vibration hazard
        if vib_res.get('vibration_hazard') == 1:
            is_triggered = True
            trigger_reasons.append(f"High-amplitude blast vibration hazard! PPV: {vib_res.get('predicted_ppv', 0.0):.2f} mm/s")
            
        # Robot safety command
        if ultra_res.get('sharp_turn_required') == 1:
            is_triggered = True
            trigger_reasons.append("Collision alert! Navigation system required sharp steering evasive action!")
            
        # 4. Handle State Transitions
        output = {
            'timestamp': timestamp,
            'gas_eval': gas_res,
            'env_eval': env_res,
            'vibration_eval': vib_res,
            'ultrasonic_eval': ultra_res,
            'triggered': is_triggered,
            'trigger_reasons': trigger_reasons,
            'device_state': self.device_state
        }
        
        if is_triggered:
            if self.device_state == "IDLE":
                print("\n" + "!" * 80)
                print("[ATR TRIGGER] Significant anomaly/hazard detected by Tier 1 monitors!")
                for r in trigger_reasons:
                    print(f"  Reason: {r}")
                print("-" * 80)
                print("[ATR STATE TRANSITION] IDLE -> ACTIVE_REASONING")
                print("  -> Swapping device memory context...")
                print("  -> Loading Llama-3.2-3B-Instruct quantized reasoning engine (~3.0 GB RAM)...")
                print("  -> Activating SciSense projection embedding layers...")
                print("!" * 80 + "\n")
                self.device_state = "ACTIVE_REASONING"
                output['device_state'] = self.device_state
                
            # Run Tier 2 Projection: Align the recent sliding history using SciSense Protocol
            print(f"[SciSense] Aligning last 10 seconds of sliding feature history at timestamp {timestamp:.2f}...")
            aligned_epochs = self.aligner.align_streams(
                self.sliding_history, 
                start_time=timestamp - 10.0, 
                end_time=timestamp + 1.0
            )
            
            # Project the most recent aligned epoch
            latest_epoch = aligned_epochs[-1]
            features = latest_epoch['aligned_features']
            
            embeddings = {}
            with torch.no_grad():
                # Project Gas (dim = 6)
                if features['gas'] is not None:
                    g_dict = features['gas']
                    g_list = [
                        g_dict.get('MQ4_CH4_ppm', g_dict.get('methane', 0.0)),
                        g_dict.get('MQ7_CO_ppm', g_dict.get('co', 0.0)),
                        g_dict.get('MQ2_LPG_ppm', g_dict.get('lpg', 0.0)),
                        g_dict.get('MQ2_Smoke_ppm', g_dict.get('smoke', 0.0)),
                        g_dict.get('MQ135_NOx_ppm', g_dict.get('nox', 0.0)),
                        g_dict.get('MG811_CO2_ppm', g_dict.get('co2', 0.0))
                    ]
                    g_vec = torch.tensor(g_list).float().unsqueeze(0)
                    embeddings['gas'] = self.gas_encoder(g_vec)
                    
                # Project Env (dim = 4)
                if features['env'] is not None:
                    e_dict = features['env']
                    e_list = [
                        e_dict.get('temp', e_dict.get('temperature', e_dict.get('Temperature[C]', e_dict.get('Temperature', 0.0)))),
                        e_dict.get('humidity', e_dict.get('Humidity[%]', e_dict.get('Humidity', 0.0))),
                        e_dict.get('pressure', e_dict.get('Pressure[hPa]', e_dict.get('Pressure', 0.0))),
                        e_dict.get('occupancy', e_dict.get('occupancy_state', e_dict.get('Occupancy', 0.0)))
                    ]
                    e_vec = torch.tensor(e_list).float().unsqueeze(0)
                    embeddings['env'] = self.env_encoder(e_vec)
                    
                # Project Vibration (dim = 15)
                if features['vibration'] is not None:
                    v_dict = features['vibration']
                    v_list = [v_dict.get(f'feature_{i}', 0.0) for i in range(15)]
                    if all(v == 0.0 for v in v_list) and 'offset' in v_dict:
                        v_list = [
                            v_dict.get('offset', 0.0), v_dict.get('max_charge', 0.0), v_dict.get('total_charge', 0.0),
                            v_dict.get('num_holes', 0.0), v_dict.get('detonator_code', 0.0), v_dict.get('trid_12', 0.0),
                            v_dict.get('trid_13', 0.0), v_dict.get('trid_14', 0.0), v_dict.get('gx', 0.0),
                            v_dict.get('gy', 0.0), v_dict.get('gelev', 0.0), v_dict.get('sx', 0.0),
                            v_dict.get('sy', 0.0), v_dict.get('selev', 0.0), v_dict.get('elevation_diff', 0.0)
                        ]
                    v_vec = torch.tensor(v_list).float().unsqueeze(0)
                    embeddings['vibration'] = self.vib_encoder(v_vec)
                else:
                    embeddings['vibration'] = torch.zeros(1, 4096)
                    
                # Project Ultrasonic (dim = 24)
                if features['ultrasonic'] is not None:
                    u_dict = features['ultrasonic']
                    u_list = [u_dict.get(f'US{i}', 0.0) for i in range(1, 25)]
                    if all(u == 0.0 for u in u_list) and 'SD_front' in u_dict:
                        u_list = [u_dict.get('SD_front', 0.0), u_dict.get('SD_left', 0.0)] + [0.0] * 22
                    u_vec = torch.tensor(u_list).float().unsqueeze(0)
                    embeddings['ultrasonic'] = self.ultra_encoder(u_vec)
                    
            output['aligned_embeddings'] = embeddings
            
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
