import os
import sys
import time
import pandas as pd
import numpy as np

# Temporarily add project root to path to import gas loaders
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from gas_sensors.data_loader import load_smoke_dataset, load_mq4_dataset
from orchestrator import ATROrchestrator
from detector_wrappers import Tier1Monitor

def load_real_data_samples(workspace_root):
    """
    Attempts to load real samples from local datasets and searches for
    rows that are verified as normal/hazard by the pre-trained models.
    """
    samples = {}
    monitor = Tier1Monitor(workspace_root)
    
    # 1. Environmental sample (Isolation Forest & Occupancy RF)
    env_csv = os.path.join(workspace_root, "temperature_humidity", "data", "data_clean", "iot_telemetry_clean.csv")
    uci_csv = os.path.join(workspace_root, "temperature_humidity", "data", "data_clean", "uci_train_clean.csv")
    
    if os.path.exists(env_csv):
        try:
            df = pd.read_csv(env_csv)
            # Find env normal and anomaly rows verified by the loaded model
            normal_env = None
            anom_env = None
            for idx in range(min(1000, len(df))):
                row_dict = df.iloc[idx].to_dict()
                eval_res = monitor.evaluate_env(row_dict)
                if eval_res.get('anomaly_detected') == 0 and normal_env is None:
                    normal_env = row_dict
                if eval_res.get('anomaly_detected') == 1 and anom_env is None:
                    anom_env = row_dict
                if normal_env is not None and anom_env is not None:
                    break
            
            samples['env_normal'] = normal_env if normal_env is not None else df.iloc[0].to_dict()
            samples['env_anomaly'] = anom_env if anom_env is not None else df.iloc[0].to_dict()
            print("  Loaded real IoT environmental telemetry samples.")
        except Exception as e:
            print(f"  Warning loading env sample: {e}")
            
    if os.path.exists(uci_csv):
        try:
            df = pd.read_csv(uci_csv)
            feats = df.drop(columns=['date', 'Occupancy'], errors='ignore').columns.tolist()
            samples['occupancy_normal'] = df[df['Occupancy'] == 0].iloc[0][feats].tolist()
            samples['occupancy_anomaly'] = df[df['Occupancy'] == 1].iloc[0][feats].tolist()
            print("  Loaded real occupancy classification features.")
        except Exception as e:
            print(f"  Warning loading occupancy sample: {e}")

    # 2. Vibration sample (PPV regressor and RF classifier)
    vib_csv = os.path.join(workspace_root, "vibration", "data", "vibration_features.csv")
    if os.path.exists(vib_csv):
        try:
            df = pd.read_csv(vib_csv)
            samples['vib_normal'] = df[df['vibration_hazard'] == 0].iloc[0].to_dict()
            samples['vib_hazard'] = df[df['vibration_hazard'] == 1].iloc[0].to_dict()
            print("  Loaded real blast vibration features.")
        except Exception as e:
            print(f"  Warning loading vibration sample: {e}")
            
    # 3. Ultrasonic sample (navigation steer classifier)
    ultra_csv = os.path.join(workspace_root, "ultrasonic_sensors", "data", "sensor_readings_24.csv")
    if os.path.exists(ultra_csv):
        try:
            df = pd.read_csv(ultra_csv, header=None)
            cols = [f'US{i}' for i in range(1, 25)] + ['command']
            df.columns = cols
            samples['ultra_normal'] = df[df['command'] == 'Move-Forward'].iloc[0].to_dict()
            samples['ultra_hazard'] = df[df['command'] == 'Sharp-Right-Turn'].iloc[0].to_dict()
            print("  Loaded real ultrasonic navigation frames.")
        except Exception as e:
            print(f"  Warning loading ultrasonic sample: {e}")
            
    # 4. Gas loaders (Methane and Smoke/Fire datasets)
    try:
        # Load smoke dataset
        X_train, X_test, y_train, y_test = load_smoke_dataset()
        norm_smoke = X_test[y_test == 0]
        if len(norm_smoke) == 0:
            norm_smoke = X_train[y_train == 0]
        samples['smoke_normal'] = norm_smoke.iloc[0].tolist() if len(norm_smoke) > 0 else [0.01] * 36
        
        haz_smoke = X_test[y_test == 1]
        if len(haz_smoke) == 0:
            haz_smoke = X_train[y_train == 1]
        samples['smoke_hazard'] = haz_smoke.iloc[0].tolist() if len(haz_smoke) > 0 else [0.95] * 36
        
        # Load methane dataset (MQ4)
        X_train, X_test, y_train, y_test = load_mq4_dataset()
        # Find a row where MQ4 model evaluates to 0
        normal_mq4 = None
        hazard_mq4 = None
        for idx in range(min(1000, len(X_test))):
            row = X_test.iloc[idx].tolist()
            eval_res = monitor.evaluate_gas(row)
            if eval_res.get('methane_hazard') == 0 and normal_mq4 is None:
                normal_mq4 = row
            if eval_res.get('methane_hazard') == 1 and hazard_mq4 is None:
                hazard_mq4 = row
            if normal_mq4 is not None and hazard_mq4 is not None:
                break
                
        samples['mq4_normal'] = normal_mq4 if normal_mq4 is not None else X_test.iloc[0].tolist()
        samples['mq4_hazard'] = hazard_mq4 if hazard_mq4 is not None else X_test.iloc[0].tolist()
        print("  Loaded real gas Methane (MQ4) & Smoke features.")
    except Exception as e:
        print(f"  Warning loading gas datasets: {e}")

    # 5. Synthetic gas scalars (FIELDMIND physics dataset)
    syn_csv = os.path.join(workspace_root, "gas_sensors", "data", "FIELDMIND_physics_dataset.csv")
    if os.path.exists(syn_csv):
        try:
            df = pd.read_csv(syn_csv)
            # Find a normal row verified by models
            normal_gas = None
            hazard_gas = None
            for idx in range(min(1000, len(df))):
                row_dict = df.iloc[idx].to_dict()
                eval_res = monitor.evaluate_gas(row_dict)
                if eval_res.get('lpg_hazard') == 0 and eval_res.get('co_nox_hazard') == 0 and normal_gas is None:
                    normal_gas = row_dict
                if (eval_res.get('lpg_hazard') == 1 or eval_res.get('co_nox_hazard') == 1) and hazard_gas is None:
                    hazard_gas = row_dict
                if normal_gas is not None and hazard_gas is not None:
                    break
                    
            samples['gas_normal_scalars'] = normal_gas if normal_gas is not None else df[df['Hazard_Alert'] == 0].iloc[0].to_dict()
            samples['gas_hazard_scalars'] = hazard_gas if hazard_gas is not None else df[df['Hazard_Alert'] == 1].iloc[0].to_dict()
            print("  Loaded real synthetic gas scalar data samples.")
        except Exception as e:
            print(f"  Warning loading synthetic gas scalars: {e}")
            
    return samples

def main():
    print("="*80)
    print("ANOMALY-TRIGGERED REASONING (ATR) - LAYER 2A DEPLOYMENT RUNNER")
    print("="*80)
    
    # 1. Initialize ATR Orchestrator
    orchestrator = ATROrchestrator(workspace_root)
    
    # 2. Try loading real data samples
    print("\nAttempting to ingest real data slices from workspace database...")
    real_samples = load_real_data_samples(workspace_root)
    
    # Extract fallback values if any loads failed
    env_norm_vals = real_samples.get('env_normal', {
        'temp': 21.5, 'humidity': 40.0, 'temp_hum_product': 860.0, 'temp_hum_ratio': 0.53, 'humidex': 22.0,
        'temp_roll_mean_5': 21.4, 'temp_roll_std_5': 0.05, 'humidity_roll_mean_5': 40.1, 'humidity_roll_std_5': 0.1
    })
    env_norm_vals['occupancy_features'] = real_samples.get('occupancy_normal', [21.5, 40.0, 320.0, 1012.0, 0.0] * 5)[:23]
    
    env_anom_vals = real_samples.get('env_anomaly', {
        'temp': 42.0, 'humidity': 92.0, 'temp_hum_product': 3864.0, 'temp_hum_ratio': 0.45, 'humidex': 68.0,
        'temp_roll_mean_5': 40.2, 'temp_roll_std_5': 1.2, 'humidity_roll_mean_5': 88.0, 'humidity_roll_std_5': 3.5
    })
    env_anom_vals['occupancy_features'] = real_samples.get('occupancy_anomaly', [42.0, 92.0, 850.0, 1008.0, 1.0] * 5)[:23]
    
    vib_norm_vals = real_samples.get('vib_normal', {
        'offset': 1250.0, 'max_charge': 40.0, 'total_charge': 120.0, 'num_holes': 3.0, 'detonator_code': 1.0,
        'trid_12': 1.0, 'trid_13': 0.0, 'trid_14': 0.0, 'gx': 324125.0, 'gy': 7412541.0, 'gelev': 210.0,
        'sx': 324150.0, 'sy': 7412560.0, 'selev': 212.0, 'scaled_distance_usbm': 114.0, 
        'scaled_distance_langefors': 80.0, 'elevation_diff': 2.0
    })
    
    vib_haz_vals = real_samples.get('vib_hazard', {
        'offset': 100.0, 'max_charge': 2500.0, 'total_charge': 10000.0, 'num_holes': 48.0, 'detonator_code': 2.0,
        'trid_12': 1.0, 'trid_13': 0.0, 'trid_14': 0.0, 'gx': 324125.0, 'gy': 7412541.0, 'gelev': 210.0,
        'sx': 324150.0, 'sy': 7412560.0, 'selev': 212.0, 'scaled_distance_usbm': 2.0, 
        'scaled_distance_langefors': 1.1, 'elevation_diff': 2.0
    })
    
    ultra_norm_vals = real_samples.get('ultra_normal', {f'US{i}': 2.4 for i in range(1, 25)})
    ultra_haz_vals = real_samples.get('ultra_hazard', {f'US{i}': 0.12 for i in range(1, 25)})
    
    gas_norm_scalars = real_samples.get('gas_normal_scalars', {
        'MQ2_LPG_ppm': 5.0, 'MQ2_Smoke_ppm': 2.0,
        'MQ4_CH4_ppm': 10.0, 'MQ7_CO_ppm': 1.2,
        'MQ135_NOx_ppm': 0.02, 'MQ3_Benzene_ppm': 0.01,
        'PM25_Dust_ugm3': 12.0, 'Temp_C': 21.5, 'Humidity_pct': 40.0,
    })
    
    gas_haz_scalars = real_samples.get('gas_hazard_scalars', {
        'MQ2_LPG_ppm': 190.0, 'MQ2_Smoke_ppm': 180.0,
        'MQ4_CH4_ppm': 1450.0, 'MQ7_CO_ppm': 82.0,
        'MQ135_NOx_ppm': 1.5, 'MQ3_Benzene_ppm': 2.4,
        'PM25_Dust_ugm3': 180.0, 'Temp_C': 35.0, 'Humidity_pct': 70.0,
    })
    
    # 3. Simulate Timeline (10 epochs)
    print("\nStarting continuous streaming simulation...")
    print("-" * 100)
    
    start_time = time.time()
    
    for step in range(1, 11):
        timestamp = start_time + step
        
        # Defaults for this epoch (Step starts completely NORMAL)
        current_env = env_norm_vals.copy()
        current_vib = vib_norm_vals.copy()
        current_ultra = ultra_norm_vals.copy()
        
        current_gas = gas_norm_scalars.copy()
        current_gas.update({
            'mq4_features': real_samples.get('mq4_normal', [0.01] * 128),
            'smoke_features': real_samples.get('smoke_normal', [0.01] * 36),
            'air_quality_features': [21.5, 40.0, 1.2, 5.0, 10.0, 0.02, 12.0]
        })
        
        event_name = "Normal telemetry"
        
        # Inject Anomaly Events at specific steps
        if step == 3:
            event_name = "INJECT: Environmental microclimate anomaly"
            current_env = env_anom_vals.copy()
            
        elif step == 5:
            event_name = "INJECT: Hazardous toxic gas build-up"
            current_gas = gas_haz_scalars.copy()
            current_gas.update({
                'mq4_features': real_samples.get('mq4_hazard', [0.85] * 128),
                'smoke_features': real_samples.get('smoke_hazard', [0.9] * 36),
                'air_quality_features': [35.0, 70.0, 82.0, 190.0, 1450.0, 1.5, 180.0]
            })
            
        elif step == 7:
            event_name = "INJECT: Heavy seismic blast event (vibration hazard)"
            current_vib = vib_haz_vals.copy()
            
        elif step == 8:
            event_name = "INJECT: Robot proximity collision risk"
            current_ultra = ultra_haz_vals.copy()
            
        print(f"Step {step:02d} | Time: {timestamp:.2f} | {event_name}")
        
        # Ingest into orchestrator
        res = orchestrator.process_stream_frame(
            current_gas, 
            current_env, 
            current_vib, 
            current_ultra, 
            timestamp
        )
        
        # Output current monitoring state
        status = res['device_state']
        triggered = res['triggered']
        print(f"  Device Memory State: {status} | Triggered: {triggered}")
        
        # If active reasoning is loaded, display the projection metrics
        if 'aligned_embeddings' in res:
            embs = res['aligned_embeddings']
            print("  [SciSense Projections Generated]")
            for mod, tensor in embs.items():
                print(f"    - Modality: {mod:<11} | Dimension: {list(tensor.shape)} | L2 Norm: {np.linalg.norm(tensor.cpu().numpy()):.2f}")
                
        print("-" * 100)
        time.sleep(0.1) # Fast simulation speed
        
    print("\nATR Activation Pipeline test completed successfully!")

if __name__ == "__main__":
    main()
