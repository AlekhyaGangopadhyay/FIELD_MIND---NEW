import time
import torch
import numpy as np
from encoders import GasEncoder, EnvironmentalEncoder, VibrationEncoder, UltrasonicEncoder
from alignment import TemporalAligner

def simulate_sensor_data(start_time, duration, interval=1.0):
    """
    Simulates multi-sensor streams with varying frequencies over a duration.
    """
    timestamps = np.arange(start_time, start_time + duration, interval)
    
    # 1. Gas stream (Methane, CO, LPG, Smoke, NOx, CO2) - sampled every 2 seconds
    gas_data = []
    for t in np.arange(start_time, start_time + duration, 2.0):
        gas_data.append({
            'timestamp': t,
            'methane': float(np.random.uniform(0.1, 1.2)),
            'co': float(np.random.uniform(10, 50)),
            'lpg': float(np.random.uniform(0.01, 0.1)),
            'smoke': float(np.random.uniform(0.0, 0.05)),
            'nox': float(np.random.uniform(10, 30)),
            'co2': float(np.random.uniform(350, 450))
        })
        
    # 2. Environmental stream (Temp, Humidity, Pressure, Occupancy) - sampled every 5 seconds
    env_data = []
    for t in np.arange(start_time, start_time + duration, 5.0):
        env_data.append({
            'timestamp': t,
            'temperature': float(np.random.uniform(22.0, 31.0)),
            'humidity': float(np.random.uniform(40.0, 80.0)),
            'pressure': float(np.random.uniform(1008.0, 1015.0)),
            'occupancy': float(np.random.choice([0.0, 1.0]))
        })
        
    # 3. Vibration stream (15 features) - sampled irregularly, e.g. at blast events
    vibration_data = []
    # Event at t=3.5 and t=7.2
    for t in [start_time + 3.5, start_time + 7.2]:
        vibration_data.append({
            'timestamp': t,
            **{f'feature_{i}': float(np.random.normal(0, 1)) for i in range(15)}
        })
        
    # 4. Ultrasonic stream (24 raw sensors) - high frequency, e.g. sampled every 0.1 seconds
    ultrasonic_data = []
    for t in np.arange(start_time, start_time + duration, 0.1):
        ultrasonic_data.append({
            'timestamp': t,
            **{f'US{i}': float(np.random.uniform(0.3, 5.0)) for i in range(1, 25)}
        })
        
    return {
        'gas': gas_data,
        'env': env_data,
        'vibration': vibration_data,
        'ultrasonic': ultrasonic_data
    }

def main():
    print("="*60)
    # Corrected SciSense Protocol title in print
    print("SCISENSE PROTOCOL - UNIFIED ALIGNMENT SPACE DEMO RUNNER")
    print("="*60)
    
    # 1. Initialize Encoders
    print("Initializing modality-specific projection encoders...")
    gas_encoder = GasEncoder(input_dim=6, hidden_dim=128, embedding_dim=4096)
    env_encoder = EnvironmentalEncoder(input_dim=4, hidden_dim=128, embedding_dim=4096)
    vib_encoder = VibrationEncoder(input_dim=15, hidden_dim=256, embedding_dim=4096)
    ultra_encoder = UltrasonicEncoder(input_dim=24, hidden_dim=256, embedding_dim=4096)
    
    # Ensure they are in evaluation mode
    gas_encoder.eval()
    env_encoder.eval()
    vib_encoder.eval()
    ultra_encoder.eval()
    
    # 2. Simulate Sensor Streams
    start_time = time.time()
    duration = 10.0 # 10 seconds of data
    print(f"Simulating 10 seconds of heterogeneous sensor streams starting at epoch: {start_time:.2f}...")
    streams = simulate_sensor_data(start_time, duration)
    
    print(f"  Gas samples:         {len(streams['gas'])}")
    print(f"  Environmental:       {len(streams['env'])}")
    print(f"  Vibration events:    {len(streams['vibration'])}")
    print(f"  Ultrasonic frames:   {len(streams['ultrasonic'])}")
    
    # 3. Synchronize Temporal Alignment
    print("\nRunning Temporal Aligner (1-second epochs)...")
    aligner = TemporalAligner(time_window_seconds=1.0)
    aligned_epochs = aligner.align_streams(streams, start_time, start_time + duration)
    print(f"  Generated {len(aligned_epochs)} aligned multi-modal intervals.")
    
    # 4. Generate Embeddings for each epoch
    print("\nProjecting aligned epochs to 4096-dimensional SciSense space:")
    print("-" * 80)
    
    with torch.no_grad():
        for i, epoch in enumerate(aligned_epochs):
            t_start = epoch['timestamp_start'] - start_time
            t_end = epoch['timestamp_end'] - start_time
            features = epoch['aligned_features']
            
            print(f"Epoch {i+1} [{t_start:.1f}s - {t_end:.1f}s]:")
            
            # A. Gas Embedding
            if features['gas'] is not None:
                gas_vector = torch.tensor(list(features['gas'].values())).unsqueeze(0)
                gas_emb = gas_encoder(gas_vector)
                print(f"  Gas Embedding Shape:          {list(gas_emb.shape)} | L2 Norm: {torch.norm(gas_emb).item():.2f}")
            else:
                print("  Gas: No reading")
                
            # B. Environmental Embedding
            if features['env'] is not None:
                env_vector = torch.tensor(list(features['env'].values())).unsqueeze(0)
                env_emb = env_encoder(env_vector)
                print(f"  Environmental Embedding:      {list(env_emb.shape)} | L2 Norm: {torch.norm(env_emb).item():.2f}")
            else:
                print("  Environmental: No reading")
                
            # C. Vibration Embedding
            if features['vibration'] is not None:
                vib_vector = torch.tensor(list(features['vibration'].values())).unsqueeze(0)
                vib_emb = vib_encoder(vib_vector)
                print(f"  Vibration Embedding:          {list(vib_emb.shape)} | L2 Norm: {torch.norm(vib_emb).item():.2f}")
            else:
                # Use a zero pad tensor if no active vibration reading
                zero_vib = torch.zeros(1, 4096)
                print(f"  Vibration (Zero-padded):       {list(zero_vib.shape)} | L2 Norm: {torch.norm(zero_vib).item():.2f}")
                
            # D. Ultrasonic Embedding
            if features['ultrasonic'] is not None:
                ultra_vector = torch.tensor(list(features['ultrasonic'].values())).unsqueeze(0)
                ultra_emb = ultra_encoder(ultra_vector)
                print(f"  Ultrasonic Embedding:         {list(ultra_emb.shape)} | L2 Norm: {torch.norm(ultra_emb).item():.2f}")
            else:
                print("  Ultrasonic: No reading")
                
            # Example Cosine Similarity fusion checking
            if features['gas'] is not None and features['ultrasonic'] is not None:
                similarity = torch.mm(gas_emb, ultra_emb.t()).item()
                print(f"  Cosine Similarity (Gas <-> Ultra): {similarity:.4f}")
                
            print("-" * 80)
            
    print("\nSciSense Alignment Pipeline test completed successfully!")

if __name__ == "__main__":
    main()
