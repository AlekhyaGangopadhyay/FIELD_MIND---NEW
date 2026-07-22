"""
make_agent_replay_from_real.py — Build agent replay dataset from real mine data.

Creates FIELDMIND_real_replay.csv with the same schema as FIELDMIND_physics_dataset.csv
so it can be a drop-in replacement for the gas agent's experience replay buffer.

Sources:
  - Part 2 CH4 bands -> MQ4_CH4_ppm (shuffled, i.i.d.)
  - Part 1 steady-state -> Temp_C, Humidity_pct (cycled to match length)
  - Clean-air baseline -> MQ2_LPG_ppm (80 +/- 5 ppm)
  - Hazard label -> ppm > 1000 (10% LEL, per generate_dataset.py)

Output: gas_sensors/data/FIELDMIND_real_replay.csv

References: docs/REAL_MINE_DATA_RETRAINING.md section 5.4
"""
import os
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = script_dir

print("=" * 60)
print("GENERATING REAL DATA REPLAY FOR GAS AGENT")
print("=" * 60)

# Load real data
bands = pd.read_csv(os.path.join(DATA_DIR, "mine_part2_bands.csv"))
base = pd.read_csv(os.path.join(DATA_DIR, "mine_part1_clean.csv"), parse_dates=["timestamp"])
base = base[~base.is_warmup]

# Shuffle CH4 data (no temporal structure in Part 2 anyway)
ch4 = bands[bands.gas == "CH4"].sample(frac=1, random_state=42).reset_index(drop=True)
n = len(ch4)

# Real Part-1 humidity/temp cycled to length — gives the agent our true env envelope
env = base[["t", "h"]].dropna()
env = env.iloc[np.arange(n) % len(env)].reset_index(drop=True)

rng = np.random.default_rng(42)

out = pd.DataFrame({
    "Timestamp":     pd.date_range("2024-01-01 06:00:00", periods=n, freq="1s"),
    "MQ2_LPG_ppm":   80.0 + rng.normal(0, 5, n),          # clean-air baseline + noise
    "MQ2_CH4_ppm":   ch4["ppm"].values * 0.85,             # MQ-2 reads ~85% of CH4
    "MQ2_CO_ppm":    12.0 + rng.normal(0, 2, n),           # baseline
    "MQ2_Smoke_ppm": 65.0 + rng.normal(0, 5, n),           # baseline
    "MQ3_Alcohol_ppm": 8.0 + rng.normal(0, 1, n),          # baseline
    "MQ3_Benzene_ppm": 4.0 + rng.normal(0, 0.5, n),       # baseline
    "MQ4_CH4_ppm":   ch4["ppm"].values,                     # real CH4 bands
    "MQ7_CO_ppm":    12.0 + rng.normal(0, 3, n),           # baseline
    "MQ135_NH3_ppm": 7.0 + rng.normal(0, 1, n),            # baseline
    "MQ135_NOx_ppm": 0.04 + rng.normal(0, 0.005, n),      # baseline
    "MQ135_CO2_ppm": 1800.0 + rng.normal(0, 50, n),       # baseline
    "MQ136_H2S_ppm": 2.0 + rng.normal(0, 0.3, n),         # baseline
    "MG811_CO2_ppm": 1800.0 + rng.normal(0, 50, n),       # baseline
    "PM25_Dust_ugm3": 35.0 + rng.normal(0, 5, n),         # baseline
    "Temp_C":        env["t"].values,                       # real temperature
    "Humidity_pct":  env["h"].values,                       # real humidity
    "Blast_Event":   0,                                     # no blast in this replay
    "Hazard_Alert":  (ch4["ppm"].values > 1000).astype(int),  # 10% LEL
})

# Clip all ppm values to non-negative
for col in out.columns:
    if "ppm" in col or "ugm3" in col:
        out[col] = out[col].clip(lower=0)

out_path = os.path.join(DATA_DIR, "FIELDMIND_real_replay.csv")
out.to_csv(out_path, index=False)

print(f"\nReplay dataset written: {out_path}")
print(f"Shape: {out.shape}")
print(f"Columns: {list(out.columns)}")
print(f"\nHazard alert distribution:")
print(out["Hazard_Alert"].value_counts().sort_index())
print(f"\nCH4 ppm stats:")
print(out["MQ4_CH4_ppm"].describe().round(1))
print(f"\nTemp/Humidity (from real Part 1):")
print(out[["Temp_C", "Humidity_pct"]].describe().round(1))
