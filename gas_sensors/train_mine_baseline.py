"""
train_mine_baseline.py — Clean-air IsolationForest from real Part 1 data.

Builds an anomaly detector fitted to our actual hardware's noise floor.
Uses Part 1 steady-state data (is_warmup=False) — this session contains
no gas event, so its correct role is a clean-air baseline / R0 reference.

This gives FIELD-MIND something it currently lacks: the ability to
distinguish a sensor fault from a gas event.

Output: gas_sensors/models/mine_baseline_iforest.joblib

References: docs/REAL_MINE_DATA_RETRAINING.md §5.1
"""
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "data")
MODELS_DIR = os.path.join(script_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("CLEAN-AIR BASELINE ANOMALY DETECTOR (IsolationForest)")
print("=" * 60)

# Load Part 1 clean data
df = pd.read_csv(os.path.join(DATA_DIR, "mine_part1_clean.csv"), parse_dates=["timestamp"])
print(f"Part 1 loaded: {len(df)} rows")

# Drop warm-up rows — section 1.5 says warm-up looks like a gas event
df = df[~df.is_warmup]
print(f"After dropping warm-up: {len(df)} rows")

# Features: all 8 sensor channels (raw ADC + physical)
FEATS = ["air_quality", "smoke", "alcohol", "flamable_gas", "MQ136_raw", "MQ7_raw", "t", "h"]
X = df[FEATS].interpolate(limit=3).dropna()
print(f"After interpolation + dropna: {len(X)} rows")
print(f"\nFeature statistics (clean-air steady state):")
print(X.describe().T[["mean", "std", "min", "max"]].round(2))

# contamination is low: this session is a clean baseline by construction.
model = make_pipeline(
    StandardScaler(),
    IsolationForest(contamination=0.01, random_state=42, n_estimators=200)
)
model.fit(X)

# Evaluate on training data (should show ~1% anomaly rate)
preds = model.predict(X)
anomaly_rate = (preds == -1).mean()
scores = model.decision_function(X)

print(f"\nBaseline anomaly rate: {anomaly_rate:.4f} ({anomaly_rate*100:.1f}%)")
print(f"Anomaly score range: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"Anomaly score mean: {scores.mean():.4f}, std: {scores.std():.4f}")

# Save model + metadata
model_data = {
    "model": model,
    "features": FEATS,
    "steady_state_stats": X.describe().T[["mean", "std", "min", "max"]].to_dict(),
    "anomaly_rate": float(anomaly_rate),
    "n_train_samples": len(X),
    "trained_at": datetime.now().isoformat(),
}
model_path = os.path.join(MODELS_DIR, "mine_baseline_iforest.joblib")
joblib.dump(model_data, model_path)
print(f"\nModel saved to: {model_path}")

# Update registry
registry_path = os.path.join(MODELS_DIR, "model_registry.json")
try:
    with open(registry_path, 'r') as f:
        registry = json.load(f)
except Exception:
    registry = {}

registry["mine_baseline_iforest"] = {
    "task_type": "anomaly_detection",
    "model_path": model_path,
    "features": FEATS,
    "targets": ["anomaly"],
    "train_shape": list(X.shape),
    "test_shape": [0, len(FEATS)],
    "metrics": {
        "anomaly_rate_on_clean_air": float(anomaly_rate),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
    },
    "remarks": (
        "IsolationForest anomaly detector trained on real clean-air baseline from "
        "mine_part1_clean.csv (steady state only, warm-up excluded). "
        "Detects deviations from our actual hardware's noise floor. "
        f"Anomaly rate on clean air: {anomaly_rate*100:.1f}%."
    ),
    "training_time_sec": 0.0,
    "trained_at": datetime.now().isoformat()
}

with open(registry_path, 'w') as f:
    json.dump(registry, f, indent=4)
print(f"Registry updated: {registry_path}")
print("=" * 60)
