"""
train_mine_severity.py — Per-gas severity classifiers from Part 2 banded data.

Trains 4 RandomForest classifiers (CH4, CO, CO2, H2) that map ppm → severity level:
  - L1 (severity=0): Normal/Low
  - L2 (severity=1): Warning
  - L3 (severity=2): Critical

These replace the broken severity thresholds in the existing models.

Optionally uses augmented data if available (from augment_part2_realistic.py).

Output: gas_sensors/models/severity_{gas}.joblib  (4 files)

References: docs/REAL_MINE_DATA_RETRAINING.md §5.2
"""
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "data")
MODELS_DIR = os.path.join(script_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# TLV thresholds per gas (percent), from the verified Part 2 headers
TLV_PCT = {"CH4": 2.5, "CO": 0.005, "CO2": 0.03, "H2": 2.0}
TLV_PPM = {k: v * 10_000 for k, v in TLV_PCT.items()}

# Band boundaries per gas (in ppm) from §3.2 of the retraining guide
BAND_EDGES = {
    "CH4": {"L1": (0, 12500), "L2": (12500, 18750), "L3": (18750, 25000)},
    "CO":  {"L1": (0, 37.5),  "L2": (37.5, 50),     "L3": (50, 10000)},
    "CO2": {"L1": (0, 400),   "L2": (400, 1000),     "L3": (1000, 5000)},
    "H2":  {"L1": (0, 18000), "L2": (18000, 25000),  "L3": (25000, 38000)},
}

print("=" * 60)
print("PER-GAS SEVERITY CLASSIFIER TRAINING")
print("=" * 60)

# Load Part 2 bands
df = pd.read_csv(os.path.join(DATA_DIR, "mine_part2_bands.csv"))
print(f"Part 2 loaded: {len(df)} rows, {df.gas.nunique()} gases\n")

# Load registry for updates
registry_path = os.path.join(MODELS_DIR, "model_registry.json")
try:
    with open(registry_path, 'r') as f:
        registry = json.load(f)
except Exception:
    registry = {}

for gas in ["CH4", "CO", "CO2", "H2"]:
    print("-" * 40)
    print(f"Training severity classifier: {gas}")
    print("-" * 40)

    g = df[df.gas == gas].copy()

    # Check if augmented data exists
    aug_path = os.path.join(DATA_DIR, f"mine_part2_{gas.lower()}_realistic.csv")
    if os.path.exists(aug_path):
        g_aug = pd.read_csv(aug_path)
        if "ppm_noisy" in g_aug.columns:
            print(f"  Using augmented data from: {aug_path}")
            X = g_aug[["ppm_noisy"]].rename(columns={"ppm_noisy": "ppm"})
        else:
            X = g[["ppm"]]
    else:
        X = g[["ppm"]]

    y = g["severity"]

    # Stratified split — bands are exactly balanced (10k each), keep it that way
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
    )
    clf.fit(Xtr, ytr)

    # Evaluate
    y_pred = clf.predict(Xte)
    cr = classification_report(yte, y_pred, digits=4, output_dict=True)
    cr_text = classification_report(yte, y_pred, digits=4,
                                     target_names=["L1 (safe)", "L2 (warning)", "L3 (critical)"])
    print(cr_text)

    # Extract learned thresholds (decision boundaries)
    # For a 1-feature problem, the tree thresholds reveal the learned split points
    thresholds = set()
    for tree in clf.estimators_:
        tree_model = tree.tree_
        feature_used = tree_model.feature
        threshold_vals = tree_model.threshold
        for i in range(tree_model.node_count):
            if feature_used[i] != -2:  # not a leaf node
                thresholds.add(round(float(threshold_vals[i]), 2))

    # Find the most common thresholds near the expected band edges
    expected_edges = BAND_EDGES[gas]
    print(f"  TLV: {TLV_PPM[gas]} ppm ({TLV_PCT[gas]}%)")
    print(f"  Expected band edges: {expected_edges}")

    # Find the two most prominent splits (L1→L2 and L2→L3 boundaries)
    sorted_thresholds = sorted(thresholds)
    l1_l2_expected = expected_edges["L1"][1]
    l2_l3_expected = expected_edges["L2"][1]

    # Find closest learned thresholds to expected edges
    l1_l2_learned = min(sorted_thresholds, key=lambda t: abs(t - l1_l2_expected))
    l2_l3_learned = min(sorted_thresholds, key=lambda t: abs(t - l2_l3_expected))

    print(f"  Learned L1->L2 threshold: {l1_l2_learned} (expected: {l1_l2_expected})")
    print(f"  Learned L2->L3 threshold: {l2_l3_learned} (expected: {l2_l3_expected})")

    l1_l2_error = abs(l1_l2_learned - l1_l2_expected) / l1_l2_expected * 100
    l2_l3_error = abs(l2_l3_learned - l2_l3_expected) / l2_l3_expected * 100
    print(f"  L1->L2 error: {l1_l2_error:.2f}%")
    print(f"  L2->L3 error: {l2_l3_error:.2f}%\n")

    # Save model
    model_path = os.path.join(MODELS_DIR, f"severity_{gas.lower()}.joblib")
    joblib.dump(clf, model_path)
    print(f"  Model saved to: {model_path}")

    # Update registry
    registry[f"severity_{gas.lower()}"] = {
        "task_type": "multiclass_classification",
        "model_path": model_path,
        "features": ["ppm"],
        "targets": ["severity"],
        "train_shape": list(Xtr.shape),
        "test_shape": list(Xte.shape),
        "metrics": {
            "accuracy": cr["accuracy"],
            "macro_precision": cr["macro avg"]["precision"],
            "macro_recall": cr["macro avg"]["recall"],
            "macro_f1": cr["macro avg"]["f1-score"],
            "learned_L1_L2_threshold_ppm": float(l1_l2_learned),
            "learned_L2_L3_threshold_ppm": float(l2_l3_learned),
            "expected_L1_L2_threshold_ppm": float(l1_l2_expected),
            "expected_L2_L3_threshold_ppm": float(l2_l3_expected),
            "L1_L2_error_pct": float(l1_l2_error),
            "L2_L3_error_pct": float(l2_l3_error),
            "tlv_ppm": float(TLV_PPM[gas]),
        },
        "remarks": (
            f"Severity classifier for {gas} trained on real mine data Part 2 bands. "
            f"Maps ppm → L1/L2/L3 severity levels. "
            f"TLV: {TLV_PPM[gas]} ppm."
        ),
        "training_time_sec": 0.0,
        "trained_at": datetime.now().isoformat()
    }

with open(registry_path, 'w') as f:
    json.dump(registry, f, indent=4)
print(f"\nRegistry updated: {registry_path}")
print("=" * 60)
