"""
train_mine_severity_dl.py — Deep Learning (PyTorch MLP) Per-Gas Severity Classifiers

Trains 4 PyTorch Deep Neural Network classifiers (CH4, CO, CO2, H2) mapping ppm -> severity:
  - L1 (severity=0): Normal/Low
  - L2 (severity=1): Warning
  - L3 (severity=2): Critical

Outputs: gas_sensors/models/severity_{gas}.joblib (4 scikit-learn compatible wrappers)

References: docs/REAL_MINE_DATA_RETRAINING.md §5.2
"""
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from dl_wrappers import PyTorchSeverityClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "data")
MODELS_DIR = os.path.join(script_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# TLV thresholds per gas (percent), from verified Part 2 headers
TLV_PCT = {"CH4": 2.5, "CO": 0.005, "CO2": 0.03, "H2": 2.0}
TLV_PPM = {k: v * 10_000 for k, v in TLV_PCT.items()}

BAND_EDGES = {
    "CH4": {"L1": (0, 12500), "L2": (12500, 18750), "L3": (18750, 25000)},
    "CO":  {"L1": (0, 37.5),  "L2": (37.5, 50),     "L3": (50, 10000)},
    "CO2": {"L1": (0, 400),   "L2": (400, 1000),     "L3": (1000, 5000)},
    "H2":  {"L1": (0, 18000), "L2": (18000, 25000),  "L3": (25000, 38000)},
}

def train_dl_severity():
    print("=" * 60)
    print("DEEP LEARNING (PYTORCH MLP) PER-GAS SEVERITY TRAINING")
    print("=" * 60)

    df_raw = pd.read_csv(os.path.join(DATA_DIR, "mine_part2_bands.csv"))

    registry_path = os.path.join(MODELS_DIR, "model_registry.json")
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except Exception:
        registry = {}

    summary_metrics = {}

    for gas in ["CH4", "CO", "CO2", "H2"]:
        print("\n" + "-" * 40)
        print(f"Training PyTorch Deep Severity Model: {gas}")
        print("-" * 40)

        aug_path = os.path.join(DATA_DIR, f"mine_part2_{gas.lower()}_realistic.csv")
        if os.path.exists(aug_path):
            g = pd.read_csv(aug_path)
            feature_col = "ppm_noisy" if "ppm_noisy" in g.columns else "ppm"
            print(f"  Loaded augmented dataset: {aug_path} ({feature_col})")
        else:
            g = df_raw[df_raw.gas == gas].copy()
            feature_col = "ppm"

        X = g[[feature_col]].rename(columns={feature_col: "ppm"})
        y = g["severity"]

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )

        clf = PyTorchSeverityClassifier(in_features=1, num_classes=3, epochs=50, batch_size=256, lr=2e-3)
        clf.fit(Xtr, ytr)

        y_pred = clf.predict(Xte)
        cr = classification_report(yte, y_pred, digits=4, output_dict=True)
        cr_text = classification_report(yte, y_pred, digits=4,
                                         target_names=["L1 (safe)", "L2 (warning)", "L3 (critical)"])
        print(cr_text)

        acc = cr["accuracy"]
        f1_macro = cr["macro avg"]["f1-score"]

        sweep_x = np.linspace(0, BAND_EDGES[gas]["L3"][1], 10000).reshape(-1, 1)
        sweep_preds = clf.predict(sweep_x)
        
        l1_l2_idx = np.where((sweep_preds[:-1] == 0) & (sweep_preds[1:] == 1))[0]
        l2_l3_idx = np.where((sweep_preds[:-1] == 1) & (sweep_preds[1:] == 2))[0]

        l1_l2_learned = float(sweep_x[l1_l2_idx[0]][0]) if len(l1_l2_idx) > 0 else BAND_EDGES[gas]["L1"][1]
        l2_l3_learned = float(sweep_x[l2_l3_idx[0]][0]) if len(l2_l3_idx) > 0 else BAND_EDGES[gas]["L2"][1]

        l1_l2_expected = BAND_EDGES[gas]["L1"][1]
        l2_l3_expected = BAND_EDGES[gas]["L2"][1]

        print(f"  Learned L1->L2 threshold: {l1_l2_learned:.2f} ppm (expected: {l1_l2_expected})")
        print(f"  Learned L2->L3 threshold: {l2_l3_learned:.2f} ppm (expected: {l2_l3_expected})")

        model_path = os.path.join(MODELS_DIR, f"severity_{gas.lower()}.joblib")
        joblib.dump(clf, model_path)
        print(f"  Saved Deep Learning model to: {model_path}")

        summary_metrics[gas] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "l1_l2_learned": l1_l2_learned,
            "l2_l3_learned": l2_l3_learned
        }

        registry[f"severity_{gas.lower()}"] = {
            "task_type": "multiclass_classification",
            "model_type": "PyTorch_Deep_MLP",
            "model_path": model_path,
            "features": ["ppm"],
            "targets": ["severity"],
            "train_shape": list(Xtr.shape),
            "test_shape": list(Xte.shape),
            "metrics": {
                "accuracy": acc,
                "macro_precision": cr["macro avg"]["precision"],
                "macro_recall": cr["macro avg"]["recall"],
                "macro_f1": f1_macro,
                "learned_L1_L2_threshold_ppm": l1_l2_learned,
                "learned_L2_L3_threshold_ppm": l2_l3_learned,
                "expected_L1_L2_threshold_ppm": float(l1_l2_expected),
                "expected_L2_L3_threshold_ppm": float(l2_l3_expected),
                "tlv_ppm": float(TLV_PPM[gas]),
            },
            "remarks": f"PyTorch Deep Neural Network (4-layer MLP + BatchNorm + GELU + Dropout) trained on real mine data for {gas}.",
            "training_time_sec": 0.0,
            "trained_at": datetime.now().isoformat()
        }

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)
    print("\n" + "=" * 60)
    print("DEEP LEARNING SEVERITY TRAINING COMPLETE")
    print("=" * 60)
    for gas, metrics in summary_metrics.items():
        print(f"  {gas:<5} Accuracy: {metrics['accuracy']*100:.2f}% | F1-Macro: {metrics['f1_macro']:.4f}")

if __name__ == "__main__":
    train_dl_severity()
