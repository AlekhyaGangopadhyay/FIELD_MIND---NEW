"""
eval_on_real_data.py — Honest evaluation of existing gas models against real mine data.

Uses Part 2 banded data (mine_part2_bands.csv) to test existing models that were
previously evaluated on synthetic-only or zero-positive test sets.

Models evaluated:
  1. gas_hazard_lpg_cng     — CH4 axis from Part 2, LPG at clean-air baseline
  2. gas_hazard_co_nox_c6h6 — CO axis from Part 2
  3. multi_gas_detector      — CH4 + CO heads from Part 2

Output: docs/REAL_DATA_EVAL.md

References: docs/REAL_MINE_DATA_RETRAINING.md §5.3
"""
import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "data")
MODELS_DIR = os.path.join(script_dir, "models")
DOCS_DIR = os.path.join(script_dir, "..", "docs")

# Load real banded data
bands = pd.read_csv(os.path.join(DATA_DIR, "mine_part2_bands.csv"))

results_md = []
results_md.append(f"# Real Data Evaluation Report\n")
results_md.append(f"**Generated:** {datetime.now().isoformat()}\n")
results_md.append(f"**Data source:** `mine_part2_bands.csv` (120,000 rows, 4 gases × 3 severity bands)\n")
results_md.append(f"**Reference:** `docs/REAL_MINE_DATA_RETRAINING.md` §5.3\n")
results_md.append("")

all_results = {}

# ──────────────────────────────────────────────────────────────────────
# 1. gas_hazard_lpg_cng (2-feature: MQ2_LPG_ppm, MQ4_CH4_ppm)
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Evaluating gas_hazard_lpg_cng on real CH4 bands")
print("=" * 60)

model_path = os.path.join(MODELS_DIR, "gas_hazard_lpg_cng.joblib")
if os.path.exists(model_path):
    m1 = joblib.load(model_path)
    ch4 = bands[bands.gas == "CH4"].copy()

    # Drive CH4 axis from real banded data; hold LPG at clean-air baseline
    X1 = pd.DataFrame({
        "MQ2_LPG_ppm": 80.0,
        "MQ4_CH4_ppm": ch4["ppm"].values
    })

    # Ground truth: OSHA/MSHA methane action level 1000 ppm (10% LEL)
    y_true_1 = (ch4["ppm"].values > 1000).astype(int)

    y_pred_1 = m1.predict(X1)

    cm1 = confusion_matrix(y_true_1, y_pred_1)
    cr1 = classification_report(y_true_1, y_pred_1, digits=4, output_dict=True)
    cr1_text = classification_report(y_true_1, y_pred_1, digits=4)

    print(f"Positives in real test set: {y_true_1.sum()} of {len(y_true_1)}")
    print(f"(Old test set had 0 positives out of 84)\n")
    print("Confusion Matrix:")
    print(cm1)
    print("\nClassification Report:")
    print(cr1_text)

    all_results["gas_hazard_lpg_cng"] = {
        "positives_in_test": int(y_true_1.sum()),
        "total_test": len(y_true_1),
        "accuracy": cr1["accuracy"],
        "precision": cr1["1"]["precision"],
        "recall": cr1["1"]["recall"],
        "f1": cr1["1"]["f1-score"],
        "confusion_matrix": cm1.tolist()
    }

    results_md.append("## 1. `gas_hazard_lpg_cng` — LPG/CNG Hazard Classifier\n")
    results_md.append(f"- **Test data:** Part 2 CH4 bands (30,000 rows), LPG held at 80 ppm baseline")
    results_md.append(f"- **Ground truth:** `MQ4_CH4_ppm > 1000` (10% LEL, per OSHA/MSHA)")
    results_md.append(f"- **Positives:** {y_true_1.sum()} of {len(y_true_1)} ({y_true_1.mean()*100:.1f}%)")
    results_md.append(f"- **Old test set:** 0 positives out of 84 → acc=1.0, precision/recall/f1=0.0\n")
    results_md.append("| Metric | Old (synthetic) | New (real data) |")
    results_md.append("|--------|----------------|-----------------|")
    results_md.append(f"| Accuracy | 1.0000 | {cr1['accuracy']:.4f} |")
    results_md.append(f"| Precision | 0.0000 | {cr1['1']['precision']:.4f} |")
    results_md.append(f"| Recall | 0.0000 | {cr1['1']['recall']:.4f} |")
    results_md.append(f"| F1-Score | 0.0000 | {cr1['1']['f1-score']:.4f} |")
    results_md.append(f"\n**Confusion Matrix:**\n")
    results_md.append("```")
    results_md.append(f"TN={cm1[0][0]:>6}  FP={cm1[0][1]:>6}")
    results_md.append(f"FN={cm1[1][0]:>6}  TP={cm1[1][1]:>6}")
    results_md.append("```\n")
else:
    print("Model not found, skipping.\n")

# ──────────────────────────────────────────────────────────────────────
# 2. gas_hazard_co_nox_c6h6 (3-feature: MQ7_CO_ppm, MQ135_NOx_ppm, MQ3_Benzene_ppm)
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("2. Evaluating gas_hazard_co_nox_c6h6 on real CO bands")
print("=" * 60)

model_path = os.path.join(MODELS_DIR, "gas_hazard_co_nox_c6h6.joblib")
if os.path.exists(model_path):
    m2 = joblib.load(model_path)
    co = bands[bands.gas == "CO"].copy()

    # Drive CO axis from real banded data; hold NOx and Benzene at baseline
    X2 = pd.DataFrame({
        "MQ7_CO_ppm": co["ppm"].values,
        "MQ135_NOx_ppm": 0.04,       # baseline from generate_dataset.py
        "MQ3_Benzene_ppm": 4.0,      # baseline from generate_dataset.py
    })

    # Ground truth: CO > 50 ppm (OSHA TWA)
    y_true_2 = (co["ppm"].values > 50).astype(int)

    y_pred_2 = m2.predict(X2)

    cm2 = confusion_matrix(y_true_2, y_pred_2)
    cr2 = classification_report(y_true_2, y_pred_2, digits=4, output_dict=True)
    cr2_text = classification_report(y_true_2, y_pred_2, digits=4)

    print(f"Positives in real test set: {y_true_2.sum()} of {len(y_true_2)}")
    print("\nConfusion Matrix:")
    print(cm2)
    print("\nClassification Report:")
    print(cr2_text)

    all_results["gas_hazard_co_nox_c6h6"] = {
        "positives_in_test": int(y_true_2.sum()),
        "total_test": len(y_true_2),
        "accuracy": cr2["accuracy"],
        "precision": cr2["1"]["precision"],
        "recall": cr2["1"]["recall"],
        "f1": cr2["1"]["f1-score"],
        "confusion_matrix": cm2.tolist()
    }

    results_md.append("---\n")
    results_md.append("## 2. `gas_hazard_co_nox_c6h6` — CO/NOx/Benzene Hazard Classifier\n")
    results_md.append(f"- **Test data:** Part 2 CO bands (30,000 rows), NOx=0.04, Benzene=4.0 (baselines)")
    results_md.append(f"- **Ground truth:** `MQ7_CO_ppm > 50` (OSHA TWA)")
    results_md.append(f"- **Positives:** {y_true_2.sum()} of {len(y_true_2)} ({y_true_2.mean()*100:.1f}%)\n")
    results_md.append("| Metric | Old (synthetic) | New (real data) |")
    results_md.append("|--------|----------------|-----------------|")
    results_md.append(f"| Accuracy | 0.9178 | {cr2['accuracy']:.4f} |")
    results_md.append(f"| Precision | 0.2472 | {cr2['1']['precision']:.4f} |")
    results_md.append(f"| Recall | 1.0000 | {cr2['1']['recall']:.4f} |")
    results_md.append(f"| F1-Score | 0.3964 | {cr2['1']['f1-score']:.4f} |")
    results_md.append(f"\n**Confusion Matrix:**\n")
    results_md.append("```")
    results_md.append(f"TN={cm2[0][0]:>6}  FP={cm2[0][1]:>6}")
    results_md.append(f"FN={cm2[1][0]:>6}  TP={cm2[1][1]:>6}")
    results_md.append("```\n")
else:
    print("Model not found, skipping.\n")

# ──────────────────────────────────────────────────────────────────────
# 3. multi_gas_detector — CH4 and CO heads
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("3. Evaluating multi_gas_detector (Methane + CO heads) on real bands")
print("=" * 60)

model_path = os.path.join(MODELS_DIR, "multi_gas_detector.joblib")
if os.path.exists(model_path):
    m3 = joblib.load(model_path)

    results_md.append("---\n")
    results_md.append("## 3. `multi_gas_detector` — Multi-Gas Presence Classifier\n")
    results_md.append("Testing Methane and CO heads against real banded data.\n")

    # multi_gas_detector uses MQ2 features: MQ2_CO_ppm, MQ2_LPG_ppm, MQ2_Smoke_ppm
    # Test CH4 head: drive MQ2_LPG_ppm from Part 2 CH4 (MQ-2 reads ~85% of MQ-4 for CH4)
    for gas_name, gas_key, feature_col, thresh, head_idx in [
        ("Methane (CH4)", "CH4", "MQ2_LPG_ppm", 117.5, 0),
        ("CO", "CO", "MQ2_CO_ppm", 15.0, 1),
    ]:
        print(f"\n--- {gas_name} head ---")
        g = bands[bands.gas == gas_key].copy()

        if gas_key == "CH4":
            # MQ-2 reads ~85% of true CH4 ppm (from generate_dataset.py)
            X3 = pd.DataFrame({
                "MQ2_CO_ppm": 12.0,        # baseline
                "MQ2_LPG_ppm": g["ppm"].values * 0.85,
                "MQ2_Smoke_ppm": 65.0,     # baseline
            })
        else:  # CO
            X3 = pd.DataFrame({
                "MQ2_CO_ppm": g["ppm"].values * 0.92,  # MQ-2 reads 92% of CO
                "MQ2_LPG_ppm": 110.0,     # baseline
                "MQ2_Smoke_ppm": 65.0,     # baseline
            })

        # Ground truth: same thresholds as generate_dataset.py
        y_true_3 = (g["ppm"].values > thresh).astype(int) if gas_key == "CH4" else (g["ppm"].values > 50).astype(int)

        y_pred_3 = m3.predict(X3)[:, head_idx]

        cm3 = confusion_matrix(y_true_3, y_pred_3)
        cr3 = classification_report(y_true_3, y_pred_3, digits=4, output_dict=True)
        cr3_text = classification_report(y_true_3, y_pred_3, digits=4)

        print(f"Positives: {y_true_3.sum()} of {len(y_true_3)}")
        print("Confusion Matrix:")
        print(cm3)
        print(cr3_text)

        all_results[f"multi_gas_detector_{gas_key}"] = {
            "positives_in_test": int(y_true_3.sum()),
            "total_test": len(y_true_3),
            "accuracy": cr3["accuracy"],
            "precision": cr3["1"]["precision"],
            "recall": cr3["1"]["recall"],
            "f1": cr3["1"]["f1-score"],
        }

        results_md.append(f"### {gas_name} Head\n")
        results_md.append(f"- **Positives:** {y_true_3.sum()} of {len(y_true_3)}")
        results_md.append(f"- **Accuracy:** {cr3['accuracy']:.4f}")
        results_md.append(f"- **Precision:** {cr3['1']['precision']:.4f}")
        results_md.append(f"- **Recall:** {cr3['1']['recall']:.4f}")
        results_md.append(f"- **F1:** {cr3['1']['f1-score']:.4f}\n")
        results_md.append("```")
        results_md.append(f"TN={cm3[0][0]:>6}  FP={cm3[0][1]:>6}")
        results_md.append(f"FN={cm3[1][0]:>6}  TP={cm3[1][1]:>6}")
        results_md.append("```\n")
else:
    print("Model not found, skipping.\n")

# ──────────────────────────────────────────────────────────────────────
# Write results
# ──────────────────────────────────────────────────────────────────────
results_md.append("---\n")
results_md.append("## Summary\n")
results_md.append("| Model | Positives | Accuracy | Precision | Recall | F1 |")
results_md.append("|-------|-----------|----------|-----------|--------|-----|")
for name, r in all_results.items():
    results_md.append(
        f"| `{name}` | {r['positives_in_test']}/{r['total_test']} "
        f"| {r['accuracy']:.4f} | {r['precision']:.4f} "
        f"| {r['recall']:.4f} | {r['f1']:.4f} |"
    )

results_md.append("\n> **Conclusion:** These results show how each model performs when it finally sees real positive samples. Models that reported perfect accuracy on zero-positive test sets now face genuine hazard data.\n")

os.makedirs(DOCS_DIR, exist_ok=True)
eval_path = os.path.join(DOCS_DIR, "REAL_DATA_EVAL.md")
with open(eval_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results_md))
print(f"\nResults written to: {eval_path}")

# Also save raw results as JSON for downstream use
json_path = os.path.join(DATA_DIR, "real_data_eval_results.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"JSON results saved to: {json_path}")
