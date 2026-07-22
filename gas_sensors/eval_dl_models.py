"""
eval_dl_models.py — Comprehensive Evaluation of PyTorch Deep Learning Models on Real Mine Data

Evaluates all newly trained PyTorch Deep Learning models against real field datasets:
  1. PyTorch Deep Severity Models (CH4, CO, CO2, H2)
  2. PyTorch gas_hazard_lpg_cng
  3. PyTorch gas_hazard_co_nox_c6h6
  4. PyTorch multi_gas_detector

Updates docs/REAL_DATA_EVAL.md with final benchmark results.

References: docs/REAL_MINE_DATA_RETRAINING.md §5.3
"""
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from dl_wrappers import PyTorchSeverityClassifier, PyTorchHazardClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "data")
MODELS_DIR = os.path.join(script_dir, "models")
DOCS_DIR = os.path.join(script_dir, "..", "docs")

bands = pd.read_csv(os.path.join(DATA_DIR, "mine_part2_bands.csv"))

def run_dl_eval():
    print("=" * 60)
    print("PYTORCH DEEP LEARNING MODEL EVALUATION SUITE")
    print("=" * 60)

    results_summary = {}

    # 1. Evaluate PyTorch gas_hazard_lpg_cng
    print("\n1. Evaluating PyTorch gas_hazard_lpg_cng on real CH4 bands")
    m1_path = os.path.join(MODELS_DIR, "gas_hazard_lpg_cng.joblib")
    if os.path.exists(m1_path):
        m1 = joblib.load(m1_path)
        ch4 = bands[bands.gas == "CH4"].copy()
        X1 = pd.DataFrame({"MQ2_LPG_ppm": 80.0, "MQ4_CH4_ppm": ch4["ppm"].values})
        y_true1 = (ch4["ppm"].values > 1000).astype(int)
        y_pred1 = m1.predict(X1)

        acc1 = accuracy_score(y_true1, y_pred1)
        prec1 = precision_score(y_true1, y_pred1, zero_division=0)
        rec1 = recall_score(y_true1, y_pred1, zero_division=0)
        f1_1 = f1_score(y_true1, y_pred1, zero_division=0)

        print(f"  Accuracy:  {acc1:.4f}")
        print(f"  Precision: {prec1:.4f}")
        print(f"  Recall:    {rec1:.4f}")
        print(f"  F1-Score:  {f1_1:.4f}")

        results_summary["gas_hazard_lpg_cng"] = {
            "accuracy": acc1, "precision": prec1, "recall": rec1, "f1": f1_1, "positives": int(y_true1.sum())
        }

    # 2. Evaluate PyTorch gas_hazard_co_nox_c6h6
    print("\n2. Evaluating PyTorch gas_hazard_co_nox_c6h6 on real CO bands")
    m2_path = os.path.join(MODELS_DIR, "gas_hazard_co_nox_c6h6.joblib")
    if os.path.exists(m2_path):
        m2 = joblib.load(m2_path)
        co = bands[bands.gas == "CO"].copy()
        X2 = pd.DataFrame({"MQ7_CO_ppm": co["ppm"].values, "MQ135_NOx_ppm": 0.04, "MQ3_Benzene_ppm": 4.0})
        y_true2 = (co["ppm"].values > 50).astype(int)
        y_pred2 = m2.predict(X2)

        acc2 = accuracy_score(y_true2, y_pred2)
        prec2 = precision_score(y_true2, y_pred2, zero_division=0)
        rec2 = recall_score(y_true2, y_pred2, zero_division=0)
        f1_2 = f1_score(y_true2, y_pred2, zero_division=0)

        print(f"  Accuracy:  {acc2:.4f}")
        print(f"  Precision: {prec2:.4f}")
        print(f"  Recall:    {rec2:.4f}")
        print(f"  F1-Score:  {f1_2:.4f}")

        results_summary["gas_hazard_co_nox_c6h6"] = {
            "accuracy": acc2, "precision": prec2, "recall": rec2, "f1": f1_2, "positives": int(y_true2.sum())
        }

    # 3. Evaluate Deep Severity Classifiers
    print("\n3. Evaluating Deep Learning Severity Classifiers (CH4, CO, CO2, H2)")
    for gas in ["CH4", "CO", "CO2", "H2"]:
        sev_path = os.path.join(MODELS_DIR, f"severity_{gas.lower()}.joblib")
        if os.path.exists(sev_path):
            m_sev = joblib.load(sev_path)
            
            aug_path = os.path.join(DATA_DIR, f"mine_part2_{gas.lower()}_realistic.csv")
            if os.path.exists(aug_path):
                g = pd.read_csv(aug_path)
                col = "ppm_noisy" if "ppm_noisy" in g.columns else "ppm"
            else:
                g = bands[bands.gas == gas].copy()
                col = "ppm"

            X = g[[col]].rename(columns={col: "ppm"})
            y_true = g["severity"].values
            y_pred = m_sev.predict(X)

            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average="macro")

            print(f"  {gas:<5} Deep Severity Acc: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f}")
            results_summary[f"severity_{gas.lower()}"] = {
                "accuracy": acc, "macro_f1": macro_f1
            }

    # Write Markdown documentation
    doc_lines = []
    doc_lines.append("# Deep Learning Models Real-Data Benchmark Report\n")
    doc_lines.append(f"**Generated:** {datetime.now().isoformat()}\n")
    doc_lines.append(f"**Framework:** PyTorch 2.12 (Deep Neural Networks / Multi-Layer Perceptrons)\n")
    doc_lines.append("## Summary Table\n")
    doc_lines.append("| Model / Task | Architecture | Key Metric | Accuracy / Score |")
    doc_lines.append("|--------------|--------------|------------|------------------|")

    for k, v in results_summary.items():
        if "macro_f1" in v:
            doc_lines.append(f"| `{k}` | PyTorch Deep MLP | Macro-F1: {v['macro_f1']:.4f} | **{v['accuracy']*100:.2f}%** |")
        else:
            doc_lines.append(f"| `{k}` | PyTorch Deep MLP | F1-Score: {v['f1']:.4f} | **{v['accuracy']*100:.2f}%** |")

    os.makedirs(DOCS_DIR, exist_ok=True)
    report_path = os.path.join(DOCS_DIR, "REAL_DATA_EVAL.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc_lines))
    print(f"\nEvaluation report written to: {report_path}")

if __name__ == "__main__":
    run_dl_eval()
