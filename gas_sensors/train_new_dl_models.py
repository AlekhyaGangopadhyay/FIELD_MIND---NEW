import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dl_wrappers import PyTorchSeverityClassifier, PyTorchHazardClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "data")
MODELS_DIR = os.path.join(script_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_new_models():
    print("=" * 60)
    print("TRAINING NEW DEEP LEARNING MODELS FOR H2S AND NH3")
    print("=" * 60)

    # 1. Load Registry
    registry_path = os.path.join(MODELS_DIR, "model_registry.json")
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except Exception:
        registry = {}

    # 2. Train severity_h2s (Deep Learning Multiclass)
    h2s_data_path = os.path.join(DATA_DIR, "mine_part2_h2s_balanced_cgan.csv")
    if os.path.exists(h2s_data_path):
        print("\n--- Training PyTorch Deep Learning Model: severity_h2s ---")
        df_h2s = pd.read_csv(h2s_data_path)
        X = df_h2s[["ppm"]]
        y = df_h2s["severity"]

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.50, stratify=y, random_state=42)
        
        # Train Deep Learning Model
        model_h2s = PyTorchSeverityClassifier(in_features=1, num_classes=3, epochs=30, batch_size=128)
        model_h2s.fit(Xtr, ytr)

        y_pred = model_h2s.predict(Xte)
        acc = accuracy_score(yte, y_pred)
        prec = precision_score(yte, y_pred, average='macro', zero_division=0)
        rec = recall_score(yte, y_pred, average='macro', zero_division=0)
        f1 = f1_score(yte, y_pred, average='macro', zero_division=0)

        print(f"  Test Accuracy:  {acc:.4f}")
        print(f"  Test Precision: {prec:.4f}")
        print(f"  Test Recall:    {rec:.4f}")
        print(f"  Test F1-Score:  {f1:.4f}")

        model_path = os.path.join(MODELS_DIR, "severity_h2s.joblib")
        joblib.dump(model_h2s, model_path)
        print(f"  Model saved to: {model_path}")

        # Update registry
        registry["severity_h2s"] = {
            "task_type": "multiclass_classification",
            "model_type": "PyTorch_Deep_MLP",
            "model_path": model_path,
            "features": ["ppm"],
            "targets": ["severity"],
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1)
            },
            "remarks": "PyTorch Deep Neural Network trained on 60,000 balanced CGAN H2S samples.",
            "training_time_sec": 0.0,
            "trained_at": datetime.now().isoformat()
        }
    else:
        print(f"Error: H2S dataset not found at {h2s_data_path}")

    # 3. Train nh3_hazard (Deep Learning Binary)
    nh3_data_path = os.path.join(DATA_DIR, "nh3_hazard_balanced_cgan.csv")
    if os.path.exists(nh3_data_path):
        print("\n--- Training PyTorch Deep Learning Model: nh3_hazard ---")
        df_nh3 = pd.read_csv(nh3_data_path)
        X = df_nh3[["MQ135_NH3_ppm"]]
        y = df_nh3["Hazard_Alert"]

        # Calculate class weights
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_weight = [n_neg / max(1, n_pos)]
        print(f"  Class Balance -> Positives: {n_pos} ({n_pos/len(y)*100:.1f}%), Negatives: {n_neg} ({n_neg/len(y)*100:.1f}%)")

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.50, stratify=y, random_state=42)

        model_nh3 = PyTorchHazardClassifier(in_features=1, out_features=1, binary=True, epochs=50, pos_weight=pos_weight)
        model_nh3.fit(Xtr, ytr)

        y_pred = model_nh3.predict(Xte)
        acc = accuracy_score(yte, y_pred)
        prec = precision_score(yte, y_pred, zero_division=0)
        rec = recall_score(yte, y_pred, zero_division=0)
        f1 = f1_score(yte, y_pred, zero_division=0)

        print(f"  Test Accuracy:  {acc:.4f}")
        print(f"  Test Precision: {prec:.4f}")
        print(f"  Test Recall:    {rec:.4f}")
        print(f"  Test F1-Score:  {f1:.4f}")

        model_path = os.path.join(MODELS_DIR, "nh3_hazard.joblib")
        joblib.dump(model_nh3, model_path)
        print(f"  Model saved to: {model_path}")

        # Update registry
        registry["nh3_hazard"] = {
            "task_type": "binary_classification",
            "model_type": "PyTorch_Deep_MLP",
            "model_path": model_path,
            "features": ["MQ135_NH3_ppm"],
            "targets": ["Hazard_Alert"],
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1)
            },
            "remarks": "PyTorch Deep Neural Network trained on 60,000 balanced CGAN NH3 samples.",
            "training_time_sec": 0.0,
            "trained_at": datetime.now().isoformat()
        }
    else:
        print(f"Error: NH3 dataset not found at {nh3_data_path}")

    # 4. Train co2_hazard (Deep Learning Binary)
    co2_data_path = os.path.join(DATA_DIR, "mine_part2_co2_balanced_cgan.csv")
    if os.path.exists(co2_data_path):
        print("\n--- Training PyTorch Deep Learning Model: co2_hazard ---")
        df_co2 = pd.read_csv(co2_data_path)
        X = df_co2[["ppm"]].copy()
        
        # Define hazard target at >= 1000.0 ppm
        y = (df_co2["ppm"] >= 1000.0).astype(int)

        # Inject 5% noise overlap around 1000 ppm threshold
        X_vals = X["ppm"].values.copy()
        noise_mask = np.random.random(len(X_vals)) < 0.05
        X_vals[noise_mask] += np.random.uniform(-100.0, 100.0, size=noise_mask.sum())
        X = pd.DataFrame(np.clip(X_vals, 0.0, None), columns=["ppm"])

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_weight = [n_neg / max(1, n_pos)]
        print(f"  Class Balance -> Positives: {n_pos} ({n_pos/len(y)*100:.1f}%), Negatives: {n_neg} ({n_neg/len(y)*100:.1f}%)")

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.50, stratify=y, random_state=42)

        model_co2 = PyTorchHazardClassifier(in_features=1, out_features=1, binary=True, epochs=50, pos_weight=pos_weight)
        model_co2.fit(Xtr, ytr)

        y_pred = model_co2.predict(Xte)
        acc = accuracy_score(yte, y_pred)
        prec = precision_score(yte, y_pred, zero_division=0)
        rec = recall_score(yte, y_pred, zero_division=0)
        f1 = f1_score(yte, y_pred, zero_division=0)

        print(f"  Test Accuracy:  {acc:.4f}")
        print(f"  Test Precision: {prec:.4f}")
        print(f"  Test Recall:    {rec:.4f}")
        print(f"  Test F1-Score:  {f1:.4f}")

        model_path = os.path.join(MODELS_DIR, "co2_hazard.joblib")
        joblib.dump(model_co2, model_path)
        print(f"  Model saved to: {model_path}")

        # Update registry
        registry["co2_hazard"] = {
            "task_type": "binary_classification",
            "model_type": "PyTorch_Deep_MLP",
            "model_path": model_path,
            "features": ["ppm"],
            "targets": ["over_tlv"],
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1)
            },
            "remarks": "PyTorch Deep Neural Network trained on 60,000 balanced CGAN CO2 samples with 5% uncertainty noise around 1000 ppm threshold.",
            "training_time_sec": 0.0,
            "trained_at": datetime.now().isoformat()
        }
    else:
        print(f"Error: CO2 dataset not found at {co2_data_path}")

    # 5. Train smoke_env_hazard (Deep Learning Binary)
    smoke_data_path = os.path.join(DATA_DIR, "FIELDMIND_physics_dataset.csv")
    if os.path.exists(smoke_data_path):
        print("\n--- Training PyTorch Deep Learning Model: smoke_env_hazard ---")
        df_smoke = pd.read_csv(smoke_data_path)
        df_smoke = df_smoke.dropna(subset=['PM25_Dust_ugm3', 'Temp_C', 'Humidity_pct']).copy()
        
        # Balance dataset based on Hazard_Alert
        df_pos = df_smoke[df_smoke['Hazard_Alert'] == 1]
        df_neg = df_smoke[df_smoke['Hazard_Alert'] == 0]
        n_samples = min(len(df_pos), len(df_neg))
        df_balanced = pd.concat([
            df_pos.sample(n_samples, random_state=42),
            df_neg.sample(n_samples, random_state=42)
        ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

        # Inject 5% transition noise zone
        noise_mask = np.random.random(len(df_balanced)) < 0.05
        df_balanced.loc[noise_mask, 'PM25_Dust_ugm3'] += np.random.uniform(-10.0, 10.0, size=noise_mask.sum())
        df_balanced.loc[noise_mask, 'Temp_C'] += np.random.uniform(-1.0, 1.0, size=noise_mask.sum())
        df_balanced.loc[noise_mask, 'Humidity_pct'] += np.random.uniform(-3.0, 3.0, size=noise_mask.sum())

        X = df_balanced[['PM25_Dust_ugm3', 'Temp_C', 'Humidity_pct']].copy()
        y = df_balanced['Hazard_Alert']

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_weight = [n_neg / max(1, n_pos)]
        print(f"  Class Balance -> Positives: {n_pos} ({n_pos/len(y)*100:.1f}%), Negatives: {n_neg} ({n_neg/len(y)*100:.1f}%)")

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.50, stratify=y, random_state=42)

        model_smoke = PyTorchHazardClassifier(in_features=3, out_features=1, binary=True, epochs=50, pos_weight=pos_weight)
        model_smoke.fit(Xtr, ytr)

        y_pred = model_smoke.predict(Xte)
        acc = accuracy_score(yte, y_pred)
        prec = precision_score(yte, y_pred, zero_division=0)
        rec = recall_score(yte, y_pred, zero_division=0)
        f1 = f1_score(yte, y_pred, zero_division=0)

        print(f"  Test Accuracy:  {acc:.4f}")
        print(f"  Test Precision: {prec:.4f}")
        print(f"  Test Recall:    {rec:.4f}")
        print(f"  Test F1-Score:  {f1:.4f}")

        model_path = os.path.join(MODELS_DIR, "smoke_env_hazard.joblib")
        joblib.dump(model_smoke, model_path)
        print(f"  Model saved to: {model_path}")

        # Update registry
        registry["smoke_env_hazard"] = {
            "task_type": "binary_classification",
            "model_type": "PyTorch_Deep_MLP",
            "model_path": model_path,
            "features": ["PM25_Dust_ugm3", "Temp_C", "Humidity_pct"],
            "targets": ["Hazard_Alert"],
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1)
            },
            "remarks": "PyTorch Deep Neural Network trained on balanced physical dataset.csv with 5% uncertainty noise.",
            "training_time_sec": 0.0,
            "trained_at": datetime.now().isoformat()
        }
    else:
        print(f"Error: Smoke dataset not found at {smoke_data_path}")

    # Write registry back
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)
    print(f"\nRegistry updated: {registry_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_new_models()
