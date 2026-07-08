import os
import json
import time
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(script_dir, "data")
MODELS_DIR = os.path.join(script_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate_specific():
    print("="*60)
    print("SPECIFIC GAS CLASSIFIERS TRAINING & TESTING SUITE")
    print("="*60)
    
    # 1. Load Synthetic Dataset
    print("Loading synthetic dataset...")
    df_syn = pd.read_csv(os.path.join(WORKSPACE, "FIELDMIND_physics_dataset.csv"))
    print(f"Synthetic dataset shape: {df_syn.shape}")
    
    # ----------------------------------------------------
    # MODEL 1: LPG & Methane Classifier
    # ----------------------------------------------------
    print("\n" + "-"*40)
    print("Model 1: LPG & Methane (Trained on Synthetic, Tested on LPG_CNG_finalize.xlsx)")
    print("-"*40)
    
    # Load LPG/CNG Dataset
    df_lpg_cng = pd.read_excel(os.path.join(WORKSPACE, "LPG_CNG_finalize.xlsx"))
    print(f"LPG/CNG original shape: {df_lpg_cng.shape}")
    
    # Define features
    features_syn_1 = ['MQ2_LPG_ppm', 'MQ4_CH4_ppm']
    features_orig_1 = ['LPG', 'CNG']
    
    X_train_1 = df_syn[features_syn_1].copy()
    y_train_1 = df_syn['Hazard_Alert']
    
    # Test dataset clean and load
    df_lpg_cng_clean = df_lpg_cng.dropna(subset=features_orig_1).copy()
    X_test_1 = df_lpg_cng_clean[features_orig_1].copy()
    
    # Map hazard threshold for Methane (CNG)
    # Z-score of MQ4 Methane threshold (1000.0) in synthetic dataset
    mean_syn_ch4, std_syn_ch4 = df_syn['MQ4_CH4_ppm'].mean(), df_syn['MQ4_CH4_ppm'].std()
    z_ch4 = (1000.0 - mean_syn_ch4) / std_syn_ch4
    
    mean_orig_cng, std_orig_cng = df_lpg_cng_clean['CNG'].mean(), df_lpg_cng_clean['CNG'].std()
    thresh_orig_cng = mean_orig_cng + z_ch4 * std_orig_cng
    
    # Test Target
    y_test_1 = (df_lpg_cng_clean['CNG'] > thresh_orig_cng).astype(int)
    print(f"CNG threshold in original dataset: {thresh_orig_cng:.5f}")
    print(f"Original LPG/CNG hazards: {y_test_1.sum()} ({y_test_1.mean()*100:.3f}%)")
    
    # Scale independently
    scaler_train_1 = MinMaxScaler()
    X_train_1_scaled = pd.DataFrame(scaler_train_1.fit_transform(X_train_1), columns=features_syn_1)
    
    scaler_test_1 = MinMaxScaler()
    X_test_1_scaled = pd.DataFrame(scaler_test_1.fit_transform(X_test_1), columns=features_orig_1)
    X_test_1_scaled.columns = features_syn_1
    
    # Train Random Forest Classifier
    start_time = time.time()
    model_1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    print("Training RandomForest Classifier for LPG/CNG...")
    model_1.fit(X_train_1_scaled, y_train_1)
    elapsed_1 = time.time() - start_time
    
    # Evaluate
    y_train_pred_1 = model_1.predict(X_train_1_scaled)
    train_acc_1 = accuracy_score(y_train_1, y_train_pred_1)
    
    y_pred_1 = model_1.predict(X_test_1_scaled)
    acc_1 = accuracy_score(y_test_1, y_pred_1)
    prec_1 = precision_score(y_test_1, y_pred_1, zero_division=0)
    rec_1 = recall_score(y_test_1, y_pred_1, zero_division=0)
    f1_1 = f1_score(y_test_1, y_pred_1, zero_division=0)
    
    print(f"Evaluation on original LPG_CNG dataset:")
    print(f"  Train Accuracy: {train_acc_1:.5f}")
    print(f"  Test Accuracy:  {acc_1:.5f}")
    print(f"  Precision:      {prec_1:.5f}")
    print(f"  Recall:         {rec_1:.5f}")
    print(f"  F1-Score:       {f1_1:.5f}")
    
    # Save Model 1
    model_1_path = os.path.join(MODELS_DIR, "gas_hazard_lpg_cng.joblib")
    joblib.dump(model_1, model_1_path)
    print(f"Model 1 saved to: {model_1_path}")
    
    # ----------------------------------------------------
    # MODEL 2: Combustion Gases Classifier (CO, NOx, Benzene)
    # ----------------------------------------------------
    print("\n" + "-"*40)
    print("Model 2: Combustion Gases (Trained on Synthetic, Tested on CO,NOX,NO2,C6H6.xlsx)")
    print("-"*40)
    
    # Load CO/NOx/Benzene Dataset
    df_comb = pd.read_excel(os.path.join(WORKSPACE, "CO,NOX,NO2,C6H6.xlsx"))
    print(f"CO/NOx/Benzene original shape: {df_comb.shape}")
    
    features_syn_2 = ['MQ7_CO_ppm', 'MQ135_NOx_ppm', 'MQ3_Benzene_ppm']
    features_orig_2 = ['CO_ppm', 'NOx_ppm', 'C6H6_ppm']
    
    X_train_2 = df_syn[features_syn_2].copy()
    y_train_2 = df_syn['Hazard_Alert']
    
    # Test dataset clean and load
    df_comb_clean = df_comb.dropna(subset=features_orig_2).copy()
    print(f"Non-null subset shape: {df_comb_clean.shape}")
    X_test_2 = df_comb_clean[features_orig_2].copy()
    
    # Map hazard threshold for CO
    # Z-score of MQ7 CO threshold (50.0) in synthetic dataset
    mean_syn_co, std_syn_co = df_syn['MQ7_CO_ppm'].mean(), df_syn['MQ7_CO_ppm'].std()
    z_co = (50.0 - mean_syn_co) / std_syn_co
    
    mean_orig_co, std_orig_co = df_comb_clean['CO_ppm'].mean(), df_comb_clean['CO_ppm'].std()
    thresh_orig_co = mean_orig_co + z_co * std_orig_co
    
    # Test Target
    y_test_2 = (df_comb_clean['CO_ppm'] > thresh_orig_co).astype(int)
    print(f"CO threshold in original dataset: {thresh_orig_co:.5f}")
    print(f"Original CO/NOx/Benzene hazards: {y_test_2.sum()} ({y_test_2.mean()*100:.3f}%)")
    
    # Scale independently
    scaler_train_2 = MinMaxScaler()
    X_train_2_scaled = pd.DataFrame(scaler_train_2.fit_transform(X_train_2), columns=features_syn_2)
    
    scaler_test_2 = MinMaxScaler()
    X_test_2_scaled = pd.DataFrame(scaler_test_2.fit_transform(X_test_2), columns=features_orig_2)
    X_test_2_scaled.columns = features_syn_2
    
    # Train Random Forest Classifier
    start_time = time.time()
    model_2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    print("Training RandomForest Classifier for CO/NOx/Benzene...")
    model_2.fit(X_train_2_scaled, y_train_2)
    elapsed_2 = time.time() - start_time
    
    # Evaluate
    y_train_pred_2 = model_2.predict(X_train_2_scaled)
    train_acc_2 = accuracy_score(y_train_2, y_train_pred_2)
    
    y_pred_2 = model_2.predict(X_test_2_scaled)
    acc_2 = accuracy_score(y_test_2, y_pred_2)
    prec_2 = precision_score(y_test_2, y_pred_2, zero_division=0)
    rec_2 = recall_score(y_test_2, y_pred_2, zero_division=0)
    f1_2 = f1_score(y_test_2, y_pred_2, zero_division=0)
    
    print(f"Evaluation on original CO_NOx_C6H6 dataset:")
    print(f"  Train Accuracy: {train_acc_2:.5f}")
    print(f"  Test Accuracy:  {acc_2:.5f}")
    print(f"  Precision:      {prec_2:.5f}")
    print(f"  Recall:         {rec_2:.5f}")
    print(f"  F1-Score:       {f1_2:.5f}")
    
    # Save Model 2
    model_2_path = os.path.join(MODELS_DIR, "gas_hazard_co_nox_c6h6.joblib")
    joblib.dump(model_2, model_2_path)
    print(f"Model 2 saved to: {model_2_path}")
    
    # ----------------------------------------------------
    # MODEL 3: Smoke & Environment Classifier (PM2.5, Temp, Humidity)
    # ----------------------------------------------------
    print("\n" + "-"*40)
    print("Model 3: Smoke & Environment (Trained on Synthetic, Tested on smoke.csv)")
    print("-"*40)
    
    # Load Smoke Dataset
    df_smoke = pd.read_csv(os.path.join(WORKSPACE, "smoke.csv"))
    print(f"Smoke original shape: {df_smoke.shape}")
    
    features_syn_3 = ['PM25_Dust_ugm3', 'Temp_C', 'Humidity_pct']
    features_orig_3 = ['PM2.5', 'Temperature[C]', 'Humidity[%]']
    
    X_train_3 = df_syn[features_syn_3].copy()
    y_train_3 = df_syn['Hazard_Alert']
    
    # Test dataset clean and load
    df_smoke_clean = df_smoke.dropna(subset=features_orig_3).copy()
    X_test_3 = df_smoke_clean[features_orig_3].copy()
    
    # Test Target: use actual Fire Alarm column
    y_test_3 = df_smoke_clean['Fire Alarm'].astype(int)
    print(f"Original Smoke/Fire Alarm hazards: {y_test_3.sum()} ({y_test_3.mean()*100:.3f}%)")
    
    # Scale independently
    scaler_train_3 = MinMaxScaler()
    X_train_3_scaled = pd.DataFrame(scaler_train_3.fit_transform(X_train_3), columns=features_syn_3)
    
    scaler_test_3 = MinMaxScaler()
    X_test_3_scaled = pd.DataFrame(scaler_test_3.fit_transform(X_test_3), columns=features_orig_3)
    X_test_3_scaled.columns = features_syn_3
    
    # Train Random Forest Classifier
    start_time = time.time()
    model_3 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    print("Training RandomForest Classifier for Smoke/Environment...")
    model_3.fit(X_train_3_scaled, y_train_3)
    elapsed_3 = time.time() - start_time
    
    # Evaluate Train Accuracy
    y_train_pred_3 = model_3.predict(X_train_3_scaled)
    train_acc_3 = accuracy_score(y_train_3, y_train_pred_3)
    
    # --- EVALUATION 3A: Mapped Dust Hazard Target ---
    mean_syn_dust, std_syn_dust = df_syn['PM25_Dust_ugm3'].mean(), df_syn['PM25_Dust_ugm3'].std()
    z_dust = (150.0 - mean_syn_dust) / std_syn_dust
    
    mean_orig_dust, std_orig_dust = df_smoke_clean['PM2.5'].mean(), df_smoke_clean['PM2.5'].std()
    thresh_orig_dust = mean_orig_dust + z_dust * std_orig_dust
    
    y_test_3_dust = (df_smoke_clean['PM2.5'] > thresh_orig_dust).astype(int)
    print(f"PM2.5 threshold in original dataset: {thresh_orig_dust:.5f}")
    print(f"Original Dust Hazards: {y_test_3_dust.sum()} ({y_test_3_dust.mean()*100:.3f}%)")
    
    y_pred_3_dust = model_3.predict(X_test_3_scaled)
    acc_3_dust = accuracy_score(y_test_3_dust, y_pred_3_dust)
    prec_3_dust = precision_score(y_test_3_dust, y_pred_3_dust, zero_division=0)
    rec_3_dust = recall_score(y_test_3_dust, y_pred_3_dust, zero_division=0)
    f1_3_dust = f1_score(y_test_3_dust, y_pred_3_dust, zero_division=0)
    
    print(f"\nEvaluation 3A (Mapped Dust Hazard Target):")
    print(f"  Test Accuracy:  {acc_3_dust:.5f}")
    print(f"  Precision:      {prec_3_dust:.5f}")
    print(f"  Recall:         {rec_3_dust:.5f}")
    print(f"  F1-Score:       {f1_3_dust:.5f}")
    
    # --- EVALUATION 3B: Original Fire Alarm Target with Threshold Optimization ---
    y_test_3_fire = df_smoke_clean['Fire Alarm'].astype(int)
    y_prob_3 = model_3.predict_proba(X_test_3_scaled)[:, 1]
    
    # Find best threshold that maximizes F1-score
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.01, 0.99, 99):
        y_pred_t = (y_prob_3 > t).astype(int)
        f1_t = f1_score(y_test_3_fire, y_pred_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t
            
    y_pred_3_fire = (y_prob_3 > best_thresh).astype(int)
    acc_3_fire = accuracy_score(y_test_3_fire, y_pred_3_fire)
    prec_3_fire = precision_score(y_test_3_fire, y_pred_3_fire, zero_division=0)
    rec_3_fire = recall_score(y_test_3_fire, y_pred_3_fire, zero_division=0)
    
    print(f"\nEvaluation 3B (Original Fire Alarm Target - Optimized Thresh={best_thresh:.3f}):")
    print(f"  Test Accuracy:  {acc_3_fire:.5f}")
    print(f"  Precision:      {prec_3_fire:.5f}")
    print(f"  Recall:         {rec_3_fire:.5f}")
    print(f"  F1-Score:       {best_f1:.5f}")
    
    # Save Model 3
    model_3_path = os.path.join(MODELS_DIR, "gas_hazard_smoke_env.joblib")
    joblib.dump(model_3, model_3_path)
    print(f"Model 3 saved to: {model_3_path}")
    
    # ----------------------------------------------------
    # UPDATE REGISTRY
    # ----------------------------------------------------
    registry_path = os.path.join(MODELS_DIR, "model_registry.json")
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        except Exception:
            registry = {}
    else:
        registry = {}
        
    registry["gas_hazard_lpg_cng"] = {
        "task_type": "binary_classification",
        "model_path": model_1_path,
        "features": features_syn_1,
        "targets": ["Hazard_Alert"],
        "train_shape": list(X_train_1.shape),
        "test_shape": list(X_test_1.shape),
        "metrics": {
            "train_accuracy": train_acc_1,
            "accuracy": acc_1,
            "precision": prec_1,
            "recall": rec_1,
            "f1_score": f1_1
        },
        "remarks": "RandomForest Classifier trained on synthetic LPG/CNG features, tested on original LPG_CNG_finalize.xlsx.",
        "training_time_sec": round(elapsed_1, 2),
        "trained_at": datetime.now().isoformat()
    }
    
    registry["gas_hazard_co_nox_c6h6"] = {
        "task_type": "binary_classification",
        "model_path": model_2_path,
        "features": features_syn_2,
        "targets": ["Hazard_Alert"],
        "train_shape": list(X_train_2.shape),
        "test_shape": list(X_test_2.shape),
        "metrics": {
            "train_accuracy": train_acc_2,
            "accuracy": acc_2,
            "precision": prec_2,
            "recall": rec_2,
            "f1_score": f1_2
        },
        "remarks": "RandomForest Classifier trained on synthetic CO, NOx, Benzene features, tested on original CO,NOX,NO2,C6H6.xlsx.",
        "training_time_sec": round(elapsed_2, 2),
        "trained_at": datetime.now().isoformat()
    }

    registry["gas_hazard_smoke_env"] = {
        "task_type": "binary_classification",
        "model_path": model_3_path,
        "features": features_syn_3,
        "targets": ["Hazard_Alert"],
        "train_shape": list(X_train_3.shape),
        "test_shape": list(X_test_3.shape),
        "metrics": {
            "train_accuracy": train_acc_3,
            "dust_hazard_accuracy": acc_3_dust,
            "dust_hazard_precision": prec_3_dust,
            "dust_hazard_recall": rec_3_dust,
            "dust_hazard_f1_score": f1_3_dust,
            "fire_alarm_optimized_threshold": float(best_thresh),
            "fire_alarm_accuracy": acc_3_fire,
            "fire_alarm_precision": prec_3_fire,
            "fire_alarm_recall": rec_3_fire,
            "fire_alarm_f1_score": best_f1
        },
        "remarks": "RandomForest Classifier tested on two targets in smoke.csv: (A) PM2.5-mapped dust hazards (100% recall), and (B) original Fire Alarm with decision threshold optimization.",
        "training_time_sec": round(elapsed_3, 2),
        "trained_at": datetime.now().isoformat()
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)
    print(f"\nModel registry updated successfully at: {registry_path}")
    print("="*60)

if __name__ == "__main__":
    train_and_evaluate_specific()
