import os
import json
import time
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

script_dir = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(script_dir, "data")
MODELS_DIR = os.path.join(script_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate_gas_detector():
    print("="*60)
    print("MULTI-GAS PRESENCE CLASSIFIER TRAINING & TESTING")
    print("="*60)
    
    # 1. Load Datasets
    print("Loading synthetic dataset...")
    df_syn = pd.read_csv(os.path.join(WORKSPACE, "FIELDMIND_physics_dataset.csv"))
    print("Loading original dataset...")
    df_orig = pd.read_excel(os.path.join(WORKSPACE, "Gas_Sensors.xlsx"))
    
    print(f"Synthetic shape: {df_syn.shape}")
    print(f"Original shape : {df_orig.shape}")
    
    # 2. Define presence thresholds in synthetic dataset
    thresholds_syn = {
        'Methane': ('MQ4_CH4_ppm', 117.5),
        'CO': ('MQ7_CO_ppm', 15.0),
        'LPG': ('MQ2_LPG_ppm', 135.0),
        'Smoke': ('MQ2_Smoke_ppm', 120.0),
        'NOx': ('MQ135_NOx_ppm', 0.07)
    }
    
    y_train_dict = {}
    for gas, (col, thresh) in thresholds_syn.items():
        y_train_dict[gas] = (df_syn[col] > thresh).astype(int)
        
    y_train = pd.DataFrame(y_train_dict)
    
    # Inputs: MQ-2 Core features
    features_syn = ['MQ2_CO_ppm', 'MQ2_LPG_ppm', 'MQ2_Smoke_ppm']
    features_orig = ['co', 'lpg', 'smoke']
    
    X_train = df_syn[features_syn].copy()
    
    print("\nTraining class balance (positives out of 50000):")
    for gas in thresholds_syn.keys():
        pos = y_train[gas].sum()
        print(f"  {gas}: {pos} ({pos/50000*100:.2f}%)")
        
    # Scale train inputs
    scaler_train = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler_train.fit_transform(X_train), columns=features_syn)
    
    # 3. Train Multi-Output Random Forest Classifier with Regularization
    start_time = time.time()
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model = MultiOutputClassifier(forest)
    print("\nTraining Regularized Multi-Output Classifier...")
    model.fit(X_train_scaled, y_train)
    elapsed = time.time() - start_time
    
    # Evaluate Train Accuracy
    y_train_pred = pd.DataFrame(model.predict(X_train_scaled), columns=y_train.columns)
    
    # Save Model
    model_path = os.path.join(MODELS_DIR, "multi_gas_detector.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # ----------------------------------------------------
    # 4. EVALUATE ON ORIGINAL DATASET (GAS_SENSORS.XLSX)
    # ----------------------------------------------------
    # Map original column names to our target gases
    orig_mapping = {
        'Methane': 'CNG',
        'CO': 'co',
        'LPG': 'lpg',
        'Smoke': 'smoke',
        'NOx': 'NO2_ppm'
    }
    
    results = {}
    
    print("\n" + "-"*40)
    print("Evaluation on Original Dataset (Gas_Sensors.xlsx)")
    print("-"*40)
    
    for gas in thresholds_syn.keys():
        syn_col, thresh_syn = thresholds_syn[gas]
        orig_col = orig_mapping[gas]
        
        # Filter original dataset to rows where the target gas is not null
        df_orig_filtered = df_orig.dropna(subset=[orig_col] + features_orig).copy()
        
        if len(df_orig_filtered) == 0:
            print(f"\nSkipping {gas}: no non-null test samples available.")
            continue
            
        # Z-score mapped threshold calculation
        mean_syn, std_syn = df_syn[syn_col].mean(), df_syn[syn_col].std()
        z = (thresh_syn - mean_syn) / std_syn
        
        mean_orig, std_orig = df_orig_filtered[orig_col].mean(), df_orig_filtered[orig_col].std()
        thresh_orig = mean_orig + z * std_orig
        
        # Build test target and inputs
        X_test = df_orig_filtered[features_orig].copy()
        y_test = (df_orig_filtered[orig_col] > thresh_orig).astype(int)
        
        # Scale test inputs
        scaler_test = MinMaxScaler()
        X_test_scaled = pd.DataFrame(scaler_test.fit_transform(X_test), columns=features_orig)
        X_test_scaled.columns = features_syn
        
        # Train accuracy for this specific gas output
        gas_idx = list(y_train.columns).index(gas)
        train_acc = accuracy_score(y_train[gas], y_train_pred[gas])
        
        # Run prediction for this gas
        # model.predict output is a numpy array of shape (N, 5)
        # We slice the column corresponding to the current gas
        y_pred = model.predict(X_test_scaled)[:, gas_idx]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"\nGas: {gas} (Tested on {len(df_orig_filtered)} rows, Mapped Thresh={thresh_orig:.5f})")
        print(f"  Train Acc: {train_acc:.5f}")
        print(f"  Test Acc:  {acc:.5f}")
        print(f"  Precision: {prec:.5f}")
        print(f"  Recall:    {rec:.5f}")
        print(f"  F1-Score:  {f1:.5f}")
        
        results[gas] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "mapped_threshold": float(thresh_orig),
            "test_samples": int(len(df_orig_filtered))
        }
        
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
        
    registry["multi_gas_detector"] = {
        "task_type": "multilabel_classification",
        "model_path": model_path,
        "features": features_syn,
        "targets": list(thresholds_syn.keys()),
        "train_shape": list(X_train.shape),
        "test_shape": list(df_orig.shape),
        "metrics": results,
        "remarks": "Multi-Output RandomForest Classifier trained on synthetic MQ-2 features to predict the presence of Methane, CO, LPG, Smoke, and NOx, validated using z-score threshold mapping on Gas_Sensors.xlsx.",
        "training_time_sec": round(elapsed, 2),
        "trained_at": datetime.now().isoformat()
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)
    print(f"\nModel registry updated successfully at: {registry_path}")
    print("="*60)

if __name__ == "__main__":
    train_and_evaluate_gas_detector()
