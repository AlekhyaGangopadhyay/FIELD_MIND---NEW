import os
import json
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import our workspace data loader
import data_loader

script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(script_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def compute_regression_accuracy(y_true, y_pred, threshold=0.10, eps=10.0):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = np.abs(y_true - y_pred)
    allowed_diff = np.maximum(np.abs(y_true) * threshold, eps)
    return np.mean(diff <= allowed_diff)

def main():
    print("="*60)
    print("COMBINED GASES REGRESSOR TRAINING SUITE (HIST GRADIENT BOOSTING)")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Load Combined Gases dataset (chronological split)
    X_train, X_test, y_train, y_test = data_loader.load_combined_gases()
    print("Dataset loaded successfully.")
    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape: {X_test.shape}")
    
    # 2. Instantiate and train HistGradientBoostingRegressor
    print("\nFitting HistGradientBoostingRegressor (max_iter=150)...")
    model = HistGradientBoostingRegressor(max_iter=150, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Evaluate on chronological test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Evaluate tolerance accuracy (within 10% or absolute margin of 10.0 ppm)
    eps = 10.0
    reg_acc = compute_regression_accuracy(y_test, y_pred, threshold=0.10, eps=eps)
    
    print(f"\nEvaluation on Test Set:")
    print(f"  Mean Absolute Error:     {mae:.5f}")
    print(f"  Root Mean Squared Error: {rmse:.5f}")
    print(f"  R2 Score:                {r2:.5f}")
    print(f"  Regression Accuracy:     {reg_acc:.5f} (within 10% or {eps} ppm margin)")
    
    # 4. Save the trained model
    model_path = os.path.join(MODELS_DIR, "combined_gases_regressor.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel successfully serialized and saved to: {model_path}")
    
    # 5. Update registry entry
    registry_path = os.path.join(MODELS_DIR, "model_registry.json")
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        except Exception:
            registry = {}
    else:
        registry = {}
        
    elapsed = time.time() - start_time
    
    registry["combined_gases"] = {
        "task_type": "regression",
        "model_path": model_path,
        "features": list(X_train.columns),
        "targets": [y_train.name],
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "accuracy_10pct": reg_acc
        },
        "remarks": f"Optimized using HistGradientBoostingRegressor (max_iter=150) and time-series engineered features (lags 1-5 for all gases, rolling averages window 3, 5). Chronological test R2 boosted to {r2*100:.2f}% and accuracy to {reg_acc*100:.2f}%.",
        "training_time_sec": round(elapsed, 2),
        "trained_at": datetime.now().isoformat()
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)
    print(f"Model registry entry updated in {registry_path}")
    print("="*60)

if __name__ == "__main__":
    main()
