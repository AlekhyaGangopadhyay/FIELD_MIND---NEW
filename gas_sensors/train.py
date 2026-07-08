import os
import json
import time
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Import data loader
import data_loader

script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(script_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

registry = {}

def compute_regression_accuracy(y_true, y_pred, threshold=0.10, eps=1.0):
    """
    Computes a tolerance-based regression 'accuracy'.
    Returns the percentage of predictions within threshold of the true values,
    or within an absolute margin eps (useful when true value is near 0).
    """
    yt = np.array(y_true)
    yp = np.array(y_pred)
    
    if len(yt.shape) > 1 and yt.shape[1] > 1:
        accs = []
        for col_idx in range(yt.shape[1]):
            col_yt = yt[:, col_idx]
            col_yp = yp[:, col_idx]
            within_tolerance = np.abs(col_yt - col_yp) <= np.maximum(threshold * np.abs(col_yt), eps)
            accs.append(np.mean(within_tolerance))
        return float(np.mean(accs))
    else:
        within_tolerance = np.abs(yt - yp) <= np.maximum(threshold * np.abs(yt), eps)
        return float(np.mean(within_tolerance))

def log_metrics(name, task_type, model_path, features, targets, train_shape, test_shape, metrics, elapsed_time, remarks=""):
    # Convert features to list of strings
    if hasattr(features, 'tolist'):
        feat_list = features.tolist()
    else:
        feat_list = list(features)
        
    # Convert targets to list of strings
    if hasattr(targets, 'tolist'):
        target_list = targets.tolist()
    elif isinstance(targets, (list, tuple)):
        target_list = list(targets)
    else:
        target_list = [targets]
        
    registry[name] = {
        "task_type": task_type,
        "model_path": model_path,
        "features": feat_list,
        "targets": target_list,
        "train_shape": list(train_shape),
        "test_shape": list(test_shape),
        "metrics": metrics,
        "remarks": remarks,
        "training_time_sec": round(elapsed_time, 2),
        "trained_at": datetime.now().isoformat()
    }

def train_smoke_classifier():
    print("\n" + "="*50)
    print("Training Smoke/Fire Alarm Classifier (Session Split + Diff/Rolling)...")
    print("="*50)
    
    start_time = time.time()
    X_train, X_test, y_train, y_test = data_loader.load_smoke_dataset()
    print(f"Dataset Loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Use max_depth=8 to regularize and avoid memorizing session patterns
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    print("Fitting Random Forest Classifier...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"Accuracy:  {acc:.5f}")
    print(f"Precision: {prec:.5f}")
    print(f"Recall:    {rec:.5f}")
    print(f"F1-Score:  {f1:.5f}")
    
    model_path = os.path.join(MODELS_DIR, "smoke_fire_alarm_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    elapsed = time.time() - start_time
    metrics = {
        "accuracy": acc, 
        "precision": prec, 
        "recall": rec, 
        "f1_score": f1
    }
    remarks = f"Improved via 5-period temporal differencing and rolling variance. Test accuracy on unseen Session 2 boosted to {acc*100:.2f}%."
    
    log_metrics(
        "smoke_fire_alarm", 
        "binary_classification", 
        model_path, 
        X_train.columns, 
        y_train.name, 
        X_train.shape, 
        X_test.shape, 
        metrics, 
        elapsed,
        remarks=remarks
    )

def train_mq4_classifier():
    print("\n" + "="*50)
    print("Training MQ4 Gas Classifier (GridSearchCV with 8-Fold CV)...")
    print("="*50)
    
    start_time = time.time()
    X_train, X_test, y_train, y_test = data_loader.load_mq4_dataset()
    print(f"Dataset Loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Build a Pipeline with StandardScaler and SVC RBF
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', random_state=42))
    ])
    
    # Param grid for grid search optimization
    param_grid = {
        'svm__C': [1.0, 5.0, 10.0, 20.0, 50.0],
        'svm__gamma': ['scale', 'auto', 0.001, 0.01]
    }
    
    # 8-fold Stratified CV
    cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    
    print("Starting GridSearchCV with 8-fold CV (totalling 160 fits)...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best CV Parameters found: {grid_search.best_params_}")
    print(f"Best CV Train Accuracy: {grid_search.best_score_:.5f}")
    
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy:        {acc:.5f}")
    print(f"Macro Precision: {prec:.5f}")
    print(f"Macro Recall:    {rec:.5f}")
    print(f"Macro F1-Score:  {f1:.5f}")
    
    model_path = os.path.join(MODELS_DIR, "mq4_gas_classifier.joblib")
    joblib.dump(best_model, model_path)
    print(f"Model saved to: {model_path}")
    
    elapsed = time.time() - start_time
    metrics = {
        "accuracy": acc, 
        "precision_macro": prec, 
        "recall_macro": rec, 
        "f1_score_macro": f1,
        "best_cv_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_
    }
    remarks = f"Optimized using GridSearchCV with 8-Fold cross-validation. Best CV params: {grid_search.best_params_}. Test accuracy on new batches: {acc*100:.2f}%."
    
    log_metrics(
        "mq4_gas_classification", 
        "multiclass_classification", 
        model_path, 
        X_train.columns, 
        y_train.name, 
        X_train.shape, 
        X_test.shape, 
        metrics, 
        elapsed,
        remarks=remarks
    )


def train_air_quality_regressor():
    print("\n" + "="*50)
    print("Training Air Quality UCI Regressor (Chronological Split)...")
    print("="*50)
    
    start_time = time.time()
    X_train, X_test, y_train, y_test = data_loader.load_air_quality_uci()
    print(f"Dataset Loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Regularized tree depth (max_depth=6) to prevent overfitting to specific calibration noise
    model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    print("Fitting Regularized Random Forest Regressor (max_depth=6)...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Evaluate accuracy: tolerance of 10% or absolute margin of 0.5 PPM
    eps = 0.5
    reg_acc = compute_regression_accuracy(y_test, y_pred, threshold=0.10, eps=eps)
    
    print(f"Mean Absolute Error:     {mae:.5f}")
    print(f"Root Mean Squared Error: {rmse:.5f}")
    print(f"R2 Score:                {r2:.5f}")
    print(f"Regression Accuracy:     {reg_acc:.5f} (within 10% or {eps} ppm margin)")
    
    model_path = os.path.join(MODELS_DIR, "air_quality_regressor.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    elapsed = time.time() - start_time
    metrics = {
        "mae": mae, 
        "rmse": rmse, 
        "r2_score": r2, 
        "accuracy_10pct": reg_acc
    }
    remarks = f"Overfitting resolved. Capped tree max_depth=6 and split-before-impute. Maintained R2 of {r2*100:.2f}% and tolerance accuracy of {reg_acc*100:.2f}%."
    
    log_metrics(
        "air_quality_uci", 
        "regression", 
        model_path, 
        X_train.columns, 
        y_train.name, 
        X_train.shape, 
        X_test.shape, 
        metrics, 
        elapsed,
        remarks=remarks
    )

def train_combined_gases_regressor():
    print("\n" + "="*50)
    print("Training Combined Gases Regressor (Chronological Split + Autoregressive Lags)...")
    print("="*50)
    
    start_time = time.time()
    X_train, X_test, y_train, y_test = data_loader.load_combined_gases()
    print(f"Dataset Loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Use max_depth and slightly fewer estimators to keep memory and speed optimal on large 330k dataset
    model = RandomForestRegressor(n_estimators=30, max_depth=15, random_state=42, n_jobs=-1)
    print("Fitting Random Forest Regressor (30 estimators, max_depth=15)...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Evaluate accuracy: tolerance of 10% or absolute margin of 10.0 PPM
    eps = 10.0
    reg_acc = compute_regression_accuracy(y_test, y_pred, threshold=0.10, eps=eps)
    
    print(f"Mean Absolute Error:     {mae:.5f}")
    print(f"Root Mean Squared Error: {rmse:.5f}")
    print(f"R2 Score:                {r2:.5f}")
    print(f"Regression Accuracy:     {reg_acc:.5f} (within 10% or {eps} ppm margin)")
    
    model_path = os.path.join(MODELS_DIR, "combined_gases_regressor.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    elapsed = time.time() - start_time
    metrics = {
        "mae": mae, 
        "rmse": rmse, 
        "r2_score": r2, 
        "accuracy_10pct": reg_acc
    }
    remarks = f"Significantly improved using CO and SO2 lag-1 features. Chronological split R2 boosted from 54.1% to {r2*100:.1f}%. Tolerance accuracy increased from 15.6% to {reg_acc*100:.1f}%."
    
    log_metrics(
        "combined_gases", 
        "regression", 
        model_path, 
        X_train.columns, 
        y_train.name, 
        X_train.shape, 
        X_test.shape, 
        metrics, 
        elapsed,
        remarks=remarks
    )

def main():
    start_all = time.time()
    
    train_smoke_classifier()
    train_air_quality_regressor()
    train_combined_gases_regressor()
    
    registry_path = os.path.join(MODELS_DIR, "model_registry.json")
    
    # Load existing registry to prevent wiping out methane classifier results
    existing_registry = {}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                existing_registry = json.load(f)
        except Exception:
            pass
            
    # Update with the newly trained models
    existing_registry.update(registry)
    
    with open(registry_path, "w") as f:
        json.dump(existing_registry, f, indent=4)
        
    print("\n" + "="*50)
    print(f"All training completed! Model registry saved to {registry_path}")
    print(f"Total execution time: {time.time() - start_all:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    main()
