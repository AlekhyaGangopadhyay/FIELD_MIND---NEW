import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

def evaluate_classifier(model_path, metadata_path, test_path, name):
    """
    Evaluates the Random Forest Classifier on a test set.
    """
    print(f"\n--- Evaluating Classifier on {name} ---")
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    
    df = pd.read_csv(test_path)
    features = metadata['features']
    
    X = df[features]
    y = df['Occupancy']
    
    # Predict
    t0 = time.time()
    y_pred = model.predict(X)
    latency = (time.time() - t0) * 1000.0 / len(X)
    
    y_prob = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Batch inference latency: {latency:.4f} ms per sample")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'batch_latency_ms': latency
    }

def evaluate_anomaly_detector(model_path, metadata_path, dataset_path, name, temp_range=(5.0, 28.0), hum_range=(15.0, 85.0), co2_max=None):
    """
    Evaluates the Isolation Forest model on a given dataset and reports detection rates
    and alignment with domain safety thresholds (Knowledge Grounding).
    """
    model_basename = os.path.basename(model_path)
    print(f"\n--- Evaluating Anomaly Detector [{model_basename}] on {name} ---")
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    
    df = pd.read_csv(dataset_path)
    features = metadata['features']
    
    # Map column names if they are capitalized (like in the UCI dataset) or vice versa
    rename_dict = {
        'Temperature': 'temp',
        'Humidity': 'humidity',
        'Temperature_roll_mean_5': 'temp_roll_mean_5',
        'Temperature_roll_std_5': 'temp_roll_std_5',
        'Humidity_roll_mean_5': 'humidity_roll_mean_5',
        'Humidity_roll_std_5': 'humidity_roll_std_5',
        'Temperature_roll_mean_30': 'temp_roll_mean_30',
        'Temperature_roll_std_30': 'temp_roll_std_30',
        'Humidity_roll_mean_30': 'humidity_roll_mean_30',
        'Humidity_roll_std_30': 'humidity_roll_std_30',
        # And backwards mapping if evaluating uci model on iot dataset
        'temp': 'Temperature',
        'humidity': 'Humidity',
        'temp_roll_mean_5': 'Temperature_roll_mean_5',
        'temp_roll_std_5': 'Temperature_roll_std_5',
        'humidity_roll_mean_5': 'Humidity_roll_mean_5',
        'humidity_roll_std_5': 'Humidity_roll_std_5',
        'temp_roll_mean_30': 'Temperature_roll_mean_30',
        'temp_roll_std_30': 'Temperature_roll_std_30',
        'humidity_roll_mean_30': 'Humidity_roll_mean_30',
        'humidity_roll_std_30': 'Humidity_roll_std_30'
    }
    
    df_mapped = df.copy()
    # Rename columns to ensure expected features exist
    for k, v in rename_dict.items():
        if k in df.columns and v not in df.columns:
            df_mapped[v] = df[k]
            
    # Map features
    for feat in features:
        if feat not in df_mapped.columns:
            if 'CO2_roll_std' in feat:
                df_mapped[feat] = 5.0
            elif 'CO2' in feat:
                df_mapped[feat] = 690.5
            elif 'temp_co2_product' in feat:
                df_mapped[feat] = 20.6 * 690.5
            else:
                df_mapped[feat] = 0.0
                
    X = df_mapped[features].fillna(0.0)
    
    # Predict (-1: anomaly, 1: normal)
    t0 = time.time()
    preds = model.predict(X)
    latency = (time.time() - t0) * 1000.0 / len(X)
    
    n_anomalies = np.sum(preds == -1)
    pct_anomalies = n_anomalies / len(df) * 100.0
    print(f"Total samples: {len(df)}")
    print(f"Anomalies detected: {n_anomalies} ({pct_anomalies:.2f}%)")
    
    # Align with custom knowledge-grounded safety rules:
    temp_col = 'Temperature' if 'Temperature' in df.columns else 'temp'
    hum_col = 'Humidity' if 'Humidity' in df.columns else 'humidity'
    
    is_extreme = (df[temp_col] > temp_range[1]) | (df[temp_col] < temp_range[0]) | (df[hum_col] > hum_range[1]) | (df[hum_col] < hum_range[0])
    if co2_max is not None and 'CO2' in df.columns:
        is_extreme = is_extreme | (df['CO2'] > co2_max)
        
    y_true = np.where(is_extreme, -1, 1)
    
    f1 = f1_score(y_true, preds, pos_label=-1, zero_division=0)
    prec = precision_score(y_true, preds, pos_label=-1, zero_division=0)
    rec = recall_score(y_true, preds, pos_label=-1, zero_division=0)
    
    print(f"Alignment with Safety Guidelines (Anomaly Class F1): {f1:.4f}")
    print(f"Safety anomaly precision: {prec:.4f}, recall: {rec:.4f}")
    print(f"Batch inference latency: {latency:.4f} ms per sample")
    
    return {
        'total_samples': len(df),
        'n_anomalies': n_anomalies,
        'pct_anomalies': pct_anomalies,
        'safety_alignment_f1': f1,
        'safety_precision': prec,
        'safety_recall': rec,
        'batch_latency_ms': latency
    }

def profile_edge_suitability(models_dir):
    """
    Profiles model file sizes and single-sample execution latency (real-time stream simulation)
    to verify Jetson Nano edge deployment viability.
    """
    print("\n--- Edge Resource & Latency Profiling ---")
    
    models = ["random_forest.joblib", "isolation_forest_iot.joblib", "isolation_forest_uci.joblib"]
    for model_name in models:
        path = os.path.join(models_dir, model_name)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024.0
            print(f"Model: {model_name}")
            print(f"  Storage Size: {size_kb:.2f} KB")
            
            # Load model and profile single-sample prediction latency
            model = joblib.load(path)
            metadata = joblib.load(path.replace(".joblib", "_metadata.joblib"))
            features = metadata['features']
            
            # Generate a single dummy test sample
            dummy_sample = pd.DataFrame([np.random.randn(len(features))], columns=features)
            
            # Warm up
            _ = model.predict(dummy_sample)
            
            # Run 1000 single-sample inference cycles
            n_trials = 1000
            t0 = time.time()
            for _ in range(n_trials):
                _ = model.predict(dummy_sample)
            elapsed = (time.time() - t0) * 1000.0 # ms
            avg_latency = elapsed / n_trials
            
            print(f"  Average Single-Sample Latency (1000 trials): {avg_latency:.4f} ms")

if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_dir = os.path.join(proj_dir, "data", "data_clean")
    models_dir = os.path.join(proj_dir, "models")
    
    # 1. Evaluate Random Forest Occupancy Classifier
    rf_model = os.path.join(models_dir, "random_forest.joblib")
    rf_metadata = os.path.join(models_dir, "random_forest_metadata.joblib")
    
    if os.path.exists(rf_model):
        test_path = os.path.join(clean_dir, "uci_test_clean.csv")
        test2_path = os.path.join(clean_dir, "uci_test2_clean.csv")
        
        if os.path.exists(test_path):
            evaluate_classifier(rf_model, rf_metadata, test_path, "UCI Test Set 1")
        if os.path.exists(test2_path):
            evaluate_classifier(rf_model, rf_metadata, test2_path, "UCI Test Set 2 (Out-of-Distribution)")
            
    # 2. Evaluate Anomaly Detectors
    if_iot_model = os.path.join(models_dir, "isolation_forest_iot.joblib")
    if_iot_metadata = os.path.join(models_dir, "isolation_forest_iot_metadata.joblib")
    if_uci_model = os.path.join(models_dir, "isolation_forest_uci.joblib")
    if_uci_metadata = os.path.join(models_dir, "isolation_forest_uci_metadata.joblib")
    
    iot_test_path = os.path.join(clean_dir, "iot_telemetry_clean.csv")
    uci_test_path = os.path.join(clean_dir, "uci_test_clean.csv")
    rpi_test_path = os.path.join(clean_dir, "log_temp_clean.csv")
    
    # Evaluate IoT Anomaly Detector
    if os.path.exists(if_iot_model):
        if os.path.exists(iot_test_path):
            evaluate_anomaly_detector(if_iot_model, if_iot_metadata, iot_test_path, "IoT Telemetry Dataset", 
                                      temp_range=(5.0, 28.0), hum_range=(15.0, 85.0))
        if os.path.exists(rpi_test_path):
            # Show cross-domain performance of generic model on Raspberry PI edge logs (expect 100% anomaly due to domain shift)
            evaluate_anomaly_detector(if_iot_model, if_iot_metadata, rpi_test_path, "Raspberry PI Logs (Cross-Domain)", 
                                      temp_range=(19.5, 22.5), hum_range=(18.0, 35.0))
            
    # Evaluate UCI Anomaly Detector
    if os.path.exists(if_uci_model):
        if os.path.exists(uci_test_path):
            evaluate_anomaly_detector(if_uci_model, if_uci_metadata, uci_test_path, "UCI Environmental Dataset (In-Domain)", 
                                      temp_range=(19.5, 22.5), hum_range=(18.0, 35.0), co2_max=1000.0)
        if os.path.exists(rpi_test_path):
            # Show cross-domain performance of a model trained on similar distribution (expect reasonable anomaly rate)
            evaluate_anomaly_detector(if_uci_model, if_uci_metadata, rpi_test_path, "Raspberry PI Logs (Respective-Domain)", 
                                      temp_range=(19.5, 22.5), hum_range=(18.0, 35.0))
            
    # 3. Profile Edge Suitability
    profile_edge_suitability(models_dir)
