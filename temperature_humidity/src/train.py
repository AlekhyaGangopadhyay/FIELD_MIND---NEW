import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

def train_isolation_forest_iot(iot_clean_path, models_dir):
    """
    Trains and saves an optimized Isolation Forest model for IoT Telemetry dataset.
    """
    print("Loading preprocessed IoT Telemetry data for Isolation Forest...")
    df = pd.read_csv(iot_clean_path)
    
    features = [
        'temp', 'humidity', 'temp_hum_product', 'temp_hum_ratio', 'humidex',
        'temp_roll_mean_5', 'temp_roll_std_5', 'humidity_roll_mean_5', 'humidity_roll_std_5'
    ]
    
    X = df[features].fillna(0.0)
    
    print(f"Fitting IoT Isolation Forest on {min(len(X), 100000)} samples...")
    X_train = X.sample(n=min(len(X), 100000), random_state=42)
    
    # Custom search
    best_f1 = 0
    best_params = {}
    best_clf = None
    
    contamination_grid = [0.01, 0.05, 0.10]
    n_estimators_grid = [50, 100]
    
    for contamination in contamination_grid:
        for n_estimators in n_estimators_grid:
            clf = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_train)
            
            # Validation set
            val_sample = df.sample(n=20000, random_state=100)
            X_val = val_sample[features].fillna(0.0)
            val_preds = clf.predict(X_val)
            
            y_val_true = np.where((val_sample['temp'] > 28.0) | (val_sample['temp'] < 5.0) | (val_sample['humidity'] > 85.0) | (val_sample['humidity'] < 15.0), -1, 1)
            score = f1_score(y_val_true, val_preds, pos_label=-1, zero_division=0)
            
            if score > best_f1:
                best_f1 = score
                best_params = {'n_estimators': n_estimators, 'contamination': contamination}
                best_clf = clf
                
    if best_clf is None:
        best_clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1).fit(X_train)
        best_params = {'n_estimators': 100, 'contamination': 0.05}
        
    print(f"Best IoT Isolation Forest parameters: {best_params} with Validation F1 of {best_f1:.4f}")
    
    model_path = os.path.join(models_dir, "isolation_forest_iot.joblib")
    joblib.dump(best_clf, model_path)
    joblib.dump({'features': features, 'best_params': best_params, 'model_type': 'IsolationForest'}, 
                os.path.join(models_dir, "isolation_forest_iot_metadata.joblib"))
    print(f"IoT Isolation Forest model saved to {model_path}")
    return model_path

def train_isolation_forest_uci(uci_train_path, models_dir):
    """
    Trains and saves an optimized Isolation Forest model for the UCI dataset.
    """
    print("Loading preprocessed UCI training data for Isolation Forest...")
    df = pd.read_csv(uci_train_path)
    
    features = [
        'Temperature', 'Humidity', 'temp_hum_product', 'temp_hum_ratio', 'humidex',
        'Temperature_roll_mean_5', 'Temperature_roll_std_5', 'Humidity_roll_mean_5', 'Humidity_roll_std_5'
    ]
    if 'CO2' in df.columns:
        features.extend(['CO2', 'CO2_roll_mean_5', 'CO2_roll_std_5', 'temp_co2_product'])
        
    X = df[features].fillna(0.0)
    
    # Custom search based on UCI safety rules
    best_f1 = 0
    best_params = {}
    best_clf = None
    
    contamination_grid = [0.01, 0.05, 0.10]
    n_estimators_grid = [50, 100]
    
    for contamination in contamination_grid:
        for n_estimators in n_estimators_grid:
            clf = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X)
            
            preds = clf.predict(X)
            
            # Anomaly rule for UCI: temp > 22.5, temp < 19.5, hum > 35.0, hum < 18.0, or CO2 > 1000
            is_extreme = (df['Temperature'] > 22.5) | (df['Temperature'] < 19.5) | (df['Humidity'] > 35.0) | (df['Humidity'] < 18.0)
            if 'CO2' in df.columns:
                is_extreme = is_extreme | (df['CO2'] > 1000.0)
            y_true = np.where(is_extreme, -1, 1)
            
            score = f1_score(y_true, preds, pos_label=-1, zero_division=0)
            
            if score > best_f1:
                best_f1 = score
                best_params = {'n_estimators': n_estimators, 'contamination': contamination}
                best_clf = clf
                
    if best_clf is None:
        best_clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1).fit(X)
        best_params = {'n_estimators': 100, 'contamination': 0.05}
        
    print(f"Best UCI Isolation Forest parameters: {best_params} with Train F1 of {best_f1:.4f}")
    
    model_path = os.path.join(models_dir, "isolation_forest_uci.joblib")
    joblib.dump(best_clf, model_path)
    joblib.dump({'features': features, 'best_params': best_params, 'model_type': 'IsolationForest'}, 
                os.path.join(models_dir, "isolation_forest_uci_metadata.joblib"))
    print(f"UCI Isolation Forest model saved to {model_path}")
    return model_path

def train_random_forest(uci_train_path, models_dir):
    """
    Trains and optimizes a Random Forest Classifier on the UCI Occupancy Detection dataset.
    """
    print("Loading preprocessed UCI training data for Random Forest Classifier...")
    df = pd.read_csv(uci_train_path)
    
    X = df.drop(columns=['date', 'Occupancy'], errors='ignore')
    y = df['Occupancy']
    
    features = X.columns.tolist()
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    print("Running GridSearchCV for Random Forest (5-fold CV)...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y)
    
    print(f"Best Random Forest Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation F1-score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    model_path = os.path.join(models_dir, "random_forest.joblib")
    joblib.dump(best_model, model_path)
    joblib.dump({'features': features, 'best_params': grid_search.best_params_, 'best_cv_score': grid_search.best_score_, 'model_type': 'RandomForestClassifier'}, 
                os.path.join(models_dir, "random_forest_metadata.joblib"))
    print(f"Random Forest Classifier saved to {model_path}")
    return model_path

if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_dir = os.path.join(proj_dir, "data", "data_clean")
    models_dir = os.path.join(proj_dir, "models")
    
    iot_clean_path = os.path.join(clean_dir, "iot_telemetry_clean.csv")
    uci_train_path = os.path.join(clean_dir, "uci_train_clean.csv")
    
    if os.path.exists(iot_clean_path):
        train_isolation_forest_iot(iot_clean_path, models_dir)
        
    if os.path.exists(uci_train_path):
        train_isolation_forest_uci(uci_train_path, models_dir)
        train_random_forest(uci_train_path, models_dir)
