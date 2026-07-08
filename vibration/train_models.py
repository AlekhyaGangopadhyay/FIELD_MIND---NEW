import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Algorithms
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_csv = os.path.join(script_dir, "data", "vibration_features.csv")
    models_dir = os.path.join(script_dir, "models")
    output_table_path = os.path.join(script_dir, "model_metrics_table.md")
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading preprocessed features...")
    df = pd.read_csv(features_csv)
    
    # 1. One-hot encode categorical feature `trid` (trace channel direction)
    # trid: 12=Z, 13=EW, 14=NS
    df = pd.get_dummies(df, columns=['trid'], prefix='trid')
    # Convert dummy columns to integers
    for col in ['trid_12', 'trid_13', 'trid_14']:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            df[col] = 0 # Fallback if some categories are missing (unlikely)
            
    print("Data Columns after dummy encoding:", df.columns.tolist())
    
    # 2. Define target variables
    # Classification: vibration_hazard (1 if PPV > 1.0, 0 otherwise)
    # Regression: log_ppv = ln(PPV)
    y_class = df['vibration_hazard'].values
    y_reg = np.log(df['ppv'].values)
    
    # 3. Define the three feature sets
    feat_set_1 = ['offset', 'max_charge', 'total_charge', 'num_holes', 'detonator_code', 'trid_12', 'trid_13', 'trid_14']
    feat_set_2 = feat_set_1 + ['gx', 'gy', 'gelev', 'sx', 'sy', 'selev']
    feat_set_3 = feat_set_2 + ['scaled_distance_usbm', 'scaled_distance_langefors', 'elevation_diff']
    
    # Map feature sets to names for the table
    feat_names = {
        1: "Standard (Offset, Charges, Holes, Detonator, Components)",
        2: "Standard + Spatial Coordinates (Source/Receiver XYZ)",
        3: "Standard + Spatial + Scaled Distances (USBM, LK, Elevation Diff)"
    }
    
    print("\n--- Training CLASSIFICATION Models (Target: PPV > 1.0 mm/s) ---")
    classification_results = []
    
    # Split for classification
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(df, y_class, test_size=0.2, random_state=42, stratify=y_class)
    
    # Classification config: (Model Name, Algorithm instance, Feature set key)
    class_models = [
        ("Logistic Regression (Baseline)", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)), 1),
        ("Decision Tree Classifier", DecisionTreeClassifier(max_depth=8, random_state=42), 1),
        ("Random Forest Classifier", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42), 2),
        ("Gradient Boosting Classifier", GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42), 3),
        ("Multi-Layer Perceptron (MLP) Classifier", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)), 3)
    ]
    
    best_clf = None
    best_clf_acc = -1.0
    best_clf_name = ""
    best_clf_feat_key = 1
    
    for name, clf, feat_key in class_models:
        feat_list = feat_set_1 if feat_key == 1 else (feat_set_2 if feat_key == 2 else feat_set_3)
        X_tr = X_train_c[feat_list].values
        X_te = X_test_c[feat_list].values
        
        print(f"Training {name} using {len(feat_list)} features...")
        clf.fit(X_tr, y_train_c)
        
        train_acc = clf.score(X_tr, y_train_c)
        test_acc = clf.score(X_te, y_test_c)
        print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        if test_acc > best_clf_acc:
            best_clf_acc = test_acc
            best_clf = clf
            best_clf_name = name
            best_clf_feat_key = feat_key
        
        classification_results.append({
            'model_name': name,
            'algo_used': clf.__class__.__name__ if not hasattr(clf, 'steps') else clf.steps[-1][1].__class__.__name__,
            'feature_used': feat_names[feat_key],
            'train_accuracy': f"{train_acc * 100:.2f}%",
            'test_accuracy': f"{test_acc * 100:.2f}%"
        })
        
    print("\n--- Training REGRESSION Models (Target: ln(PPV)) ---")
    regression_results = []
    
    # Split for regression
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(df, y_reg, test_size=0.2, random_state=42)
    
    # Regression config: (Model Name, Algorithm instance, Feature set key)
    reg_models = [
        ("Ridge Regression (Baseline)", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), 1),
        ("Decision Tree Regressor", DecisionTreeRegressor(max_depth=8, random_state=42), 1),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42), 2),
        ("Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42), 3),
        ("Multi-Layer Perceptron (MLP) Regressor", make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)), 3)
    ]
    
    best_reg = None
    best_reg_r2 = -1.0
    best_reg_name = ""
    best_reg_feat_key = 1
    
    for name, reg, feat_key in reg_models:
        feat_list = feat_set_1 if feat_key == 1 else (feat_set_2 if feat_key == 2 else feat_set_3)
        X_tr = X_train_r[feat_list].values
        X_te = X_test_r[feat_list].values
        
        print(f"Training {name} using {len(feat_list)} features...")
        reg.fit(X_tr, y_train_r)
        
        train_r2 = reg.score(X_tr, y_train_r)
        test_r2 = reg.score(X_te, y_test_r)
        print(f"  Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}")
        
        if test_r2 > best_reg_r2:
            best_reg_r2 = test_r2
            best_reg = reg
            best_reg_name = name
            best_reg_feat_key = feat_key
        
        regression_results.append({
            'model_name': name,
            'algo_used': reg.__class__.__name__ if not hasattr(reg, 'steps') else reg.steps[-1][1].__class__.__name__,
            'feature_used': feat_names[feat_key],
            'train_accuracy': f"{train_r2:.4f} (R2)",
            'test_accuracy': f"{test_r2:.4f} (R2)"
        })
        
    # Clean up and save only the best models
    print(f"\nCleaning up existing models in {models_dir}...")
    for filename in os.listdir(models_dir):
        if filename.endswith(".joblib"):
            file_path = os.path.join(models_dir, filename)
            try:
                os.remove(file_path)
                print(f"  Deleted: {filename}")
            except Exception as e:
                print(f"  Error deleting {filename}: {e}")
                
    print(f"\nSaving best classification model: {best_clf_name} (Acc: {best_clf_acc:.4f})")
    clean_clf_name = "best_" + best_clf_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(best_clf, os.path.join(models_dir, f"{clean_clf_name}.joblib"))
    
    print(f"Saving best regression model: {best_reg_name} (R2: {best_reg_r2:.4f})")
    clean_reg_name = "best_" + best_reg_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(best_reg, os.path.join(models_dir, f"{clean_reg_name}.joblib"))
        
    print("\nStep 5: Writing metrics comparison table to markdown file...")
    # Generate Markdown Table content
    md_content = []
    md_content.append("# Blast Vibration Prediction Model Metrics\n")
    md_content.append("This document summarizes the performance of the trained machine learning models on the Mount Erzberg production blast vibration dataset. Models were trained using features from trace headers and coordinates merged with aggregated blasthole information from `BLASTS.txt`.\n")
    
    md_content.append("## 1. Classification Models")
    md_content.append("*Target variable*: **Vibration Hazard** (`vibration_hazard = 1` if Peak Particle Velocity (PPV) > 1.0 mm/s, else `0`)\n")
    md_content.append("| Model Name | Algorithm Used | Features Used | Train Accuracy | Test Accuracy | Status |")
    md_content.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in classification_results:
        status = "⭐⭐ **Best (Saved)**" if r['model_name'] == best_clf_name else "Not Saved"
        md_content.append(f"| {r['model_name']} | {r['algo_used']} | {r['feature_used']} | {r['train_accuracy']} | {r['test_accuracy']} | {status} |")
    
    md_content.append("\n## 2. Regression Models")
    md_content.append("*Target variable*: **ln(PPV)** (log-transformed Peak Particle Velocity in mm/s)\n")
    md_content.append("| Model Name | Algorithm Used | Features Used | Train R2 Score | Test R2 Score | Status |")
    md_content.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in regression_results:
        status = "⭐⭐ **Best (Saved)**" if r['model_name'] == best_reg_name else "Not Saved"
        md_content.append(f"| {r['model_name']} | {r['algo_used']} | {r['feature_used']} | {r['train_accuracy']} | {r['test_accuracy']} | {status} |")
        
    with open(output_table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    print(f"Metrics table successfully written to {output_table_path}")

if __name__ == "__main__":
    main()
