import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

# Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    models_dir = os.path.join(script_dir, "models")
    metrics_table_path = os.path.join(script_dir, "model_metrics_table.md")
    
    os.makedirs(models_dir, exist_ok=True)
    
    datasets = {
        2: "sensor_readings_2.csv",
        4: "sensor_readings_4.csv",
        24: "sensor_readings_24.csv"
    }
    
    results = []
    
    for num_features, filename in datasets.items():
        csv_path = os.path.join(data_dir, filename)
        if not os.path.exists(csv_path):
            print(f"Error: Dataset {filename} not found in {data_dir}. Skipping...")
            continue
            
        print(f"\n==========================================")
        print(f"Processing Dataset: {filename} ({num_features} Features)")
        print(f"==========================================")
        
        # Load and clean DataFrame
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Target is always 'Class'
        target_col = 'Class'
        if target_col not in df.columns:
            raise KeyError(f"Expected target column '{target_col}' not found in {filename}")
            
        df[target_col] = df[target_col].str.strip()
        
        # Split into features X and target y
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        
        # Encode label strings to integers
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Save label encoder class mappings
        class_mapping = {int(i): label for i, label in enumerate(le.classes_)}
        print("Class mapping:", class_mapping)
        
        # Split data 80% train, 20% test with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Define classifiers to train
        classifiers = [
            ("Logistic Regression (Baseline)", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))),
            ("Decision Tree Classifier", DecisionTreeClassifier(max_depth=10, random_state=42)),
            ("Random Forest Classifier", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)),
            ("Gradient Boosting Classifier", GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)),
            ("Multi-Layer Perceptron (MLP) Classifier", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)))
        ]
        
        best_model = None
        best_acc = -1.0
        best_model_name = ""
        
        for name, clf in classifiers:
            print(f"Training {name}...")
            clf.fit(X_train, y_train)
            
            train_acc = clf.score(X_train, y_train)
            test_acc = clf.score(X_test, y_test)
            print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = clf
                best_model_name = name
                
            results.append({
                'dataset': f"{num_features}-sensor configuration",
                'model_name': name,
                'algo_used': clf.__class__.__name__ if not hasattr(clf, 'steps') else clf.steps[-1][1].__class__.__name__,
                'train_acc': f"{train_acc * 100:.2f}%",
                'test_acc': f"{test_acc * 100:.2f}%",
                'is_best': False # Will set after finding the best for each dataset
            })
            
        # Save the best model for this dataset
        print(f"\nSaving best model for {num_features}-sensor dataset: {best_model_name} (Test Acc: {best_acc * 100:.2f}%)")
        model_filename = f"best_ultrasonic_{num_features}.joblib"
        model_path = os.path.join(models_dir, model_filename)
        
        # Save model along with its class mapping as metadata
        model_data = {
            'model': best_model,
            'classes': class_mapping,
            'features': X.columns.tolist()
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        
        # Update results list to mark the best model
        for r in results:
            if r['dataset'] == f"{num_features}-sensor configuration" and r['model_name'] == best_model_name:
                r['is_best'] = True
                
    # Generate model_metrics_table.md
    print("\nWriting metrics table to markdown file...")
    md_content = []
    md_content.append("# Robot Navigation Ultrasonic Sensor Classification Metrics\n")
    md_content.append("This document summarizes the performance of various machine learning models trained on the SCITOS-G5 robot navigation dataset using 2, 4, and 24 ultrasonic sensor configurations.\n")
    md_content.append("The target variable is the robot navigation decision Class: `Move-Forward`, `Slight-Right-Turn`, `Sharp-Right-Turn`, or `Slight-Left-Turn`.\n")
    
    current_dataset = None
    for r in results:
        if r['dataset'] != current_dataset:
            current_dataset = r['dataset']
            md_content.append(f"\n## {current_dataset}")
            md_content.append("| Model Name | Algorithm Used | Train Accuracy | Test Accuracy | Status |")
            md_content.append("| :--- | :--- | :--- | :--- | :--- |")
            
        status = "⭐⭐ **Best (Saved)**" if r['is_best'] else "Not Saved"
        md_content.append(f"| {r['model_name']} | {r['algo_used']} | {r['train_acc']} | {r['test_acc']} | {status} |")
        
    with open(metrics_table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
        
    print(f"Metrics table successfully written to {metrics_table_path}")

if __name__ == "__main__":
    main()
