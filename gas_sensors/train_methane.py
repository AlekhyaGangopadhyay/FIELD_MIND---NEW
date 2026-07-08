import os
import json
import time
from datetime import datetime
import joblib
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import warnings

# Suppress future warnings from SVC probability parameter
warnings.filterwarnings("ignore", category=FutureWarning)

# Import our workspace data loader
import data_loader

script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(script_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    print("="*60)
    print("METHANE GAS CLASSIFIER TRAINING SUITE (HYBRID SVM + MLP VOTING)")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Load MQ4 Methane Gas dataset (batches 1-8 train, batches 9-10 test)
    X_train, X_test, y_train, y_test = data_loader.load_mq4_dataset()
    print(f"Dataset loaded successfully.")
    print(f"Train Shape (Batches 1-8): {X_train.shape}")
    print(f"Test Shape (Batches 9-10): {X_test.shape}")
    
    # 2. Define the Hybrid Base Models
    # We ensemble five classifiers to improve robustness under temporal drift:
    # 2x RBF SVMs (C=10.0, 20.0), 1x Linear SVM (C=1.0), and 2x MLPs (hidden size 256x128 and 128x64)
    svm10_model = SVC(kernel='rbf', C=10.0, probability=True, random_state=42)
    svm20_model = SVC(kernel='rbf', C=20.0, probability=True, random_state=42)
    svm_lin_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    mlp256_model = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=0.01, max_iter=500, random_state=42)
    mlp128_model = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.01, max_iter=500, random_state=42)
    
    # Combined inside a Soft Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('svm10', svm10_model),
            ('svm20', svm20_model),
            ('svm_lin', svm_lin_model),
            ('mlp256', mlp256_model),
            ('mlp128', mlp128_model)
        ],
        voting='soft',
        n_jobs=-1
    )
    
    # Unified Pipeline so features are scaled once and passed to the ensemble
    hybrid_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('voting', voting_clf)
    ])
    
    # 3. Stratified 8-Fold Cross-Validation to evaluate CV accuracy
    cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    
    print("\nEvaluating the optimized hybrid ensemble model via 8-fold Cross-Validation...")
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(hybrid_pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    mean_cv_score = np.mean(cv_scores)
    
    print("-" * 40)
    print(f"  8-Fold CV Mean Train Acc: {mean_cv_score:.5f}")
    print("-" * 40)
    
    # 4. Fit the model on the full training set (Batches 1-8)
    print("\nFitting final model on the entire training set...")
    hybrid_pipeline.fit(X_train, y_train)
    best_model = hybrid_pipeline
    
    # 5. Evaluate the best hybrid model on test set (new collection batches 9-10)
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nEvaluation on Test Set (Batches 9-10):")
    print(f"  Accuracy:        {acc:.5f}")
    print(f"  Macro Precision: {prec:.5f}")
    print(f"  Macro Recall:    {rec:.5f}")
    print(f"  Macro F1-Score:  {f1:.5f}")
    
    # 5. Save the trained hybrid pipeline model
    model_path = os.path.join(MODELS_DIR, "mq4_gas_classifier.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nModel pipeline successfully serialized and saved to: {model_path}")
    
    # 6. Update registry entry
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
    
    registry["mq4_gas_classification"] = {
        "task_type": "multiclass_classification",
        "model_path": model_path,
        "features": list(X_train.columns),
        "targets": [y_train.name],
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "metrics": {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_score_macro": f1,
            "best_cv_params": {
                "svm10_C": 10.0,
                "svm20_C": 20.0,
                "svm_lin_C": 1.0,
                "mlp_alpha": 0.01,
                "mlp256_hidden": [256, 128],
                "mlp128_hidden": [128, 64]
            },
            "best_cv_score": mean_cv_score
        },
        "remarks": "Optimized Hybrid Soft Voting 5-model ensemble combining 2x RBF SVM, 1x Linear SVM, and 2x MLP Classifiers, evaluated using 8-fold cross-validation.",
        "training_time_sec": round(elapsed, 2),
        "trained_at": datetime.now().isoformat()
    }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)
    print(f"Model registry entry updated in {registry_path}")
    print("="*60)

if __name__ == "__main__":
    main()
