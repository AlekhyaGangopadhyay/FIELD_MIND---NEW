"""Train the production eight-channel multi-gas presence detector."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from dl_wrappers import PyTorchHazardClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data" / "multi_gas_detector_real_v2.csv"
MODEL_PATH = SCRIPT_DIR / "models" / "multi_gas_detector.joblib"
REGISTRY_PATH = SCRIPT_DIR / "models" / "model_registry.json"
FEATURES = ["CH4_ppm", "CO_ppm", "CO2_ppm", "H2_ppm", "H2S_ppm", "NH3_ppm", "LPG_ppm", "CNG_ppm"]
TARGETS = ["target_Methane", "target_CO", "target_CO2", "target_H2", "target_H2S", "target_NH3", "target_LPG", "target_CNG"]


def train_detector(epochs: int = 40) -> dict:
    frame = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + TARGETS).difference(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    X = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = frame[TARGETS].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    positives = y_train.sum(axis=0).to_numpy(dtype=float)
    negatives = len(y_train) - positives
    weights = np.sqrt(negatives / np.maximum(positives, 1.0))

    model = PyTorchHazardClassifier(
        in_features=len(FEATURES),
        out_features=len(TARGETS),
        binary=True,
        epochs=epochs,
        batch_size=256,
        lr=2e-3,
        pos_weight=weights,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_true = y_test.to_numpy()
    elementwise = float((y_true == predictions).mean())
    exact = float(np.all(y_true == predictions, axis=1).mean())
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        registry = {}
    registry["multi_gas_detector"] = {
        "task_type": "multilabel_classification",
        "model_type": "PyTorch_DeepHazardNet",
        "model_path": "gas_sensors/models/multi_gas_detector.joblib",
        "features": FEATURES,
        "targets": TARGETS,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "metrics": {
            "elementwise_accuracy": elementwise,
            "exact_subset_accuracy": exact,
            "precision_macro": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
            "per_target_accuracy": {
                target: float((y_true[:, i] == predictions[:, i]).mean())
                for i, target in enumerate(TARGETS)
            },
        },
        "remarks": "Eight-output multi-label detector trained from multi_gas_detector_real_v2.csv.",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=4) + "\n", encoding="utf-8")
    return registry["multi_gas_detector"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    result = train_detector(args.epochs)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

