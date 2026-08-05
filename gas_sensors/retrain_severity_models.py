"""Retrain the five production gas-severity heads from corrected labels.

The corrected datasets in ``gas_sensors/data`` are the source of truth.  This
script deliberately trains from the concentration-based ``severity`` column,
not from the old location/band index.  It is safe to rerun and updates the
model registry only after each model has been evaluated and serialized.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from dl_wrappers import PyTorchSeverityClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"

GAS_CONFIG: Dict[str, Tuple[float, float, float]] = {
    "ch4": (10_000.0, 15_000.0, 25_000.0),
    "co": (25.0, 50.0, 50.0),
    "co2": (1_000.0, 4_000.0, 5_000.0),
    "h2": (4_000.0, 20_000.0, 20_000.0),
    "h2s": (10.0, 20.0, 20.0),
}


def _load_dataset(gas: str) -> tuple[pd.DataFrame, str]:
    path = DATA_DIR / f"mine_part2_{gas}_balanced_cgan_corrected.csv"
    if not path.exists():
        raise FileNotFoundError(f"corrected severity dataset not found: {path}")
    frame = pd.read_csv(path)
    required = {"ppm", "severity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    feature = "ppm_noisy" if "ppm_noisy" in frame.columns else "ppm"
    frame = frame[[feature, "severity"]].replace([np.inf, -np.inf], np.nan).dropna()
    frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame["severity"] = pd.to_numeric(frame["severity"], errors="coerce")
    frame = frame.dropna().astype({"severity": int})
    if not set(frame["severity"].unique()).issubset({0, 1, 2}):
        raise ValueError(f"{path.name} contains labels outside 0, 1, 2")
    return frame, feature


def train_one(gas: str, epochs: int = 50, test_size: float = 0.25) -> dict:
    frame, source_feature = _load_dataset(gas)
    X = frame[[source_feature]].rename(columns={source_feature: "ppm"})
    y = frame["severity"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    classifier = PyTorchSeverityClassifier(
        in_features=1,
        num_classes=3,
        epochs=epochs,
        batch_size=256,
        lr=2e-3,
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    model_path = MODELS_DIR / f"severity_{gas}.joblib"
    joblib.dump(classifier, model_path)

    lower, upper, tlv = GAS_CONFIG[gas]
    return {
        "task_type": "multiclass_classification",
        "model_type": "PyTorch_Deep_MLP",
        "model_path": f"gas_sensors/models/severity_{gas}.joblib",
        "features": ["ppm"],
        "targets": ["severity"],
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "metrics": {
            "accuracy": float(report["accuracy"]),
            "macro_precision": float(report["macro avg"]["precision"]),
            "macro_recall": float(report["macro avg"]["recall"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "expected_L1_L2_threshold_ppm": lower,
            "expected_L2_L3_threshold_ppm": upper,
            "tlv_ppm": tlv,
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        },
        "remarks": (
            "PyTorch Deep Neural Network trained from the corrected, "
            f"concentration-labelled {gas.upper()} dataset."
        ),
        "training_time_sec": 0.0,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": f"gas_sensors/data/mine_part2_{gas}_balanced_cgan_corrected.csv",
        "source_feature": source_feature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per gas head")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    registry_path = MODELS_DIR / "model_registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        registry = {}

    for gas in GAS_CONFIG:
        print(f"Training severity_{gas}...")
        registry[f"severity_{gas}"] = train_one(gas, epochs=args.epochs)

    registry_path.write_text(json.dumps(registry, indent=4) + "\n", encoding="utf-8")
    print(f"Updated {registry_path}")


if __name__ == "__main__":
    main()
