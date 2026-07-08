# FIELD-MIND — Underground Mining Multimodal Sensor ML Pipelines

FIELD-MIND is an offline multimodal intelligence platform developed for underground mining operations. This repository hosts machine learning pipelines, preprocessing scripts, datasets, and serialized models across three primary sensor domains: **Gas Safety**, **Environmental (Temperature & Humidity)**, and **Blast Vibration**.

---

## Sensor Modules Overview

### 1. Gas Sensors (`gas_sensors/`)
Processes real-time multi-gas inputs (MQ-2, MQ-3, MQ-4, MQ-7, MQ-135, MQ-136, MG811 arrays) to identify hazards:
- **Synthetic Data Generation**: Models Gaussian plumes, ventilation cycles, machinery startup emissions, and sensor drift.
- **Methane Detection**: Employs a hybrid **SVM + MLP Voting Classifier** to predict methane levels robustly.
- **Specific Gas Classifiers**: Targets LPG/CNG, combustion gases (CO/Benzene), and smoke/fire hazards.
- **Multi-Gas Detector**: Tracks 5 gases simultaneously (Methane, CO, LPG, Smoke, NOx) using a MultiOutput Random Forest model.

### 2. Temperature & Humidity (`temperature_humidity/`)
Monitors occupational safety and environmental anomaly conditions:
- **Anomaly Detection**: Trains unsupervised **Isolation Forests** on rolling mean, variance, and humidex indices to capture microclimate anomalies.
- **Occupancy Classification**: A **Random Forest Classifier** trained on CO2, light, temperature, and humidity levels to predict office/tunnel occupancy.

### 3. Vibration (`vibration/`)
Predicts blast-induced seismic vibration levels (Peak Particle Velocity - PPV) near mine sites:
- **SEG-Y Binary Reader**: Custom parsing of big-endian IBM float headers from SEG-Y seismogram trace outputs.
- **Feature Extraction**: Calculates maximum charge, number of blast holes, trace directions (`trid`), and receptor coordinates.
- **Prediction Model**: Trains a **Gradient Boosting Regressor** for continuous PPV prediction and a **Random Forest Classifier** for threshold-based hazard alert triggers.

### 4. Ultrasonic Sensors (`ultrasonic_sensors/`)
Classifies robot navigation decisions based on 2, 4, or 24-sensor configurations:
- **Decision Classification**: Trains multiple ML algorithms (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, MLP) to predict robot commands (`Move-Forward`, `Slight-Right-Turn`, `Sharp-Right-Turn`, `Slight-Left-Turn`).

---

## Directory Structure

The repository has been organized into modular sensor packages:

```
FIELD_MIND (Project)/
├── docs/                         # Project proposal and background documentation
│   └── FIELD_MIND_proposal.docx  # Full project proposal and technical spec
├── gas_sensors/                  # Gas concentration prediction & classification
│   ├── data/                     # CSV & Excel datasets (physics-informed synthetic + raw)
│   ├── models/                   # Serialized ML models & model registry metadata
│   ├── data_loader.py            # Custom chronological & session split loaders
│   ├── generate_dataset.py       # Physics-informed synthetic dataset generator
│   ├── train.py                  # Smoke, VOC, Air Quality, and combined pipelines
│   ├── train_combined.py         # HistGradientBoosting regressor training script
│   ├── train_gas_detector.py     # Multi-Output RandomForest detector training
│   ├── train_methane.py          # Hybrid SVM + MLP Voting classifier training
│   ├── train_specific.py         # Specific gas safety hazard classifier training
│   └── Evaluation.md             # Gas-specific classifier evaluation results
├── temperature_humidity/         # Environmental anomaly detection
│   ├── data/                     # Clean and raw datasets (Kaggle & UCI formats)
│   ├── models/                   # Unsupervised isolation forests & Random Forest classifiers
│   ├── src/
│   │   ├── preprocess.py         # Environmental logs cleanup & feature extraction
│   │   ├── train.py              # Isolation Forest & RF training pipeline
│   │   ├── evaluate.py           # Anomaly detector & RF classifier evaluator
│   │   └── run_pipeline.py       # End-to-end execution orchestrator
│   └── model_metrics_table.md    # Performance report for environmental models
├── vibration/                    # Seismic blast vibration analysis
│   ├── data/                     # SEG-Y raw trace records and BLASTS.txt coordinates
│   ├── models/                   # Best classifier & regressor joblib files
│   ├── train_models.py           # Classifier & regressor pipelines (vibration hazards & PPV)
│   ├── vibration_data_prep.py    # Custom binary SEG-Y reader & feature extractor
│   └── model_metrics_table.md    # Blast vibration prediction evaluation report
└── ultrasonic_sensors/           # Robot navigation decision classification
    ├── data/                     # CSV datasets (2, 4, 24 sensor readings)
    ├── models/                   # Serialized classifier models
    ├── train_models.py           # Training pipeline for multiple classifiers
    └── model_metrics_table.md    # Classification performance report
```

---

## How to Run

A Python interpreter (version 3.10+) with required packages (`pandas`, `numpy`, `scikit-learn`, `joblib`, `openpyxl`) is required. 

Ensure you run all scripts from the project root.

### Running Gas Sensor Pipelines
```bash
# Generate the synthetic dataset
python gas_sensors/generate_dataset.py

# Train baseline classifiers
python gas_sensors/train.py

# Train specific hazard and multi-gas classifiers
python gas_sensors/train_specific.py
python gas_sensors/train_gas_detector.py
```

### Running Temperature & Humidity Pipeline
```bash
# Run the end-to-end preprocessing, training, and evaluation pipeline
python temperature_humidity/src/run_pipeline.py
```

### Running Vibration Analysis
```bash
# Extract features from raw blasts.sgy files
python vibration/vibration_data_prep.py

# Train vibration prediction models
python vibration/train_models.py
```

### Running Ultrasonic Sensor Analysis
```bash
# Train ultrasonic sensor classification models
python ultrasonic_sensors/train_models.py
```

---

## Git LFS Tracking

Large binary datasets (SEG-Y files, Excel datasets, model files) are tracked using **Git Large File Storage (LFS)**. Ensure `git-lfs` is installed and initialized before pushing/pulling:
```bash
git lfs install
```
