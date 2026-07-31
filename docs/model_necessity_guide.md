# FIELD-MIND — ML Model Inventory, Performance, and Verdict Guide

> **Document Scope**: Analysis of all 30 model artifacts in the `FIELD_MIND` project across Gas, Vibration, Ultrasonic, and Temperature/Humidity domains.
> **Purpose**: Technical classification of models to keep in production vs. those that can be safely removed or quarantined.

---

## Model Inventory and Verdict Matrix

| Domain | Model File Name | Training Dataset | Performance (Train / Test) | Verdict | Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gas** | `gas_hazard_lpg_cng.joblib` | `mine_part2_ch4_balanced_cgan.csv` | **99.98%** / **99.97%** Acc | 🟢 **KEEP (Critical)** | Primary safety gate for methane and LPG explosion prevention. Active trigger for Tier-1 ATR. |
| **Gas** | `gas_hazard_co_nox_c6h6.joblib` | `mine_part2_co_balanced_cgan.csv` | **93.90%** / **93.81%** Acc (Syn)<br>**99.55%** Acc / **0.9932** F1 (Real) | 🟢 **KEEP (Critical)** | Essential toxic gas warning monitor. Highly validated against real mine telemetry. |
| **Gas** | `multi_gas_detector.joblib` | `FIELDMIND_physics_dataset.csv` | **97.40%** / **97.32%** Acc | 🟢 **KEEP (Critical)** | Fuses five gases into a multi-label classification vector to identify compound hazards. |
| **Gas** | `mine_baseline_iforest.joblib` | `mine_part1_clean.csv` | — / **98.95%** Alignment | 🟢 **KEEP (Critical)** | Unsupervised Isolation Forest baseline to detect deviations from steady-state clean air. |
| **Gas** | `severity_ch4.joblib` | `mine_part2_ch4_realistic.csv` | **99.10%** / **99.07%** Acc | 🟢 **KEEP (Critical)** | Categorizes methane concentrations into graded alarm severities (L1, L2, L3). |
| **Gas** | `severity_co.joblib` | `mine_part2_co_realistic.csv` | **92.10%** / **91.92%** Acc | 🟢 **KEEP (Critical)** | Categorizes carbon monoxide levels. Crucial for triggering evacuation severity layers. |
| **Gas** | `severity_co2.joblib` | `mine_part2_co2_realistic.csv` | **90.50%** / **90.27%** Acc | 🟢 **KEEP (Critical)** | Categorizes CO₂ buildup risk in confined spaces. |
| **Gas** | `severity_h2.joblib` | `mine_part2_h2_realistic.csv` | **95.80%** / **95.72%** Acc | 🟢 **KEEP (Critical)** | Categorizes hydrogen risk levels from battery charger corridors and blast areas. |
| **Gas** | `severity_h2s.joblib` | `mine_part2_h2s_balanced_cgan.csv` | **99.72%** / **99.77%** Acc | 🟢 **KEEP (Critical)** | Categorizes highly toxic hydrogen sulfide concentration thresholds. |
| **Gas** | `nh3_hazard.joblib` | `nh3_hazard_balanced_cgan.csv` | **98.86%** / **98.86%** Acc | 🟢 **KEEP (Critical)** | Binary toxic ammonia warning. Directly maps to the 25 ppm NIOSH safety guidelines. |
| **Gas** | `co2_hazard.joblib` | `mine_part2_co2_balanced_cgan.csv` | **99.74%** / **99.74%** Acc | 🟢 **KEEP (Critical)** | Serves as the primary early warning trigger for high-concentration CO₂ asphyxiation. |
| **Gas** | `smoke_env_hazard.joblib` | `FIELDMIND_physics_dataset.csv` | **99.86%** / **99.86%** Acc | 🟢 **KEEP (Critical)** | Drives Goal 3 (Dust/Smoke detection). Fuses PM2.5 particulate count with environmental temp. |
| **Gas (MQ4)** | `mq4_gas_classifier.joblib` | `Methane_MQ4/Dataset/` (10 batches) | SVM+MLP Ensemble | 🟡 **KEEP (Conditional)** | Operates on 128-D spectral raw signal characteristics. Keep for physical input verification, but cross-check with ppm models. |
| **Vibration** | `best_random_forest_classifier.joblib` | `vibration_features.csv` | **96.92%** / **93.01%** Acc | 🟢 **KEEP (Critical)** | Primary blast hazard classification indicator (PPV > 1.0 mm/s). Crucial for Goal 2. |
| **Vibration** | `best_gradient_boosting_regressor.joblib` | `vibration_features.csv` | R²=**0.9415** / R²=**0.9165** | 🟢 **KEEP (Critical)** | Regresses log-transformed Peak Particle Velocity to forecast seismic blast energy propagation. |
| **Ultrasonic** | `best_ultrasonic_24.joblib` | `sensor_readings_24.csv` | **100.00%** / **99.54%** Acc | 🟢 **KEEP (Critical)** | Essential for robot obstacle navigation using the complete 24-sensor ring input. |
| **Ultrasonic** | `best_ultrasonic_2.joblib` | `sensor_readings_2.csv` | **100.00%** / **100.00%** Acc | 🟡 **KEEP (Fallback)** | Minimally loads to guide navigation if a subset of sonar hardware fails. Keep file on disk. |
| **Temp/Hum** | `random_forest.joblib` | UCI Occupancy (`datatraining.txt`) | **99.01%** / **97.11%** Acc | 🟢 **KEEP (Critical)** | Used to infer human occupancy, ensuring evacuation guidance matches personnel presence. |
| **Temp/Hum** | `isolation_forest_uci.joblib` | UCI Environmental data | **73.87%** / **89.19%** Acc | 🟢 **KEEP (Critical)** | Unsupervised environmental anomaly check. Wire this as primary in `Tier1Monitor` (better profile). |
| **Temp/Hum** | `isolation_forest_iot.joblib` | `iot_telemetry_clean.csv` | — / **93.38%** Acc | 🟡 **KEEP (Fallback)** | Cross-domain evaluation drops to 67.56% on Pi sensors. Keep as secondary baseline. |
| **Ultrasonic** | `best_ultrasonic_4.joblib` | `sensor_readings_4.csv` | **100.00%** / **100.00%** Acc | ❌ **REMOVE (Redundant)** | The 4-sensor model is loaded by default but is fully superseded by the 24-sensor production model. Remove from system startup to free RAM. |
| **Temp/Hum** | `isolation_forest.joblib` | N/A | N/A | ❌ **REMOVE (Redundant)** | Unused template file from testing. Not wired to any agent or monitor class. |
| **Gas (DL)** | `part1_warmup_dl_best.joblib` | `mine_part1_balanced_gan.csv` | **100.00%** / **100.00%** Acc | ❌ **REMOVE (Unnecessary)** | Unnecessary ML overhead. Identifying the sensor warmup transient should be done via a simple time check: `if elapsed_time < 120s`. |
| **Gas (DL)** | `ch4_severity_dl_best.joblib` | `mine_part2_ch4_realistic.csv` | **99.68%** / **99.71%** Acc | ❌ **REMOVE (Redundant)** | Duplicates the role of the production-ready `severity_ch4.joblib` model. |
| **Gas (DL)** | `co_severity_dl_best.joblib` | `mine_part2_co_realistic.csv` | **97.71%** / **97.64%** Acc | ❌ **REMOVE (Redundant)** | Duplicates the role of the production-ready `severity_co.joblib` model. |
| **Gas (DL)** | `co2_severity_dl_best.joblib` | `mine_part2_co2_realistic.csv` | **94.90%** / **95.01%** Acc | ❌ **REMOVE (Redundant)** | Duplicates the role of the production-ready `severity_co2.joblib` model. |
| **Gas (DL)** | `h2_severity_dl_best.joblib` | `mine_part2_h2_realistic.csv` | **97.76%** / **97.65%** Acc | ❌ **REMOVE (Redundant)** | Duplicates the role of the production-ready `severity_h2.joblib` model. |
| **Gas (DL)** | `ch4_over_tlv_dl_best.joblib` | `mine_part2_ch4_balanced_cgan.csv` | **99.97%** / **99.97%** Acc | ❌ **REMOVE (Unnecessary)** | Unused binary checker; contains data leakage (uses `pct` features derived directly from target boundary). |
| **Gas (DL)** | `co_over_tlv_dl_best.joblib` | `mine_part2_co_balanced_cgan.csv` | **99.86%** / **99.79%** Acc | ❌ **REMOVE (Unnecessary)** | Unused binary checker; contains data leakage. |
| **Gas (DL)** | `co2_over_tlv_dl_best.joblib` | `mine_part2_co2_balanced_cgan.csv` | **99.96%** / **99.97%** Acc | ❌ **REMOVE (Unnecessary)** | Unused binary checker; contains data leakage. |
| **Gas (DL)** | `h2_over_tlv_dl_best.joblib` | `mine_part2_h2_balanced_cgan.csv` | **99.99%** / **99.97%** Acc | ❌ **REMOVE (Unnecessary)** | Unused binary checker; contains data leakage. |

---

## Summary Action Plan

### 1. Quarantine Unnecessary & Redundant Models (11 Files)
Create an `experimental/` or `deprecated/` subfolder at `gas_sensors/models/experimental/` and move the following files there to clear disk space and clutter:
*   All 9 tournament files (`part1_warmup_dl_best.joblib`, `*_severity_dl_best.joblib`, `*_over_tlv_dl_best.joblib`)
*   `temperature_humidity/models/isolation_forest.joblib` (and its `.joblib_metadata`)

### 2. Startup Optimization (1 File)
Modify the system monitor initialization to ignore the redundant low-resolution configuration:
*   Remove `best_ultrasonic_4.joblib` from the loading list in `detector_wrappers.py` (leave it on disk only as a physical spare fallback).

### 3. Rewire Environmental Baselines (2 Files)
*   Promote `isolation_forest_uci.joblib` to be the primary model for the temperature/humidity monitor.
*   Demote `isolation_forest_iot.joblib` to fallback because of its high cross-domain performance drop on standard microcontroller setups.
