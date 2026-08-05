# FIELD-MIND Gas Sensor Evaluation

**Snapshot:** 2026-08-05
**Scope:** current gas-sensor artifacts present in `gas_sensors/models/` and loaded by the runtime integration.

This document is the current evaluation record for the gas-sensor stack. It uses the serialized model files and the metadata in [`models/model_registry.json`](models/model_registry.json) as the source of truth. Obsolete, missing, and tournament-only model entries are not included as production models.

## Current production model inventory

There are 11 current gas models. The five severity heads share the same corrected-label training design but have separate gas-specific thresholds and learned decision boundaries.

| Artifact | Task | Model / architecture | Inputs | Test result |
| --- | --- | --- | --- | --- |
| `multi_gas_detector.joblib` | Eight-output multilabel gas presence | PyTorch LayerNorm/SiLU MLP | 8 ppm channels: CH4, CO, CO2, H2, H2S, NH3, LPG, CNG | Elementwise accuracy **98.81%**; exact eight-target match **90.78%**; P/R/F1 **99.87% / 95.34% / 97.45%** |
| `mine_baseline_iforest.joblib` | Clean-air anomaly detection | StandardScaler + IsolationForest | 8 real mine sensor channels | Clean-air normal rate **98.95%**; anomaly/false-alarm rate **1.05%** |
| `severity_ch4.joblib` | CH4 severity, 3 classes | PyTorch Deep MLP | `ppm` | Accuracy **99.28%**; macro F1 **0.9896**; learned boundaries about **10,103 / 15,064 ppm**; TLV **25,000 ppm** |
| `severity_co.joblib` | CO severity, 3 classes | PyTorch Deep MLP | `ppm` | Accuracy **88.00%**; macro F1 **0.8746**; learned boundaries about **18.9 / 46.3 ppm**; TLV **50 ppm** |
| `severity_co2.joblib` | CO2 severity, 3 classes | PyTorch Deep MLP | `ppm` | Accuracy **97.42%**; macro F1 **0.9668**; learned boundaries about **965 / 3,876 ppm**; TLV **5,000 ppm** |
| `severity_h2.joblib` | H2 severity, 3 classes | PyTorch Deep MLP | `ppm` | Accuracy **99.08%**; macro F1 **0.9856**; learned boundaries about **3,537 / 19,646 ppm**; TLV **20,000 ppm** |
| `severity_h2s.joblib` | H2S severity, 3 classes | PyTorch Deep MLP | `ppm` | Accuracy **99.65%**; macro F1 **0.9965**; learned boundaries about **9.6 / 19.2 ppm**; TLV **20 ppm** |
| `nh3_hazard.joblib` | NH3 binary hazard alert | PyTorch Deep MLP | `MQ135_NH3_ppm` | Accuracy **98.86%**; precision **99.07%**; recall **98.65%**; F1 **0.9886** |
| `co2_hazard.joblib` | CO2 binary hazard alert | PyTorch Deep MLP | `ppm` | Accuracy **99.74%**; precision **99.81%**; recall **99.45%**; F1 **0.9963** |
| `smoke_env_hazard.joblib` | Dust/environment binary hazard alert | PyTorch Deep MLP | PM2.5 dust, temperature, humidity | Accuracy **99.86%**; precision **99.78%**; recall **99.94%**; F1 **0.9986** |
| `mq4_gas_classifier.joblib` | MQ4 methane multiclass classification | StandardScaler + soft-voting ensemble: 2 RBF SVMs, linear SVM, 2 MLPs | 128 MQ4 response features | Accuracy **77.37%**; macro F1 **0.7789**; 8-fold train CV **99.47%** |

### Interpretation notes

- `multi_gas_detector` is the unified presence detector. Its exact-subset score is stricter than its elementwise score because all eight gas outputs must be correct for a row to count as correct.
- The severity models use concentration-based corrected labels (`severity` 0/1/2), not the old location/band index. Their registry metrics are held-out test metrics from the retraining scripts.
- The MQ4 result is deliberately reported against batches 9-10, which are held out by batch rather than randomly mixed with the training data. The high cross-validation score therefore should not be confused with the 77.37% temporal test accuracy.
- Historical tournament artifacts are not loaded by the current gas runtime and are not counted as production models.

## Training dataset table

The counts below are taken from the files currently under `gas_sensors/data/` and the split logic in the corresponding training scripts. For balanced datasets, “rows used” means rows after the script's balancing/filtering step.

| Model(s) | Dataset/source | Source rows | Features used | Target | Training/evaluation split | Purpose |
| --- | --- | ---: | --- | --- | --- | --- |
| `multi_gas_detector` | [`multi_gas_detector_real_v2.csv`](data/multi_gas_detector_real_v2.csv) | 50,000 | 8 ppm channels: `CH4_ppm`, `CO_ppm`, `CO2_ppm`, `H2_ppm`, `H2S_ppm`, `NH3_ppm`, `LPG_ppm`, `CNG_ppm` | 8 binary presence columns: `target_Methane` through `target_CNG` | 37,500 train / 12,500 test (75/25) | Unified multi-gas presence detection |
| `severity_ch4` | [`mine_part2_ch4_balanced_cgan_corrected.csv`](data/mine_part2_ch4_balanced_cgan_corrected.csv) | 60,000 | `ppm_noisy` (normalized to model input `ppm`) | `severity` (0/1/2) | 45,000 train / 15,000 test (75/25, stratified) | Corrected CH4 concentration severity |
| `severity_co` | [`mine_part2_co_balanced_cgan_corrected.csv`](data/mine_part2_co_balanced_cgan_corrected.csv) | 60,000 | `ppm_noisy` (normalized to `ppm`) | `severity` (0/1/2) | 45,000 train / 15,000 test (75/25, stratified) | Corrected CO concentration severity |
| `severity_co2` | [`mine_part2_co2_balanced_cgan_corrected.csv`](data/mine_part2_co2_balanced_cgan_corrected.csv) | 60,000 | `ppm_noisy` (normalized to `ppm`) | `severity` (0/1/2) | 45,000 train / 15,000 test (75/25, stratified) | Corrected CO2 concentration severity |
| `severity_h2` | [`mine_part2_h2_balanced_cgan_corrected.csv`](data/mine_part2_h2_balanced_cgan_corrected.csv) | 60,000 | `ppm_noisy` (normalized to `ppm`) | `severity` (0/1/2) | 45,000 train / 15,000 test (75/25, stratified) | Corrected H2 concentration severity |
| `severity_h2s` | [`mine_part2_h2s_balanced_cgan_corrected.csv`](data/mine_part2_h2s_balanced_cgan_corrected.csv) | 60,000 | `ppm` | `severity` (0/1/2) | 45,000 train / 15,000 test (75/25, stratified) | Corrected H2S concentration severity |
| `mine_baseline_iforest` | [`mine_part1_clean.csv`](data/mine_part1_clean.csv) | 1,815 raw; 1,721 steady-state | `air_quality`, `smoke`, `alcohol`, `flamable_gas`, `MQ136_raw`, `MQ7_raw`, `t`, `h` | Unlabelled normal baseline; IsolationForest anomaly score | Fit and evaluated on the 1,721 steady-state rows; 94 warm-up rows excluded | Hardware noise-floor baseline and sensor-fault detection |
| `nh3_hazard` | [`nh3_hazard_balanced_cgan.csv`](data/nh3_hazard_balanced_cgan.csv) | 60,000 | `MQ135_NH3_ppm` | `Hazard_Alert` | 30,000 train / 30,000 test (50/50, stratified) | NH3 hazard alert |
| `co2_hazard` | [`mine_part2_co2_balanced_cgan_corrected.csv`](data/mine_part2_co2_balanced_cgan_corrected.csv) | 60,000 | `ppm` with 5% threshold uncertainty noise during training | `over_tlv`-style binary hazard label at the configured 1,000 ppm boundary | 30,000 train / 30,000 test (50/50, stratified) | Early CO2 hazard alert |
| `smoke_env_hazard` | [`FIELDMIND_physics_dataset.csv`](data/FIELDMIND_physics_dataset.csv) | 50,000 raw; 7,104 balanced | `PM25_Dust_ugm3`, `Temp_C`, `Humidity_pct` | `Hazard_Alert` | 3,552 train / 3,552 test (50/50 after class balancing) | Dust, temperature, and humidity hazard alert |
| `mq4_gas_classifier` | [`Methane_MQ4/Dataset/`](data/Methane_MQ4/Dataset/) batches 1-10 | 13,910 total | `feature_1` ... `feature_128` | `label` | 9,840 train (batches 1-8) / 4,070 test (batches 9-10) | Batch-held-out MQ4 methane classification |

## Reproducibility references

- Multi-gas training: [`train_multi_gas_detector.py`](train_multi_gas_detector.py)
- Severity retraining: [`retrain_severity_models.py`](retrain_severity_models.py)
- Baseline anomaly detector: [`train_mine_baseline.py`](train_mine_baseline.py)
- NH3, CO2, and smoke/environment heads: [`train_new_dl_models.py`](train_new_dl_models.py)
- MQ4 ensemble: [`train_methane.py`](train_methane.py) and [`data_loader.py`](data_loader.py)
- Serialized artifacts and recorded metrics: [`models/model_registry.json`](models/model_registry.json)

## Validation status

- All 11 production artifact names in this document resolve under `gas_sensors/models/`.
- Dataset filenames and row counts were checked against the current `gas_sensors/data/` directory.
- The old missing hazard entries, obsolete tournament winner rows, and deprecated vibration-model section were removed from this evaluation document.
- No model file was deleted as part of this documentation cleanup; only current production artifacts are recorded here.
