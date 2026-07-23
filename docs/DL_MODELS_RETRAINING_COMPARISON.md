# PyTorch Deep Learning Model Retraining & Architecture Tournament Report
## Evaluation on CGAN Balanced Gas Telemetry Datasets

This document summarizes the retraining and architectural comparison of PyTorch Deep Learning models trained on balanced datasets (`mine_part1_balanced_gan.csv` and `mine_part2_*_balanced_cgan.csv`). For each target, **ONLY the single best-performing deep learning model** has been saved to disk.

**Leakage-safe update (2026-07-23):** the four over-TLV hazard artifacts (`ch4_over_tlv`, `co_over_tlv`, `co2_over_tlv`, `h2_over_tlv`) were retrained using only the respective dataset `ppm` column as input. Label/threshold columns (`over_tlv`, `severity`, `level`, `pct`, `tlv_pct`, `tlv_ppm`) and generated alternate concentration (`ppm_noisy`) were excluded from features. The previous saved over-TLV artifacts also had incompatible tournament weights inside the generic hazard wrapper and failed inference with state-dict shape errors.

---
## Executive Summary of Winning Models

| Target Key | Target Feature | Dataset / Gas | Winning DL Architecture | Test Accuracy | Macro F1-Score | ROC-AUC | Saved Model Artifact |
|---|---|---|---|---|---|---|---|
| `part1_warmup` | `warmup` | Part | **ResNet1DMLP** | **100.00%** | **1.0000** | **1.0000** | `gas_sensors/models/part1_warmup_dl_best.joblib` |
| `ch4_severity` | `severity` | Methane | **LayerNormSwishMLP** | **98.54%** | **0.9854** | **0.9992** | `gas_sensors/models/ch4_severity_dl_best.joblib` |
| `ch4_over_tlv` | `over_tlv` | Methane | **DeepHazardNet** | **99.21%** | **0.9922** | **1.0000** | `gas_sensors/models/ch4_over_tlv_dl_best.joblib` |
| `co_severity` | `severity` | Carbon | **LayerNormSwishMLP** | **70.49%** | **0.7074** | **0.8771** | `gas_sensors/models/co_severity_dl_best.joblib` |
| `co_over_tlv` | `over_tlv` | Carbon Monoxide | **DeepHazardNet** | **90.10%** | **0.9099** | **1.0000** | `gas_sensors/models/co_over_tlv_dl_best.joblib` |
| `co2_severity` | `severity` | Carbon | **LayerNormSwishMLP** | **80.87%** | **0.8026** | **0.9460** | `gas_sensors/models/co2_severity_dl_best.joblib` |
| `co2_over_tlv` | `over_tlv` | Carbon Dioxide | **DeepHazardNet** | **99.23%** | **0.9924** | **1.0000** | `gas_sensors/models/co2_over_tlv_dl_best.joblib` |
| `h2_severity` | `severity` | Hydrogen | **ResNet1DMLP** | **73.87%** | **0.7311** | **0.9060** | `gas_sensors/models/h2_severity_dl_best.joblib` |
| `h2_over_tlv` | `over_tlv` | Hydrogen | **DeepHazardNet** | **100.00%** | **1.0000** | **1.0000** | `gas_sensors/models/h2_over_tlv_dl_best.joblib` |

---

## Detailed Model Training vs. Testing Performance & Dataset Splits

| Model Name | Arch Used | Training Acc | Testing Acc | Training Dataset | Testing Dataset | Train Test Split Ratio |
|---|---|---|---|---|---|:---:|
| `part1_warmup_dl_best.joblib` | **ResNet1DMLP** | **99.96%** | **100.00%** | `mine_part1_balanced_gan.csv` (122,025 rows) | `mine_part1_balanced_gan.csv` (40,675 rows) | 75:25 (0.75 / 0.25) |
| `ch4_severity_dl_best.joblib` | **LayerNormSwishMLP** | **98.59%** | **98.54%** | `mine_part2_ch4_balanced_cgan.csv` (45,000 rows) | `mine_part2_ch4_balanced_cgan.csv` (15,000 rows) | 75:25 (0.75 / 0.25) |
| `ch4_over_tlv_dl_best.joblib` | **DeepHazardNet** | **99.32%** | **99.21%** | `mine_part2_ch4_balanced_cgan.csv` (45,000 rows) | `mine_part2_ch4_balanced_cgan.csv` (15,000 rows) | 75:25 (0.75 / 0.25) |
| `co_severity_dl_best.joblib` | **LayerNormSwishMLP** | **70.96%** | **70.49%** | `mine_part2_co_balanced_cgan.csv` (45,000 rows) | `mine_part2_co_balanced_cgan.csv` (15,000 rows) | 75:25 (0.75 / 0.25) |
| `co_over_tlv_dl_best.joblib` | **DeepHazardNet** | **90.43%** | **90.10%** | `mine_part2_co_balanced_cgan.csv` (45,000 rows) | `mine_part2_co_balanced_cgan.csv` (15,000 rows) | 75:25 (0.75 / 0.25) |
| `co2_severity_dl_best.joblib` | **LayerNormSwishMLP** | **81.20%** | **80.87%** | `mine_part2_co2_balanced_cgan.csv` (45,000 rows) | `mine_part2_co2_balanced_cgan.csv` (15,000 rows) | 75:25 (0.75 / 0.25) |
| `co2_over_tlv_dl_best.joblib` | **DeepHazardNet** | **99.22%** | **99.23%** | `mine_part2_co2_balanced_cgan.csv` (45,000 rows) | `mine_part2_co2_balanced_cgan.csv` (15,000 rows) | 75:25 (0.75 / 0.25) |
| `h2_severity_dl_best.joblib` | **ResNet1DMLP** | **74.30%** | **73.87%** | `mine_part2_h2_balanced_cgan.csv` (45,000 rows) | `mine_part2_h2_balanced_cgan.csv` (15,000 rows) | 75:25 (0.75 / 0.25) |
| `h2_over_tlv_dl_best.joblib` | **DeepHazardNet** | **100.00%** | **100.00%** | `mine_part2_h2_balanced_cgan.csv` (45,000 rows) | `mine_part2_h2_balanced_cgan.csv` (15,000 rows) | 75:25 (0.75 / 0.25) |

---

## Top 3 Algorithm Comparisons per Target Model

### Target: `PART1_WARMUP` — Part 1 Clean Telemetry Warmup State Classifier

| Rank | Architecture Candidate | Test Accuracy | Precision | Recall | Macro F1-Score | ROC-AUC | Status |
|:---:|---|---|---|---|---|---|---|
| **Rank 1** | **ResNet1DMLP** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **SAVED (WINNER)** |
| **Rank 2** | **LayerNormSwishMLP** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Discarded |
| **Rank 3** | **SelfAttentionMLP** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Discarded |


### Target: `CH4_SEVERITY` — Methane (CH4) Multi-Class Severity Model (0, 1, 2)

| Rank | Architecture Candidate | Test Accuracy | Precision | Recall | Macro F1-Score | ROC-AUC | Status |
|:---:|---|---|---|---|---|---|---|
| **Rank 1** | **LayerNormSwishMLP** | 98.54% | 0.9854 | 0.9854 | 0.9854 | 0.9992 | **SAVED (WINNER)** |
| **Rank 2** | **SelfAttentionMLP** | 98.24% | 0.9825 | 0.9824 | 0.9824 | 0.9989 | Discarded |
| **Rank 3** | **WideAndDeepNet** | 97.66% | 0.9770 | 0.9766 | 0.9766 | 0.9970 | Discarded |


### Target: `CH4_OVER_TLV` — Methane (CH4) Over TLV Hazard Classifier (0, 1)

Leakage-safe retrain: saved artifact uses only `ppm` as input. Confusion matrix on the 15,000-row test split: `[[7382, 118], [0, 7500]]`.

| Rank | Architecture Candidate | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
|:---:|---|---|---|---|---|---|---|
| **Rank 1** | **DeepHazardNet** | 99.21% | 0.9845 | 1.0000 | 0.9922 | 1.0000 | **SAVED (LEAKAGE-SAFE)** |
| Previous | LayerNormSwishMLP | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Replaced: invalid saved wrapper/old leakage-risk report |


### Target: `CO_SEVERITY` — Carbon Monoxide (CO) Multi-Class Severity Model (0, 1, 2)

| Rank | Architecture Candidate | Test Accuracy | Precision | Recall | Macro F1-Score | ROC-AUC | Status |
|:---:|---|---|---|---|---|---|---|
| **Rank 1** | **LayerNormSwishMLP** | 70.49% | 0.7355 | 0.7049 | 0.7074 | 0.8771 | **SAVED (WINNER)** |
| **Rank 2** | **SelfAttentionMLP** | 70.23% | 0.7298 | 0.7023 | 0.7048 | 0.8757 | Discarded |
| **Rank 3** | **Conv1DNet** | 66.11% | 0.7171 | 0.6611 | 0.6613 | 0.8476 | Discarded |


### Target: `CO_OVER_TLV` — Carbon Monoxide (CO) Over TLV Hazard Classifier (0, 1)

Leakage-safe retrain: saved artifact uses only `ppm` as input. Confusion matrix on the 15,000-row test split: `[[6015, 1485], [0, 7500]]`.

| Rank | Architecture Candidate | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
|:---:|---|---|---|---|---|---|---|
| **Rank 1** | **DeepHazardNet** | 90.10% | 0.8347 | 1.0000 | 0.9099 | 1.0000 | **SAVED (LEAKAGE-SAFE)** |
| Previous | LayerNormSwishMLP | 93.61% | 0.8866 | 1.0000 | 0.9399 | 1.0000 | Replaced: invalid saved wrapper/old leakage-risk report |


### Target: `CO2_SEVERITY` — Carbon Dioxide (CO2) Multi-Class Severity Model (0, 1, 2)

| Rank | Architecture Candidate | Test Accuracy | Precision | Recall | Macro F1-Score | ROC-AUC | Status |
|:---:|---|---|---|---|---|---|---|
| **Rank 1** | **LayerNormSwishMLP** | 80.87% | 0.8060 | 0.8087 | 0.8026 | 0.9460 | **SAVED (WINNER)** |
| **Rank 2** | **SelfAttentionMLP** | 78.27% | 0.7820 | 0.7827 | 0.7820 | 0.9367 | Discarded |
| **Rank 3** | **ResNet1DMLP** | 77.93% | 0.7786 | 0.7793 | 0.7775 | 0.9361 | Discarded |


### Target: `CO2_OVER_TLV` — Carbon Dioxide (CO2) Over TLV Hazard Classifier (0, 1)

Leakage-safe retrain: saved artifact uses only `ppm` as input. Confusion matrix on the 15,000-row test split: `[[7385, 115], [0, 7500]]`.

| Rank | Architecture Candidate | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
|:---:|---|---|---|---|---|---|---|
| **Rank 1** | **DeepHazardNet** | 99.23% | 0.9849 | 1.0000 | 0.9924 | 1.0000 | **SAVED (LEAKAGE-SAFE)** |
| Previous | LayerNormSwishMLP | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Replaced: invalid saved wrapper/old leakage-risk report |


### Target: `H2_SEVERITY` — Hydrogen (H2) Multi-Class Severity Model (0, 1, 2)

| Rank | Architecture Candidate | Test Accuracy | Precision | Recall | Macro F1-Score | ROC-AUC | Status |
|:---:|---|---|---|---|---|---|---|
| **Rank 1** | **ResNet1DMLP** | 73.87% | 0.7566 | 0.7387 | 0.7311 | 0.9060 | **SAVED (WINNER)** |
| **Rank 2** | **LayerNormSwishMLP** | 73.61% | 0.7530 | 0.7361 | 0.7280 | 0.9084 | Discarded |
| **Rank 3** | **Conv1DNet** | 66.08% | 0.7295 | 0.6608 | 0.6539 | 0.8991 | Discarded |


### Target: `H2_OVER_TLV` — Hydrogen (H2) Over TLV Hazard Classifier (0, 1)

Leakage-safe retrain: saved artifact uses only `ppm` as input. Confusion matrix on the 15,000-row test split: `[[7500, 0], [0, 7500]]`.

| Rank | Architecture Candidate | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
|:---:|---|---|---|---|---|---|---|
| **Rank 1** | **DeepHazardNet** | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **SAVED (LEAKAGE-SAFE)** |
| Previous | ResNet1DMLP | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Replaced: invalid saved wrapper/old leakage-risk report |


---

## Candidate Deep Learning Architecture Specifications

1. **ResNet1DMLP (Deep Residual Skip Network)**:
   - Input linear projection -> 2x ResNet 1D blocks with BatchNorm1d, GELU activations, and shortcut residual additions.
   - Prevents gradient degradation in multi-layer representation learning.

2. **LayerNormSwishMLP (Swish Deep MLP)**:
   - Layer Normalization + SiLU (Swish) non-linearities + Dropout regularization.
   - Provides smooth gradient flow and robust feature scale invariance.

3. **WideAndDeepNet (Wide & Deep Architecture)**:
   - Direct wide linear connection fused with deep non-linear feature pathways.
   - Captures both linear feature thresholds and complex non-linear sensor correlations.

4. **Conv1DNet (1D Convolutional Neural Network)**:
   - 1D Convolutions (`Conv1d`) -> BatchNorm1d -> Adaptive Average Pooling -> Dense classification head.
   - Captures spatial feature receptive fields across multi-sensor channels.

5. **SelfAttentionMLP (Multi-Head Self-Attention Transformer)**:
   - Multi-Head Self-Attention (`MultiheadAttention`) projection with LayerNorm and Feed-Forward Network.
   - Computes dynamic cross-feature attention weights.

6. **DeepHazardNet (Leakage-Safe Saved Hazard Wrapper)**:
   - BatchNorm/GELU feed-forward PyTorch binary classifier exposed through `PyTorchHazardClassifier`.
   - Used for the corrected over-TLV artifacts so serialized weights match the inference wrapper.

---

## Final Model Registry Verification

All winning model artifacts have been serialized using `joblib` into `gas_sensors/models/` and registered in `model_registry.json`. Ready for production agent integration.
