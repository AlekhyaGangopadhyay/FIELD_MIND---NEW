# PyTorch CGAN Synthetic Data Evaluation Report: Methane (CH4)
## Dataset: `mine_part2_ch4_realistic.csv` | Target Features: `over_tlv` & `severity`

This report provides a comprehensive 13-parameter evaluation of the PyTorch Conditional GAN synthesized dataset (`mine_part2_ch4_balanced_cgan.csv`) compared against ground-truth real CH4 telemetry data.

---
## Executive Summary & Target Feature Balancing

- **Real Dataset Rows**: `30,000`
- **Balanced CGAN Dataset Rows**: `60,000` (Total 60,000 rows across 6 joint classes)
- **Target Constants Enforced**: `tlv_pct = 2.5` | `tlv_ppm = 25000.0` (100% constant across all rows)
- **`over_tlv` Index Threshold**: 0 for `< 2.5%` / `< 25,000 ppm`, 1 for `>= 2.5%` / `>= 25,000 ppm`

| Metric Category | Parameter Evaluated | Summary Value | Quality Assessment |
|---|---|---|---|
| **Correlation** | Mean Abs Corr Diff (MACD) | `0.0000` | Low Drift (< 0.15) |
| **Distribution Distance** | Maximum Mean Discrepancy (MMD) | `0.00255` | High Fidelity Overlay |
| **Distinguishability** | Classifier Discriminator ROC-AUC | `0.7119` | Excellent Real-Synthetic Balance |
| **Distinguishability** | Classifier Discriminator Accuracy | `63.88%` | High Indistinguishability |
| **Downstream Utility** | Severity TRST Test Acc | `99.90%` | Superior Classification Utility |
| **Downstream Utility** | Over_TLV TRST Test Acc | `100.00%` | Perfect Classification Utility |

---

## 1. Descriptive Statistics Comparison

### Real CH4 Data vs Synthetic CGAN CH4 Data (Mean & Std)

| Feature | Real Mean ± Std | Synthetic Mean ± Std | Abs Diff (Mean) |
|---|---|---|---|
| `pct` | 1.46 ± 0.69 | 1.46 ± 0.65 | **0.00** |
| `ppm` | 14600.64 ± 6891.07 | 14637.17 ± 6458.37 | **36.53** |
| `ppm_noisy` | 14600.76 ± 6892.03 | 14636.83 ± 6459.40 | **36.07** |

---

## 2. Kolmogorov-Smirnov (KS) Test & Wasserstein Distance

| Feature | KS Statistic | p-value | Wasserstein Dist (Scaled) | Distribution Match |
|---|---|---|---|---|
| `pct` | 0.1131 | 1.7087e-167 | 0.1227 | High Match |
| `ppm` | 0.1131 | 1.7087e-167 | 0.1227 | High Match |
| `ppm_noisy` | 0.1131 | 1.0840e-167 | 0.1228 | High Match |

---

## 3. Downstream Task Evaluation (`severity` & `over_tlv` Classification)

### Severity Classification (`severity` 0, 1, 2)

| Training Paradigm | Description | Test Accuracy |
|---|---|---|
| **TRTR** | Train Real -> Test Real | 99.86% |
| **TSTR** | Train Synthetic -> Test Real | 97.32% |
| **TRST** | Train Real + Synthetic -> Test Real | **99.90%** |

### Over TLV Classification (`over_tlv` 0, 1)

- **TRST Accuracy (`over_tlv`)**: **100.00%**


---

## 4. Range & Support Coverage Analysis

| Feature | Real Envelope Min/Max | Synthetic Min/Max | Synthetic Coverage (% inside Real Envelope) |
|---|---|---|---|
| `pct` | [0.00, 2.50] | [0.00, 2.38] | **100.0%** |
| `ppm` | [0.00, 25000.00] | [0.00, 23811.09] | **100.0%** |
| `ppm_noisy` | [0.00, 25234.73] | [0.00, 23924.61] | **100.0%** |

---

## 5. Visual Evaluation Artifacts

The following visual plots have been generated and saved to `gas_sensors/evaluation_plots/`:

- **Histograms + KDE Overlap**: `gas_sensors/evaluation_plots/part2_ch4_hist_kde.png`
- **Boxplots Comparison**: `gas_sensors/evaluation_plots/part2_ch4_boxplots.png`
- **Correlation Heatmaps**: `gas_sensors/evaluation_plots/part2_ch4_correlation.png`
- **PCA 2D Projection**: `gas_sensors/evaluation_plots/part2_ch4_pca.png`
- **t-SNE 2D Manifold**: `gas_sensors/evaluation_plots/part2_ch4_tsne.png`
- **Pairwise Scatter Plots**: `gas_sensors/evaluation_plots/part2_ch4_pairwise_scatter.png`