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
| **Correlation** | Mean Abs Corr Diff (MACD) | `0.0016` | Low Drift (< 0.15) |
| **Distribution Distance** | Maximum Mean Discrepancy (MMD) | `0.90224` | High Fidelity Overlay |
| **Distinguishability** | Classifier Discriminator ROC-AUC | `1.0000` | Excellent Real-Synthetic Balance |
| **Distinguishability** | Classifier Discriminator Accuracy | `99.93%` | High Indistinguishability |
| **Downstream Utility** | Severity TRST Test Acc | `100.00%` | Superior Classification Utility |
| **Downstream Utility** | Over_TLV TRST Test Acc | `99.93%` | Perfect Classification Utility |

---

## 1. Descriptive Statistics Comparison

### Real CH4 Data vs Synthetic CGAN CH4 Data (Mean & Std)

| Feature | Real Mean ± Std | Synthetic Mean ± Std | Abs Diff (Mean) |
|---|---|---|---|
| `pct` | 1.46 ± 0.69 | 3.41 ± 0.64 | **1.95** |
| `ppm` | 14600.64 ± 6891.07 | 34139.20 ± 6415.87 | **19538.57** |
| `ppm_noisy` | 14524.32 ± 6669.58 | 33659.95 ± 6208.01 | **19135.63** |

---

## 2. Kolmogorov-Smirnov (KS) Test & Wasserstein Distance

| Feature | KS Statistic | p-value | Wasserstein Dist (Scaled) | Distribution Match |
|---|---|---|---|---|
| `pct` | 0.9988 | 0.0000e+00 | 2.8354 | Moderate Match |
| `ppm` | 0.9988 | 0.0000e+00 | 2.8354 | Moderate Match |
| `ppm_noisy` | 0.9640 | 0.0000e+00 | 2.8691 | Moderate Match |

---

## 3. Downstream Task Evaluation (`severity` & `over_tlv` Classification)

### Severity Classification (`severity` 0, 1, 2)

| Training Paradigm | Description | Test Accuracy |
|---|---|---|
| **TRTR** | Train Real -> Test Real | 99.81% |
| **TSTR** | Train Synthetic -> Test Real | 24.47% |
| **TRST** | Train Real + Synthetic -> Test Real | **100.00%** |

### Over TLV Classification (`over_tlv` 0, 1)

- **TRST Accuracy (`over_tlv`)**: **99.93%**


---

## 4. Range & Support Coverage Analysis

| Feature | Real Envelope Min/Max | Synthetic Min/Max | Synthetic Coverage (% inside Real Envelope) |
|---|---|---|---|
| `pct` | [0.00, 2.50] | [1.87, 4.80] | **3.6%** |
| `ppm` | [0.00, 25000.00] | [18749.25, 47993.81] | **3.6%** |
| `ppm_noisy` | [0.00, 26850.03] | [18377.55, 47317.70] | **18.1%** |

---

## 5. Visual Evaluation Artifacts

The following visual plots have been generated and saved to `gas_sensors/evaluation_plots/`:

- **Histograms + KDE Overlap**: `gas_sensors/evaluation_plots/part2_ch4_hist_kde.png`
- **Boxplots Comparison**: `gas_sensors/evaluation_plots/part2_ch4_boxplots.png`
- **Correlation Heatmaps**: `gas_sensors/evaluation_plots/part2_ch4_correlation.png`
- **PCA 2D Projection**: `gas_sensors/evaluation_plots/part2_ch4_pca.png`
- **t-SNE 2D Manifold**: `gas_sensors/evaluation_plots/part2_ch4_tsne.png`
- **Pairwise Scatter Plots**: `gas_sensors/evaluation_plots/part2_ch4_pairwise_scatter.png`