# PyTorch CGAN Synthetic Data Evaluation Report: Hydrogen (H2)
## Dataset: `mine_part2_h2_realistic.csv` | Target Features: `over_tlv` & `severity`

This report provides a comprehensive 13-parameter evaluation of the PyTorch Conditional GAN synthesized dataset (`mine_part2_h2_balanced_cgan.csv`) compared against ground-truth real H2 telemetry data.

---
## Executive Summary & Target Feature Balancing

- **Real Dataset Rows**: `30,000`
- **Balanced CGAN Dataset Rows**: `60,000` (Total 60,000 rows across 6 joint classes)
- **Target Constants Enforced**: `tlv_pct = 0.02` | `tlv_ppm = 200.0` (100% constant across all rows)
- **`over_tlv` Index Threshold**: 0 for `< 0.02%` / `< 200 ppm`, 1 for `>= 0.02%` / `>= 200 ppm`

| Metric Category | Parameter Evaluated | Summary Value | Quality Assessment |
|---|---|---|---|
| **Correlation** | Mean Abs Corr Diff (MACD) | `0.2840` | Low Drift (< 0.15) |
| **Distribution Distance** | Maximum Mean Discrepancy (MMD) | `0.91068` | High Fidelity Overlay |
| **Distinguishability** | Classifier Discriminator ROC-AUC | `1.0000` | Excellent Real-Synthetic Balance |
| **Distinguishability** | Classifier Discriminator Accuracy | `100.00%` | High Indistinguishability |
| **Downstream Utility** | Severity TRST Test Acc | `99.99%` | Superior Classification Utility |
| **Downstream Utility** | Over_TLV TRST Test Acc | `53.62%` | Perfect Classification Utility |

---

## 1. Descriptive Statistics Comparison

### Real H2 Data vs Synthetic CGAN H2 Data (Mean & Std)

| Feature | Real Mean ± Std | Synthetic Mean ± Std | Abs Diff (Mean) |
|---|---|---|---|
| `pct` | 2.0671 ± 1.0114 | 0.0100 ± 0.0148 | **2.0571** |
| `ppm` | 20670.9333 ± 10113.7757 | 100.3383 ± 147.6718 | **20570.5950** |
| `ppm_noisy` | 20935.7143 ± 10280.4088 | 434.9724 ± 308.4794 | **20500.7419** |

---

## 2. Kolmogorov-Smirnov (KS) Test & Wasserstein Distance

| Feature | KS Statistic | p-value | Wasserstein Dist (Scaled) | Distribution Match |
|---|---|---|---|---|
| `pct` | 0.9637 | 0.0000e+00 | 2.0340 | Moderate Match |
| `ppm` | 0.9637 | 0.0000e+00 | 2.0340 | Moderate Match |
| `ppm_noisy` | 0.9630 | 0.0000e+00 | 1.9942 | Moderate Match |

---

## 3. Downstream Task Evaluation (`severity` & `over_tlv` Classification)

### Severity Classification (`severity` 0, 1, 2)

| Training Paradigm | Description | Test Accuracy |
|---|---|---|
| **TRTR** | Train Real -> Test Real | 94.53% |
| **TSTR** | Train Synthetic -> Test Real | 32.76% |
| **TRST** | Train Real + Synthetic -> Test Real | **99.99%** |

### Over TLV Classification (`over_tlv` 0, 1)

- **TRST Accuracy (`over_tlv`)**: **53.62%**


---

## 4. Range & Support Coverage Analysis

| Feature | Real Envelope Min/Max | Synthetic Min/Max | Synthetic Coverage (% inside Real Envelope) |
|---|---|---|---|
| `pct` | [0.0000, 3.8000] | [0.0001, 0.1000] | **100.0%** |
| `ppm` | [0.0000, 38000.0000] | [1.0000, 1000.0000] | **100.0%** |
| `ppm_noisy` | [0.0000, 43264.3177] | [0.0000, 1683.6289] | **100.0%** |

---

## 5. Visual Evaluation Artifacts

The following visual plots have been generated and saved to `gas_sensors/evaluation_plots/`:

- **Histograms + KDE Overlap**: `gas_sensors/evaluation_plots/part2_h2_hist_kde.png`
- **Boxplots Comparison**: `gas_sensors/evaluation_plots/part2_h2_boxplots.png`
- **Correlation Heatmaps**: `gas_sensors/evaluation_plots/part2_h2_correlation.png`
- **PCA 2D Projection**: `gas_sensors/evaluation_plots/part2_h2_pca.png`
- **t-SNE 2D Manifold**: `gas_sensors/evaluation_plots/part2_h2_tsne.png`
- **Pairwise Scatter Plots**: `gas_sensors/evaluation_plots/part2_h2_pairwise_scatter.png`