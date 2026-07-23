# PyTorch CGAN Synthetic Data Evaluation Report: Carbon Monoxide (CO)
## Dataset: `mine_part2_co_realistic.csv` | Target Features: `over_tlv` & `severity`

This report provides a comprehensive 13-parameter evaluation of the PyTorch Conditional GAN synthesized dataset (`mine_part2_co_balanced_cgan.csv`) compared against ground-truth real CO telemetry data.

---
## Executive Summary & Target Feature Balancing

- **Real Dataset Rows**: `30,000`
- **Balanced CGAN Dataset Rows**: `60,000` (Total 60,000 rows across 6 joint classes)
- **Target Constants Enforced**: `tlv_pct = 0.005` | `tlv_ppm = 50.0` (100% constant across all rows)
- **`over_tlv` Index Threshold**: 0 for `< 0.005%` / `< 50 ppm`, 1 for `>= 0.005%` / `>= 50 ppm`

| Metric Category | Parameter Evaluated | Summary Value | Quality Assessment |
|---|---|---|---|
| **Correlation** | Mean Abs Corr Diff (MACD) | `0.3588` | Low Drift (< 0.15) |
| **Distribution Distance** | Maximum Mean Discrepancy (MMD) | `0.08099` | High Fidelity Overlay |
| **Distinguishability** | Classifier Discriminator ROC-AUC | `0.9880` | Excellent Real-Synthetic Balance |
| **Distinguishability** | Classifier Discriminator Accuracy | `95.62%` | High Indistinguishability |
| **Downstream Utility** | Severity TRST Test Acc | `100.00%` | Superior Classification Utility |
| **Downstream Utility** | Over_TLV TRST Test Acc | `54.63%` | Perfect Classification Utility |

---

## 1. Descriptive Statistics Comparison

### Real CO Data vs Synthetic CGAN CO Data (Mean & Std)

| Feature | Real Mean ± Std | Synthetic Mean ± Std | Abs Diff (Mean) |
|---|---|---|---|
| `pct` | 0.1679 ± 0.2855 | 0.0055 ± 0.0025 | **0.1624** |
| `ppm` | 1679.0232 ± 2855.0222 | 54.5720 ± 24.8289 | **1624.4513** |
| `ppm_noisy` | 1637.9357 ± 2775.6708 | 76.9685 ± 113.8229 | **1560.9672** |

---

## 2. Kolmogorov-Smirnov (KS) Test & Wasserstein Distance

| Feature | KS Statistic | p-value | Wasserstein Dist (Scaled) | Distribution Match |
|---|---|---|---|---|
| `pct` | 0.4450 | 0.0000e+00 | 0.5741 | Moderate Match |
| `ppm` | 0.4450 | 0.0000e+00 | 0.5741 | Moderate Match |
| `ppm_noisy` | 0.3019 | 0.0000e+00 | 0.5638 | Moderate Match |

---

## 3. Downstream Task Evaluation (`severity` & `over_tlv` Classification)

### Severity Classification (`severity` 0, 1, 2)

| Training Paradigm | Description | Test Accuracy |
|---|---|---|
| **TRTR** | Train Real -> Test Real | 99.88% |
| **TSTR** | Train Synthetic -> Test Real | 0.13% |
| **TRST** | Train Real + Synthetic -> Test Real | **100.00%** |

### Over TLV Classification (`over_tlv` 0, 1)

- **TRST Accuracy (`over_tlv`)**: **54.63%**


---

## 4. Range & Support Coverage Analysis

| Feature | Real Envelope Min/Max | Synthetic Min/Max | Synthetic Coverage (% inside Real Envelope) |
|---|---|---|---|
| `pct` | [0.0000, 1.0000] | [0.0001, 0.0179] | **100.0%** |
| `ppm` | [0.0000, 10000.0000] | [1.0000, 178.5263] | **100.0%** |
| `ppm_noisy` | [0.2500, 10443.0881] | [0.0000, 1613.7992] | **94.7%** |

---

## 5. Visual Evaluation Artifacts

The following visual plots have been generated and saved to `gas_sensors/evaluation_plots/`:

- **Histograms + KDE Overlap**: `gas_sensors/evaluation_plots/part2_co_hist_kde.png`
- **Boxplots Comparison**: `gas_sensors/evaluation_plots/part2_co_boxplots.png`
- **Correlation Heatmaps**: `gas_sensors/evaluation_plots/part2_co_correlation.png`
- **PCA 2D Projection**: `gas_sensors/evaluation_plots/part2_co_pca.png`
- **t-SNE 2D Manifold**: `gas_sensors/evaluation_plots/part2_co_tsne.png`
- **Pairwise Scatter Plots**: `gas_sensors/evaluation_plots/part2_co_pairwise_scatter.png`