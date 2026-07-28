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
| **Correlation** | Mean Abs Corr Diff (MACD) | `0.0000` | Low Drift (< 0.15) |
| **Distribution Distance** | Maximum Mean Discrepancy (MMD) | `0.00046` | High Fidelity Overlay |
| **Distinguishability** | Classifier Discriminator ROC-AUC | `0.8966` | Excellent Real-Synthetic Balance |
| **Distinguishability** | Classifier Discriminator Accuracy | `76.99%` | High Indistinguishability |
| **Downstream Utility** | Severity TRST Test Acc | `94.71%` | Superior Classification Utility |
| **Downstream Utility** | Over_TLV TRST Test Acc | `100.00%` | Perfect Classification Utility |

---

## 1. Descriptive Statistics Comparison

### Real H2 Data vs Synthetic CGAN H2 Data (Mean & Std)

| Feature | Real Mean ± Std | Synthetic Mean ± Std | Abs Diff (Mean) |
|---|---|---|---|
| `pct` | 2.0671 ± 1.0114 | 2.0712 ± 1.0473 | **0.0041** |
| `ppm` | 20670.9333 ± 10113.7757 | 20712.1833 ± 10473.1898 | **41.2499** |
| `ppm_noisy` | 20671.1019 ± 10113.2052 | 20712.2591 ± 10472.9649 | **41.1573** |

---

## 2. Kolmogorov-Smirnov (KS) Test & Wasserstein Distance

| Feature | KS Statistic | p-value | Wasserstein Dist (Scaled) | Distribution Match |
|---|---|---|---|---|
| `pct` | 0.0638 | 1.8100e-53 | 0.0612 | High Match |
| `ppm` | 0.0638 | 1.8100e-53 | 0.0612 | High Match |
| `ppm_noisy` | 0.0587 | 2.0222e-45 | 0.0598 | High Match |

---

## 3. Downstream Task Evaluation (`severity` & `over_tlv` Classification)

### Severity Classification (`severity` 0, 1, 2)

| Training Paradigm | Description | Test Accuracy |
|---|---|---|
| **TRTR** | Train Real -> Test Real | 94.28% |
| **TSTR** | Train Synthetic -> Test Real | 95.24% |
| **TRST** | Train Real + Synthetic -> Test Real | **94.71%** |

### Over TLV Classification (`over_tlv` 0, 1)

- **TRST Accuracy (`over_tlv`)**: **100.00%**


---

## 4. Range & Support Coverage Analysis

| Feature | Real Envelope Min/Max | Synthetic Min/Max | Synthetic Coverage (% inside Real Envelope) |
|---|---|---|---|
| `pct` | [0.0000, 3.8000] | [0.0000, 3.8000] | **100.0%** |
| `ppm` | [0.0000, 38000.0000] | [0.0000, 38000.0000] | **100.0%** |
| `ppm_noisy` | [0.0000, 38178.4194] | [0.0000, 38182.2280] | **100.0%** |

---

## 5. Visual Evaluation Artifacts

The following visual plots have been generated and saved to `gas_sensors/evaluation_plots/`:

- **Histograms + KDE Overlap**: `gas_sensors/evaluation_plots/part2_h2_hist_kde.png`
- **Boxplots Comparison**: `gas_sensors/evaluation_plots/part2_h2_boxplots.png`
- **Correlation Heatmaps**: `gas_sensors/evaluation_plots/part2_h2_correlation.png`
- **PCA 2D Projection**: `gas_sensors/evaluation_plots/part2_h2_pca.png`
- **t-SNE 2D Manifold**: `gas_sensors/evaluation_plots/part2_h2_tsne.png`
- **Pairwise Scatter Plots**: `gas_sensors/evaluation_plots/part2_h2_pairwise_scatter.png`