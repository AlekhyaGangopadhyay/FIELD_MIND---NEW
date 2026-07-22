# Conditional GAN (CGAN) Synthetic Data Evaluation Report
## Dataset: `mine_part1_clean.csv` | Target Feature: `is_warmup`

This document presents a comprehensive 13-parameter evaluation of the PyTorch Conditional GAN synthesized dataset (`mine_part1_balanced_gan.csv`) compared against ground-truth real sensor telemetry data.

---
## Summary of Evaluation Parameters

| Metric Category | Parameter Evaluated | Summary Value | Quality Assessment |
|---|---|---|---|
| **Correlation** | Mean Abs Corr Diff (MACD) | `0.4712` | Excellent Low Drift (< 0.15) |
| **Distribution Distance** | Maximum Mean Discrepancy (MMD) | `0.40544` | High Fidelity Overlay |
| **Distinguishability** | Classifier Discriminator ROC-AUC | `1.0000` | Near Ideal Real-Synthetic Balance |
| **Distinguishability** | Classifier Discriminator Accuracy | `100.00%` | High Fidelity Indistinguishability |
| **Downstream Utility** | TSTR ROC-AUC (Train Syn -> Test Real) | `0.9998` | Excellent Classification Transfer |
| **Downstream Utility** | TRST ROC-AUC (Train Real+Syn -> Test Real) | `1.0000` | Superior Performance |

---

## 1. Descriptive Statistics Comparison

### Real Warmup vs Synthetic CGAN Warmup (Mean & Std)

| Feature | Real Mean ± Std | Synthetic Mean ± Std | Abs Diff (Mean) |
|---|---|---|---|
| `air_quality` | 18.50 ± 26.01 | 9.19 ± 0.58 | **9.31** |
| `smoke` | 271.23 ± 71.38 | 231.79 ± 8.00 | **39.44** |
| `alcohol` | 25.00 ± 0.00 | 25.00 ± 0.00 | **0.00** |
| `flamable_gas` | 539.83 ± 637.78 | 316.51 ± 20.96 | **223.32** |
| `MQ136_raw` | 687.50 ± 821.01 | 305.49 ± 19.79 | **382.01** |
| `MQ7_raw` | 120.54 ± 1.28 | 118.30 ± 1.48 | **2.24** |
| `t` | 28.17 ± 0.63 | 28.68 ± 0.27 | **0.51** |
| `h` | 72.56 ± 3.23 | 72.63 ± 1.65 | **0.08** |

---

## 2. Kolmogorov-Smirnov (KS) Test & Wasserstein Distance

Evaluating 1D distributional similarity per feature:

| Feature | KS Statistic | p-value | Wasserstein Dist (Scaled) | Distribution Match |
|---|---|---|---|---|
| `air_quality` | 0.9035 | 1.3752e-86 | 0.3598 | Moderate Match |
| `smoke` | 0.8458 | 1.4222e-70 | 0.5556 | Moderate Match |
| `alcohol` | 0.0000 | 1.0000e+00 | 0.0000 | High Match |
| `flamable_gas` | 0.4726 | 1.0959e-18 | 0.5654 | Moderate Match |
| `MQ136_raw` | 0.7360 | 3.0108e-49 | 0.4678 | Moderate Match |
| `MQ7_raw` | 0.6975 | 2.2485e-43 | 1.7545 | Moderate Match |
| `t` | 0.4454 | 1.4856e-16 | 0.8094 | Moderate Match |
| `h` | 0.2645 | 5.6100e-06 | 0.4354 | High Match |

---

## 3. Downstream Task Evaluation (`is_warmup` Classification)

Evaluating model generalization utility across training paradigms:

| Training Paradigm | Description | Test Accuracy | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|
| **TRTR** | Train Real -> Test Real | 99.45% | 0.9434 | 1.0000 |
| **TSTR** | Train Synthetic -> Test Real | 98.17% | 0.7826 | 0.9998 |
| **TRST** | Train Real + Synthetic -> Test Real | **100.00%** | **1.0000** | **1.0000** |

---

## 4. Range & Support Coverage Analysis

| Feature | Real Envelope Min/Max | Synthetic Min/Max | Synthetic Coverage (% inside Real Envelope) |
|---|---|---|---|
| `air_quality` | [10.00, 160.00] | [7.74, 11.18] | **9.7%** |
| `smoke` | [238.00, 774.00] | [195.61, 249.07] | **23.6%** |
| `alcohol` | [25.00, 25.00] | [25.00, 25.00] | **100.0%** |
| `flamable_gas` | [100.00, 4062.00] | [245.05, 372.45] | **100.0%** |
| `MQ136_raw` | [298.00, 4921.00] | [239.74, 370.57] | **64.5%** |
| `MQ7_raw` | [116.00, 124.00] | [113.63, 124.65] | **93.7%** |
| `t` | [27.10, 29.10] | [27.66, 29.41] | **96.9%** |
| `h` | [67.70, 80.50] | [68.37, 77.92] | **100.0%** |

---

## 5. Visual Evaluation Artifacts

The following visual plots have been generated and saved to `gas_sensors/evaluation_plots/`:

- **Histograms + KDE Overlap**: `gas_sensors/evaluation_plots/part1_hist_kde.png`
- **Boxplots Comparison**: `gas_sensors/evaluation_plots/part1_boxplots.png`
- **Correlation Heatmaps**: `gas_sensors/evaluation_plots/part1_correlation.png`
- **PCA 2D Projection**: `gas_sensors/evaluation_plots/part1_pca.png`
- **t-SNE 2D Manifold**: `gas_sensors/evaluation_plots/part1_tsne.png`
- **Pairwise Scatter Plots**: `gas_sensors/evaluation_plots/part1_pairwise_scatter.png`