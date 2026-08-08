# Conditional GAN (CGAN) Synthetic Data Evaluation Report
## Dataset: `mine_part1_clean.csv` | Target Feature: `is_warmup`

This document presents a comprehensive 13-parameter evaluation of the PyTorch Conditional GAN synthesized dataset (`mine_part1_balanced_gan.csv`) compared against ground-truth real sensor telemetry data.

---
## Summary of Evaluation Parameters

| Metric Category | Parameter Evaluated | Summary Value | Quality Assessment |
|---|---|---|---|
| **Correlation** | Mean Abs Corr Diff (MACD) | `0.3744` | Excellent Low Drift (< 0.15) |
| **Distribution Distance** | Maximum Mean Discrepancy (MMD) | `1.03872` | High Fidelity Overlay |
| **Distinguishability** | Classifier Discriminator ROC-AUC | `1.0000` | Near Ideal Real-Synthetic Balance |
| **Distinguishability** | Classifier Discriminator Accuracy | `100.00%` | High Fidelity Indistinguishability |
| **Downstream Utility** | TSTR ROC-AUC (Train Syn -> Test Real) | `1.0000` | Excellent Classification Transfer |
| **Downstream Utility** | TRST ROC-AUC (Train Real+Syn -> Test Real) | `1.0000` | Superior Performance |

---

## 1. Descriptive Statistics Comparison

### Real Warmup vs Synthetic CGAN Warmup (Mean & Std)

| Feature | Real Mean ± Std | Synthetic Mean ± Std | Abs Diff (Mean) |
|---|---|---|---|
| `air_quality_ppm` | 282.99 ± 1382.21 | 43.60 ± 16.48 | **239.39** |
| `smoke_ppm` | 13.70 ± 59.86 | 6.48 ± 0.91 | **7.22** |
| `alcohol_ppm` | 0.00 ± 0.00 | 0.01 ± 0.04 | **0.01** |
| `flamable_gas_ppm` | 529926.56 ± 4784443.77 | 52232.16 ± 40310.24 | **477694.40** |
| `MQ136_ppm` | 391936.81 ± 3750768.81 | 11512.23 ± 18976.69 | **380424.58** |
| `MQ7_ppm` | 0.66 ± 0.01 | 0.66 ± 0.01 | **0.00** |
| `t` | 28.17 ± 0.63 | 28.17 ± 0.56 | **0.00** |
| `h` | 72.56 ± 3.23 | 72.85 ± 2.60 | **0.29** |

---

## 2. Kolmogorov-Smirnov (KS) Test & Wasserstein Distance

Evaluating 1D distributional similarity per feature:

| Feature | KS Statistic | p-value | Wasserstein Dist (Scaled) | Distribution Match |
|---|---|---|---|---|
| `air_quality_ppm` | 0.8476 | 5.8291e-71 | 0.2212 | Moderate Match |
| `smoke_ppm` | 0.4811 | 2.1887e-19 | 0.1436 | Moderate Match |
| `alcohol_ppm` | 0.8451 | 2.2419e-70 | 98720.0725 | Moderate Match |
| `flamable_gas_ppm` | 0.6091 | 5.4819e-32 | 0.1168 | Moderate Match |
| `MQ136_ppm` | 0.5728 | 5.3535e-28 | 0.1064 | Moderate Match |
| `MQ7_ppm` | 0.2089 | 7.0347e-04 | 0.3461 | High Match |
| `t` | 0.1307 | 8.7361e-02 | 0.2186 | High Match |
| `h` | 0.1554 | 2.4288e-02 | 0.2175 | High Match |

---

## 3. Downstream Task Evaluation (`is_warmup` Classification)

Evaluating model generalization utility across training paradigms:

| Training Paradigm | Description | Test Accuracy | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|
| **TRTR** | Train Real -> Test Real | 99.45% | 0.9434 | 1.0000 |
| **TSTR** | Train Synthetic -> Test Real | 98.53% | 0.8333 | 1.0000 |
| **TRST** | Train Real + Synthetic -> Test Real | **100.00%** | **1.0000** | **1.0000** |

---

## 4. Range & Support Coverage Analysis

| Feature | Real Envelope Min/Max | Synthetic Min/Max | Synthetic Coverage (% inside Real Envelope) |
|---|---|---|---|
| `air_quality_ppm` | [2.69, 11953.43] | [0.00, 95.24] | **99.6%** |
| `smoke_ppm` | [4.75, 578.68] | [2.36, 9.21] | **96.2%** |
| `alcohol_ppm` | [0.00, 0.00] | [0.00, 0.35] | **0.0%** |
| `flamable_gas_ppm` | [16.10, 46376242.94] | [0.00, 214234.92] | **84.4%** |
| `MQ136_ppm` | [54.01, 36368750.19] | [0.00, 125147.60] | **42.7%** |
| `MQ7_ppm` | [0.61, 0.69] | [0.62, 0.71] | **99.1%** |
| `t` | [27.10, 29.10] | [25.74, 29.27] | **94.1%** |
| `h` | [67.70, 80.50] | [66.73, 80.60] | **98.6%** |

---

## 5. Visual Evaluation Artifacts

The following visual plots have been generated and saved to `gas_sensors/evaluation_plots/`:

- **Histograms + KDE Overlap**: `gas_sensors/evaluation_plots/part1_hist_kde.png`
- **Boxplots Comparison**: `gas_sensors/evaluation_plots/part1_boxplots.png`
- **Correlation Heatmaps**: `gas_sensors/evaluation_plots/part1_correlation.png`
- **PCA 2D Projection**: `gas_sensors/evaluation_plots/part1_pca.png`
- **t-SNE 2D Manifold**: `gas_sensors/evaluation_plots/part1_tsne.png`
- **Pairwise Scatter Plots**: `gas_sensors/evaluation_plots/part1_pairwise_scatter.png`