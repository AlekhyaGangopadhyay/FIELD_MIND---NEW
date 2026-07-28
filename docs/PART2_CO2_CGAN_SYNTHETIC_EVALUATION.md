# PyTorch CGAN Synthetic Data Evaluation Report: Carbon Dioxide (CO2)
## Dataset: `mine_part2_co2_realistic.csv` | Target Features: `over_tlv` & `severity`

This report provides a comprehensive 13-parameter evaluation of the PyTorch Conditional GAN synthesized dataset (`mine_part2_co2_balanced_cgan.csv`) compared against ground-truth real CO2 telemetry data.

---
## Executive Summary & Target Feature Balancing

- **Real Dataset Rows**: `30,000`
- **Balanced CGAN Dataset Rows**: `60,000` (Total 60,000 rows across 6 joint classes)
- **Target Constants Enforced**: `tlv_pct = 0.02` | `tlv_ppm = 200.0` (100% constant across all rows)
- **`over_tlv` Index Threshold**: 0 for `< 0.02%` / `< 200 ppm`, 1 for `>= 0.02%` / `>= 200 ppm`

| Metric Category | Parameter Evaluated | Summary Value | Quality Assessment |
|---|---|---|---|
| **Correlation** | Mean Abs Corr Diff (MACD) | `0.0000` | Low Drift (< 0.15) |
| **Distribution Distance** | Maximum Mean Discrepancy (MMD) | `0.00273` | High Fidelity Overlay |
| **Distinguishability** | Classifier Discriminator ROC-AUC | `0.9084` | Excellent Real-Synthetic Balance |
| **Distinguishability** | Classifier Discriminator Accuracy | `80.06%` | High Indistinguishability |
| **Downstream Utility** | Severity TRST Test Acc | `99.98%` | Superior Classification Utility |
| **Downstream Utility** | Over_TLV TRST Test Acc | `100.00%` | Perfect Classification Utility |

---

## 1. Descriptive Statistics Comparison

### Real CO2 Data vs Synthetic CGAN CO2 Data (Mean & Std)

| Feature | Real Mean ± Std | Synthetic Mean ± Std | Abs Diff (Mean) |
|---|---|---|---|
| `pct` | 0.1305 ± 0.1480 | 0.1408 ± 0.1596 | **0.0102** |
| `ppm` | 1305.4667 ± 1480.2808 | 1407.9121 ± 1596.2003 | **102.4454** |
| `ppm_noisy` | 1305.7445 ± 1480.2010 | 1407.8639 ± 1596.2931 | **102.1195** |

---

## 2. Kolmogorov-Smirnov (KS) Test & Wasserstein Distance

| Feature | KS Statistic | p-value | Wasserstein Dist (Scaled) | Distribution Match |
|---|---|---|---|---|
| `pct` | 0.0900 | 4.2715e-106 | 0.1099 | High Match |
| `ppm` | 0.0900 | 4.2715e-106 | 0.1099 | High Match |
| `ppm_noisy` | 0.0887 | 3.8905e-103 | 0.1072 | High Match |

---

## 3. Downstream Task Evaluation (`severity` & `over_tlv` Classification)

### Severity Classification (`severity` 0, 1, 2)

| Training Paradigm | Description | Test Accuracy |
|---|---|---|
| **TRTR** | Train Real -> Test Real | 88.66% |
| **TSTR** | Train Synthetic -> Test Real | 89.91% |
| **TRST** | Train Real + Synthetic -> Test Real | **99.98%** |

### Over TLV Classification (`over_tlv` 0, 1)

- **TRST Accuracy (`over_tlv`)**: **100.00%**


---

## 4. Range & Support Coverage Analysis

| Feature | Real Envelope Min/Max | Synthetic Min/Max | Synthetic Coverage (% inside Real Envelope) |
|---|---|---|---|
| `pct` | [0.0000, 0.5000] | [0.0000, 0.5000] | **100.0%** |
| `ppm` | [0.0000, 5000.0000] | [0.0000, 5000.0000] | **100.0%** |
| `ppm_noisy` | [0.0000, 5030.8155] | [0.0000, 5031.1896] | **100.0%** |

---

## 5. Visual Evaluation Artifacts

The following visual plots have been generated and saved to `gas_sensors/evaluation_plots/`:

- **Histograms + KDE Overlap**: `gas_sensors/evaluation_plots/part2_co2_hist_kde.png`
- **Boxplots Comparison**: `gas_sensors/evaluation_plots/part2_co2_boxplots.png`
- **Correlation Heatmaps**: `gas_sensors/evaluation_plots/part2_co2_correlation.png`
- **PCA 2D Projection**: `gas_sensors/evaluation_plots/part2_co2_pca.png`
- **t-SNE 2D Manifold**: `gas_sensors/evaluation_plots/part2_co2_tsne.png`
- **Pairwise Scatter Plots**: `gas_sensors/evaluation_plots/part2_co2_pairwise_scatter.png`