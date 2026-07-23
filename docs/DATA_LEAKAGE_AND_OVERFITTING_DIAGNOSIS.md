# Data Leakage & Overfitting Diagnosis Report

## Executive Summary & Findings

An empirical investigation was conducted to analyze why `part1_warmup_dl_best.joblib` and `ch4_over_tlv_dl_best.joblib` achieve nearly **100% training and testing accuracy**.

### Verdict Summary

| Model Target | Near 100% Acc Cause | Is it Overfitting? | Is it Data Leakage? | Explanation & Recommended Fix |
|---|---|:---:|:---:|---|
| `ch4_over_tlv_dl_best.joblib` | **Deterministic Target Definition** | **NO** | **YES (Feature Redundancy)** | `over_tlv` is mathematically defined as `(pct >= 2.5)`. Passing `pct` and `ppm` into input `X` gives the model the exact ground-truth threshold variable. **Fix**: Train over_tlv classifiers using *only* noisy sensor output (`ppm_noisy`). |
| `part1_warmup_dl_best.joblib` | **Physical Feature Separability** | **NO** | **NO** | Sensor warmup dynamics exhibit distinct thermal stabilization (`t <= 29.1°C` vs `t > 29.1°C`, corr = -0.945). The classes are physically linearly separable. |

---

## 1. Investigation: `ch4_over_tlv_dl_best.joblib`

### 1.1 Root Cause Analysis
In `mine_part2_ch4_balanced_cgan.csv`, the target feature `over_tlv` is generated deterministically:

$$\text{over\_tlv} = \begin{cases} 1 & \text{if } \text{pct} \ge 2.5 \text{ (or } \text{ppm} \ge 25,000\text{)} \\ 0 & \text{if } \text{pct} < 2.5 \text{ (or } \text{ppm} < 24,990\text{)} \end{cases}$$

Empirical feature ranges in the dataset:
- `over_tlv == 1`: Min `pct` = **2.500%** (Min `ppm` = **25,000.0**)
- `over_tlv == 0`: Max `pct` = **2.499%** (Max `ppm` = **24,990.0**)

When the input feature matrix is defined as `X = ["pct", "ppm", "ppm_noisy"]`, the model receives `pct` and `ppm` directly. The neural network learns the simple step threshold `ppm >= 25000` with **100% precision** on both training and test sets.

### 1.2 Realistic Deployment Test (Using Only `ppm_noisy`)
If a deployed sensor only measures noisy electrical readings (`ppm_noisy`) without pre-calibrated ground-truth `pct`/`ppm`:
- `over_tlv == 1`: Min `ppm_noisy` = 22,802.7
- `over_tlv == 0`: Max `ppm_noisy` = 26,850.0

Noise overlaps around the 25,000 ppm threshold. Training on `ppm_noisy` alone yields realistic sensor uncertainty.

---

## 2. Investigation: `part1_warmup_dl_best.joblib`

### 2.1 Root Cause Analysis
In `mine_part1_clean.csv` / `mine_part1_balanced_gan.csv`, physical MQ gas sensors undergo a thermal stabilization phase when first powered on.

Empirical feature distributions by `is_warmup` state:

| Feature | Warmup (`is_warmup == True`) | Normal (`is_warmup == False`) | Correlation with `is_warmup` |
|---|:---:|:---:|:---:|
| Temperature (`t`) | **27.1°C – 29.1°C** | **29.1°C – 30.4°C** | **-0.945** |
| Humidity (`h`) | **67.7% – 80.5%** | **55.8% – 70.0%** | **+0.638** |
| Flammable Gas (`flamable_gas`) | Initial high resistance spike | Stabilized baseline | **+0.559** |

Because temperature `t` and initial sensor resistance form a distinct boundary (`t <= 29.1°C`), the warmup state is **physically linearly separable** from normal operation. This is an accurate reflection of MQ sensor physics, not overfitting or data leakage.
