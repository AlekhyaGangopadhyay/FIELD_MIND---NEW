# Real Data Evaluation Report

**Generated:** 2026-07-28T14:09:12.024853

**Data source:** `mine_part2_bands.csv` (120,000 rows, 4 gases × 3 severity bands)

**Reference:** `docs/REAL_MINE_DATA_RETRAINING.md` §5.3


## 1. `gas_hazard_lpg_cng` — LPG/CNG Hazard Classifier

- **Test data:** Part 2 CH4 bands (30,000 rows), LPG held at 80 ppm baseline
- **Ground truth:** `MQ4_CH4_ppm > 1000` (10% LEL, per OSHA/MSHA)
- **Positives:** 20076 of 30000 (66.9%)
- **Old test set:** 0 positives out of 84 → acc=1.0, precision/recall/f1=0.0

| Metric | Old (synthetic) | New (real data) |
|--------|----------------|-----------------|
| Accuracy | 1.0000 | 1.0000 |
| Precision | 0.0000 | 1.0000 |
| Recall | 0.0000 | 1.0000 |
| F1-Score | 0.0000 | 1.0000 |

**Confusion Matrix:**

```
TN=  9924  FP=     0
FN=     0  TP= 20076
```

---

## 2. `gas_hazard_co_nox_c6h6` — CO/NOx/Benzene Hazard Classifier

- **Test data:** Part 2 CO bands (30,000 rows), NOx=0.04, Benzene=4.0 (baselines)
- **Ground truth:** `MQ7_CO_ppm > 50` (OSHA TWA)
- **Positives:** 9996 of 30000 (33.3%)

| Metric | Old (synthetic) | New (real data) |
|--------|----------------|-----------------|
| Accuracy | 0.9178 | 0.9955 |
| Precision | 0.2472 | 1.0000 |
| Recall | 1.0000 | 0.9865 |
| F1-Score | 0.3964 | 0.9932 |

**Confusion Matrix:**

```
TN= 20004  FP=     0
FN=   135  TP=  9861
```

---

## 3. `multi_gas_detector` — Multi-Gas Presence Classifier

Testing Methane and CO heads against real banded data.

### Methane (CH4) Head

- **Positives:** 29832 of 30000
- **Accuracy:** 0.9948
- **Precision:** 1.0000
- **Recall:** 0.9947
- **F1:** 0.9974

```
TN=   168  FP=     0
FN=   157  TP= 29675
```

### CO Head

- **Positives:** 9996 of 30000
- **Accuracy:** 0.4745
- **Precision:** 0.3880
- **Recall:** 1.0000
- **F1:** 0.5591

```
TN=  4239  FP= 15765
FN=     0  TP=  9996
```

---

## Summary

| Model | Positives | Accuracy | Precision | Recall | F1 |
|-------|-----------|----------|-----------|--------|-----|
| `gas_hazard_lpg_cng` | 20076/30000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `gas_hazard_co_nox_c6h6` | 9996/30000 | 0.9955 | 1.0000 | 0.9865 | 0.9932 |
| `multi_gas_detector_CH4` | 29832/30000 | 0.9948 | 1.0000 | 0.9947 | 0.9974 |
| `multi_gas_detector_CO` | 9996/30000 | 0.4745 | 0.3880 | 1.0000 | 0.5591 |

> **Conclusion:** These results show how each model performs when it finally sees real positive samples. Models that reported perfect accuracy on zero-positive test sets now face genuine hazard data.
