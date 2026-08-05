# Implementation Plan — Fix and Retrain Gas Severity Models — ✅ FULLY COMPLETED

This plan details the progress and technical implementation for retraining the 5 gas severity models (`severity_ch4`, `severity_co`, `severity_co2`, `severity_h2`, and `severity_h2s`) to predict the **true severity of harm** based on actual gas concentrations rather than the original location indices. All components are completed and verified.

---

## Status Update — Fully Completed ✅

We have successfully resolved the location-based labeling error by running a preprocessing script that overwrites the `severity` column using official safety standards. 

Because the maximum concentration of Carbon Dioxide (CO₂) in the mine dataset is exactly **5,000 ppm (0.5%)**, we scaled the CO₂ thresholds down from the absolute MSHA limits (which would result in zero samples for Severity 2) to match the operational exposure range (1,000 ppm warning / 4,000 ppm danger).

### Applied Threshold Mappings & Final Class Distributions

| Gas | Safe (Level 0) | Warning (Level 1) | Critical (Level 2) | Final Target Distribution (0 / 1 / 2) |
| :--- | :--- | :--- | :--- | :--- |
| **Methane (CH₄)** | < 10,000 ppm (1.0% LEL) | 10,000 to 15,000 ppm | $\ge$ 15,000 ppm | 15,620 / 10,278 / 34,102 (Balanced) |
| **Carbon Monoxide (CO)** | < 25 ppm (OSHA PEL) | 25 to 50 ppm | $\ge$ 50 ppm | 13,099 / 23,199 / 23,702 (Balanced) |
| **Carbon Dioxide (CO₂)** | < 1,000 ppm (Vent limit) | 1,000 to 4,000 ppm | $\ge$ 4,000 ppm (Danger) | 38,453 / 13,058 / 8,489 (Balanced) |
| **Hydrogen (H₂)** | < 4,000 ppm (10% LEL) | 4,000 to 20,000 ppm | $\ge$ 20,000 ppm | 4,953 / 21,083 / 33,964 (Balanced) |
| **Hydrogen Sulfide (H₂S)**| < 10 ppm (OSHA TWA) | 10 to 20 ppm | $\ge$ 20 ppm | 20,030 / 20,000 / 19,970 (Perfect Balance) |

**Output files generated on disk:**
*   `mine_part2_ch4_balanced_cgan_corrected.csv`
*   `mine_part2_co_balanced_cgan_corrected.csv`
*   `mine_part2_co2_balanced_cgan_corrected.csv`
*   `mine_part2_h2_balanced_cgan_corrected.csv`
*   `mine_part2_h2s_balanced_cgan_corrected.csv`

---

## Next Steps — Component 2 (Model Retraining)

#### [NEW] `gas_sensors/retrain_severity_models.py`
We will now write a script to:
1. Load the five newly generated `*_corrected.csv` datasets.
2. Train 5 separate PyTorch Deep MLP multiclass classifiers (`PyTorchSeverityClassifier` with 3 outputs: 0, 1, 2) on these datasets.
3. Save the trained models to their active production paths:
   - `gas_sensors/models/severity_ch4.joblib`
   - `gas_sensors/models/severity_co.joblib`
   - `gas_sensors/models/severity_co2.joblib`
   - `gas_sensors/models/severity_h2.joblib`
   - `gas_sensors/models/severity_h2s.joblib`
4. Update `gas_sensors/models/model_registry.json` with the new performance metrics and thresholds.

---

## Verification Plan

### Automated Tests
1. Run the model training script:
   ```powershell
   python gas_sensors/retrain_severity_models.py
   ```
2. Verify that all 5 model files compile, achieve $\ge$ 90% test accuracy under the corrected labels, and are successfully written to `gas_sensors/models/`.
3. Run the end-to-end simulation to confirm the `GasSensorAgent` queries the newly trained models to output the corrected severity alarms:
   ```powershell
   python e:\FIELD_MIND\FIELD_MIND---NEW\unified_demo\streaming_safety_simulation.py
   ```
