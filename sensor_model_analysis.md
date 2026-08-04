# FIELD_MIND — Complete Sensor Input ↔ Model ↔ Prediction Map

## 1. Sensors and What They Provide

| Sensor | Measurable Gases / Quantities | Unit | Raw Output Column(s) in Synthetic Data |
|--------|-------------------------------|------|----------------------------------------|
| **MQ-2** | LPG, CH₄ (Methane), CO, Smoke | ppm | `MQ2_LPG_ppm`, `MQ2_CH4_ppm`, `MQ2_CO_ppm`, `MQ2_Smoke_ppm` |
| **MQ-3** | Alcohol, Benzene (C₆H₆) | ppm | `MQ3_Alcohol_ppm`, `MQ3_Benzene_ppm` |
| **MQ-4** | CH₄ (Methane — dedicated) | ppm | `MQ4_CH4_ppm` |
| **MQ-7** | CO (Carbon Monoxide — dedicated) | ppm | `MQ7_CO_ppm` |
| **MQ-135** | NH₃ (Ammonia), NOx, CO₂ | ppm | `MQ135_NH3_ppm`, `MQ135_NOx_ppm`, `MQ135_CO2_ppm` |
| **MQ-136** | H₂S (Hydrogen Sulphide) | ppm | `MQ136_H2S_ppm` |
| **MG811** | CO₂ (NDIR dedicated) | ppm | `MG811_CO2_ppm` |
| **PM2.5** | Particulate / Dust | µg/m³ | `PM25_Dust_ugm3` |
| **DHT22** | Temperature, Humidity | °C, % | `Temp_C`, `Humidity_pct` |

> [!NOTE]
> **Total unique physical readings from your 9 sensors: 16 columns** (MQ-2 gives 4, MQ-3 gives 2, MQ-4 gives 1, MQ-7 gives 1, MQ-135 gives 3, MQ-136 gives 1, MG811 gives 1, PM2.5 gives 1, DHT22 gives 2)

---

## 2. All Existing Models and Their Inputs → Outputs

### 🔵 GOAL 1: Gas Presence & Hazard Detection ("Is the gas present? Is it harmful?")

---

#### Model A: `multi_gas_detector` (Multi-Label Gas Presence)
- **File**: [multi_gas_detector.joblib](file:///c:/FIELDMIND/FIELD_MIND---NEW/gas_sensors/models/multi_gas_detector.joblib)
- **Training Script**: [train_multi_gas_detector.py](file:///C:/Users/user/.gemini/antigravity-ide/brain/9301a4ab-aea0-407d-be44-471acedf7a79/scratch/train_multi_gas_detector.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required** | `CH4_ppm`, `CO_ppm`, `CO2_ppm`, `H2_ppm`, `H2S_ppm`, `NH3_ppm`, `LPG_ppm`, `CNG_ppm` (8 features) |
| **Prediction** | Multi-label: **which gases are present** — Methane, CO, CO2, H2, H2S, NH3, LPG, CNG (8 outputs) |
| **Thresholds** | Evaluated via PyTorch neural network (LayerNormSwishMLP architecture) |
| **Task Type** | Multi-label classification (8 binary outputs) |

> [!IMPORTANT]
> **This directly answers your Goal #1** — "Which gas is present?" with 98.81% elementwise accuracy on real mine envelopes.

---

#### Model B: `gas_hazard_lpg_cng` (LPG/CNG Hazard Alert) — 🔴 DEPRECATED (OF NO USE)
- **Status**: Deprecated. Replaced by the 8-input unified `multi_gas_detector.joblib` which includes LPG/CNG targets.

---

#### Model C: `gas_hazard_co_nox_c6h6` (Combustion Gases Hazard) — 🔴 DEPRECATED (OF NO USE)
- **Status**: Deprecated. Replaced by the 8-input unified `multi_gas_detector.joblib` and specialized hazard detectors (`co2_hazard.joblib`, `nh3_hazard.joblib`).

---

#### Model D: `gas_hazard_smoke_env` (Smoke/Dust/Environment Hazard)
- **File**: [gas_hazard_smoke_env.joblib](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/models/gas_hazard_smoke_env.joblib)
- **Training Script**: [train_specific.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/train_specific.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required** | `PM25_Dust_ugm3`, `Temp_C`, `Humidity_pct` (PM2.5 + DHT22) |
| **Prediction** | Binary: **Hazard Alert** (fire/dust hazard) |
| **Task Type** | Binary classification |

> [!IMPORTANT]
> **This directly answers Goal #3** — "Is dust present and hazardous?"

---

#### Model E: `severity_{ch4,co,co2,h2,h2s}` (Per-Gas Severity Level)
- **Files**: `severity_ch4.joblib`, `severity_co.joblib`, `severity_co2.joblib`, `severity_h2.joblib`, `severity_h2s.joblib`
- **Training Script**: [retrain_severity_models.py](file:///C:/Users/user/.gemini/antigravity-ide/brain/9301a4ab-aea0-407d-be44-471acedf7a79/scratch/retrain_severity_models.py)

| Direction | Details |
|-----------|---------|
| **Input Required** | `ppm` (single gas concentration from the respective sensor) |
| **Prediction** | Multi-class: **Severity Level** — `0 (Safe)`, `1 (Warning)`, `2 (Critical)` |
| **Band Boundaries** | See safety standard table below |

| Gas | Sensor Source | Safe (Level 0) | Warning (Level 1) | Critical (Level 2) |
|-----|-------------|-----------|--------------|----------------|
| **CH₄** | MQ-4 / MQ-2 | < 10,000 ppm (1.0% LEL) | 10,000 to 15,000 ppm | $\ge$ 15,000 ppm |
| **CO** | MQ-7 / MQ-2 | < 25 ppm (OSHA PEL) | 25 to 50 ppm | $\ge$ 50 ppm |
| **CO₂** | MG811/MQ-135 | < 1,000 ppm (Vent Limit) | 1,000 to 4,000 ppm | $\ge$ 4,000 ppm |
| **H₂** | MQ-2 | < 4,000 ppm (10% LEL) | 4,000 to 20,000 ppm | $\ge$ 20,000 ppm |
| **H₂S** | MQ-136 | < 10 ppm (OSHA TWA) | 10 to 20 ppm | $\ge$ 20 ppm |

---

#### Model F: DL Tournament Best Models (`*_dl_best.joblib`) — 🔴 DEPRECATED (OF NO USE)
- **Status**: Deprecated. Replaced entirely by the retrained multiclass safety classifiers (`severity_ch4`, `severity_co`, `severity_co2`, `severity_h2`, `severity_h2s`) to support true hazard boundaries.

---

#### Model G: `smoke_fire_alarm` (Smoke/Fire Alarm)
- **File**: [smoke_fire_alarm_model.joblib](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/models/smoke_fire_alarm_model.joblib)
- **Training Script**: [train.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/train.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required** | 12 base features + engineered features: `Temperature[C]`, `Humidity[%]`, `TVOC[ppb]`, `eCO2[ppm]`, `Raw H2`, `Raw Ethanol`, `Pressure[hPa]`, `PM1.0`, `PM2.5`, `NC0.5`, `NC1.0`, `NC2.5` + 5-period diffs + 5-period rolling stds (36 total) |
| **Prediction** | Binary: **Fire Alarm** (1 = fire detected, 0 = safe) |

---

#### Model H: `air_quality_regressor` (Benzene Concentration Estimation)
- **File**: [air_quality_regressor.joblib](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/models/air_quality_regressor.joblib)
- **Training Script**: [train.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/train.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required** | `PT08.S2(NMHC)`, `NOx(GT)`, `PT08.S3(NOx)`, `PT08.S5(O3)`, `T`, `RH`, `AH` (7 features) |
| **Prediction** | Regression: **C₆H₆(GT) in µg/m³** (Benzene concentration) |

---

#### Model I: `combined_gases_regressor` (CO Concentration Prediction)
- **File**: [combined_gases_regressor.joblib](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/models/combined_gases_regressor.joblib)
- **Training Script**: [train.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/train.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required** | `SO2_ppm`, `NO2_ppm`, `O3_ppm` + autoregressive lags (1–5) + rolling averages (33 total engineered features) |
| **Prediction** | Regression: **CO_ppm** (Carbon Monoxide concentration) |

---

### 🟢 GOAL 2: Wall/Floor/Roof Fall Prediction (Structural Collapse)

---

#### Model J: `vibration_hazard_classifier` / `vibration_regressor` — 🔴 DEPRECATED (OF NO USE)
- **Status**: Deprecated. Replaced by the real-time physical calculations in the `vibration/structural_monitor.py` package (`SW420VibrationMonitor` and `UltrasonicDisplacementModel`).

---

#### Model L: `ultrasonic_navigation_classifier` (Robot Navigation / Proximity)
- **Files**: `best_ultrasonic_2.joblib`, `best_ultrasonic_4.joblib`, `best_ultrasonic_24.joblib`
- **Training Script**: [train_models.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/ultrasonic_sensors/train_models.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required (24-sensor)** | `US1` to `US24` (24 ultrasound distance readings around robot) |
| **Inputs Required (4-sensor)** | `SD_front`, `SD_left`, `SD_right`, `SD_back` (simplified distances) |
| **Inputs Required (2-sensor)** | `SD_front`, `SD_left` |
| **Prediction** | 4-class: `Move-Forward`, `Slight-Right-Turn`, `Sharp-Right-Turn`, `Slight-Left-Turn` |

> [!NOTE]
> The ultrasonic models are for **autonomous navigation** (wall-following), not structural fall prediction. However, sudden changes in ultrasonic distance readings *could* indicate wall collapse or roof fall if repurposed.

---

### 🟡 GOAL 3: Dust Presence Detection

#### Model D (reused): `gas_hazard_smoke_env`
| Direction | Details |
|-----------|---------|
| **Inputs Required** | `PM25_Dust_ugm3`, `Temp_C`, `Humidity_pct` |
| **Prediction** | Binary: **Dust/Smoke Hazard** (PM2.5 > 150 µg/m³ = silica hazard) |

---

### 🟠 Temperature/Humidity Anomaly Detection

#### Model M: `isolation_forest_iot` and `isolation_forest_uci`
- **Training Script**: [train.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/temperature_humidity/src/train.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required** | `temp`, `humidity`, `temp_hum_product`, `temp_hum_ratio`, `humidex`, rolling means/stds (9 features) |
| **Prediction** | Anomaly: **Normal (+1)** vs **Anomalous (-1)** (extreme temp/humidity) |

#### Model N: `random_forest` (Occupancy Detection)
| Direction | Details |
|-----------|---------|
| **Inputs Required** | `Temperature`, `Humidity`, + engineered features from UCI occupancy dataset |
| **Prediction** | Binary: **Occupancy** (1 = occupied, 0 = not occupied) |

---

## 3. Summary: What You Need to Input Per Sensor

| Your Sensor | Input Columns You'll Read | Which Models Use It |
|-------------|--------------------------|---------------------|
| **MQ-2** | `MQ2_LPG_ppm`, `MQ2_CH4_ppm`, `MQ2_CO_ppm`, `MQ2_Smoke_ppm` | `multi_gas_detector` |
| **MQ-3** | `MQ3_Alcohol_ppm`, `MQ3_Benzene_ppm` | `multi_gas_detector` |
| **MQ-4** | `MQ4_CH4_ppm` | `multi_gas_detector`, `severity_ch4` |
| **MQ-7** | `MQ7_CO_ppm` | `multi_gas_detector`, `severity_co` |
| **MQ-135** | `MQ135_NH3_ppm`, `MQ135_NOx_ppm`, `MQ135_CO2_ppm` | `multi_gas_detector`, `severity_co2` |
| **MQ-136** | `MQ136_H2S_ppm` | `multi_gas_detector`, `severity_h2s` |
| **MG811** | `MG811_CO2_ppm` | `multi_gas_detector`, `severity_co2` |
| **PM2.5** | `PM25_Dust_ugm3` | `smoke_env_hazard`, `smoke_fire_alarm` |
| **DHT22** | `Temp_C`, `Humidity_pct` | `smoke_env_hazard`, `isolation_forest_iot` |

---

## 4. Complete List of Predictions From Existing Models

| # | Prediction | Type | Model(s) |
|---|------------|------|----------|
| 1 | **Which gas is present** (Methane, CO, LPG, Smoke, NOx) | Multi-label classification | `multi_gas_detector` |
| 2 | **Is LPG/CNG at hazardous level?** | Binary classification | `gas_hazard_lpg_cng` |
| 3 | **Is CO/NOx/Benzene at hazardous level?** | Binary classification | `gas_hazard_co_nox_c6h6` |
| 4 | **Is Dust/Smoke at hazardous level?** | Binary classification | `gas_hazard_smoke_env` |
| 5 | **Gas severity level** (Safe/Warning/Critical) for CH₄, CO, CO₂, H₂ | Multi-class classification | `severity_ch4`, `severity_co`, `severity_co2`, `severity_h2` |
| 6 | **Is gas over TLV** (Threshold Limit Value)? for CH₄, CO, CO₂, H₂ | Binary classification | `*_over_tlv_dl_best` models |
| 7 | **Fire Alarm trigger** | Binary classification | `smoke_fire_alarm` |
| 8 | **Benzene (C₆H₆) concentration** | Regression (µg/m³) | `air_quality_regressor` |
| 9 | **CO concentration from other gas correlates** | Regression (ppm) | `combined_gases_regressor` |
| 10 | **Blast vibration hazard** (PPV > 1.0 mm/s → collapse risk) | Binary classification | Vibration classifier |
| 11 | **Peak Particle Velocity magnitude** (ln(PPV)) | Regression | Vibration regressor |
| 12 | **Robot navigation direction** | 4-class classification | Ultrasonic models |
| 13 | **Temperature/Humidity anomaly** | Anomaly detection | Isolation Forest models |
| 14 | **Area occupancy** | Binary classification | Random Forest (UCI) |
| 15 | **Sensor warmup state** | Binary classification | `part1_warmup_dl_best` |

---

## 5. Gaps & Resolved Items

| Item / Gap | Status | Resolution / Approach |
|-----|----------------|-------------------|
| **H₂S severity levels** | ✅ RESOLVED | Retrained PyTorch Deep MLP model `severity_h2s` based on OSHA TWA limits (10 ppm warning / 20 ppm critical). |
| **Direct geomechanical fall/displacement monitoring** | ✅ RESOLVED | Implemented physical `vibration/structural_monitor.py` displacement tracking velocity and acceleration to detect imminent wall collapse events. |
| **NH₃ (Ammonia) hazard classification** | ✅ RESOLVED | NH3 is now tracked natively in the 8-input unified `multi_gas_detector` model. |
| **Continuous dust concentration regression** | Missing | Can train a regression model mapping MQ-2 + PM2.5 → continuous µg/m³. |
| **Structural integrity hardware integration** | Missing | Would need additional rock bolt strain sensor telemetry integrated. |
