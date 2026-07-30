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
- **File**: [multi_gas_detector.joblib](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/models/multi_gas_detector.joblib)
- **Training Script**: [train_gas_detector.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/train_gas_detector.py) and [train_hazard_dl.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/train_hazard_dl.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required** | `MQ2_CO_ppm`, `MQ2_LPG_ppm`, `MQ2_Smoke_ppm` (3 features from MQ-2) |
| **Prediction** | Multi-label: **which gases are present** — Methane ✅/❌, CO ✅/❌, LPG ✅/❌, Smoke ✅/❌, NOx ✅/❌ |
| **Thresholds** | CH₄ > 117.5 ppm, CO > 15 ppm, LPG > 135 ppm, Smoke > 120 ppm, NOx > 0.07 ppm |
| **Task Type** | Multi-label classification (5 binary outputs) |

> [!IMPORTANT]
> **This directly answers your Goal #1** — "Which gas is present?"

---

#### Model B: `gas_hazard_lpg_cng` (LPG/CNG Hazard Alert)
- **File**: [gas_hazard_lpg_cng.joblib](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/models/gas_hazard_lpg_cng.joblib)
- **Training Script**: [train_hazard_dl.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/train_hazard_dl.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required** | `MQ2_LPG_ppm`, `MQ4_CH4_ppm` (2 features from MQ-2 + MQ-4) |
| **Prediction** | Binary: **Hazard Alert** (1 = dangerous, 0 = safe) |
| **Threshold Logic** | CH₄ ≥ 12,500 ppm (1.25% LEL) OR LPG ≥ 110 ppm |
| **Task Type** | Binary classification |

---

#### Model C: `gas_hazard_co_nox_c6h6` (Combustion Gases Hazard)
- **File**: [gas_hazard_co_nox_c6h6.joblib](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/models/gas_hazard_co_nox_c6h6.joblib)
- **Training Script**: [train_hazard_dl.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/train_hazard_dl.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required** | `MQ7_CO_ppm`, `MQ135_NOx_ppm`, `MQ3_Benzene_ppm` (3 features from MQ-7 + MQ-135 + MQ-3) |
| **Prediction** | Binary: **Hazard Alert** (1 = dangerous, 0 = safe) |
| **Threshold Logic** | CO ≥ 50 ppm OR NOx ≥ 0.10 ppm OR Benzene ≥ 5.0 ppm |
| **Task Type** | Binary classification |

> [!IMPORTANT]
> **Models B and C directly answer "Is their presence harmful?"** — they output Hazard Alerts.

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

#### Model E: `severity_{ch4,co,co2,h2}` (Per-Gas Severity Level)
- **Files**: `severity_ch4.joblib`, `severity_co.joblib`, `severity_co2.joblib`, `severity_h2.joblib`
- **Training Script**: [train_mine_severity.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/train_mine_severity.py)

| Direction | Details |
|-----------|---------|
| **Input Required** | `ppm` (single gas concentration from the respective sensor) |
| **Prediction** | Multi-class: **Severity Level** — `L1 (Safe)`, `L2 (Warning)`, `L3 (Critical)` |
| **Band Boundaries** | See table below |

| Gas | Sensor Source | L1 (Safe) | L2 (Warning) | L3 (Critical) |
|-----|-------------|-----------|--------------|----------------|
| **CH₄** | MQ-4 | 0 – 12,500 ppm | 12,500 – 18,750 ppm | 18,750 – 25,000 ppm |
| **CO** | MQ-7 | 0 – 37.5 ppm | 37.5 – 50 ppm | 50 – 10,000 ppm |
| **CO₂** | MG811/MQ-135 | 0 – 400 ppm | 400 – 1,000 ppm | 1,000 – 5,000 ppm |
| **H₂** | (derived) | 0 – 18,000 ppm | 18,000 – 25,000 ppm | 25,000 – 38,000 ppm |

---

#### Model F: DL Tournament Best Models (`*_dl_best.joblib`)
- **Training Script**: [retrain_dl_models.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/retrain_dl_models.py)

| Model Key | Input Features | Target | Output |
|-----------|----------------|--------|--------|
| `part1_warmup` | `air_quality, smoke, alcohol, flamable_gas, MQ136_raw, MQ7_raw, t, h` (8 features) | `is_warmup` | Binary: Is sensor in warmup phase? |
| `ch4_severity` | `ppm` (1 feature) | `severity` | 3-class: L1/L2/L3 |
| `ch4_over_tlv` | `ppm` (1 feature) | `over_tlv` | Binary: Over Threshold Limit Value? |
| `co_severity` | `ppm` (1 feature) | `severity` | 3-class: L1/L2/L3 |
| `co_over_tlv` | `ppm` (1 feature) | `over_tlv` | Binary: Over TLV? |
| `co2_severity` | `ppm` (1 feature) | `severity` | 3-class: L1/L2/L3 |
| `co2_over_tlv` | `ppm` (1 feature) | `over_tlv` | Binary: Over TLV? |
| `h2_severity` | `ppm` (1 feature) | `severity` | 3-class: L1/L2/L3 |
| `h2_over_tlv` | `ppm` (1 feature) | `over_tlv` | Binary: Over TLV? |

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

#### Model J: `vibration_hazard_classifier` (Blast Vibration Hazard)
- **File**: [best_gradient_boosting_classifier.joblib](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/vibration/models/) (best model)
- **Training Script**: [train_models.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/vibration/train_models.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required** | `offset`, `max_charge`, `total_charge`, `num_holes`, `detonator_code`, `trid_12/13/14` (component direction), `gx, gy, gelev, sx, sy, selev` (coordinates), `scaled_distance_usbm`, `scaled_distance_langefors`, `elevation_diff` (up to 17 features) |
| **Prediction** | Binary: **Vibration Hazard** (PPV > 1.0 mm/s → wall/roof fall risk) |

#### Model K: `vibration_regressor` (PPV Magnitude Prediction)
- **File**: [best_gradient_boosting_regressor.joblib](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/vibration/models/) (best model)
- **Training Script**: [train_models.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/vibration/train_models.py)

| Direction | Details |
|-----------|---------|
| **Inputs Required** | Same as Model J (up to 17 features) |
| **Prediction** | Regression: **ln(PPV)** — Peak Particle Velocity in mm/s |

> [!WARNING]
> **For actual wall/floor/roof fall prediction**, the vibration models predict whether blast-induced vibration exceeds safety thresholds (PPV > 1.0 mm/s). This is a **proxy indicator** — high PPV correlates with structural collapse risk. You do **not** currently have a dedicated "wall fall" or "roof fall" classifier. You would need additional data (e.g., accelerometer time-series on walls, geophone data, rock bolt strain gauges) for a direct fall prediction model.

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

---

#### Model D (reused): `gas_hazard_smoke_env`
| Direction | Details |
|-----------|---------|
| **Inputs Required** | `PM25_Dust_ugm3`, `Temp_C`, `Humidity_pct` |
| **Prediction** | Binary: **Dust/Smoke Hazard** (PM2.5 > 150 µg/m³ = silica hazard) |

> [!TIP]
> The PM2.5 dust sensor reading is directly used. The model classifies whether dust is at hazardous levels (> 150 µg/m³). If you want *continuous* dust level prediction (not just binary hazard), you can use the raw `PM25_Dust_ugm3` reading directly from the PM2.5 sensor.

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
| **MQ-2** | `MQ2_LPG_ppm`, `MQ2_CH4_ppm`, `MQ2_CO_ppm`, `MQ2_Smoke_ppm` | A (gas presence), B (LPG/CNG hazard), D (smoke/dust) |
| **MQ-3** | `MQ3_Alcohol_ppm`, `MQ3_Benzene_ppm` | C (combustion hazard) |
| **MQ-4** | `MQ4_CH4_ppm` | B (LPG/CNG hazard), E (CH₄ severity) |
| **MQ-7** | `MQ7_CO_ppm` | C (combustion hazard), E (CO severity), F (CO over TLV) |
| **MQ-135** | `MQ135_NH3_ppm`, `MQ135_NOx_ppm`, `MQ135_CO2_ppm` | A (gas presence), C (combustion hazard), E (CO₂ severity) |
| **MQ-136** | `MQ136_H2S_ppm` | F (warmup detection — as raw reading) |
| **MG811** | `MG811_CO2_ppm` | E (CO₂ severity — interchangeable with MQ-135 CO₂) |
| **PM2.5** | `PM25_Dust_ugm3` | D (smoke/dust hazard), G (fire alarm) |
| **DHT22** | `Temp_C`, `Humidity_pct` | D (smoke/dust hazard), M (anomaly detection) |

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

## 5. Gaps — What You DON'T Have Yet

> [!CAUTION]
> The following are **not directly predictable** from existing models:

| Gap | What's Missing | Suggested Approach |
|-----|----------------|-------------------|
| **Direct wall/floor/roof fall prediction** | No labeled "collapse" dataset. Vibration model is a proxy only. | Need accelerometer/geophone time-series data with labeled collapse events, or use vibration PPV threshold as trigger. |
| **H₂S severity levels** | Severity models exist for CH₄, CO, CO₂, H₂ but **not** H₂S | Train a `severity_h2s` model using MQ-136 data with bands (e.g., 0–10 ppm safe, 10–20 warning, >20 critical per MSHA). |
| **NH₃ (Ammonia) hazard classification** | MQ-135 reads NH₃ but no dedicated hazard model for it | Can be added with threshold at 50 ppm (OSHA TWA). |
| **Continuous dust concentration regression** | Only binary "hazardous or not" exists | Can train a regression model mapping MQ-2 + PM2.5 → continuous µg/m³. |
| **Structural integrity monitoring** | No rock bolt strain or geophone data | Would need additional hardware sensors (strain gauges, seismometers). |
