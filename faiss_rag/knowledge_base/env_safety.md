# Environmental Safety — Temperature, Humidity & Occupancy Standards

## Overview

Underground mine environments present unique thermal and atmospheric challenges. FIELD-MIND's Environmental Sensor Agent monitors temperature and humidity continuously using Isolation Forest anomaly detection and an occupancy classifier, benchmarked against international occupational health standards.

---

## Temperature Standards

### OSHA and NIOSH Guidelines (USA)

| Condition | Temperature (°C) | Risk Level |
|-----------|------------------|------------|
| Optimal work zone | 18 – 24 °C | Safe |
| Caution zone | 24 – 28 °C | Heat stress possible |
| Warning zone | > 28 °C | Heat strain likely |
| Danger zone | > 35 °C | Heat stroke risk |
| Cold stress | < 10 °C | Cold-related illness risk |
| Frostbite risk | < 0 °C | Immediate hazard |

**FIELD-MIND anomaly thresholds** (from Isolation Forest training):
- Temperature anomaly if `temp > 28°C` OR `temp < 5°C`

### South African Mine Health and Safety Act (MHSA)

Section 11.5 of MHSA requires:
- Wet bulb temperature shall not exceed 27.5°C at the workplace
- Effective temperature (ET) shall not exceed 28°C
- For virgin rock temperatures > 35°C, environmental engineering controls are mandatory
- Heat stress index must be monitored using wet bulb globe temperature (WBGT)

### Heat Stress Index Calculation

**Humidex** (used in FIELD-MIND feature engineering):
```
Humidex = T + 0.33 × e - 4.0
e = 6.105 × exp(17.27 × RH / (237.7 + RH))
```
Where T = temperature (°C), RH = relative humidity (%).

| Humidex | Comfort Level |
|---------|---------------|
| < 29    | Comfortable |
| 30 – 39 | Some discomfort |
| 40 – 45 | Great discomfort, avoid exertion |
| > 45    | Dangerous, heat stroke likely |

---

## Humidity Standards

| Relative Humidity (%) | Condition | Effect |
|-----------------------|-----------|--------|
| < 15 | Very dry | Dust suspension increases, respiratory irritation |
| 15 – 85 | Safe operating range | Normal operations |
| > 85 | Very humid | Heat stress amplified, equipment corrosion |
| > 90 | Extreme | Water condensation on sensors, slip hazards |

**FIELD-MIND anomaly thresholds**:
- Humidity anomaly if `humidity > 85%` OR `humidity < 15%`

---

## Ventilation and Environmental Control

### Primary Ventilation

- **Minimum air quantity**: Based on the greatest of:
  - Diesel equipment ratings (0.06 m³/min per kW)
  - Number of personnel (4 m³/min per person minimum)
  - Regulatory minimum for tunnel cross-section
- **Air velocity**: 0.5 m/s minimum at active working faces

### Refrigeration Requirements

When virgin rock temperature exceeds 37°C:
- Chilled water or bulk air cooling plants required
- Target cooling to < 27.5°C WBGT at workface
- Secondary cooling (spot coolers) for faces with high heat loads

### Humidity Control

- Fogging systems used to suppress dust (increases humidity)
- Trade-off between dust control and heat stress must be managed
- Target: 60 – 75% RH in active workings

---

## CO2 Monitoring and Occupancy Detection

### CO2 Safety Thresholds

| CO2 Level (ppm) | Condition | Action |
|-----------------|-----------|--------|
| 400 – 800 | Outdoor / normal indoor | No action |
| 800 – 1,000 | Slight stuffiness | Increase ventilation |
| 1,000 – 2,500 | Deteriorating air quality | Mandatory ventilation boost |
| 2,500 – 5,000 | Headaches, poor concentration | Evacuate if possible |
| > 5,000 | OSHA PEL exceeded | Immediate evacuation |
| > 40,000 | IDLH | Life-threatening |

### FIELD-MIND Occupancy Classification

The `random_forest.joblib` model classifies tunnel occupancy (0=unoccupied, 1=occupied) using features including CO2, temperature, humidity, and light level. This enables:
- Prioritising ventilation in occupied areas
- Alerting when unexpected occupancy is detected in sealed sections
- Correlating gas hazard events with personnel presence

---

## FIELD-MIND Environmental Sensor System

### Models

| Model | Type | Input Features | Target |
|-------|------|----------------|--------|
| `isolation_forest_iot.joblib` | IsolationForest | 9 env features | anomaly_flag (-1 = anomaly) |
| `random_forest.joblib` | RF Classifier | 23 features | occupancy (0/1) |

### 9 Isolation Forest Features

1. temp
2. humidity
3. temp_hum_product (temp × humidity)
4. temp_hum_ratio (temp / humidity)
5. humidex
6. temp_roll_mean_5 (5-reading rolling mean of temperature)
7. temp_roll_std_5 (5-reading rolling std of temperature)
8. humidity_roll_mean_5
9. humidity_roll_std_5

### EnvSensorAgent

The `EnvSensorAgent` runs continuous Isolation Forest inference each tick. Because Isolation Forest is unsupervised, its self-learning refit ignores labels — it refits on raw feature distributions (200 samples). This adapts the model to seasonal or operational shifts in the mine's baseline thermal environment.

### Datasets

- Training: `iot_telemetry_clean.csv` (IoT Telemetry dataset — Kaggle, 3.9M rows, sampled to 100k)
- Occupancy: `uci_train_clean.csv` (UCI Occupancy Detection dataset)
