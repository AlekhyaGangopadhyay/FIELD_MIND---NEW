# FIELD_MIND — Brief Sensor ↔ Model ↔ Prediction Map

This document lists the sensor models currently deployed in the project, detailing exactly which sensor maps to which model, and what that model is detecting.

---

### 1. Gas & Dust Sensors

| Sensor | Model | What it is detecting |
| :--- | :--- | :--- |
| **MQ-2** | `multi_gas_detector` | Presence of LPG, Methane ($CH_4$), Carbon Monoxide (CO), and Smoke (binary flag) |
| **MQ-2** | `severity_h2` | Hydrogen ($H_2$) concentration severity levels (`0` Safe, `1` Warning, `2` Critical) |
| **MQ-3** | `mine_baseline_iforest` | Baseline hardware telemetry anomalies (deviations from clean-air noise floor) |
| **MQ-4** | `multi_gas_detector` | Methane presence (binary flag) |
| **MQ-4** | `severity_ch4` | Methane exposure severity levels (`0` Safe, `1` Warning, `2` Critical) |
| **MQ-4** | `mq4_gas_classifier` | Methane transient state classification (sensor warmup vs. active gas profiles) |
| **MQ-7** | `multi_gas_detector` | Carbon Monoxide presence (binary flag) |
| **MQ-7** | `severity_co` | Carbon Monoxide exposure severity levels (`0` Safe, `1` Warning, `2` Critical) |
| **MQ-135** | `multi_gas_detector` | Ammonia ($NH_3$), NOx, and Carbon Dioxide ($CO_2$) presence (binary flags) |
| **MQ-135** | `nh3_hazard` | High-concentration toxic Ammonia hazard alerts (binary flag) |
| **MQ-135** | `severity_co2` | Carbon Dioxide exposure severity levels (`0` Safe, `1` Warning, `2` Critical) |
| **MQ-136** | `multi_gas_detector` | Hydrogen Sulphide ($H_2S$) presence (binary flag) |
| **MQ-136** | `severity_h2s` | Hydrogen Sulphide exposure severity levels (`0` Safe, `1` Warning, `2` Critical) |
| **MG811** | `multi_gas_detector` | Carbon Dioxide ($CO_2$) presence (binary flag) |
| **MG811** | `co2_hazard` | Carbon Dioxide over-TLV exposure hazard warnings (binary flag) |
| **MG811** | `severity_co2` | Carbon Dioxide exposure severity levels (`0` Safe, `1` Warning, `2` Critical) |
| **PM2.5** | `smoke_env_hazard` | Toxic dust particulate levels exceeding NIOSH limits (binary flag) |

---

### 2. Environmental Sensors

| Sensor | Model | What it is detecting |
| :--- | :--- | :--- |
| **DHT22** | `env_iforest` | Ambient microclimate temperature/humidity pattern anomalies (binary flag) |
| **DHT22** | `env_occupancy` | Area occupancy status (binary flag: occupied vs. unoccupied) |
| **DHT22** | `smoke_env_hazard` | Combined heat and humidity combustion hazard conditions (binary flag) |

---

### 3. Vibration & Structural Sensors

| Sensor | Model | What it is detecting |
| :--- | :--- | :--- |
| **SW-420** | `SW420VibrationMonitor` | Ground shock severity levels (`0` Safe, `1` Warning, `2` Critical) and instant shock alerts |
| **Ultrasonic sensor (wall displacement)** | `UltrasonicDisplacementModel` | Imminent geomechanical wall/roof collapse risk (computed using displacement velocity and acceleration rates) or local blockages |
