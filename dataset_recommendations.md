# IEEE DataPort Dataset Recommendations for FIELD-MIND

## Why You're Getting `REVIEW_MODEL_DISAGREEMENT`

Your ML models were trained on **domain-specific datasets** where the "hazard" labels were defined by **the dataset authors' criteria** — not by OSHA/NIOSH/IS-6922 thresholds. For example:

| Model | Training Data | Model's "Hazard" Boundary | Safety Evaluator Threshold | Mismatch? |
|---|---|---|---|---|
| `co_nox_hazard` | CO, NOx, Benzene data from [CO,NOX,NO2,C6H6.xlsx](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/data/CO,NOX,NO2,C6H6.xlsx) | Fires at CO ~5 ppm | CO > 25 ppm (OSHA) | ✅ Yes |
| `methane_hazard` | MQ4 spectral features from [Gas_Sensors.xlsx](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/data/Gas_Sensors.xlsx) | Doesn't fire even at 10,000 ppm | CH4 > 1,000 ppm (OSHA) | ✅ Yes |
| `vibration_hazard` | Blast data from [Mt. Erzberg](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/vibration/data/README.txt) | Fires at PPV ~6 mm/s | PPV > 1 mm/s (FIELD-MIND) | ⚠️ Borderline |
| `anomaly_detected` | IoT temp/humidity data | Flags synthetic values as anomalous | Temp > 28°C, Humidity > 85% | ✅ Yes |

**The fix:** You need datasets where hazard labels are **defined at the same thresholds** your safety evaluator uses.

---

## Recommended Datasets from IEEE DataPort

### 1. Gas Sensors — CO/NOx/Methane Hazard Detection

> [!IMPORTANT]
> This is the **highest priority** — `co_nox_hazard` is the main cause of `REVIEW_MODEL_DISAGREEMENT`.

#### a) Electronic Nose / Gas Sensor Array Datasets
- **Search on IEEE DataPort:** `"electronic nose gas sensor array"`
- Contains MQ-series sensor readings (MQ2, MQ3, MQ4, MQ5, MQ6, MQ7, MQ8, MQ135) with gas concentration labels
- **Why it helps:** Multimodal gas readings with concentration values you can re-label using OSHA thresholds

#### b) Gas Sensor Array under Dynamic Gas Mixtures
- **Platform:** IEEE DataPort / UCI ML Repository
- **Search:** `"gas sensor array dynamic mixtures"`
- Features: Time-series from 16 chemical sensors exposed to CH4, CO, and ethylene at varying concentrations
- **Why it helps:** Has actual ppm concentration values — you can set your own binary hazard labels at OSHA limits (CO > 25 ppm = 1, CO ≤ 25 ppm = 0)

#### c) MultimodalGasData (Mendeley / IEEE)
- **Search:** `"MultimodalGasData MQ series"`
- Contains numerical values from MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135 sensors
- **Why it helps:** Covers the exact sensor types your models use

---

### 2. Environmental — Temperature/Humidity Anomaly Detection

Your `env_iforest` (Isolation Forest) and `env_occupancy` (Random Forest) were trained on generic IoT/smart-building data.

#### a) Semantic-Aware Sensor Anomaly Detection Dataset
- **Platform:** IEEE DataPort
- **Search:** `"semantic-aware sensor anomaly ESP32"`
- Contains: Temperature, humidity, optical sensor data with **ground-truth fault labels**
- **Why it helps:** Has labeled normal vs anomalous conditions — retrain your Isolation Forest with mining-range temperature/humidity distributions

#### b) Edge-IIoTset
- **Platform:** IEEE DataPort (Reference: `ieee-dataport.org/8939`)
- **Search:** `"Edge-IIoTset IoT"`
- Contains: Temperature, humidity sensors with multiple IoT scenario data
- **Why it helps:** Industrial IoT context closer to mining than smart-home occupancy

#### c) Occupancy Detection Dataset
- **Platform:** UCI ML Repository (your current [data source](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/temperature_humidity/data/2%20-%20UCI))
- **Note:** Your existing occupancy model comes from this. For better alignment, re-label using mining-specific ranges (occupied mine zone vs unoccupied)

---

### 3. Vibration — Blast PPV Prediction

Your current data comes from the **Mt. Erzberg iron ore mine** (Austria) — this is already one of the best publicly available blast vibration datasets.

#### a) Blast Vibration Prediction Datasets
- **Search on IEEE DataPort:** `"blast vibration PPV tunnel"`
- Also search: `"ground vibration blasting prediction dataset"`
- **Alternative:** MDPI open-access papers often include supplementary blast vibration CSV data (search Google Scholar: `"blast vibration dataset" site:mdpi.com`)

#### b) Your Current Erzberg Dataset
- **Already excellent** for PPV regression
- **Issue:** The `vibration_hazard` **classifier** was trained with labels that don't match your 1 mm/s early-warning threshold
- **Fix:** Re-derive the binary label from your existing regressor output: `vibration_hazard = 1 if predicted_ppv > 1 else 0`

---

### 4. Ultrasonic — Robot Navigation

Your current data is from the **UCI Wall-Following Robot (SCITOS G5)** — 24 ultrasonic sensors.

#### a) Wall-Following Robot Navigation Dataset
- **Platform:** UCI ML Repository (you already have this)
- **Search on IEEE DataPort:** `"wall following robot ultrasonic navigation"`
- **Note:** This is the de facto standard dataset for this task. It's already well-suited.

#### b) Dynamic Indoor Robot Navigation Dataset
- **Platform:** Kaggle
- **Search:** `"dynamic indoor robot navigation ultrasonic LIDAR"`
- Contains: Ultrasonic proximity + LIDAR + collision flags
- **Why it helps:** Includes explicit collision labels useful for your `sharp_turn_required` classifier

---

### 5. Smoke / Fire Detection

#### a) Smoke Detection IoT Dataset
- **Search on IEEE DataPort:** `"smoke fire detection sensor PM2.5"`
- Also search Kaggle: `"smoke detection dataset IoT"` — there's a well-known dataset with Temperature, Humidity, PM2.5, CO, and Fire/No-Fire labels
- **Why it helps:** Re-label using your PM2.5 threshold to match FIELD-MIND smoke evaluation

---

## Recommended Re-Training Strategy

> [!TIP]
> You don't necessarily need **new** datasets. The fastest fix is to **re-label your existing data** using OSHA/NIOSH thresholds.

### Quick Fix (No New Data Needed)

```python
# Re-label your existing CO/NOx dataset with OSHA thresholds
df['co_nox_hazard'] = ((df['MQ7_CO_ppm'] > 25) | (df['MQ135_NOx_ppm'] > 3)).astype(int)

# Re-label methane using OSHA PEL
df['methane_hazard'] = (df['MQ4_CH4_ppm'] > 1000).astype(int)

# Re-derive vibration hazard from PPV
df['vibration_hazard'] = (df['predicted_ppv'] > 1.0).astype(int)
```

Then retrain the same ML algorithms (Random Forest, Gradient Boosting, etc.) with these corrected labels.

### Full Fix (New Data from IEEE DataPort)

| Priority | Model | Search Query on IEEE DataPort |
|---|---|---|
| 🔴 **P0** | `co_nox_hazard` | `"gas sensor CO NOx hazard classification"` |
| 🔴 **P0** | `methane_hazard` | `"MQ4 methane gas sensor array"` |
| 🟡 **P1** | `env_iforest` | `"IoT sensor anomaly detection temperature humidity"` |
| 🟡 **P1** | `vibration_hazard` | `"blast vibration PPV classification"` |
| 🟢 **P2** | `smoke_alarm` | `"smoke fire detection PM2.5 sensor"` |
| 🟢 **P2** | `ultrasonic` | Already well-suited (UCI Wall-Following) |
