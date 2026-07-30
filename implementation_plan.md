# Add Input Validation, Sensor Fault Detection, and MQ-4 Methane Classifier to GasSensorAgent

This plan details the implementation steps to resolve Gap 6 (No input validation), Gap 7 (No sensor fault detection), and Gap 12 (MQ4 classifier not loaded) to ensure the FIELD-MIND gas sensor pipeline is completely foolproof and resilient against hardware or telemetry failures.

## User Review Required

> [!NOTE]
> **Are new datasets required?**
> **No.** The existing datasets in the workspace are completely sufficient:
> 1. **MQ-4 Methane Classifier**: The training dataset `Methane_MQ4/Dataset/` (batches 1-10) already exists in `gas_sensors/data/` and will be used by `train_methane.py` to train the ensembled multiclass model `mq4_gas_classifier.joblib`.
> 2. **Input Validation and Fault Detection**: These are physics-based and datasheet-based checks, so they run dynamically in memory and do not require training datasets.

---

## Proposed Changes

### 1. Reusable Validation Component

#### [NEW] [input_validator.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/sensor_agents/input_validator.py)
- Create a reusable health checking module containing:
  - Range validation and clamping based on MQ sensor datasheets (clamping negative/NaN readings to safe minimums).
  - Stuck sensor detection (checking if the value remains identical over the last 10 ticks).
  - Dead sensor detection (detecting analog readings that drop below minimum active datasheet thresholds).
  - Spike anomaly detection (flagging sudden fluctuations exceeding $5\sigma$ of the rolling history).
  - `SensorHealthReport` dataclass encapsulating the status.

---

### 2. Gas Agent Component

#### [MODIFY] [gas_agent.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/sensor_agents/gas_agent.py)
- **Model Loading**: Add `"mq4_classifier": "mq4_gas_classifier.joblib"` to `_load_models()`.
- **Perception Validation**:
  - Initialize histories (`self._sensor_histories`) and fault status (`self._sensor_faults`) for each feature.
  - In `perceive()`, validate each incoming sensor value against the validator.
  - Maintain a rolling memory buffer of the last 50 raw sensor values per channel to calculate spikes and stuck conditions.
  - Extract the 128-dimensional MQ-4 features array (`feature_1` ... `feature_128`) if they are present in the incoming raw sensor dictionary.
- **Inference**:
  - Run the `mq4_classifier` if the 128-dimensional features are present, adding `mq4_class` to the result dictionary.
- **Confidence Scoring**:
  - Adjust confidence weights to include `sensor_fault` (weight 0.09) to raise system alert confidence if a sensor starts failing.
- **Alert Descriptions**:
  - Update `_build_reason()` to append detailed sensor fault descriptions if any validation check fails.

---

### 3. Training Suite Component

#### [RUN] [train_methane.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/train_methane.py)
- Run the existing training script to fit the ensembled Voting Classifier (SVM + MLP) on the 10 batches of the `Methane_MQ4` dataset and output `mq4_gas_classifier.joblib` to the `gas_sensors/models/` directory.

---

## Verification Plan

### Automated Tests
- Run `python train_methane.py` to ensure `mq4_gas_classifier.joblib` is trained, saved, and registered.
- Run a new test suite [test_validation_faults.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/gas_sensors/test_validation_faults.py) to simulate:
  1. Negative/NaN inputs $\rightarrow$ verify clamping and range flags.
  2. Stuck sensor (10 identical consecutive values) $\rightarrow$ verify stuck flag.
  3. Dead sensor (0.0 ppm for analog channels) $\rightarrow$ verify dead flag.
  4. Out-of-bounds inputs $\rightarrow$ verify range flag.
  5. Correct multi-class prediction of `mq4_classifier` on 128-dimensional mock inputs.
