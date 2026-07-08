# Anomaly-Triggered Reasoning (ATR) - Tier 2 Activation

The **Anomaly-Triggered Reasoning (ATR)** module represents Layer 2A of the FIELD-MIND edge intelligence architecture. It is a power-aware scheduling layer that coordinates low-power background monitoring and high-power deep reasoning on resource-constrained hardware (e.g., Jetson Nano).

---

## ⚙️ How it Works

```
           [ STREAMING INPUTS ]
                    │
                    ▼
     ┌──────────────────────────────┐
     │  Tier 1 Continuous Monitor   │ <--- (Runs 100% of the time, low power)
     │  (Lightweight Classifiers)   │
     └──────────────┬───────────────┘
                    │
           [ Anomaly Triggered? ]
              /          \
            No           Yes
            /              \
           ▼                ▼
     ┌───────────┐    ┌──────────────────────────────┐
     │ Keep IDLE │    │ ATR Orchestrator State Change │
     │  State    │    │ (IDLE -> ACTIVE_REASONING)   │
     └───────────┘    └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │  Boot Reasoning Engine (3B)  │
                      │  Align & Encode SciSense (1s)│
                      └──────────────────────────────┘
```

### 1. Tier 1 Monitoring (Background)
* Continuous, low-power monitoring using lightweight `scikit-learn` algorithms (Random Forests, Gradient Boosting, Isolation Forests).
* These models run directly on the raw, unaligned sensor telemetry.

### 2. Tier 2 Active Reasoning (Triggered)
* If any classifier detects a hazard, the orchestrator triggers state transition.
* It allocates memory and spins up the **SciSense Protocol (Layer 1)** temporal aligner and projection layers to map recent sensor history into the shared 4,096-D embedding space.
* The aligned embeddings are formatted for the **LangGraph Reasoning Agent (Layer 3)** to diagnose and recommend corrective instructions.

---

## 📦 Loaded Pre-trained Models

This module integrates all previously trained models across the workspace:

* **Gas safety**:
  * `mq4_gas_classifier.joblib`: Methane presence classifier (128 features).
  * `smoke_fire_alarm_model.joblib`: Smoke and fire alarm predictor (36 features).
  * `gas_hazard_lpg_cng.joblib`: LPG/CNG hazard detector (2 features).
  * `gas_hazard_co_nox_c6h6.joblib`: Combustion gas hazard detector (3 features).
  * `gas_hazard_smoke_env.joblib`: Smoke and environment classifier (3 features).
  * `air_quality_regressor.joblib`: Continuous air quality score regressor (7 features).
* **Environmental**:
  * `isolation_forest_iot.joblib`: Microclimate anomaly detector (9 features).
  * `random_forest.joblib`: Office/Tunnel occupancy classifier (23 features).
* **Seismic Vibration**:
  * `best_random_forest_classifier.joblib`: Threshold blast vibration hazard classifier (14 features).
  * `best_gradient_boosting_regressor.joblib`: Continuous Peak Particle Velocity (PPV) regressor (17 features).
* **Robot Navigation**:
  * `best_ultrasonic_24.joblib`: 24-sensor steering command classifier.
  * `best_ultrasonic_4.joblib`: 4-sensor steering command classifier.
  * `best_ultrasonic_2.joblib`: 2-sensor steering command classifier.

---

## ⚡ Memory State Transitions

* **`IDLE`**: Encoders and Large Language Model (LLM) weights are suspended or unloaded from RAM. The device runs at minimal power consumption.
* **`ACTIVE_REASONING`**: The orchestrator triggers when a hazard occurs, dynamically loads the 3B quantized LLM reasoning weights, projects aligned temporal windows using PyTorch, and performs contextual diagnosis.

---

## 🚀 How to Run the Demo

Run the end-to-end streaming simulation and state machine:

```bash
python atr_activation/demo_atr.py
```
