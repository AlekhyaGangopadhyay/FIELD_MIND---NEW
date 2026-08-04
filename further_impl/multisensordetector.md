# Implementation Plan — Train Multi-Gas Detector & Implement Standalone Agent (Option B)

This plan outlines how we will train the `multi_gas_detector.joblib` model using a PyTorch multi-label deep learning architecture on the compiled dataset and create a new standalone agent (`MultiGasDetectorAgent`) to run in parallel with the existing system.

---

## User Review Required

> [!IMPORTANT]
> **Standalone Agent Pattern (Option B)**:
> We will create a new standalone agent `MultiGasDetectorAgent` ([multi_gas_agent.py](file:///e:/FIELD_MIND/FIELD_MIND---NEW/sensor_agents/multi_gas_agent.py)) that:
> 1. Loads the newly trained `multi_gas_detector.joblib` on startup.
> 2. Subscribes to raw sensor readings published on the `AgentBus`.
> 3. Runs inference to predict active gases and broadcasts presence warnings to the bus.

---

## Proposed Changes

### Component 1 — Model Training

#### [NEW] [train_multi_gas_detector.py](file:///C:/Users/iamal/.gemini/antigravity-ide/brain/6e8885f3-a741-423e-b11d-251b8e6bebe4/scratch/train_multi_gas_detector.py) (Scratch Script)
Create a training script to:
1. Load `multi_gas_detector_real_v2.csv` (50,000 rows).
2. Split features and targets:
   - **Features**: `['CH4_ppm', 'CO_ppm', 'CO2_ppm', 'H2_ppm', 'H2S_ppm', 'NH3_ppm', 'LPG_ppm', 'CNG_ppm']`
   - **Targets**: `['target_Methane', 'target_CO', 'target_CO2', 'target_H2', 'target_H2S', 'target_NH3', 'target_LPG', 'target_CNG']`
3. Train `PyTorchHazardClassifier` (8 inputs, 8 outputs, Binary Cross-Entropy loss) using the `DeepHazardNet` LayerNorm+SiLU architecture.
4. Serialize and save the trained model to `gas_sensors/models/multi_gas_detector.joblib`.
5. Update `gas_sensors/models/model_registry.json` with the new schema, feature keys, and performance metrics.

---

### Component 2 — Standalone Agent & Wrapper Integration

#### [NEW] [multi_gas_agent.py](file:///e:/FIELD_MIND/FIELD_MIND---NEW/sensor_agents/multi_gas_agent.py)
Create a new agent class inheriting from `SensorAgentBase` that:
* Loads the trained `multi_gas_detector.joblib` on instantiation.
* Normalized inputs from raw telemetry into the 8 input features.
* Runs inference to classify which gases are active.
* Emits a `MULTIGAS_ALERT` on the `AgentBus` containing the list of all identified active gases.

#### [MODIFY] [detector_wrappers.py](file:///e:/FIELD_MIND/FIELD_MIND---NEW/atr_activation/detector_wrappers.py)
* Add a `gas_multi` evaluation block inside the `evaluate_gas()` method to feed the 8 sensor inputs, run inference, and return the predicted active gas dictionary.

---

## Verification Plan

### Automated Tests
1. Run the training scratch script:
   ```powershell
   python C:\Users\iamal\.gemini\antigravity-ide\brain\6e8885f3-a741-423e-b11d-251b8e6bebe4\scratch\train_multi_gas_detector.py
   ```
2. Verify the model file compiles and is saved at `gas_sensors/models/multi_gas_detector.joblib`.
3. Run the end-to-end multi-agent streaming simulation to verify the new agent runs in parallel, consumes sensor streams, and outputs active gas predictions without crashing:
   ```powershell
   python e:\FIELD_MIND\FIELD_MIND---NEW\unified_demo\streaming_safety_simulation.py
   ```
