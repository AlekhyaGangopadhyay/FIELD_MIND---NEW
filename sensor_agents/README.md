# Sensor Agents — FIELD-MIND Autonomous Multi-Agent Layer

This folder houses the **Autonomous Sensor AI Agent Layer** of the FIELD-MIND system. Every sensor domain is modeled as an independent, self-learning agent running an event-driven loop and communicating over a shared message broker.

---

## Architecture Overview

All agents inherit from a unified base class implementing the **Observe-Reason-Act-Learn** loop:
1. **Observe**: Ingests new telemetry data from the sensor pipelines.
2. **Reason**: Runs the raw telemetry through pre-trained local ML models to calculate hazard levels.
3. **Act**: If a hazard score exceeds safety thresholds (confidence $\ge 0.5$ for 2+ consecutive ticks), publishes a `BusMessage` ALERT.
4. **Learn**: Saves the sample to an **Experience Replay Buffer** (size=200). When the buffer is full, it triggers background retraining of the local model and hot-swaps it atomically without interrupting operations.

---

## Agent Directory & Responsibilities

| File | Agent Class | Primary Responsibility | Underlyings Models | Dataset for Learning |
|---|---|---|---|---|
| [`gas_agent.py`](file:///c:/Users/Student/Desktop/FIELD_MIND - NEW/sensor_agents/gas_agent.py) | **`GasSensorAgent`** | Tracks gas toxic thresholds and fires alarms for methane, LPG, and CO. | Methane voting classifier, LPG/CNG SVM, CO/NOx, Smoke/Fire | `FIELDMIND_physics_dataset.csv` |
| [`env_agent.py`](file:///c:/Users/Student/Desktop/FIELD_MIND - NEW/sensor_agents/env_agent.py) | **`EnvSensorAgent`** | Monitors microclimate comfort, temp/humidity anomalies, and cabin occupancy. | Isolation Forest, Occupancy Random Forest | `iot_telemetry_clean.csv` |
| [`vibration_agent.py`](file:///c:/Users/Student/Desktop/FIELD_MIND - NEW/sensor_agents/vibration_agent.py) | **`VibrationSensorAgent`** | Predicts blast Peak Particle Velocity (PPV) and logs ground hazards. | Random Forest Classifier, Gradient Boosting Regressor | `vibration_features.csv` |
| [`ultrasonic_agent.py`](file:///c:/Users/Student/Desktop/FIELD_MIND - NEW/sensor_agents/ultrasonic_agent.py) | **`UltrasonicSensorAgent`** | Governs robotic platform collision risk and steering commands. | 24-Sensor Navigation Random Forest | `sensor_readings_24.csv` |
| [`ekg_agent.py`](file:///c:/Users/Student/Desktop/FIELD_MIND - NEW/sensor_agents/ekg_agent.py) | **`EKGAgent`** | Subscribes to the `AgentBus` and writes all triggered alerts directly to EKG memory. | Expedition Knowledge Graph (NetworkX) | N/A (Reactive Logger) |
| [`mine_orchestrator_agent.py`](file:///c:/Users/Student/Desktop/FIELD_MIND - NEW/sensor_agents/mine_orchestrator_agent.py) | **`MineOrchestratorAgent`** | Aggregates and weights alert levels from all agents to control emergency triggers. | Custom Weighted Hazard Score Fusion | N/A (Global Coordinator) |

---

## Supporting Infrastructure

- [`agent_base.py`](file:///c:/Users/Student/Desktop/FIELD_MIND - NEW/sensor_agents/agent_base.py): Core interface for base agent loop, thread pooling, and experience replay buffer management.
- [`agent_bus.py`](file:///c:/Users/Student/Desktop/FIELD_MIND - NEW/sensor_agents/agent_bus.py): Local event-driven publish/subscribe message broker coordinating asynchronous inter-agent signaling.
- [`demo_agents.py`](file:///c:/Users/Student/Desktop/FIELD_MIND - NEW/sensor_agents/demo_agents.py): Streaming simulation script showing all agents observing data, raising alerts, logging to EKG, and retraining their ML models live.
