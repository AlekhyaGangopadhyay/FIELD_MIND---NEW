# FIELD-MIND System Overview

## What is FIELD-MIND?

**FIELD-MIND** (Field Intelligence for Environmental and Logistical Detection — Multimodal INtelligence Device) is an offline multimodal AI platform designed for underground mining safety and operations. It runs entirely on-device (NVIDIA Jetson Nano, 4 GB RAM) without internet connectivity, making it suitable for remote and deep underground environments.

---

## System Architecture

FIELD-MIND operates as a six-layer pipeline:

```
Layer 1: Data Ingestion & Alignment
  └── SciSense Protocol — aligns 4 heterogeneous sensor streams into
      a shared 4,096-dimensional embedding space

Layer 2A: Continuous Monitoring (Tier 1 — always running)
  └── Sensor AI Agents — autonomous agents with Observe→Reason→Act→Learn
      cycles for Gas, Env, Vibration, and Ultrasonic sensors

Layer 2B: Memory (Tier 2)
  └── Expedition Knowledge Graph — persistent mine memory (NetworkX)
  └── FAISS Vector Database — offline semantic search over scientific docs

Layer 3: Scientific Reasoning Core (triggered on EMERGENCY)
  └── LangGraph agent loop with Llama-3.2-3B-Instruct (4-bit quantized)
  └── OBSERVE → EKG-RETRIEVE → RAG-RETRIEVE → HYPOTHESIZE → SUGGEST
```

---

## Sensor Domains

### Gas Sensors (MQ Series)

FIELD-MIND monitors 7 gas types using MQ-2, MQ-3, MQ-4, MQ-7, MQ-135, MQ-136, and MG811 sensor arrays:
- **Methane (CH4)**: SVM + MLP Voting Classifier, 128 features
- **LPG/CNG**: Random Forest, 2 features (MQ2_LPG_ppm, MQ4_CH4_ppm)
- **CO/NOx/Benzene**: Random Forest, 3 features
- **Smoke/Dust**: Random Forest, 3 features (PM2.5, Temp, Humidity)
- **Air Quality Score**: Gradient Boosting Regressor, 7 features

### Environmental Sensors (DHT22, BMP280)

- **Temperature**: Range -40°C to +80°C, ±0.5°C accuracy
- **Humidity**: Range 0–100% RH, ±2% accuracy
- **Pressure**: BMP280, range 300–1100 hPa
- Anomaly detection via Isolation Forest (9 rolling features)
- Occupancy classification via Random Forest (23 features)

### Vibration Sensors (Geophones)

- **SEG-Y seismogram** data from blast events
- Peak Particle Velocity (PPV) prediction: Gradient Boosting Regressor
- Hazard classification: Random Forest (PPV > 1.0 mm/s threshold)
- 17-feature model including USBM and Langefors scaled distances

### Ultrasonic Sensors (HC-SR04 Array)

- **24-sensor** configuration for 360° robot navigation
- Classification of navigation decisions: Move-Forward, Slight-Right-Turn, Slight-Left-Turn, Sharp-Right-Turn
- Collision risk detection when Sharp-Right-Turn predicted or distance < 0.3 m

---

## AI Agent System

Each sensor domain runs as an autonomous AI agent:

### GasSensorAgent
- Wraps all 6 gas ML models
- Self-learns from `FIELDMIND_physics_dataset.csv`
- Ground truth: `Hazard_Alert` column
- Alert: confidence ≥ 0.5 for 2+ consecutive ticks

### EnvSensorAgent
- IsolationForest + RF Occupancy
- Self-learns from `iot_telemetry_clean.csv`
- Unsupervised refit (IsolationForest — no labels needed)
- Alert: anomaly detected in temp/humidity rolling window

### VibrationSensorAgent
- RF Classifier + GB Regressor for PPV
- Self-learns from `vibration_features.csv`
- Ground truth: `vibration_hazard` (PPV > 1.0 mm/s)
- Alert: vibration_hazard = 1 or PPV > threshold

### UltrasonicSensorAgent
- 24-sensor RF classifier
- Self-learns from `sensor_readings_24.csv`
- Ground truth: Class == 'Sharp-Right-Turn' → collision_risk = 1
- Alert: sharp turn required or min_distance < 0.3 m

### MineOrchestratorAgent
- Fuses all 4 agent alerts with weighted scoring:
  - Gas: 0.35, Vibration: 0.30, Env: 0.20, Ultrasonic: 0.15
- Global states: IDLE → ACTIVE_REASONING → EMERGENCY
- EMERGENCY triggered at global_score ≥ 0.60

### EKGAgent
- Subscribes to all ALERT messages on AgentBus
- Writes confirmed hazard events to Expedition Knowledge Graph
- Provides historical context to other agents via QUERY/RESPONSE

---

## Expedition Knowledge Graph (EKG)

Persistent property graph with 8 node types:
- TunnelSegment, SensorNode, BlastEvent, VibrationEvent
- GasAnomaly, EnvironmentalReading, NavigationEvent, Equipment

Key relationships:
- OCCURRED_IN → links events to tunnel segments
- CAUSED_BY → blast → vibration causal chain
- CORRELATED_WITH → gas anomaly temporal correlation
- NAVIGATION_IN → robot navigation in tunnel segment

After a full demo run (300 ticks):
- 1,389+ nodes, 1,444+ edges
- 260 NavigationEvent nodes from UltrasonicSensorAgent

---

## FAISS RAG (Retrieval-Augmented Generation)

Offline semantic search over safety literature:
- Index type: FAISS IndexFlatIP (cosine similarity)
- Embedding model: all-MiniLM-L6-v2 (384-dim, 22 MB, CPU-only)
- Knowledge base: gas safety, vibration standards, env safety, navigation protocols

Query example:
```python
retriever = RAGRetriever(index_path, metadata_path, embedder)
results = retriever.retrieve("methane concentration safe limit underground", top_k=5)
context = retriever.format_context(results)
# → Used as grounding context for Llama-3.2-3B reasoning
```

---

## Operational Modes

| State | Description | Power Mode |
|-------|-------------|------------|
| IDLE | All 4 sensor agents run lightweight inference | Low power |
| ACTIVE_REASONING | SciSense embeddings activated | Medium power |
| EMERGENCY | Full LLM reasoning + RAG retrieval | Full power |

---

## Running the System

```bash
# Full AI agent system demo (300 ticks, 5 hazard episodes)
py -X utf8 sensor_agents/demo_agents.py

# Build FAISS RAG index from knowledge base
py -X utf8 faiss_rag/demo_rag.py

# Expedition Knowledge Graph demo
py -X utf8 expedition_knowledge_graph/demo_ekg.py

# SciSense protocol alignment demo
py scisense_protocol/demo_alignment.py

# ATR activation demo
py atr_activation/demo_atr.py
```
