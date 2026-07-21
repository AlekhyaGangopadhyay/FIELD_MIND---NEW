# Unified Demo — FIELD-MIND End-to-End Showcase

This directory contains two runnable demos that exercise every completed layer of FIELD-MIND in a single pipeline:

1. **Interactive Safety Hub** (`interactive_safety_hub.py`) — manual, single-node, single-timestamp CLI walkthrough.
2. **Streaming Safety Simulation** (`streaming_safety_simulation.py`) — automated, multi-node, multi-timestamp stress-test with randomized data and hazard injection.

Both demos run the real Tier-1 ML models, query the FAISS RAG vector database, consult the Expedition Knowledge Graph (EKG), and produce a full conversational safety response.

---

## Files

```
unified_demo/
├── interactive_safety_hub.py         # Manual CLI: user enters sensor values, gets safety analysis
├── streaming_safety_simulation.py    # Automated: 3 nodes × 10 timestamps, randomized hazard drift
├── data/
│   ├── simulation_history.json       # Default output from streaming simulation
│   ├── simulation_history_test.json  # Test-run artefact
│   └── ...                           # Additional run artefacts
└── README.md                         # This file
```

---

## 1. Streaming Safety Simulation (Multi-Node, Multi-Timestamp)

### What it does

| Feature | Detail |
|---------|--------|
| **Nodes** | 3 sensor nodes in a single tunnel (`NODE_1`, `NODE_2`, `NODE_3`) |
| **Timestamps** | 10 samples per node, 2-second intervals (20 seconds total) |
| **Data generation** | Fully randomized (deterministic seed) — no manual input required |
| **Hazard injection** | Node-specific gradual drift that crosses warning/critical thresholds |
| **ML inference** | Runs all Tier-1 models (SVM, MLP, RF, Isolation Forest, GBR) per sample |
| **Protocol checks** | `SafetyProtocolEvaluator` compares readings + model outputs against OSHA/NIOSH/IS limits |
| **Trend analysis** | Builds a cross-node, multi-timestamp trend summary (rising/falling/stable) |
| **Final response** | Sends the complete trend + worst-case state to `MineSafetyChatAssistant` |
| **History** | Saves all 30 samples (3 × 10) to a JSON file for later analysis |

### Hazard drift by node

| Node | Injected hazard | What happens |
|------|-----------------|-------------|
| **NODE_1** | Gas accumulation | CH4 ramps from ~400 → ~6500 ppm; CO rises past 35 ppm |
| **NODE_2** | Thermal + blast drift | Temperature climbs past 28°C; PPV rises above 25 mm/s |
| **NODE_3** | Closing clearance | `min_distance` drops from ~2.5 m → 0.18 m (collision range) |

### How to run

```bash
# Fast mode — skip real-time pacing, stress-test models immediately
python -X utf8 unified_demo/streaming_safety_simulation.py --fast

# Real-time mode — 2-second pacing between timestamps (20 seconds total)
python -X utf8 unified_demo/streaming_safety_simulation.py

# Custom seed and output path
python -X utf8 unified_demo/streaming_safety_simulation.py --fast --seed 42 --history-out unified_demo/data/my_run.json
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--nodes` | `3` | Number of sensor nodes in the tunnel |
| `--timestamps` | `10` | Number of sampling intervals per node |
| `--interval` | `2.0` | Seconds between timestamps |
| `--tunnel-id` | `TUNNEL_SIM_01` | Shared tunnel identifier |
| `--seed` | `69` | Deterministic random seed for reproducibility |
| `--fast` | off | Skip wall-clock pacing (run all timestamps as fast as possible) |
| `--history-out` | `data/simulation_history.json` | JSON output path for the time-series history |

### Example output (abbreviated)

```
FIELD-MIND MULTI-NODE STREAMING SAFETY SIMULATION
Tunnel: TUNNEL_SIM_01 | Nodes: 3 | Samples: 10 | Interval: 2.0s
Mode: fast | Seed: 69

[Setup] Loading Tier-1 models...
[Setup] Loading safety reasoning resources...
[Setup] FAISS embedder ready in 2150 ms.

[t=00 +  0s] TUNNEL_SIM_01_NODE_1   CH4=    379 CO= 10.9 PPV= 11.78 clearance=2.44m status=REVIEW_MODEL_DISAGREEMENT
[t=00 +  0s] TUNNEL_SIM_01_NODE_2   CH4=    385 CO= 13.7 PPV= 11.78 clearance=2.39m status=REVIEW_MODEL_DISAGREEMENT
[t=00 +  0s] TUNNEL_SIM_01_NODE_3   CH4=    408 CO=  9.8 PPV= 11.78 clearance=2.51m status=REVIEW_MODEL_DISAGREEMENT
...
[t=09 + 18s] TUNNEL_SIM_01_NODE_1   CH4=   6480 CO= 36.9 PPV= 11.78 clearance=2.51m status=CRITICAL
[t=09 + 18s] TUNNEL_SIM_01_NODE_2   CH4=    409 CO= 13.4 PPV= 25.04 clearance=2.57m status=CRITICAL
[t=09 + 18s] TUNNEL_SIM_01_NODE_3   CH4=    429 CO= 11.2 PPV=  9.10 clearance=0.18m status=CRITICAL

================================================================================
MULTI-NODE TREND SUMMARY
================================================================================
TUNNEL_SIM_01_NODE_1: CH4 379->6480 ppm (rising); CO 10.9->36.9 ppm (rising); ...
TUNNEL_SIM_01_NODE_2: CH4 385->409 ppm (stable); PPV 11.78->25.04 mm/s (rising); ...
TUNNEL_SIM_01_NODE_3: clearance 2.51->0.18 m (falling); ...
Interpretation: gradual gas/PPV escalation is visible at NODE_1, NODE_2.
Interpretation: robot clearance is shrinking at NODE_3; late-window readings deserve priority.

================================================================================
FIELD-MIND MULTI-NODE RESPONSE
================================================================================
FIELD-MIND safety assessment for TUNNEL_SIM_01
Overall status: **CRITICAL**

### Findings
- **methane (CH4)**: 6479.5 ppm - CRITICAL / MODEL_MISS. ...
- **carbon monoxide (CO)**: 36.9 ppm - WARNING / ALERT. ...
- **PPV**: 11.78 mm/s - WARNING / ALERT. ...

### Multi-node trend
(full cross-node trend summary injected here)

### Recommended actions
1. Review gas model disagreement...
2. Increase ventilation and alert personnel...
3. Pause further blasting...

### Grounding
FAISS evidence consulted: vibration_limits.md (0.66), env_safety.md (0.62), gas_safety.md (0.48).
================================================================================
```

### History JSON structure

Each of the 30 records in the output JSON follows this schema:

```json
{
  "timestamp": "2026-07-21T10:05:54.312262+00:00",
  "tunnel_id": "TUNNEL_SIM_01",
  "node_id": "TUNNEL_SIM_01_NODE_1",
  "readings": {
    "MQ4_CH4_ppm": 378.953,
    "MQ7_CO_ppm": 10.903,
    "MQ2_LPG_ppm": 80.879,
    "MQ135_NOx_ppm": 2.277,
    "MQ3_Benzene_ppm": 0.448,
    "PM25_Dust_ugm3": 38.493,
    "temp": 22.071,
    "humidity": 53.445,
    "max_charge": 42.578,
    "num_holes": 18.673,
    "distance": 341.402,
    "min_distance": 2.438,
    "predicted_ppv": 11.78
  },
  "predictions": {
    "methane_hazard": 0,
    "smoke_alarm": 1,
    "lpg_hazard": 0,
    "co_nox_hazard": 1,
    "air_quality_score": 0.678,
    "anomaly_detected": 1,
    "occupancy_state": 1,
    "vibration_hazard": 1,
    "predicted_ppv": 11.78,
    "steering_decision": "Slight-Left-Turn",
    "sharp_turn_required": 0
  },
  "status": "REVIEW_MODEL_DISAGREEMENT"
}
```

---

## 2. Interactive Safety Hub (Manual, Single-Node)

### What it does

A terminal-based CLI where you manually enter sensor values for one tunnel segment. FIELD-MIND runs the Tier-1 ML models, consults RAG + EKG, and returns a conversational safety analysis.

### How to run

```bash
python -X utf8 unified_demo/interactive_safety_hub.py
```

### Demo input scenario — Critical Gas Leak

| Prompt | Input Value | Safety Implication |
|--------|-------------|---------------------|
| **Tunnel Segment ID** | `TUNNEL_A1` | Specifies location |
| **Methane level (CH4)** | `12500.0` | 🚨 CRITICAL (evacuation threshold) |
| **Carbon Monoxide (CO)** | `35.0` | Elevated |
| **LPG/CNG level** | `80.0` | Nominal |
| **Nitrogen Oxides (NOx)** | `2.1` | Nominal |
| **Benzene** | `0.5` | Nominal |
| **PM2.5 Dust load** | `35.0` | Nominal |
| **Ambient Temperature** | `29.5` | ⚠ CAUTION (heat stress threshold) |
| **Relative Humidity %** | `55.0` | Nominal |
| **Maximum charge weight** | `45.0` | Blast weight parameters |
| **Number of blast holes** | `18.0` | Blast parameters |
| **Seismic distance** | `350.0` | Distance from geophone |
| **Obstacle Distance** | `2.5` | Nominal |

### Expected output

```
[Gas ML Model Predictions]
  • Methane Hazard Classifier : 1  (🚨 Hazard Alert)
  • LPG/CNG Hazard Classifier : 0
  • Smoke/Fire Alarm          : 0
  • CO/NOx Hazard Classifier  : 0
  • Predicted Air Quality Score: 36.42

[Environmental ML Model Predictions]
  • Isolation Forest Anomaly  : 1  (🚨 Anomaly Alert)
  • Occupancy State           : 0

[Blast Vibration ML Model Predictions]
  • PPV Hazard Classifier     : 1  (🚨 Vibration Alert)
  • Predicted PPV Regressor   : 15.80 mm/s (🚨 Exceeds structural limits)

[Robot Navigation ML Model Predictions]
  • Predicted Steering command: Move-Forward
  • Collision Risk Warning    : 0
```

The assistant then responds with data analysis, regulation references, and an action plan.

---

## Architecture

Both demos share the same execution pipeline:

```
  Sensor Data (manual or randomized)
           │
           v
  ┌──────────────────────────────┐
  │  Tier-1 ML Models            │
  │  (SVM, MLP, RF, IF, GBR)    │
  │  via atr_activation/         │
  └──────────────────────────────┘
           │
           v
  ┌──────────────────────────────┐
  │  SafetyProtocolEvaluator     │
  │  Deterministic limit checks  │
  │  + model-vs-reading compare  │
  │  via faiss_rag/              │
  └──────────────────────────────┘
           │
           v
  ┌──────────────────────────────┐
  │  FAISS RAG Retriever         │
  │  OSHA/NIOSH/IS grounding     │
  │  via faiss_rag/              │
  └──────────────────────────────┘
           │
           v
  ┌──────────────────────────────┐
  │  Expedition Knowledge Graph  │
  │  Segment history & memory    │
  │  via expedition_knowledge_   │
  │      graph/                  │
  └──────────────────────────────┘
           │
           v
  ┌──────────────────────────────┐
  │  MineSafetyChatAssistant     │
  │  Conversational response     │
  │  with trend context          │
  │  via reasoning_core/         │
  └──────────────────────────────┘
           │
           v
      Operator Output
```

The streaming simulation additionally builds a **cross-node trend summary** before the final assistant call, giving the LLM visibility into rising/falling patterns across all nodes and timestamps.
