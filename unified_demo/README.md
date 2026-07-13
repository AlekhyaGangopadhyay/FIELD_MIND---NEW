# Interactive Safety Hub — FIELD-MIND Unified Demo

The **Interactive Safety Hub** is a terminal-based interface designed to showcase all the completed layers of FIELD-MIND working together in a unified loop:
1. **Multimodal Ingestion**: Collects user-provided parameters for Gas, Environmental, Vibration, and Navigation sensors.
2. **Pre-trained ML Inference**: Runs inputs through the actual pre-trained machine learning models (SVM, MLP, RF, Isolation Forest, Gradient Boosting Regressor) via `Tier1Monitor`.
3. **Conversational RAG Grounding**: Searches the local FAISS vector database for relevant safety protocols and segment histories from the Expedition Knowledge Graph (EKG).
4. **Safety Assistant Response**: Formulates a detailed safety analysis and step-by-step action plan to present to the user.

---

## Files

```
unified_demo/
├── interactive_safety_hub.py   # CLI entry point to ingest data, run actual ML models, and query safety regulations
└── README.md                   # Setup guide and demo input scenarios
```

---

## How to Run

Ensure you are in the project root directory, then execute:
```bash
py -X utf8 unified_demo/interactive_safety_hub.py
```

---

## 💡 Demo Input & Walkthrough Scenario

Here is a step-by-step test scenario representing a **Critical Gas Leak, Temperature Rise, and Ground Vibration** incident.

### Step 1: Input Telemetry Parameters
When prompted, copy and paste the following parameters into the terminal:

| Prompt | Input Value | Safety Implication |
|--------|-------------|--------------------|
| **Tunnel Segment ID** | `TUNNEL_A1` | Specifies location |
| **Methane level (CH4)** | `12500.0` | **🚨 CRITICAL** (evacuation threshold) |
| **Carbon Monoxide (CO)** | `35.0` | Elevated |
| **LPG/CNG level** | `80.0` | Nominal |
| **Nitrogen Oxides (NOx)** | `2.1` | Nominal |
| **Benzene** | `0.5` | Nominal |
| **PM2.5 Dust load** | `35.0` | Nominal |
| **Ambient Temperature** | `29.5` | **⚠ CAUTION** (heat stress threshold) |
| **Relative Humidity %** | `55.0` | Nominal |
| **Maximum charge weight** | `45.0` | Blast weight parameters |
| **Number of blast holes** | `18.0` | Blast parameters |
| **Seismic distance** | `350.0` | Distance from geophone |
| **Obstacle Distance** | `2.5` | Nominal |

---

### Step 2: Running ML Models Output
The CLI runs the inputs through the actual project models, displaying the predictions:

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

---

### Step 3: Assistant Conversational Reasoning
Accept the default query to generate the action plan:
```
Enter your question for FIELD-MIND:
[Press ENTER to use default: "Explain the hazard cause of the alerts at TUNNEL_A1 and provide an emergency evacuation or response action plan."]
```

---

### Step 4: Final Output Response
The assistant combines EKG memory and FAISS safety measures to output the conversation block:

```
================================================================================
🤖 FIELD-MIND RESPONSE
================================================================================
Hello. This is **FIELD-MIND**, your on-site safety assistant. Let me analyze the current safety parameters for **TUNNEL_A1** and answer your request.

### 📊 Active Data Inputs Analysis
• **Methane (CH4)**: 12500.0 ppm [🚨 CRITICAL].
• **Carbon Monoxide (CO)**: 35.0 ppm [NOMINAL].
• **Ambient Temperature**: 29.5°C.
• **Memory Logs**: EKG context for segment TUNNEL_A1: active hazards count=1, gas anomalies=1, vibration events=0.

### 🛡️ Regulation Safety Measures
• **Gas Safety (OSHA/NIOSH)**: Methane concentrations exceeding 1.0% (10,000 ppm) require immediate evacuation. CO must remain below 25 ppm for safe post-blast entry.
• **Environmental Comfort (OSHA)**: Ambient work temperatures above 28°C require worker hydration cycles (15 minutes rest per hour) and secondary air conditioning controls.

### 📢 Recommended Safety Actions
1. **EVACUATE** segment immediately. Power down all non-intrinsically safe electrical grids.
2. Override and speed up auxiliary exhaust ventilation fans.
3. Initiate worker hydration and cooling protocols in the evacuation staging area.
================================================================================
```

