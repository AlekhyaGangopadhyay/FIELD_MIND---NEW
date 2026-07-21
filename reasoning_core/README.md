# Scientific Reasoning Core — FIELD-MIND Layer 3

The **Scientific Reasoning Core** is FIELD-MIND's on-device reasoning engine. It coordinates multi-step diagnostic reasoning using a **LangGraph** workflow loop. The core activates when the system enters `EMERGENCY` or `ACTIVE_REASONING` states, drawing context from the Expedition Knowledge Graph (EKG) and the FAISS RAG database to formulate root-cause hypotheses and generate prioritized, actionable safety suggestions.

All operations run **100% offline** on local CPU hardware.

---

## Reasoning Workflow Loop

```
               [ MineOrchestratorAgent Alert ]
                              │
                              v
             +─────────────────────────────────+
             |         1. OBSERVE              |
             |  Reads raw anomalies & context  |
             +─────────────────────────────────+
                              │
                              v
             +─────────────────────────────────+
             |       2. EKG-RETRIEVE           |
             | Fetches segment historical risk |
             +─────────────────────────────────+
                              │
                              v
             +─────────────────────────────────+
             |       3. RAG-RETRIEVE           |
             | Queries FAISS safety guidelines |
             +─────────────────────────────────+
                              │
                              v
             +─────────────────────────────────+
             |        4. HYPOTHESIZE           |
             |  Formulates root-cause theory   |
             +─────────────────────────────────+
                              │
                              v
             +─────────────────────────────────+
             |          5. SUGGEST             |
             |  Generates safety mitigations   |
             +─────────────────────────────────+
                              │
                              v
             +─────────────────────────────────+
             |        6. UPDATE_EKG            |
             | Saves resolution & links segment|
             +─────────────────────────────────+
                              │
                              v
                 [ Actionable Advice Output ]
```

---

## File Structure

```
reasoning_core/
├── __init__.py          # Package exports (ScientificReasoningCore, AgentState, OfflineLLMRunner, MineSafetyChatAssistant)
├── state.py             # AgentState schema defining context, history, and outputs
├── llm_runner.py        # GGUF llama.cpp inference engine with robust Expert Rule fallback
├── agent_loop.py        # LangGraph StateGraph state machine and node implementations
├── chat_assistant.py    # Conversational agent analyzing data inputs, safety measures, and multi-node trends
├── demo_reasoning.py    # Workflow demo simulating critical gas, blast, and robot alerts
├── demo_chat.py         # Conversational dialogue scenarios demonstrating the safety chat assistant
└── README.md            # Workflow architecture, usage guidelines, and specifications
```

---

## Core Nodes & Logic

| Node | Action |
|------|--------|
| **Observe** | Scans active inputs from `anomalies` dictionary and logs parameter states. |
| **EKG-Retrieve** | Inspects local graph storage, executes `get_segment_risk_profile` and `get_blast_history`, checking for recent blasting activity or recurring gas anomalies. |
| **RAG-Retrieve** | Queries the local FAISS index for safety literature corresponding to active sensor modalities (e.g. gas, vibration, environment, robot collision). |
| **Hypothesize** | Passes EKG context, RAG rules, and sensor anomalies to the LLM/expert reasoner to diagnose the event's root cause. |
| **Suggest** | Compiles actionable safety, ventilation, structural, or navigational recommendations. |
| **Update-EKG** | Injects a new `ReasoningResolution` node containing the hypothesis and safety suggestions into the Expedition Knowledge Graph, linking it to the active `TunnelSegment`. |

---

## Configuration & Offline Inference

The reasoning core uses `OfflineLLMRunner` to execute model queries offline:

1. **Quantized LLM Execution (GGUF)**: If `llama-cpp-python` is installed and a GGUF model path (such as `Llama-3.2-3B-Instruct-GGUF`) is provided during initialization, the runner runs model inference on CPU threads.
2. **Domain-Informed Expert Fallback**: If no model path is provided or compiling GGUF packages fails, the runner defaults to a robust, rules-based expert system mapping standard mining limits and EKG historical associations to structured Markdown suggestions.

---

## Usage

### Run the Demos

**Reasoning Workflow Demo**:
Executes simulated emergency scenarios representing methane gas build-up, blast vibration thresholds, and robot proximity collision risks:
```bash
py -X utf8 reasoning_core/demo_reasoning.py
```

**Conversational Assistant Demo**:
Runs conversation scenarios where the assistant answers natural language queries about mine safety by analyzing data inputs and EKG/RAG context:
```bash
py -X utf8 reasoning_core/demo_chat.py
```

---

### Python API

#### Batch Reasoning Loop
```python
from reasoning_core import ScientificReasoningCore

# Initialize reasoning core
core = ScientificReasoningCore(
    workspace_root = "/path/to/FIELD_MIND",
    model_path     = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"  # Optional GGUF path
)

# Run complete diagnostic loop
result = core.reason(
    anomalies = {
        "MQ4_CH4_ppm": 12500.0,    # 1.25% methane
        "MQ7_CO_ppm": 45.0,
        "temp": 29.0
    },
    segment_id = "TUNNEL_A1"
)

# Access outcomes
print(f"Hypothesis: {result['hypothesis']}")
print(f"Suggestions: {result['suggestions']}")
```

#### Conversational Safety Assistant
```python
from reasoning_core import MineSafetyChatAssistant

# Initialize conversational assistant
assistant = MineSafetyChatAssistant(
    workspace_root = "/path/to/FIELD_MIND",
    model_path     = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"  # Optional GGUF path
)

# Simple single-node query (original API, still works)
reply = assistant.chat(
    user_message = "What safety measures apply to the current methane readings?",
    segment_id   = "TUNNEL_A1",
    active_anomalies = {
        "MQ4_CH4_ppm": 12500.0,
        "MQ7_CO_ppm": 35.0
    }
)
print(reply)
```

#### Multi-node streaming query (new)

When calling from the streaming simulation, pass the full sensor readings,
model predictions, and cross-node trend context so the assistant can reason
about gradual rises, falling clearance, and cross-node differences:

```python
reply = assistant.chat(
    user_message     = "Analyze the 10-timestamp trend across 3 nodes.",
    segment_id       = "TUNNEL_SIM_01",
    active_anomalies = {"MQ4_CH4_ppm": 6480, "MQ7_CO_ppm": 36.9},
    model_predictions = {
        "methane_hazard": 0,
        "co_nox_hazard": 1,
        "vibration_hazard": 1,
        "predicted_ppv": 11.78,
        "steering_decision": "Slight-Left-Turn",
    },
    sensor_readings = {
        "MQ4_CH4_ppm": 6480,
        "MQ7_CO_ppm": 36.9,
        "temp": 22.0,
        "predicted_ppv": 11.78,
        "min_distance": 2.51,
    },
    trend_context = (
        "NODE_1: CH4 379->6480 ppm (rising); CO 10.9->36.9 ppm (rising)\n"
        "NODE_2: PPV 11.78->25.04 mm/s (rising)\n"
        "NODE_3: clearance 2.51->0.18 m (falling)"
    ),
)
print(reply)
```

The `trend_context` string is included verbatim in the prompt and in the
structured assessment response, giving the LLM and the rule-engine full
visibility into the time-series progression across all nodes.
