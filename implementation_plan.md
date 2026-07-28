# Implementation Plan - Jetson Orin Nano 8GB, Qwen2.5-7B-Instruct, & Autonomous Self-Learning Architecture

This plan details the upgrade to **NVIDIA Jetson Orin Nano (8GB Unified RAM)** running **`Qwen2.5-7B-Instruct-Q4_K_M.gguf`**, along with an **Autonomous Self-Learning & Real-Time Reflection Engine** that enables agents to learn on-the-fly when real-world conditions differ from model predictions.

---

## 1. Architectural Feature: Autonomous Self-Learning & Reflection Loop

### **How Real-Time Self-Learning Works**
When telemetry arrives and the current model prediction contradicts verified ground truth (e.g., false alarm due to humidity spray, or unexpected gas pocket behavior):

```
 ┌─────────────────────────────┐
 │  Streaming Real-Time Data   │
 └──────────────┬──────────────┘
                │
                ▼
 ┌─────────────────────────────┐
 │  Model Prediction Mismatch? │ <--- (Real-world reality differs from prediction)
 └──────────────┬──────────────┘
                │
                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  LLM Self-Reflection Node (Qwen2.5-7B-Instruct)            │
 │  Analyzes misprediction & formulates reflective safety rule │
 └──────────────┬──────────────────────────────┬───────────────┘
                │                              │
                ▼                              ▼
 ┌─────────────────────────────┐  ┌────────────────────────────┐
 │  FAISS Self-Learned RAG     │  │  Agent Experience Replay   │
 │  Stores learned rule in     │  │  Retrains ML model on-the- │
 │  vector memory for retrieval│  │  fly with corrected data   │
 └─────────────────────────────┘  └────────────────────────────┘
```

1. **LLM Self-Reflection Node (`REFLECT`)**: `Qwen2.5-7B-Instruct` analyzes the discrepancy between prediction and reality, formulating a structured **Self-Learned Correction Rule**.
2. **FAISS Vector Experience Storage**: The learned rule is immediately embedded and saved into FAISS vector memory (`faiss_rag`), ensuring future reasoning queries retrieve this experience and **never make the same mistake twice**.
3. **Experience Replay & Online Retraining**: Corrected `(features, true_label)` pairs are pushed to the agent's replay buffer, automatically retraining the local classifier when the buffer fills.

---

## 2. Jetson Orin Nano (8GB) & Qwen2.5-7B Memory Profile

| Component | Hardware / Model Allocator | Memory Footprint |
| :--- | :--- | :---: |
| **Edge Hardware Platform** | **NVIDIA Jetson Orin Nano (8GB Unified RAM, 1024 CUDA cores)** | Total: **8,192 MB** |
| **Primary LLM Engine** | **`Qwen2.5-7B-Instruct-Q4_K_M.gguf` (4-bit INT4 quantization)** | **~4,350 MB** |
| **KV Cache & Context Window** | 2,048-token context (`n_ctx=2048`, `n_gpu_layers=-1`) | ~400 MB |
| **PyTorch Tier 1 Monitors** | 8 Gas + Vibration + Env + Ultrasonic Models | ~400 MB |
| **SciSense Embedding Aligner** | 4,096-D Projection Layers (`all-MiniLM-L6-v2`) | ~400 MB |
| **FAISS Vector RAG Index** | Static Literature + **Self-Learned Experience Memory** | ~120 MB |
| **OS & CUDA Base** | Ubuntu 22.04 LTS (JetPack 6.x) | ~1,500 MB |
| **Available Headroom** | Dynamic buffer for real-time EKG graph updates | **~1,022 MB** |

---

## 3. Proposed Code & Module Changes

### **Self-Learning Engine & Agents**

#### [MODIFY] [faiss_rag/retriever.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/faiss_rag/retriever.py) & [embedder.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/faiss_rag/embedder.py)
* Add `add_learned_experience(rule_text)` method to dynamically embed and persist self-learned rules into the active FAISS vector index.

#### [MODIFY] [reasoning_core/agent_loop.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/reasoning_core/agent_loop.py) & [llm_runner.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/reasoning_core/llm_runner.py)
* Add **`REFLECT`** node to the LangGraph state graph.
* Update `OfflineLLMRunner` to support `task_type="reflection"` using `Qwen2.5-7B-Instruct-Q4_K_M.gguf` with CUDA GPU offload (`n_gpu_layers=-1`).

#### [MODIFY] [sensor_agents/agent_base.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/sensor_agents/agent_base.py) & [gas_agent.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/sensor_agents/gas_agent.py)
* Add `feedback_correction(sensor_data, actual_situation, true_label)` method to trigger the LLM self-reflection pipeline and online replay retraining.

#### [NEW] [reasoning_core/demo_self_learning.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/reasoning_core/demo_self_learning.py)
* Create an interactive demonstration showing the agent receiving real-time discrepancy feedback, reflecting via `Qwen2.5-7B-Instruct`, storing the new rule in FAISS, retraining its ML model, and successfully handling identical telemetry on the next turn.

### **Orchestrator & Documentation**

#### [MODIFY] [atr_activation/orchestrator.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/atr_activation/orchestrator.py) & [demo_atr.py](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/atr_activation/demo_atr.py)
* Update orchestrator state transition logs to reference `Qwen2.5-7B-Instruct-Q4_K_M.gguf` and Jetson Orin Nano 8GB RAM.

#### [MODIFY] [reasoning_core/README.md](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/reasoning_core/README.md) & [atr_activation/README.md](file:///c:/Users/Student/Desktop/FIELD_MIND%20-%20NEW/atr_activation/README.md)
* Update documentation specs for Jetson Orin Nano 8GB, `Qwen2.5-7B-Instruct`, and the Self-Learning Reflection Loop.

---

## 4. Verification Plan

### Automated Execution
1. Run `python reasoning_core/demo_self_learning.py` to verify the end-to-end self-reflection, FAISS memory update, and online model retraining loop.
2. Run `python atr_activation/demo_atr.py` to verify ATR state transitions under the Jetson Orin Nano 8GB memory budget.
