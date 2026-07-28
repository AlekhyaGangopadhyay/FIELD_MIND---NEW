# 🚀 NVIDIA Jetson Orin Nano Deployment & QLoRA Fine-Tuned LLM Specification

**Project**: FIELD-MIND — Offline Multimodal Agentic AI for Underground Mining  
**Target Edge Platform**: NVIDIA Jetson Orin Nano (8GB Unified LPDDR5 Memory, 1024 CUDA Cores)  
**Primary LLM Reasoning Engine**: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (Mining QLoRA Fine-Tuned 7B GGUF Model)  
**Primary Storage Media**: 512 GB MicroSD Card (UHS-I U3 / A2 rated)  

---

## 1. Hardware & Deployment Overview

FIELD-MIND is designed to execute **100% offline** on resource-constrained edge hardware deployed inside underground mine headings. The system operates on an **NVIDIA Jetson Orin Nano (8GB)** equipped with a **512 GB MicroSD card**, providing unified memory management across PyTorch deep learning monitors, vector RAG databases, graph stores, and quantized LLM engines.

---

## 2. 512 GB MicroSD Storage Footprint Breakdown

The entire FIELD-MIND software suite, including operating system binaries, CUDA runtimes, Python virtual environments, fine-tuned LLM GGUF models, and multi-modal datasets, consumes **~52 GB of disk space**, leaving **~460 GB (89% free space)** on a 512 GB card.

| Software / System Component | Disk Storage Required | Percent of 512 GB Card |
| :--- | :---: | :---: |
| **Ubuntu 22.04 LTS + JetPack 6.x OS & CUDA Toolkit** | ~30.0 GB | 5.8% |
| **Python Virtual Environment (`PyTorch`, `faiss`, `llama-cpp-python`)** | ~12.0 GB | 2.3% |
| **`Qwen2.5-7B-Instruct-Q4_K_M.gguf` Fine-Tuned Model File** | **~4.35 GB** | **0.8%** |
| **8 PyTorch Production Core Models + SciSense Encoders** | ~0.5 GB | 0.1% |
| **FAISS Vector Index & RAG Safety Knowledge Base** | ~0.2 GB | 0.04% |
| **Expedition Knowledge Graph & Mine Telemetry Datasets** | ~5.0 GB | 1.0% |
| **TOTAL SOFTWARE STORAGE FOOTPRINT** | **~52.05 GB** | **10.1%** |
| **FREE STORAGE REMAINING ON 512 GB CARD** | **~459.95 GB** | **89.9% (Available)** |

> [!TIP]
> **Performance Recommendation:** Using a **UHS-I U3 / A2** rated MicroSD card (e.g. *SanDisk Extreme* or *Samsung EVO Select*) enables read speeds up to 160 MB/s, loading the 4.35 GB GGUF model into Jetson RAM in **2 to 3 seconds** upon system boot up.

---

## 3. Jetson Orin Nano (8GB) RAM & VRAM Memory Budget

The Jetson Orin Nano utilizes a **Unified LPDDR5 Memory Architecture** (68 GB/s bandwidth) shared dynamically between the 6-core Arm CPU and the 1024-core Ampere GPU.

```
 ┌─────────────────────────────────────────────────────────────┐
 │ NVIDIA Jetson Orin Nano Unified LPDDR5 Memory (8,192 MB)    │
 └──────────────────────────────┬──────────────────────────────┘
                                │
   ├── OS & CUDA System Base    : ~1,500 MB (18.3%)
   ├── Fine-Tuned Qwen2.5-7B LLM: ~4,350 MB (53.1%) [Q4_K_M GGUF]
   ├── KV Cache (n_ctx=2048)    :   ~400 MB ( 4.9%) [n_gpu_layers=-1]
   ├── PyTorch Tier 1 Monitors  :   ~400 MB ( 4.9%) [8 Gas + Env + Vib + Nav]
   ├── SciSense Projection Head :   ~400 MB ( 4.9%) [4,096-D Alignment]
   ├── FAISS Vector RAG Index   :   ~120 MB ( 1.5%) [Local Safety DB]
   └── FREE RAM HEADROOM        : ~1,022 MB (12.5%) [Dynamic Telemetry Buffer]
```

---

## 4. Qwen2.5-7B-Instruct Dual Operating Modes

`Qwen2.5-7B-Instruct` fulfills **two distinct operational roles** in the system:

### Mode 1: Autonomous Diagnostic Mode (Background / ATR Triggered)
When Tier 1 monitors detect an anomaly, the Anomaly-Triggered Reasoning (ATR) orchestrator swaps device state (`IDLE` $\rightarrow$ `ACTIVE_REASONING`), passing aligned sensor telemetry, EKG graph history, and FAISS safety rules to Qwen. The model formulates root-cause hypotheses and safety mitigations in the background.

### Mode 2: Conversational Safety Assistant Mode (Interactive User Interface)
Mine operators, safety engineers, and field supervisors can **chat directly with Qwen** in natural language via `unified_demo/interactive_safety_hub.py` or `reasoning_core/agent_loop.py` (`MineSafetyChatAssistant`). Qwen answers queries conversationally in real-time, backed by the FAISS RAG retriever and EKG long-term memory.

---

## 5. QLoRA Fine-Tuning Pipeline & GGUF Conversion

Fine-tuning `Qwen2.5-7B-Instruct` adapts the model's neural weights to underground mining protocols (MSHA/DGMS threshold boundaries, goaf ventilation rules, gas sensor cross-sensitivity drift).

### Training vs. Deployment Specifications

| Phase | Compute Platform | Memory Required | Duration |
| :--- | :--- | :---: | :---: |
| **QLoRA Fine-Tuning Phase** | 1x NVIDIA GPU (RTX 3060 / 4070 or Colab T4) | ~7.5 - 8.0 GB VRAM | ~15 - 20 minutes |
| **Edge Deployment Phase** | NVIDIA Jetson Orin Nano (8GB) | ~4.35 GB VRAM | Continuous Edge Execution |

### Step-by-Step Fine-Tuning Workflow

1. **Instruction Dataset Generation (`gas_sensors/generate_qwen_qlora_dataset.py`)**:
   Generates 1,500+ structured ChatML multi-turn instruction pairs saved to `gas_sensors/data/qwen_mining_instructions.jsonl`.
2. **QLoRA Training (`gas_sensors/train_qwen_qlora.py`)**:
   Loads base `Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4 quantization, attaches LoRA adapters ($r=16, \alpha=16$), trains for 3 epochs via `SFTTrainer`, and exports merged FP16 weights (`gas_sensors/models/qwen2_5_mining_merged`).
3. **GGUF Quantization (`gas_sensors/convert_qwen_to_gguf.py`)**:
   Converts FP16 weights to GGUF format and quantizes down to `Q4_K_M` (`gas_sensors/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`).
4. **Edge Execution**:
   The GGUF model is loaded by `llama-cpp-python` with CUDA GPU offloading (`n_gpu_layers=-1`) on Jetson Orin Nano.

---

## 6. How to Run Verification Demos on Edge

```bash
# 1. Run Autonomous Self-Learning & Reflection Demo
py -X utf8 reasoning_core/demo_self_learning.py

# 2. Run Interactive Conversational Safety Hub
py -X utf8 unified_demo/interactive_safety_hub.py

# 3. Run Streaming Multi-Agent Simulation
py -X utf8 sensor_agents/demo_agents.py --ticks 300
```
