# 📊 FIELD-MIND: Storage & Memory Estimation for Jetson Nano Deployment

This report estimates the on-device storage requirements and memory constraints for deploying the **FIELD-MIND** software suite on the target hardware: **NVIDIA Jetson Nano Module (4 GB LPDDR4 RAM, 16 GB eMMC 5.1 Flash, 128-core Maxwell GPU)**.

---

## 🔍 Context and Project Scope

Based on the [FIELD_MIND_software_roadmap.md](file:///d:/Users/SUPRATIK/FieldMind/FIELD_MIND---NEW/docs/FIELD_MIND_software_roadmap.md) and [FIELD_MIND_proposal.docx](file:///d:/Users/SUPRATIK/FieldMind/FIELD_MIND---NEW/docs/FIELD_MIND_proposal.docx), the software stack is an **offline multimodal agentic AI platform** for underground mining. 

Key layers running on the Jetson Nano include:
1. **Continuous Anomaly Monitors (ATR Tier 1)**: 13 serialized `scikit-learn` classifiers running on continuous telemetry.
2. **SciSense Projection Encoders (Layer 1)**: PyTorch projection networks aligning sensor inputs into a 4096-D shared space.
3. **Expedition Knowledge Graph (EKG - Layer 2B)**: Directed property graph mapping tunnel geometry and historical alerts (serialized as JSON).
4. **FAISS Vector Database (RAG - Layer 2C)**: Index of mining literature, manuals, and safety guidelines.
5. **Scientific Reasoning Core (Layer 3)**: A `LangGraph` agent loop executing a quantized **Llama-3.2-3B-Instruct** LLM in GGUF format via `llama.cpp` / `llama-cpp-python`.
6. **Rugged Tablet Web Dashboard & ASR**: FastAPI server and offline voice processing (ASR Whisper-cpp / Vosk).

---

## 💾 Storage Requirement Breakdown (eMMC Footprint)

In a production environment, the **raw training datasets** (such as the ~1.2 GB gas sensors data or large SEG-Y vibration traces) **do not need to be deployed** on the Jetson Nano. Only the pre-trained models, database indices, web dashboard, and system runtimes are required.

Here is the itemized estimation of the required storage space:

| Component | Description | Estimated Size (Min) | Estimated Size (Max) | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **1. Operating System (JetPack L4T)** | Ubuntu 18.04 + NVIDIA drivers | **4.5 GB** | **6.0 GB** | Assumes a headless, minimal/trimmed installation (no GUI desktop environment, developer samples, or redundant docs removed). |
| **2. Machine Learning Runtimes** | PyTorch (with CUDA support) | **1.8 GB** | **2.2 GB** | Pre-built NVIDIA PyTorch wheel for Jetson Nano. |
| **3. Core Python Libraries** | Scikit-learn, SciPy, NumPy, Pandas, NetworkX | **500 MB** | **700 MB** | Essential scientific stack packages. |
| **4. Reasoning Core Runtime** | `llama-cpp-python` & compiled bindings | **200 MB** | **300 MB** | Compiled with CUDA/cuBLAS support for Maxwell GPU acceleration. |
| **5. Web/UI Frameworks & APIs** | FastAPI, Uvicorn, LangGraph, etc. | **150 MB** | **200 MB** | Backend server and orchestration packages. |
| **6. FIELD-MIND Codebase** | Python scripts, config, utility files | **5 MB** | **10 MB** | Excluding raw datasets and logs. |
| **7. Pre-trained ML Models** | 13 `joblib` files + SciSense PyTorch weights | **90 MB** | **120 MB** | Sklearn models (`~75 MB`) + SciSense MLP projections (`~15 MB`). |
| **8. Retrieval & Knowledge Bases** | Sentence Transformers + FAISS Index + EKG JSON | **250 MB** | **450 MB** | Literature database, vector embeddings, and persistent JSON graph store. |
| **9. Speech-to-Text ASR Model** | Whisper-tiny / base GGML weights | **75 MB** | **150 MB** | Used for local offline speech parsing. |
| **10. Mandatory System Swap File** | Virtual memory allocation on disk | **4.0 GB** | **6.0 GB** | **CRITICAL**: Required to prevent Out-Of-Memory (OOM) crashes when loading the 3B LLM on 4GB RAM. |
| **Total Estimated Footprint** | **All software parts + Swap space** | **11.57 GB** | **16.13 GB** | **Saturates the 16 GB eMMC capacity.** |

---

## ⚠️ Key Engineering Constraints & Risks

> [!WARNING]
> ### 1. 16GB eMMC Storage Saturation
> A naive installation of standard JetPack, Python libraries, PyTorch, a 3B LLM, and a mandatory 4GB swap space will exceed the **16 GB eMMC** limit (requiring ~13.5 to 16.0 GB). Even with aggressive optimization, running close to 90%+ storage capacity degrades eMMC write speeds, shortens flash memory lifespan (wear leveling issues), and risks disk-full crashes during logging or database serialization.

> [!IMPORTANT]
> ### 2. 4GB RAM & Model Swapping Bottleneck
> A 3B quantized LLM (Q4_K_M GGUF) requires **~2.1 GB** of memory. When combined with the OS (~500 MB), PyTorch encoders (~150 MB), FAISS indices (~200 MB), ASR (~150 MB), and FastAPI (~100 MB), the peak memory requirement during the `ACTIVE_REASONING` phase reaches **~3.2 GB - 3.7 GB**. 
> - This fits within the 4 GB limit but requires aggressive memory management.
> - The **Anomaly-Triggered Reasoning (ATR)** design (detailed in [orchestrator.py](file:///d:/Users/SUPRATIK/FieldMind/FIELD_MIND---NEW/atr_activation/orchestrator.py)) is absolutely critical: it must load the LLM into RAM *only* when an anomaly occurs and immediately unload it (`del model` + `gc.collect()` + `cuda.empty_cache()`) when returning to `IDLE` state.
> - **The Catch**: Reading a 2.1 GB LLM file from the eMMC 5.1 (~250 MB/s read speed) into RAM on every trigger will introduce an **8 to 10-second cold-start latency** before reasoning begins.

---

## 🛠️ Recommended Deployment Strategies

To ensure system stability, speed, and safety in underground environments, the following three hardware/deployment setups are proposed:

### Strategy 1: Hybrid Storage (OS on eMMC + Models/Swap on MicroSD) — *Recommended*
* **Setup**: Keep the trimmed JetPack OS and Python packages on the 16GB eMMC (utilizing its fast ~250MB/s speeds for booting and libraries). Configure the system to store the 3B GGUF model, FAISS indices, and the mandatory Swapfile on the **external 256GB Micro SD Card** (UHS-I U3 grade) budgeted in the proposal.
* **Storage Allocation**: 
  - **eMMC (16GB)**: OS + Python + Core Libraries (~7.5 GB used, ~8.5 GB free).
  - **Micro SD (256GB)**: GGUF Models + Swapfile + FAISS Index + Ingested telemetry databases.
* **Trade-off**: Slows down the LLM load-time during `ACTIVE_REASONING` state transitions (~20–30 seconds due to MicroSD read limits of 80MB/s).

### Strategy 2: USB 3.0 SSD Booting (Best Performance)
* **Setup**: Bypass the 16GB eMMC entirely. Flash the JetPack OS onto a fast **M.2 SATA or NVMe SSD in a USB 3.0 enclosure** (often supported by Jetson Carrier boards) and configure the Jetson Nano to boot from USB.
* **Storage Allocation**: All files, libraries, swap, and models reside on the SSD (>128GB capacity, >400MB/s read/write).
* **Trade-off**: Adds slightly to hardware costs (~₹3,000 for SSD and casing) but eliminates the storage bottleneck, eliminates eMMC wear-out risks, and reduces LLM load latency to **under 5 seconds**.

### Strategy 3: Downscaling to a 1B Parameter LLM
* **Setup**: If forced to run strictly on the 16GB eMMC without external media:
  - Downscale the LLM from `Llama-3.2-3B` to **`Llama-3.2-1B-Instruct` (Q4_K_M GGUF)**.
  - The model size drops from **2.1 GB to ~700 MB**.
  - System swap space can be trimmed safely to **2.5 GB**.
  - Total storage footprint drops to **~9.0 GB**, leaving a comfortable ~7.0 GB free on the eMMC.
* **Trade-off**: Lower reasoning capabilities and citation quality compared to the 3B model, but provides low-latency loading (under 3 seconds on eMMC) and runs without external storage.
