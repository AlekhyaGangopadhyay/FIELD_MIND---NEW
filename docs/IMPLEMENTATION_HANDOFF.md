# FIELD-MIND Implementation Handoff

This is the executable status of the QLoRA/GGUF implementation plan for the
Jetson Orin Nano target.

## Implemented

- `gas_sensors/generate_qwen_qlora_dataset.py` creates deterministic local
  ChatML JSONL training data. It defaults to 1,500 examples and writes
  atomically, refusing accidental overwrite.
- `gas_sensors/train_qwen_qlora.py` runs QLoRA only on a CUDA training host,
  with NF4 double quantization, LoRA rank 16, evaluation split, checkpointing,
  and optional merged FP16 export.
- `gas_sensors/convert_qwen_to_gguf.py` validates llama.cpp tools and refuses
  to publish an unexpectedly small GGUF artifact.
- `gas_sensors/eval_qwen_mining_llm.py` measures JSON validity, classification
  accuracy, and GGUF throughput. If the model is absent it reports a clearly
  labelled deterministic fallback result rather than claiming an edge score.
- `jetson_preflight.py` checks JetPack/Jetson identity, Python version, free
  storage, GGUF presence, CUDA tooling, and available memory.
- `OfflineLLMRunner` now resolves paths from the workspace, lazy-loads the
  GGUF, exposes health information, and can unload it to reclaim unified RAM.

## Build and deployment sequence

Run dataset generation and QLoRA training on a CUDA workstation:

```bash
python gas_sensors/generate_qwen_qlora_dataset.py --overwrite
python gas_sensors/train_qwen_qlora.py --merge
python gas_sensors/convert_qwen_to_gguf.py --llama-cpp /opt/llama.cpp
```

Copy only the final `Qwen2.5-7B-Instruct-Q4_K_M.gguf` to the Jetson. Do not
copy the training cache, raw datasets, or merged FP16 checkpoint unless they
are explicitly needed for maintenance.

On the Jetson, run:

```bash
python jetson_preflight.py
python gas_sensors/eval_qwen_mining_llm.py
python reasoning_core/demo_self_learning.py
python atr_activation/demo_atr.py
```

The expected deployment memory behavior is IDLE monitoring first, with the
GGUF loaded only when reasoning is requested. Keep `FIELDMIND_LLM_CONTEXT` at
1024 for the first 8 GB test; increase it only after recording free unified
memory and sustained inference latency.

## Acceptance gates for the ORIN test

1. Preflight exits zero with at least 20 GB free storage and the expected GGUF
   artifact present.
2. The evaluator reports `gguf_cases` equal to the number of model-backed
   cases, not zero, and records throughput.
3. Tier-1 monitoring continues when the GGUF is absent or fails to load; the
   expert fallback remains available.
4. No unexplained OOM, swap thrashing, or increasing latency occurs during a
   30-minute streaming run.
5. Safety actions are checked by a qualified operator against the mine's
   approved procedures before live deployment.

