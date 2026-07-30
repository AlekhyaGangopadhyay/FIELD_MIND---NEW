"""QLoRA training entry point for FIELD-MIND.

Training is intentionally a separate developer/workstation step.  Do not run
this on the 8 GB Jetson; deploy only the resulting GGUF artifact there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require_training_stack() -> None:
    missing = []
    for name in ("torch", "transformers", "datasets", "peft", "trl", "bitsandbytes"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError("Missing training packages: " + ", ".join(missing) + ". Install requirements-training.txt on the training GPU.")


def _load_messages(path: Path):
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=str(path), split="train")
    if len(dataset) < 10:
        raise ValueError("Training dataset must contain at least 10 examples.")
    return dataset


def _format(example):
    messages = example["messages"]
    return {"text": "\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)}


def train(args: argparse.Namespace) -> Path:
    _require_training_stack()
    import torch
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for QLoRA training. Run this script on the training GPU, not the Jetson deployment target.")
    dataset = _load_messages(args.dataset).map(_format, remove_columns=[c for c in _load_messages(args.dataset).column_names if c != "text"])
    split = dataset.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant, device_map="auto", torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    training_args = TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate, logging_steps=10, save_strategy="epoch", evaluation_strategy="epoch", fp16=True, gradient_checkpointing=True, report_to="none", seed=args.seed)
    trainer_kwargs = dict(model=model, args=training_args, train_dataset=split["train"], eval_dataset=split["test"], peft_config=lora, tokenizer=tokenizer, dataset_text_field="text", max_seq_length=args.max_seq_length, packing=True)
    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError:
        # TRL >= 0.17 renamed tokenizer/max_seq_length arguments.
        trainer_kwargs.pop("tokenizer", None)
        trainer_kwargs.pop("max_seq_length", None)
        trainer_kwargs["processing_class"] = tokenizer
        trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    adapter_dir = args.output / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    if args.merge:
        merged_dir = args.output / "merged_fp16"
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="auto")
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(str(merged_dir))
    (args.output / "training_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    return adapter_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset", type=Path, default=root / "gas_sensors" / "data" / "qwen_mining_instructions.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=Path, default=root / "gas_sensors" / "models" / "qwen2_5_mining")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge", action="store_true", help="Also export merged FP16 weights for GGUF conversion.")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    if not args.dataset.exists():
        parser.error(f"Dataset not found: {args.dataset}")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Training {args.base_model} with QLoRA; output={args.output}")
    print(f"Deployment note: only the final GGUF should be copied to the Jetson.")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
