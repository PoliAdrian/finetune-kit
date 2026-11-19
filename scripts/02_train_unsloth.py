#!/usr/bin/env python3
"""
02_train_unsloth.py
-------------------
Fast fine-tuning of small open models (FunctionGemma 270M, Gemma 1B/4B, Qwen 0.5B/3B)
using Unsloth + TRL SFTTrainer, with direct export to GGUF for local CPU inference.

Run on Google Colab (free T4) or any cloud GPU (RunPod / Verda / Modal / Lambda):
  # Full fine-tune for tiny 270M model (~5-15 min):
  python scripts/02_train_unsloth.py \
      --model unsloth/functiongemma-270m-it \
      --mode full \
      --epochs 3 \
      --quantization q8_0

  # QLoRA for 1B-4B models (~10-20 min on free T4):
  python scripts/02_train_unsloth.py \
      --model unsloth/gemma-3-1b-it \
      --mode lora \
      --epochs 2 \
      --quantization q4_k_m
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small model with Unsloth and export GGUF.")
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/functiongemma-270m-it",
        help="Hugging Face base model ID (Unsloth mirrors avoid gated delays).",
    )
    parser.add_argument("--train-file", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--gguf-dir", type=Path, default=Path("models/gguf"))
    parser.add_argument(
        "--mode",
        choices=["full", "lora"],
        default="full",
        help="'full' is great for <=500M models (like FunctionGemma 270M); use 'lora' for >=1B.",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None, help="Auto-selected by mode if omitted.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument(
        "--quantization",
        type=str,
        default="q8_0",
        help="GGUF quantization: q8_0 (best for 270M, ~290MB), q4_k_m (best for 1B-4B), f16.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Optional HF repo ID to upload GGUF (e.g. 'yourname/my-hackathon-model'). Requires HF_TOKEN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Import heavy GPU libs only when running training
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    load_in_4bit = args.mode == "lora"
    full_finetuning = args.mode == "full"
    lr = args.lr if args.lr is not None else (5e-5 if full_finetuning else 2e-4)

    print(f"Loading base model: {args.model} (mode={args.mode}, lr={lr})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,  # auto-detect bf16/fp16
        load_in_4bit=load_in_4bit,
        full_finetuning=full_finetuning,
    )

    if args.mode == "lora":
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_r,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    # Load JSONL datasets
    data_files = {"train": str(args.train_file)}
    if args.val_file.exists():
        data_files["validation"] = str(args.val_file)
    ds = load_dataset("json", data_files=data_files)

    def format_chat(batch: dict) -> dict:
        texts = []
        for msgs in batch["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    ds = ds.map(format_chat, batched=True, remove_columns=ds["train"].column_names)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        args=SFTConfig(
            output_dir=str(args.output_dir),
            dataset_text_field="text",
            max_length=args.max_seq_len,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            logging_steps=20,
            eval_strategy="epoch" if "validation" in ds else "no",
            save_strategy="epoch",
            bf16=use_bf16,
            fp16=not use_bf16,
            optim="adamw_8bit" if args.mode == "lora" else "adamw_torch",
            seed=42,
            report_to="none",
        ),
    )

    print("Starting training...")
    stats = trainer.train()
    print(f"Training complete! Metrics: {stats.metrics}")

    # Export GGUF for fast local CPU inference (llama.cpp / node-llama-cpp / Ollama)
    args.gguf_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting GGUF ({args.quantization}) to {args.gguf_dir} ...")
    model.save_pretrained_gguf(
        str(args.gguf_dir),
        tokenizer,
        quantization_method=args.quantization,
    )

    if args.push_to_hub:
        token = os.getenv("HF_TOKEN")
        if not token:
            print("[warn] --push-to-hub set but HF_TOKEN env var is missing. Skipping upload.")
        else:
            print(f"Uploading GGUF to Hugging Face Hub: {args.push_to_hub}")
            model.push_to_hub_gguf(
                args.push_to_hub,
                tokenizer,
                quantization_method=args.quantization,
                token=token,
            )

    print("All done!")


if __name__ == "__main__":
    main()
