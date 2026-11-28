"""QLoRA fine-tune of Llama 3.2 3B Instruct on teacher labels.

Owner: Bereket. Spec: product spec, Task 5 (QLoRA fine-tune of Llama 3.2 3B).

Base model: unsloth/Llama-3.2-3B-Instruct  (weight-identical mirror of
meta-llama/Llama-3.2-3B-Instruct; the official repo is gated, and Meta's approval
for my HF account was still pending at project end, so the report documents the swap.)

Trains on teacher_labels.csv split=='train' rows (2018-2022, Gemini-labeled).
Writes the LoRA adapter to --output-dir and optionally pushes it to HF Hub.

Usage:
    python qlora_finetune.py --output-dir adapters/llama32-3b-leantrader \
        --labels data/interim/teacher_labels.csv \
        --news data/interim/news_clean.csv \
        --epochs 2 \
        --push-to-hub user/repo-name          # optional; set HF_TOKEN first
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from news_prompt import SYSTEM_PROMPT, format_input

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
MAX_LEN = 1024


def build_examples(labels_df: pd.DataFrame, news_df: pd.DataFrame, split: str) -> list[dict]:
    """Return chat-format rows for the SFTTrainer, one per labeled article."""
    rows = labels_df[labels_df["split"] == split].copy()
    news_subset = news_df[["news_id", "ticker", "company", "title", "summary"]]
    merged = rows.merge(news_subset, on=["news_id", "ticker"], how="left")
    # Fallback for teacher_labels that already carry the fields
    for col in ("company", "title", "summary"):
        if col + "_x" in merged.columns:
            merged[col] = merged[col + "_x"].fillna(merged[col + "_y"])
    examples: list[dict] = []
    for r in merged.itertuples(index=False):
        title = str(getattr(r, "title", "") or "")
        if title == "nan":
            title = ""
        summary = str(getattr(r, "summary", "") or "")
        if summary == "nan":
            summary = ""
        company = str(getattr(r, "company", "") or r.ticker)
        if company == "nan":
            company = r.ticker
        user = format_input(r.ticker, company, title, summary)
        label = {
            "relevant": bool(int(r.relevant)),
            "event_type": r.event_type,
            "sentiment": round(float(r.sentiment), 3),
            "confidence": round(float(r.confidence), 3),
        }
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(label)},
            ]
        })
    return examples


def main():
    ap = argparse.ArgumentParser(description="QLoRA fine-tune of Llama 3.2 3B on teacher labels.")
    ap.add_argument("--labels", type=Path, default=Path("data/interim/teacher_labels.csv"))
    ap.add_argument("--news", type=Path, default=Path("data/interim/news_clean.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("adapters/llama32-3b-leantrader"))
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--push-to-hub", help="HF repo id (e.g. 'user/repo'). Needs HF_TOKEN.")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
    )
    from trl import SFTConfig, SFTTrainer

    labels = pd.read_csv(args.labels)
    news = pd.read_csv(args.news, low_memory=False)
    train_examples = build_examples(labels, news, "train")
    check_examples = build_examples(labels, news, "check")
    print(f"train: {len(train_examples):,} rows   check: {len(check_examples):,} rows")
    train_ds = Dataset.from_list(train_examples)
    check_ds = Dataset.from_list(check_examples)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_cfg = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=25,
        save_strategy="epoch",
        bf16=True,
        fp16=False,
        max_length=MAX_LEN,
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=1,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=check_ds,
        args=sft_cfg,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"Saved adapter to {args.output_dir}")

    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub)
        tok.push_to_hub(args.push_to_hub)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
