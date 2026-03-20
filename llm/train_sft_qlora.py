from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, default="artifacts/sft/ckpt")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--ga", type=int, default=16)
    args = p.parse_args()

    can_bnb = hasattr(torch.nn.Module, "set_submodule")
    bnb = None
    if can_bnb:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb is not None:
        model_kwargs["quantization_config"] = bnb
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    ds = load_dataset("json", data_files=args.data, split="train")

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SFTConfig(
        output_dir=out_dir.as_posix(),
        max_length=int(args.max_seq_len),
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.bs),
        gradient_accumulation_steps=int(args.ga),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
        formatting_func=lambda ex: ex["messages"],
    )
    trainer.train()
    trainer.save_model(out_dir.as_posix())


if __name__ == "__main__":
    main()
