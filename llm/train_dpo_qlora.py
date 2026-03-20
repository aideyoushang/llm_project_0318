from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _format_prompt(prompt: str) -> str:
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, default="artifacts/pref/dpo_ckpt")
    p.add_argument("--max_prompt_len", type=int, default=512)
    p.add_argument("--max_len", type=int, default=1536)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--ga", type=int, default=16)
    args = p.parse_args()

    try:
        import torch.distributed.fsdp as fsdp  # type: ignore

        if not hasattr(fsdp, "FSDPModule"):
            class FSDPModule:  # type: ignore
                pass
            fsdp.FSDPModule = FSDPModule  # type: ignore[attr-defined]
    except Exception:
        pass

    from trl import DPOConfig, DPOTrainer

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

    def map_row(ex):
        return {
            "prompt": _format_prompt(str(ex["prompt"])),
            "chosen": str(ex["chosen"]),
            "rejected": str(ex["rejected"]),
        }

    ds = ds.map(map_row, remove_columns=ds.column_names)

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

    cfg = DPOConfig(
        output_dir=out_dir.as_posix(),
        per_device_train_batch_size=int(args.bs),
        gradient_accumulation_steps=int(args.ga),
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.lr),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        report_to=[],
        max_length=int(args.max_len),
        max_prompt_length=int(args.max_prompt_len),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(out_dir.as_posix())


if __name__ == "__main__":
    main()
