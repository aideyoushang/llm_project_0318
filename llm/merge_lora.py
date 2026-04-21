from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--sft", type=str, required=True)
    p.add_argument("--dpo", type=str, required=True)
    p.add_argument("--out", type=str, default="artifacts/fused/fused_lora")
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()

    alpha = float(args.alpha)
    if alpha < 0 or alpha > 1:
        raise SystemExit("alpha must be in [0,1]")

    sft_cfg = PeftConfig.from_pretrained(args.sft)
    dpo_cfg = PeftConfig.from_pretrained(args.dpo)
    if sft_cfg.peft_type != dpo_cfg.peft_type:
        raise SystemExit("SFT and DPO peft_type mismatch")
    if sft_cfg.base_model_name_or_path != dpo_cfg.base_model_name_or_path:
        raise SystemExit("SFT and DPO base model mismatch")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, args.sft, adapter_name="sft")
    model.load_adapter(args.dpo, adapter_name="dpo")

    model.set_adapter("sft")
    sft_state = get_peft_model_state_dict(model, adapter_name="sft")
    model.set_adapter("dpo")
    dpo_state = get_peft_model_state_dict(model, adapter_name="dpo")

    fused_state = {}
    for k, v in sft_state.items():
        if k not in dpo_state:
            fused_state[k] = v
            continue
        dpo_v = dpo_state[k]
        if isinstance(v, torch.Tensor) and isinstance(dpo_v, torch.Tensor):
            fused_state[k] = (1 - alpha) * v + alpha * dpo_v
        else:
            fused_state[k] = v

    model.set_adapter("sft")
    set_peft_model_state_dict(model, fused_state, adapter_name="sft")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir.as_posix())
    tokenizer.save_pretrained(out_dir.as_posix())
    meta = {
        "sft": args.sft,
        "dpo": args.dpo,
        "alpha": alpha,
        "out": out_dir.as_posix(),
    }
    (out_dir / "merge_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

