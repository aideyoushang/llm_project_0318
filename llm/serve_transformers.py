from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_prompt(messages: list[dict[str, str]]) -> str:
    out = []
    for m in messages:
        role = str(m.get("role") or "").strip().lower()
        content = str(m.get("content") or "")
        if not role:
            role = "user"
        if role == "system":
            out.append(f"<|system|>\n{content}\n")
        elif role == "assistant":
            out.append(f"<|assistant|>\n{content}\n")
        else:
            out.append(f"<|user|>\n{content}\n")
    out.append("<|assistant|>\n")
    return "".join(out)


def _iter_generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text


def create_app(model_path: str, lora_path: str | None = None) -> FastAPI:
    app = FastAPI(title="local-llm-server")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(device)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model.eval()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        prompt = _build_prompt([m for m in messages if isinstance(m, dict)])
        max_new_tokens = int(payload.get("max_tokens") or payload.get("max_new_tokens") or 512)
        temperature = float(payload.get("temperature") or 0.2)
        top_p = float(payload.get("top_p") or 0.9)
        content = _iter_generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }

    @app.post("/v1/chat/completions/stream")
    def chat_completions_stream(payload: dict[str, Any]) -> StreamingResponse:
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        prompt = _build_prompt([m for m in messages if isinstance(m, dict)])
        max_new_tokens = int(payload.get("max_tokens") or payload.get("max_new_tokens") or 512)
        temperature = float(payload.get("temperature") or 0.2)
        top_p = float(payload.get("top_p") or 0.9)

        def gen():
            text = _iter_generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for ch in text:
                data = json.dumps({"type": "delta", "content": ch}, ensure_ascii=False)
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--lora", type=str, default="")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    args = p.parse_args()

    import uvicorn

    lora_path = args.lora.strip() or None
    uvicorn.run(create_app(args.model, lora_path=lora_path), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
