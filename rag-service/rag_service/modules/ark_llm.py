from __future__ import annotations

import json
import ssl
import urllib.request
from dataclasses import dataclass
from typing import Any

from rag_service.modules.runtime_config import load_runtime_config


@dataclass(frozen=True)
class ArkConfig:
    base_url: str
    api_key: str
    model: str


class ArkResponsesClient:
    def __init__(self, config: ArkConfig | None = None) -> None:
        if config is None:
            rc = load_runtime_config()
            base_url = rc.ark_base_url or "https://ark.cn-beijing.volces.com/api/v3/responses"
            api_key = (rc.ark_api_key or "").strip()
            model = (rc.ark_model or "").strip()
            config = ArkConfig(base_url=str(base_url), api_key=api_key, model=model)
        self._config = config

    def is_configured(self) -> bool:
        return bool(self._config.api_key and self._config.model and self._config.base_url)

    def response_text(self, user_text: str, *, timeout_s: float = 60.0) -> str:
        if not self.is_configured():
            raise RuntimeError("Missing ARK_API_KEY or ARK_MODEL for ArkResponsesClient")

        payload = {
            "model": self._config.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_text,
                        }
                    ],
                }
            ],
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._config.base_url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(raw)
        text = self._extract_text(obj)
        return text.strip()

    def _extract_text(self, obj: Any) -> str:
        if isinstance(obj, dict):
            if isinstance(obj.get("output_text"), str):
                return obj["output_text"]
            out = obj.get("output")
            if isinstance(out, list):
                for item in out:
                    t = self._extract_text(item)
                    if t:
                        return t
            content = obj.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        if isinstance(c.get("text"), str) and c.get("text"):
                            return c["text"]
                        if isinstance(c.get("output_text"), str) and c.get("output_text"):
                            return c["output_text"]
                    t = self._extract_text(c)
                    if t:
                        return t
            if isinstance(obj.get("text"), str) and obj.get("text"):
                return obj["text"]
            for v in obj.values():
                t = self._extract_text(v)
                if t:
                    return t
        if isinstance(obj, list):
            for it in obj:
                t = self._extract_text(it)
                if t:
                    return t
        return ""
