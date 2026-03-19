from __future__ import annotations

import json
import re
from typing import Any

from rag_service.modules.ark_llm import ArkResponsesClient


class GeneratorModule:
    def __init__(self) -> None:
        self._llm = ArkResponsesClient()

    def generate(self, question: str, contexts: list[str], references: list[dict[str, Any]]) -> dict[str, Any]:
        if not contexts or not references:
            return {"answer": "I don't have enough evidence from reviews to answer this question.", "used_refs": []}

        if not self._llm.is_configured():
            return {"answer": "\n".join(contexts[:1]), "used_refs": [0]}

        evidence = []
        for i, ref in enumerate(references):
            chunk_text = str(ref.get("chunk_text") or "")
            if not chunk_text:
                continue
            evidence.append(
                {
                    "id": i,
                    "chunk_id": ref.get("chunk_id"),
                    "doc_id": ref.get("doc_id"),
                    "post_date": ref.get("post_date"),
                    "text": chunk_text,
                }
            )
        if not evidence:
            return {"answer": "\n".join(contexts[:1]), "used_refs": [0]}

        prompt = "\n".join(
            [
                "You are answering a question using only the provided review excerpts.",
                "Return ONLY valid minified JSON with this schema:",
                '{"answer": string, "used_refs": number[]}',
                "Rules:",
                "- Answer in English.",
                "- Each sentence in answer must include inline citations like [id].",
                "- used_refs must include all ids referenced in the answer.",
                f"Question: {question}",
                "Evidence:",
                json.dumps(evidence, ensure_ascii=False),
            ]
        )
        try:
            text = self._llm.response_text(prompt, timeout_s=45.0)
        except Exception:
            return {"answer": "\n".join(contexts[:1]), "used_refs": [0]}
        payload = self._try_parse_json(text)
        if not payload:
            return {"answer": "\n".join(contexts[:1]), "used_refs": [0]}
        answer = str(payload.get("answer") or "").strip()
        used = payload.get("used_refs") or []
        used_refs: list[int] = []
        if isinstance(used, list):
            for u in used:
                try:
                    idx = int(u)
                except Exception:
                    continue
                if idx not in used_refs:
                    used_refs.append(idx)
        if not answer:
            return {"answer": "\n".join(contexts[:1]), "used_refs": [0]}
        return {"answer": answer, "used_refs": used_refs}

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        t = text.strip()
        if not t:
            return None
        if "```" in t:
            parts = t.split("```")
            if len(parts) >= 3:
                fenced = parts[1].strip()
                lines = fenced.splitlines()
                if lines and lines[0].strip().lower() in {"json", "jsonc"}:
                    lines = lines[1:]
                t = "\n".join(lines).strip()
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            start = t.find("{")
            end = t.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(t[start : end + 1])
                if isinstance(obj, dict):
                    return obj
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None
