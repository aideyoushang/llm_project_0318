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
            return {"answer": "I don't have enough evidence from reviews to answer this question.", "used_refs": [], "claims": []}

        if not self._llm.is_configured():
            return {"answer": "\n".join(contexts[:1]), "used_refs": [0], "claims": []}

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
            return {"answer": "\n".join(contexts[:1]), "used_refs": [0], "claims": []}

        prompt = "\n".join(
            [
                "You are extracting evidence-grounded claims using only the provided review excerpts.",
                "Return ONLY valid minified JSON with this schema:",
                '{"claims":[{"text":string,"ref_ids":number[]}]}',
                "Rules:",
                "- Answer in English.",
                "- Create 2 to 6 claims.",
                "- Each claim text must be a standalone sentence WITHOUT any [id] citations.",
                "- Each claim must cite 1 to 3 ref_ids from the provided evidence ids.",
                "- ref_ids must be integers from the provided evidence ids.",
                "- Do not invent facts not supported by evidence.",
                f"Question: {question}",
                "Evidence:",
                json.dumps(evidence, ensure_ascii=False),
            ]
        )
        try:
            text = self._llm.response_text(prompt, timeout_s=45.0, json_object=True)
        except Exception:
            return {"answer": "\n".join(contexts[:1]), "used_refs": [0], "claims": []}
        payload = self._try_parse_json(text)
        if not payload:
            return {"answer": "\n".join(contexts[:1]), "used_refs": [0], "claims": []}

        claims = self._parse_claims(payload.get("claims"), max_ref_id=len(references) - 1)
        if not claims:
            return {"answer": "\n".join(contexts[:1]), "used_refs": [0], "claims": []}

        used_refs: list[int] = []
        for c in claims:
            for rid in c.get("ref_ids", []):
                if rid not in used_refs:
                    used_refs.append(rid)

        answer = self.build_answer_from_claims(claims)
        if not answer:
            return {"answer": "\n".join(contexts[:1]), "used_refs": [0], "claims": []}
        return {"answer": answer, "used_refs": used_refs, "claims": claims}

    def stream_answer(self, question: str, references: list[dict[str, Any]]):
        if not references or not self._llm.is_configured():
            return
            yield

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
            return
            yield

        prompt = "\n".join(
            [
                "You are answering a question using only the provided review excerpts.",
                "Rules:",
                "- Answer in English.",
                "- Each sentence must include inline citations like [id].",
                "- Do NOT output JSON. Do NOT output markdown fences.",
                f"Question: {question}",
                "Evidence:",
                json.dumps(evidence, ensure_ascii=False),
            ]
        )
        for delta in self._llm.response_text_stream(prompt, timeout_s=120.0):
            yield delta

    def extract_used_refs(self, answer: str) -> list[int]:
        refs = set()
        for m in re.findall(r"\[(\d+)\]", answer):
            try:
                refs.add(int(m))
            except Exception:
                continue
        return sorted(refs)

    def extract_claims_from_answer(self, answer: str, *, max_ref_id: int) -> list[dict[str, Any]]:
        chunks = []
        for part in re.split(r"(?<=[.!?])\s+", answer.strip()):
            s = part.strip()
            if not s:
                continue
            ref_ids = []
            for m in re.findall(r"\[(\d+)\]", s):
                try:
                    rid = int(m)
                except Exception:
                    continue
                if 0 <= rid <= max_ref_id and rid not in ref_ids:
                    ref_ids.append(rid)
            text = re.sub(r"\s*\[\d+\]\s*", " ", s).strip()
            text = re.sub(r"\s{2,}", " ", text).strip()
            if not text or not ref_ids:
                continue
            if text.endswith((".", "!", "?")):
                text = text[:-1].strip()
            chunks.append({"text": text, "ref_ids": ref_ids})
        return chunks

    def build_answer_from_claims(self, claims: list[dict[str, Any]]) -> str:
        lines = []
        for c in claims:
            text = str(c.get("text") or "").strip()
            if not text:
                continue
            ref_ids = c.get("ref_ids") or []
            if not isinstance(ref_ids, list) or not ref_ids:
                continue
            cites = "".join([f"[{int(r)}]" for r in ref_ids if isinstance(r, int)])
            if not cites:
                continue
            line = text.rstrip(". ").strip()
            lines.append(f"{line}. {cites}".strip())
        return " ".join(lines).strip()

    def _parse_claims(self, raw: Any, *, max_ref_id: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            ref_ids_raw = item.get("ref_ids") or []
            if not isinstance(ref_ids_raw, list):
                continue
            ref_ids: list[int] = []
            for r in ref_ids_raw:
                try:
                    rid = int(r)
                except Exception:
                    continue
                if 0 <= rid <= max_ref_id and rid not in ref_ids:
                    ref_ids.append(rid)
            if not ref_ids:
                continue
            out.append({"text": text, "ref_ids": ref_ids})
        return out

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
