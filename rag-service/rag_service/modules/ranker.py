from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from rag_service.modules.ark_llm import ArkResponsesClient
from rag_service.modules.runtime_config import load_runtime_config


class RankerModule:
    def __init__(self) -> None:
        self._llm = ArkResponsesClient()
        self._last_mode = "heuristic"

    def last_mode(self) -> str:
        return self._last_mode

    def rerank(self, question: str, candidates: list[dict[str, Any]], intent: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not candidates:
            return []
        intent = intent or {}
        constraints_obj = intent.get("constraints")
        constraints: dict[str, Any] = constraints_obj if isinstance(constraints_obj, dict) else {}
        recency_level = str(constraints.get("recency_level") or "none")

        cfg = load_runtime_config()
        llm_enabled = cfg.enable_llm_rerank is True and self._llm.is_configured()

        ranked = self._heuristic_rerank(question, candidates, recency_level=recency_level)
        if llm_enabled:
            llm_ranked = self._llm_rerank(question, ranked[:20], intent=intent)
            if llm_ranked:
                self._last_mode = "llm"
                return llm_ranked + ranked[20:]
        self._last_mode = "heuristic"
        return ranked

    def _heuristic_rerank(self, question: str, candidates: list[dict[str, Any]], recency_level: str) -> list[dict[str, Any]]:
        dates: list[datetime] = []
        for c in candidates:
            meta = c.get("metadata") or {}
            d = meta.get("post_date")
            if isinstance(d, str):
                try:
                    dates.append(datetime.fromisoformat(d.replace("Z", "+00:00")))
                except Exception:
                    pass
        most_recent = max(dates) if dates else None

        def score_item(c: dict[str, Any]) -> float:
            base = float(c.get("score") or 0.0)
            sources = c.get("sources") or {}
            if isinstance(sources, dict):
                base += 0.05 * sum(float(v.get("w") or 0.0) for v in sources.values() if isinstance(v, dict))
            if recency_level in {"clear", "implied"} and most_recent:
                meta = c.get("metadata") or {}
                d = meta.get("post_date")
                if isinstance(d, str):
                    try:
                        dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
                        delta_days = (most_recent - dt).days
                        if delta_days < 0:
                            delta_days = 0
                        base += 0.1 * (1.0 / (1.0 + delta_days / 30.0))
                    except Exception:
                        pass
            return base

        return sorted(candidates, key=score_item, reverse=True)

    def _llm_rerank(self, question: str, candidates: list[dict[str, Any]], intent: dict[str, Any]) -> list[dict[str, Any]] | None:
        items = []
        for i, c in enumerate(candidates):
            meta = c.get("metadata") or {}
            text = str(meta.get("chunk_text") or c.get("text") or "")
            if not text:
                continue
            items.append(
                {
                    "id": i,
                    "doc_id": meta.get("doc_id"),
                    "chunk_id": meta.get("chunk_id"),
                    "post_date": meta.get("post_date"),
                    "text": text[:800],
                }
            )
        if len(items) < 3:
            return None

        constraints_obj = intent.get("constraints")
        constraints: dict[str, Any] = constraints_obj if isinstance(constraints_obj, dict) else {}
        rating_fields = constraints.get("rating_fields") or []
        if not isinstance(rating_fields, list):
            rating_fields = []
        recency_level = str(constraints.get("recency_level") or "none")
        schema = '{"ranked_ids": number[]}'
        prompt = "\n".join(
            [
                "You are a reranker for hotel review retrieval results.",
                "Return ONLY valid minified JSON and nothing else.",
                f"Schema: {schema}",
                "Rules:",
                "- ranked_ids must be a permutation of provided ids.",
                "- Prefer evidence that directly answers the question.",
                "- Prefer content about the requested aspects if present.",
                "- Prefer more recent stays if recency is implied/clear.",
                f"Question: {question}",
                f"Aspects: {', '.join([str(x) for x in rating_fields if isinstance(x, str)])}",
                f"Recency: {recency_level}",
                "Candidates:",
                json.dumps(items, ensure_ascii=False),
            ]
        )
        try:
            text = self._llm.response_text(prompt, timeout_s=45.0)
        except Exception:
            return None
        payload = self._try_parse_json(text)
        if not payload:
            return None
        ranked_ids = payload.get("ranked_ids")
        if not isinstance(ranked_ids, list):
            return None
        order: list[int] = []
        seen: set[int] = set()
        for x in ranked_ids:
            try:
                idx = int(x)
            except Exception:
                continue
            if 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                order.append(idx)
        if len(order) < 3:
            return None
        return [candidates[i] for i in order] + [c for j, c in enumerate(candidates) if j not in seen]

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
