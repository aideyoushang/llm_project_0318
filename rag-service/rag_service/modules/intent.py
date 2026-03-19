from __future__ import annotations

import json
import re
from typing import Any

from rag_service.modules.ark_llm import ArkResponsesClient
from rag_service.modules.runtime_config import load_runtime_config


class IntentModule:
    def __init__(self) -> None:
        self._llm = ArkResponsesClient()

    def classify(self, question: str) -> dict[str, Any]:
        q = question.strip()
        q_lower = q.lower()
        cfg = load_runtime_config()
        if cfg.force_english is True and self._looks_like_chinese(q):
            return {
                "use_retrieval": False,
                "intent_type": "language_mismatch",
                "query": q,
                "constraints": {"language": "en"},
                "subqueries": [{"q": q, "weight": 1.0, "type": "original"}],
                "intent_source": "rule",
            }

        use_llm = self._use_llm_intent()
        if use_llm:
            llm_result = self._classify_with_llm(q)
            if llm_result is not None:
                llm_result["intent_source"] = "ark"
                return llm_result

        rule_result = self._classify_with_rules(q, q_lower)
        rule_result["intent_source"] = "rule"
        if use_llm:
            rule_result["intent_fallback"] = "ark_failed"
        return rule_result

    def _use_llm_intent(self) -> bool:
        mode = (load_runtime_config().intent_mode or "").strip().lower()
        if mode in {"llm", "ark"}:
            return self._llm.is_configured()
        return False

    def _classify_with_llm(self, question: str) -> dict[str, Any] | None:
        allowed_fields = ["overall", "cleanliness", "value", "location", "rooms", "sleep_quality"]
        user_prompt = "\n".join(
            [
                "You are a query understanding module for hotel review RAG.",
                "All user questions are in English. Respond in English.",
                "Return ONLY valid minified JSON with this schema:",
                "{"
                '"use_retrieval": boolean,'
                '"intent_type": "chat"|"domain_qa"|"analytics",'
                '"query": string,'
                '"constraints": {"recency_level":"clear"|"implied"|"none","rating_fields": string[]},'
                '"subqueries": [{"q": string, "weight": number, "type": string}]'
                "}",
                f"Allowed rating_fields values: {allowed_fields}",
                "Rules:",
                "- If user is greeting/chitchat, set use_retrieval=false and intent_type=chat.",
                "- Otherwise use_retrieval=true and intent_type=domain_qa.",
                "- subqueries must be 1 to 3 items, weights in (0,1].",
                f"User question: {question}",
            ]
        )
        try:
            text = self._llm.response_text(user_prompt, timeout_s=30.0)
        except Exception:
            return None
        payload = self._try_parse_json(text)
        if payload is None:
            return None
        try:
            use_retrieval = bool(payload.get("use_retrieval"))
            intent_type = str(payload.get("intent_type") or "domain_qa")
            constraints_obj = payload.get("constraints")
            constraints: dict[str, Any] = constraints_obj if isinstance(constraints_obj, dict) else {}
            recency_level = str(constraints.get("recency_level") or "none")
            rating_fields_raw = constraints.get("rating_fields") or []
            if not isinstance(rating_fields_raw, list):
                rating_fields_raw = []
            rating_fields: list[str] = []
            for f in rating_fields_raw:
                if isinstance(f, str) and f in allowed_fields:
                    rating_fields.append(f)
            if not rating_fields:
                rating_fields = ["overall"]

            subqueries_raw = payload.get("subqueries") or []
            subqueries: list[dict[str, Any]] = []
            if isinstance(subqueries_raw, list):
                for item in subqueries_raw:
                    if not isinstance(item, dict):
                        continue
                    q = str(item.get("q") or "").strip()
                    if not q:
                        continue
                    try:
                        w = float(item.get("weight") or 1.0)
                    except Exception:
                        w = 1.0
                    if w <= 0:
                        continue
                    if w > 1:
                        w = 1.0
                    subqueries.append({"q": q, "weight": w, "type": str(item.get("type") or "llm")})
            if not subqueries:
                subqueries = [{"q": question, "weight": 1.0, "type": "original"}]
            subqueries = subqueries[:3]

            return {
                "use_retrieval": use_retrieval,
                "intent_type": intent_type,
                "query": question,
                "constraints": {"recency_level": recency_level, "rating_fields": rating_fields},
                "subqueries": subqueries,
            }
        except Exception:
            return None

    def _classify_with_rules(self, q: str, q_lower: str) -> dict[str, Any]:
        intent_type = "domain_qa"
        use_retrieval = True

        if self._is_smalltalk(q_lower):
            intent_type = "chat"
            use_retrieval = False

        recency_level = self._detect_recency_level(q_lower)
        rating_fields = self._detect_rating_fields(q_lower)
        constraints = {
            "recency_level": recency_level,
            "rating_fields": rating_fields,
        }

        subqueries = self._expand_queries(q, rating_fields, recency_level)

        return {
            "use_retrieval": use_retrieval,
            "intent_type": intent_type,
            "query": q,
            "constraints": constraints,
            "subqueries": subqueries,
        }

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        t = text.strip()
        if not t:
            return None
        if t.startswith("```"):
            t = t.strip("`").strip()
        try:
            obj = json.loads(t)
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

    def _is_smalltalk(self, q_lower: str) -> bool:
        greetings = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        if any(g in q_lower for g in greetings):
            return True
        if len(q_lower) <= 8 and q_lower in {"hi", "hello", "hey", "yo"}:
            return True
        return False

    def _looks_like_chinese(self, text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def _detect_recency_level(self, q_lower: str) -> str:
        if re.search(r"\b(20\d{2})\b", q_lower):
            return "clear"
        if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", q_lower):
            return "clear"
        if re.search(r"\b(this year|last year|this month|last month|this week|last week)\b", q_lower):
            return "clear"
        if re.search(r"\b(recent|recently|latest|newest|nowadays|these days)\b", q_lower):
            return "implied"
        return "none"

    def _detect_rating_fields(self, q_lower: str) -> list[str]:
        mapping: list[tuple[str, list[str]]] = [
            ("cleanliness", ["clean", "dirty", "hygiene"]),
            ("value", ["value", "worth", "price", "expensive", "cheap"]),
            ("location", ["location", "near", "close to", "distance"]),
            ("rooms", ["room", "rooms", "bed", "suite"]),
            ("sleep_quality", ["sleep", "quiet", "noise", "noisy"]),
            ("overall", ["overall", "rating", "score", "recommend"]),
        ]
        fields: list[str] = []
        for field, keywords in mapping:
            if any(k in q_lower for k in keywords):
                fields.append(field)
        if not fields:
            fields = ["overall"]
        seen: set[str] = set()
        deduped: list[str] = []
        for f in fields:
            if f not in seen:
                seen.add(f)
                deduped.append(f)
        return deduped

    def _expand_queries(self, question: str, rating_fields: list[str], recency_level: str) -> list[dict[str, Any]]:
        qs: list[dict[str, Any]] = []
        qs.append({"q": question, "weight": 1.0, "type": "original"})

        if rating_fields:
            aspect_hint = ", ".join(rating_fields[:3])
            qs.append({"q": f"{question} Focus on aspects: {aspect_hint}.", "weight": 0.7, "type": "aspect_hint"})

        if recency_level in {"clear", "implied"}:
            qs.append({"q": f"{question} Recent information preferred.", "weight": 0.6, "type": "recency_hint"})

        return qs[:3]
