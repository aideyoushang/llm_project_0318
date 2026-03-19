from __future__ import annotations

import re
from typing import Any


class IntentModule:
    def classify(self, question: str) -> dict[str, Any]:
        q = question.strip()
        q_lower = q.lower()

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

    def _is_smalltalk(self, q_lower: str) -> bool:
        greetings = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "你好",
            "在吗",
            "你是谁",
            "你能做什么",
        ]
        if any(g in q_lower for g in greetings):
            return True
        if len(q_lower) <= 8 and q_lower in {"hi", "hello", "hey", "yo"}:
            return True
        return False

    def _detect_recency_level(self, q_lower: str) -> str:
        if re.search(r"\b(20\d{2})\b", q_lower):
            return "clear"
        if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", q_lower):
            return "clear"
        if re.search(r"\b(this year|last year|this month|last month|this week|last week)\b", q_lower):
            return "clear"
        if re.search(r"\b(recent|recently|latest|newest|nowadays|these days)\b", q_lower):
            return "implied"
        if re.search(r"(最近|近期|最新|今年|本月|上月|本周|上周)", q_lower):
            return "implied"
        return "none"

    def _detect_rating_fields(self, q_lower: str) -> list[str]:
        mapping: list[tuple[str, list[str]]] = [
            ("cleanliness", ["clean", "dirty", "hygiene", "卫生", "干净", "脏"]),
            ("value", ["value", "worth", "price", "expensive", "cheap", "性价比", "价格", "贵", "便宜"]),
            ("location", ["location", "near", "close to", "distance", "交通", "位置", "附近", "离"]),
            ("rooms", ["room", "rooms", "bed", "suite", "房间", "床", "套房"]),
            ("sleep_quality", ["sleep", "quiet", "noise", "noisy", "睡眠", "安静", "噪音", "吵"]),
            ("overall", ["overall", "rating", "score", "recommend", "推荐", "评分", "总评"]),
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
