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
            llm_result, llm_err = self._classify_with_llm(q)
            if llm_result is not None:
                llm_result["intent_source"] = "ark"
                return llm_result

        rule_result = self._classify_with_rules(q, q_lower)
        rule_result["intent_source"] = "rule"
        if use_llm:
            rule_result["intent_fallback"] = "ark_failed"
            if llm_err:
                rule_result["intent_fallback_detail"] = llm_err
        return rule_result

    def _use_llm_intent(self) -> bool:
        mode = (load_runtime_config().intent_mode or "").strip().lower()
        if mode in {"llm", "ark"}:
            return self._llm.is_configured()
        return False

    def _classify_with_llm(self, question: str) -> tuple[dict[str, Any] | None, str | None]:
        allowed_fields = ["overall", "cleanliness", "value", "location", "rooms", "sleep_quality"]
        is_zh = self._looks_like_chinese(question)
        schema = (
            "{"
            '"use_retrieval": boolean,'
            '"intent_type": "chat"|"domain_qa"|"analytics",'
            '"query": string,'
            '"constraints": {"recency_level":"clear"|"implied"|"none","rating_fields": string[]},'
            '"subqueries": [{"q": string, "weight": number, "type": string}]'
            "}"
        )

        example = (
            '{"use_retrieval":true,"intent_type":"domain_qa","query":"How is the hotel location recently?",'
            '"constraints":{"recency_level":"implied","rating_fields":["location"]},'
            '"subqueries":[{"q":"recent hotel location reviews","weight":1.0,"type":"retrieval"},'
            '{"q":"hotel location ratings in recent reviews","weight":0.9,"type":"rating"}]}'
        )

        user_prompt = "\n".join(
            [
                "You are a query understanding module for hotel review RAG.",
                "Generate subqueries in English for retrieval over an English review corpus.",
                "If the user question is not in English, still generate English subqueries.",
                "You MUST output a single line of minified JSON and nothing else.",
                "Do NOT wrap in markdown fences. Do NOT add explanations.",
                "The output must start with '{' and end with '}'.",
                f"Schema: {schema}",
                f"Allowed rating_fields values: {allowed_fields}",
                "Rules:",
                "- If user is greeting/chitchat, set use_retrieval=false and intent_type=chat.",
                "- Otherwise use_retrieval=true and intent_type=domain_qa.",
                "- subqueries must be 1 to 3 items, weights in (0,1].",
                "- subqueries must be in English.",
                f"Example output: {example}",
                f"User question: {question}",
                "Note: question_language=zh" if is_zh else "Note: question_language=en",
            ]
        )
        try:
            text = self._llm.response_text(user_prompt, timeout_s=30.0)
        except Exception as e:
            msg = str(e)
            msg = msg.replace("\n", " ").strip()
            if len(msg) > 300:
                msg = msg[:300]
            return None, msg

        payload = self._try_parse_json(text)
        if payload is None:
            repair_prompt = "\n".join(
                [
                    "Convert the following text into a single line of valid minified JSON that matches the exact schema.",
                    "Output ONLY JSON. No markdown. No extra text.",
                    f"Schema: {schema}",
                    f"Allowed rating_fields values: {allowed_fields}",
                    "Text to convert:",
                    text,
                ]
            )
            try:
                repaired = self._llm.response_text(repair_prompt, timeout_s=30.0)
            except Exception:
                repaired = ""
            payload = self._try_parse_json(repaired)
        if payload is None:
            return None, "Ark response is not valid JSON"
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
            }, None
        except Exception as e:
            msg = str(e).replace("\n", " ").strip()
            if len(msg) > 300:
                msg = msg[:300]
            return None, msg

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
