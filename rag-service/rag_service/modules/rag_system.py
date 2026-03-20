from __future__ import annotations

import json
from typing import Any

from rag_service.modules.intent import IntentModule
from rag_service.modules.retriever import RetrieverModule
from rag_service.modules.ranker import RankerModule
from rag_service.modules.generator import GeneratorModule


class RagSystem:
    def __init__(self) -> None:
        self.intent = IntentModule()
        self.retriever = RetrieverModule()
        self.ranker = RankerModule()
        self.generator = GeneratorModule()

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question") or "").strip()
        if not question:
            return {"answer": "", "references": [], "intent": {"intent_type": "empty", "use_retrieval": False}}

        intent = self.intent.classify(question)
        if bool(payload.get("intent_only")):
            return {"answer": "", "references": [], "intent": intent}
        if bool(payload.get("debug_retriever")):
            self.retriever._ensure_loaded()
            return {"answer": "", "references": [], "intent": intent, "retriever_status": self.retriever.status()}
        if intent.get("intent_type") == "language_mismatch":
            return {
                "answer": "Please ask your question in English so I can retrieve evidence from the English review corpus.",
                "references": [],
                "intent": intent,
            }
        contexts = []
        references = []
        if intent.get("use_retrieval", True):
            candidates = self.retriever.retrieve(question, intent=intent)
            ranked = self.ranker.rerank(question, candidates, intent=intent)
            contexts = [c.get("text", "") for c in ranked]
            references = []
            for c in ranked:
                meta = dict(c.get("metadata", {}) or {})
                if "sources" not in meta and isinstance(c.get("sources"), dict):
                    meta["sources"] = c.get("sources")
                if "score" not in meta and c.get("score") is not None:
                    try:
                        meta["score"] = float(c.get("score"))
                    except Exception:
                        pass
                references.append(meta)

        gen = self.generator.generate(question=question, contexts=contexts, references=references)
        answer = str(gen.get("answer") or "")
        used_refs = gen.get("used_refs") or []
        if used_refs:
            filtered = []
            for idx in used_refs:
                if isinstance(idx, int) and 0 <= idx < len(references):
                    filtered.append(references[idx])
            references = filtered if filtered else references
        return {"answer": answer, "references": references, "intent": intent}

    def chat_stream(self, payload: dict[str, Any]):
        question = str(payload.get("question") or "").strip()
        if not question:
            yield self._sse({"type": "answer_chunk", "content": ""})
            yield self._sse({"type": "references", "content": []})
            yield self._sse({"type": "intent", "content": {"intent_type": "empty", "use_retrieval": False}})
            yield self._sse("[DONE]", raw=True)
            return

        yield self._sse({"type": "stage", "content": "intent"})
        intent = self.intent.classify(question)
        if bool(payload.get("intent_only")):
            yield self._sse({"type": "intent", "content": intent})
            yield self._sse("[DONE]", raw=True)
            return
        if bool(payload.get("debug_retriever")):
            self.retriever._ensure_loaded()
            yield self._sse({"type": "intent", "content": intent})
            yield self._sse({"type": "retriever_status", "content": self.retriever.status()})
            yield self._sse("[DONE]", raw=True)
            return
        if intent.get("intent_type") == "language_mismatch":
            yield self._sse({"type": "answer_chunk", "content": "Please ask your question in English so I can retrieve evidence from the English review corpus."})
            yield self._sse({"type": "references", "content": []})
            yield self._sse({"type": "intent", "content": intent})
            yield self._sse("[DONE]", raw=True)
            return

        references: list[dict[str, Any]] = []
        if intent.get("use_retrieval", True):
            yield self._sse({"type": "stage", "content": "retrieve"})
            candidates = self.retriever.retrieve(question, intent=intent)
            yield self._sse({"type": "stage", "content": "rerank"})
            ranked = self.ranker.rerank(question, candidates, intent=intent)
            for c in ranked:
                meta = dict(c.get("metadata", {}) or {})
                if "sources" not in meta and isinstance(c.get("sources"), dict):
                    meta["sources"] = c.get("sources")
                if "score" not in meta and c.get("score") is not None:
                    try:
                        meta["score"] = float(c.get("score"))
                    except Exception:
                        pass
                references.append(meta)

        answer_parts: list[str] = []
        any_streamed = False
        try:
            yield self._sse({"type": "stage", "content": "generate"})
            for delta in self.generator.stream_answer(question=question, references=references):
                any_streamed = True
                answer_parts.append(delta)
                yield self._sse({"type": "answer_chunk", "content": delta})
        except Exception:
            any_streamed = False

        if not any_streamed:
            yield self._sse({"type": "stage", "content": "generate_fallback"})
            contexts = [str(r.get("chunk_text") or "") for r in references if str(r.get("chunk_text") or "")]
            gen = self.generator.generate(question=question, contexts=contexts, references=references)
            answer = str(gen.get("answer") or "")
            used_refs = gen.get("used_refs") or []
            if used_refs:
                filtered = []
                for idx in used_refs:
                    if isinstance(idx, int) and 0 <= idx < len(references):
                        filtered.append(references[idx])
                references = filtered if filtered else references
            chunk_size = 200
            for i in range(0, len(answer), chunk_size):
                yield self._sse({"type": "answer_chunk", "content": answer[i : i + chunk_size]})
        else:
            full_answer = "".join(answer_parts)
            used_refs = self.generator.extract_used_refs(full_answer)
            if used_refs:
                filtered = []
                for idx in used_refs:
                    if 0 <= idx < len(references):
                        filtered.append(references[idx])
                references = filtered if filtered else references

        yield self._sse({"type": "stage", "content": "finalize"})
        yield self._sse({"type": "references", "content": references})
        yield self._sse({"type": "intent", "content": intent})
        yield self._sse("[DONE]", raw=True)

    def _sse(self, payload: Any, *, raw: bool = False) -> str:
        if raw:
            return f"data: {payload}\n\n"
        data = json.dumps(payload, ensure_ascii=False)
        return f"data: {data}\n\n"
