from __future__ import annotations

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
