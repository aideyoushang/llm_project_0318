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
        contexts = []
        references = []
        if intent.get("use_retrieval", True):
            candidates = self.retriever.retrieve(question, intent=intent)
            ranked = self.ranker.rerank(question, candidates)
            contexts = [c.get("text", "") for c in ranked]
            references = [c.get("metadata", {}) for c in ranked]

        answer = self.generator.generate(question=question, contexts=contexts)
        return {"answer": answer, "references": references, "intent": intent}
