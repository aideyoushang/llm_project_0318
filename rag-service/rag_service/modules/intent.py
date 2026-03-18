from __future__ import annotations

from typing import Any


class IntentModule:
    def classify(self, question: str) -> dict[str, Any]:
        return {"use_retrieval": True, "query": question}

