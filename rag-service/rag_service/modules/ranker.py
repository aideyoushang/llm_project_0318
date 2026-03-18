from __future__ import annotations

from typing import Any


class RankerModule:
    def rerank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return candidates

