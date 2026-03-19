from __future__ import annotations

from datetime import datetime
from typing import Any


class RankerModule:
    def rerank(self, question: str, candidates: list[dict[str, Any]], intent: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not candidates:
            return []
        intent = intent or {}
        constraints_obj = intent.get("constraints")
        constraints: dict[str, Any] = constraints_obj if isinstance(constraints_obj, dict) else {}
        recency_level = str(constraints.get("recency_level") or "none")

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

        ranked = sorted(candidates, key=score_item, reverse=True)
        return ranked
