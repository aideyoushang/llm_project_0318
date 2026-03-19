from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from heapq import nlargest
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievedItem:
    key: str
    text: str
    metadata: dict[str, Any]
    source: str
    rank: int


class RetrieverModule:
    def __init__(self) -> None:
        self._loaded = False
        self._bm25 = None
        self._bm25_meta: list[dict[str, Any]] | None = None
        self._vector_index = None
        self._vector_meta: list[dict[str, Any]] | None = None
        self._summary_index = None
        self._summary_meta: list[dict[str, Any]] | None = None
        self._encoder = None

    def retrieve(self, question: str, intent: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self._ensure_loaded()
        intent = intent or {}
        subqueries = intent.get("subqueries") or [{"q": question, "weight": 1.0}]
        planned: list[tuple[str, float]] = []
        for sq in subqueries:
            q = str(sq.get("q") or "").strip()
            if not q:
                continue
            try:
                w = float(sq.get("weight") or 1.0)
            except Exception:
                w = 1.0
            if w <= 0:
                continue
            planned.append((q, w))
        if not planned:
            planned = [(question, 1.0)]

        rrf_k = 60
        fused: dict[str, dict[str, Any]] = {}

        def add_ranked(items: list[RetrievedItem], route_weight: float, query_weight: float) -> None:
            w = route_weight * query_weight
            for item in items:
                score = w / (rrf_k + item.rank)
                slot = fused.get(item.key)
                if slot is None:
                    slot = {
                        "text": item.text,
                        "metadata": dict(item.metadata),
                        "score": 0.0,
                        "sources": {},
                    }
                    fused[item.key] = slot
                slot["score"] += score
                slot["sources"][item.source] = {"rank": item.rank, "w": w}

        for q, q_weight in planned:
            add_ranked(self._retrieve_bm25(q, top_k=30), route_weight=1.0, query_weight=q_weight)
            add_ranked(self._retrieve_vector(q, top_k=30), route_weight=1.2, query_weight=q_weight)
            add_ranked(self._retrieve_summary(q, top_k=6), route_weight=0.6, query_weight=q_weight)

        top = nlargest(8, fused.items(), key=lambda kv: float(kv[1]["score"]))
        out: list[dict[str, Any]] = []
        for _, payload in top:
            out.append(
                {
                    "text": payload["text"],
                    "metadata": payload["metadata"],
                    "score": float(payload["score"]),
                    "sources": payload.get("sources", {}),
                }
            )
        return out

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        root = self._project_root()

        bm25_dir = root / "artifacts" / "bm25"
        bm25_pkl = bm25_dir / "bm25.pkl"
        bm25_meta = bm25_dir / "doc_meta.jsonl"
        if bm25_pkl.exists() and bm25_meta.exists():
            try:
                with bm25_pkl.open("rb") as f:
                    self._bm25 = pickle.load(f)
                self._bm25_meta = self._read_jsonl(bm25_meta)
            except Exception:
                self._bm25 = None
                self._bm25_meta = None

        vector_dir = root / "artifacts" / "vector"
        vector_index_path = vector_dir / "index.faiss"
        vector_meta = vector_dir / "doc_meta.jsonl"
        vector_cfg = vector_dir / "config.json"
        if vector_index_path.exists() and vector_meta.exists():
            try:
                import faiss  # type: ignore

                self._vector_index = faiss.read_index(vector_index_path.as_posix())
                self._vector_meta = self._read_jsonl(vector_meta)
                model = None
                if vector_cfg.exists():
                    model = json.loads(vector_cfg.read_text(encoding="utf-8")).get("model")
                if model:
                    self._encoder = self._load_encoder(str(model))
            except Exception:
                self._vector_index = None
                self._vector_meta = None

        summary_dir = root / "artifacts" / "summary_vector"
        summary_index_path = summary_dir / "index.faiss"
        summary_meta = summary_dir / "doc_meta.jsonl"
        summary_cfg = summary_dir / "config.json"
        if summary_index_path.exists() and summary_meta.exists():
            try:
                import faiss  # type: ignore

                self._summary_index = faiss.read_index(summary_index_path.as_posix())
                self._summary_meta = self._read_jsonl(summary_meta)
                if self._encoder is None and summary_cfg.exists():
                    model = json.loads(summary_cfg.read_text(encoding="utf-8")).get("model")
                    if model:
                        self._encoder = self._load_encoder(str(model))
            except Exception:
                self._summary_index = None
                self._summary_meta = None

        self._loaded = True

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _load_encoder(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Missing dependency: sentence-transformers. Install with: pip install sentence-transformers"
            ) from e
        return SentenceTransformer(model_name)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _retrieve_bm25(self, query: str, top_k: int) -> list[RetrievedItem]:
        if not self._bm25 or not self._bm25_meta:
            return []
        payload = self._bm25
        inverted = payload.get("inverted") or {}
        idf = payload.get("idf") or {}
        doc_len = payload.get("doc_len") or []
        avgdl = float(payload.get("avgdl") or 0.0)
        if avgdl <= 0 or not doc_len:
            return []

        k1 = 1.2
        b = 0.75
        scores: dict[int, float] = {}
        terms = self._tokenize(query)
        if not terms:
            return []
        for term in terms:
            postings = inverted.get(term)
            if not postings:
                continue
            term_idf = float(idf.get(term) or 0.0)
            if term_idf <= 0:
                continue
            for doc_idx, tf in postings:
                dl = float(doc_len[doc_idx]) if doc_idx < len(doc_len) else 0.0
                denom = float(tf) + k1 * (1.0 - b + b * (dl / avgdl))
                if denom <= 0:
                    continue
                s = term_idf * (float(tf) * (k1 + 1.0) / denom)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + s

        if not scores:
            return []

        top = nlargest(top_k, scores.items(), key=lambda kv: kv[1])
        out: list[RetrievedItem] = []
        for rank, (doc_idx, _) in enumerate(top, start=1):
            if doc_idx >= len(self._bm25_meta):
                continue
            meta = dict(self._bm25_meta[doc_idx])
            chunk_text = str(meta.get("chunk_text") or "")
            chunk_id = meta.get("chunk_id")
            key = f"chunk:{chunk_id}" if chunk_id is not None else f"bm25:{doc_idx}"
            meta["source"] = "bm25"
            out.append(RetrievedItem(key=key, text=chunk_text, metadata=meta, source="bm25", rank=rank))
        return out

    def _encode_query(self, query: str):
        if self._encoder is None:
            return None
        emb = self._encoder.encode([query], normalize_embeddings=True)
        try:
            import numpy as np  # type: ignore

            return np.asarray(emb, dtype="float32")
        except ModuleNotFoundError as e:
            raise RuntimeError("Missing dependency: numpy. Install with: pip install numpy") from e

    def _retrieve_vector(self, query: str, top_k: int) -> list[RetrievedItem]:
        if self._vector_index is None or not self._vector_meta:
            return []
        emb = self._encode_query(query)
        if emb is None:
            return []
        distances, indices = self._vector_index.search(emb, top_k)
        out: list[RetrievedItem] = []
        for rank, idx in enumerate(indices[0].tolist(), start=1):
            if idx < 0 or idx >= len(self._vector_meta):
                continue
            meta = dict(self._vector_meta[idx])
            chunk_text = str(meta.get("chunk_text") or "")
            chunk_id = meta.get("chunk_id")
            key = f"chunk:{chunk_id}" if chunk_id is not None else f"vector:{idx}"
            meta["source"] = "vector"
            meta["vector_score"] = float(distances[0][rank - 1])
            out.append(RetrievedItem(key=key, text=chunk_text, metadata=meta, source="vector", rank=rank))
        return out

    def _retrieve_summary(self, query: str, top_k: int) -> list[RetrievedItem]:
        if self._summary_index is None or not self._summary_meta:
            return []
        emb = self._encode_query(query)
        if emb is None:
            return []
        distances, indices = self._summary_index.search(emb, top_k)
        out: list[RetrievedItem] = []
        for rank, idx in enumerate(indices[0].tolist(), start=1):
            if idx < 0 or idx >= len(self._summary_meta):
                continue
            meta = dict(self._summary_meta[idx])
            summary = str(meta.get("summary") or "")
            group_id = meta.get("group_id")
            key = f"summary:{group_id}" if group_id is not None else f"summary:{idx}"
            meta["source"] = "summary"
            meta["vector_score"] = float(distances[0][rank - 1])
            out.append(RetrievedItem(key=key, text=summary, metadata=meta, source="summary", rank=rank))
        return out
