from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/tripadvisor")
    parser.add_argument("--output-path", default="data/rag/hotel_summaries.parquet")
    parser.add_argument("--output-index-dir", default="artifacts/summary_vector")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-reviews-per-hotel", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=2000)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_batches(input_dir: Path, batch_size: int) -> Iterable[dict[str, list[Any]]]:
    try:
        import pyarrow.dataset as ds
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency: pyarrow. Install with: pip install pyarrow") from e

    dataset = ds.dataset(input_dir.as_posix(), format="parquet")
    for batch in dataset.to_batches(batch_size=batch_size):
        yield batch.to_pydict()


def normalize_review(title: Any, text: Any, review: Any) -> str:
    if isinstance(review, str) and review.strip():
        return review.strip()
    parts: list[str] = []
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())
    return "\n".join(parts)


def normalize_lang(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        if v.startswith("__label__"):
            return v.replace("__label__", "")
        return v
    return str(value)


def build_summary(reviews: list[str], max_chars: int) -> str:
    reviews = sorted(reviews, key=len, reverse=True)
    buf: list[str] = []
    size = 0
    for review in reviews:
        if size + len(review) + 1 > max_chars:
            break
        buf.append(review)
        size += len(review) + 1
    return "\n".join(buf)


def load_encoder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install with: pip install sentence-transformers"
        ) from e
    return SentenceTransformer(model_name)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    output_index_dir = Path(args.output_index_dir)
    ensure_dir(output_path.parent)
    ensure_dir(output_index_dir)

    hotel_reviews: dict[str, list[str]] = defaultdict(list)
    hotel_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    counters = {
        "seen_rows": 0,
        "kept_reviews": 0,
        "skipped_lang": 0,
        "skipped_empty_review": 0,
    }

    for batch in iter_batches(input_dir, args.batch_size):
        rows = len(next(iter(batch.values()))) if batch else 0
        for i in range(rows):
            counters["seen_rows"] += 1
            record_lang = normalize_lang(batch.get("lang", [None])[i])
            if args.lang and record_lang not in (None, args.lang):
                counters["skipped_lang"] += 1
                continue
            hotel_id = str(batch["hotel_id"][i])
            review = normalize_review(batch.get("title", [None])[i], batch.get("text", [None])[i], batch.get("review", [None])[i])
            if not review:
                counters["skipped_empty_review"] += 1
                continue
            hotel_reviews[hotel_id].append(review)
            counters["kept_reviews"] += 1
            if "overall" in batch:
                overall = batch["overall"][i]
                if overall is not None:
                    hotel_stat = hotel_stats[hotel_id]
                    hotel_stat.setdefault("overall_sum", 0.0)
                    hotel_stat.setdefault("overall_cnt", 0)
                    hotel_stat["overall_sum"] += float(overall)
                    hotel_stat["overall_cnt"] += 1

    rows: list[dict[str, Any]] = []
    for hotel_id, reviews in hotel_reviews.items():
        reviews = reviews[: args.max_reviews_per_hotel]
        summary = build_summary(reviews, args.max_chars)
        stats = hotel_stats.get(hotel_id, {})
        avg_overall = None
        if stats.get("overall_cnt"):
            avg_overall = stats["overall_sum"] / max(stats["overall_cnt"], 1)
            avg_overall = round(avg_overall, 2)
        rows.append(
            {
                "hotel_id": hotel_id,
                "summary": summary,
                "review_count": len(reviews),
                "avg_overall": avg_overall,
            }
        )

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency: pyarrow. Install with: pip install pyarrow") from e

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path.as_posix(), compression="zstd")

    try:
        import faiss
        import numpy as np
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency: faiss or numpy. Install with: pip install faiss-cpu numpy") from e

    encoder = load_encoder(args.model)
    dim = encoder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    meta_path = output_index_dir / "doc_meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as meta_f:
        batch_size = 128
        total = 0
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            texts = [r["summary"] for r in batch_rows]
            emb = encoder.encode(texts, normalize_embeddings=True, batch_size=batch_size)
            emb = np.asarray(emb, dtype="float32")
            index.add(emb)
            for r in batch_rows:
                meta_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1

    faiss.write_index(index, (output_index_dir / "index.faiss").as_posix())
    config = {
        "input_dir": input_dir.as_posix(),
        "output_path": output_path.as_posix(),
        "output_index_dir": output_index_dir.as_posix(),
        "model": args.model,
        "total_docs": len(rows),
        "lang": args.lang,
        "stats": {
            "seen_rows": int(counters["seen_rows"]),
            "kept_reviews": int(counters["kept_reviews"]),
            "skipped_lang": int(counters["skipped_lang"]),
            "skipped_empty_review": int(counters["skipped_empty_review"]),
            "unique_hotels": int(len(rows)),
        },
    }
    (output_index_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
