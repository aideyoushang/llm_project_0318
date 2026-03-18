from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/tripadvisor")
    parser.add_argument("--output-path", default="data/rag/chunks.parquet")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2000)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return str(value)


def normalize_text(title: Any, text: Any, review: Any) -> str:
    if isinstance(review, str) and review.strip():
        return review.strip()
    parts: list[str] = []
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())
    return "\n".join(parts)


def sentence_split(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def build_chunks(text: str, max_chars: int, overlap: int, min_chars: int) -> list[str]:
    sentences = sentence_split(text)
    if not sentences:
        return []
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sent in sentences:
        if size + len(sent) + 1 > max_chars and current:
            chunk = " ".join(current).strip()
            if len(chunk) >= min_chars:
                chunks.append(chunk)
            overlap_text = chunk[-overlap:] if overlap > 0 else ""
            current = [overlap_text] if overlap_text else []
            size = len(overlap_text)
        current.append(sent)
        size += len(sent) + 1
    if current:
        chunk = " ".join(current).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
    return chunks


def iter_batches(input_dir: Path, batch_size: int) -> Iterable[dict[str, list[Any]]]:
    try:
        import pyarrow.dataset as ds
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency: pyarrow. Install with: pip install pyarrow") from e

    dataset = ds.dataset(input_dir.as_posix(), format="parquet")
    for batch in dataset.to_batches(batch_size=batch_size):
        yield batch.to_pydict()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency: pyarrow. Install with: pip install pyarrow") from e

    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []
    chunk_id = 0
    doc_id = 0

    for batch in iter_batches(input_dir, args.batch_size):
        rows = len(next(iter(batch.values()))) if batch else 0
        for i in range(rows):
            record = {k: v[i] for k, v in batch.items()}
            if args.lang and record.get("lang") not in (None, args.lang):
                continue
            text = normalize_text(record.get("title"), record.get("text"), record.get("review"))
            if not text:
                continue
            chunks = build_chunks(text, args.max_chars, args.overlap, args.min_chars)
            if not chunks:
                continue
            for idx, chunk in enumerate(chunks):
                buffer.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "hotel_id": record.get("hotel_id"),
                        "user_id": record.get("user_id"),
                        "post_date": to_iso(record.get("post_date")),
                        "stay_year": record.get("stay_year"),
                        "overall": record.get("overall"),
                        "cleanliness": record.get("cleanliness"),
                        "value": record.get("value"),
                        "location": record.get("location"),
                        "rooms": record.get("rooms"),
                        "sleep_quality": record.get("sleep_quality"),
                        "freq": record.get("freq"),
                        "char": record.get("char"),
                        "review": text,
                        "chunk_text": chunk,
                        "chunk_char": len(chunk),
                    }
                )
                chunk_id += 1
            doc_id += 1

        if len(buffer) >= 5000:
            table = pa.Table.from_pylist(buffer)
            if writer is None:
                writer = pq.ParquetWriter(output_path.as_posix(), table.schema, compression="zstd")
            writer.write_table(table)
            buffer = []

    if buffer:
        table = pa.Table.from_pylist(buffer)
        if writer is None:
            writer = pq.ParquetWriter(output_path.as_posix(), table.schema, compression="zstd")
        writer.write_table(table)

    if writer is not None:
        writer.close()

    meta = {
        "input_dir": input_dir.as_posix(),
        "output_path": output_path.as_posix(),
        "lang": args.lang,
        "max_chars": args.max_chars,
        "overlap": args.overlap,
        "min_chars": args.min_chars,
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

