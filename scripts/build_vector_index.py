from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="data/rag/chunks.parquet")
    parser.add_argument("--output-dir", default="artifacts/vector")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_batches(input_path: Path, batch_size: int) -> Iterable[dict[str, list[Any]]]:
    try:
        import pyarrow.dataset as ds
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency: pyarrow. Install with: pip install pyarrow") from e

    dataset = ds.dataset(input_path.as_posix(), format="parquet")
    for batch in dataset.to_batches(batch_size=batch_size):
        yield batch.to_pydict()


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
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    try:
        import faiss
        import numpy as np
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency: faiss or numpy. Install with: pip install faiss-cpu numpy") from e

    encoder = load_encoder(args.model)
    dim = encoder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    meta_path = output_dir / "doc_meta.jsonl"
    total = 0
    with meta_path.open("w", encoding="utf-8") as meta_f:
        for batch in iter_batches(input_path, args.batch_size):
            texts = batch["chunk_text"]
            embeddings = encoder.encode(texts, normalize_embeddings=True, batch_size=args.batch_size)
            embeddings = np.asarray(embeddings, dtype="float32")
            index.add(embeddings)
            for i in range(len(texts)):
                meta = {
                    "doc_id": int(batch["doc_id"][i]),
                    "chunk_id": int(batch["chunk_id"][i]),
                    "hotel_id": batch["hotel_id"][i],
                    "user_id": batch["user_id"][i],
                    "post_date": batch["post_date"][i],
                    "chunk_text": texts[i],
                    "chunk_char": int(batch["chunk_char"][i]),
                }
                meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                total += 1

    faiss.write_index(index, (output_dir / "index.faiss").as_posix())
    config = {
        "input_path": input_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "model": args.model,
        "total_docs": total,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

