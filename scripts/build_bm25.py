from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="data/rag/chunks.parquet")
    parser.add_argument("--output-dir", default="artifacts/bm25")
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2000)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def tokenize(text: str, min_len: int) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= min_len]


def iter_batches(input_path: Path, batch_size: int) -> Iterable[dict[str, list[Any]]]:
    try:
        import pyarrow.dataset as ds
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency: pyarrow. Install with: pip install pyarrow") from e

    dataset = ds.dataset(input_path.as_posix(), format="parquet")
    for batch in dataset.to_batches(batch_size=batch_size):
        yield batch.to_pydict()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_len: list[int] = []
    doc_meta_path = output_dir / "doc_meta.jsonl"
    total_docs = 0
    df_counter: Counter[str] = Counter()

    with doc_meta_path.open("w", encoding="utf-8") as meta_f:
        for batch in iter_batches(input_path, args.batch_size):
            rows = len(next(iter(batch.values()))) if batch else 0
            for i in range(rows):
                chunk_text = batch["chunk_text"][i]
                tokens = tokenize(chunk_text, args.min_token_len)
                if not tokens:
                    continue
                tf = Counter(tokens)
                for term, freq in tf.items():
                    inverted[term].append((total_docs, int(freq)))
                for term in tf.keys():
                    df_counter[term] += 1
                doc_len.append(len(tokens))

                meta = {
                    "doc_id": int(batch["doc_id"][i]),
                    "chunk_id": int(batch["chunk_id"][i]),
                    "hotel_id": batch["hotel_id"][i],
                    "user_id": batch["user_id"][i],
                    "post_date": batch["post_date"][i],
                    "chunk_text": chunk_text,
                    "chunk_char": int(batch["chunk_char"][i]),
                }
                meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                total_docs += 1

    avgdl = sum(doc_len) / len(doc_len) if doc_len else 0.0
    idf = {t: math.log(1 + (total_docs - df + 0.5) / (df + 0.5)) for t, df in df_counter.items()}

    payload = {
        "total_docs": total_docs,
        "avgdl": avgdl,
        "doc_len": doc_len,
        "idf": idf,
        "inverted": inverted,
    }

    with (output_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(payload, f)

    config = {
        "input_path": input_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "min_token_len": args.min_token_len,
        "total_docs": total_docs,
        "avgdl": avgdl,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

