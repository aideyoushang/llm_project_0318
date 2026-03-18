from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Callable


@dataclass(frozen=True)
class DownloadResult:
    dataset_name: str
    split: str
    output_dir: str
    output_format: str
    files: list[str]
    num_rows: int
    columns: list[str]
    created_at_utc: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_batches(dataset: Any, batch_size: int) -> Iterable[dict[str, list[Any]]]:
    for batch in dataset.iter(batch_size=batch_size):
        yield batch


def json_default() -> Callable[[Any], Any]:
    pd_timestamp: type | None = None
    try:
        import pandas as pd  # type: ignore

        pd_timestamp = pd.Timestamp  # type: ignore[attr-defined]
    except Exception:
        pd_timestamp = None

    np = None
    try:
        import numpy as _np  # type: ignore

        np = _np
    except Exception:
        np = None

    pa = None
    try:
        import pyarrow as _pa  # type: ignore

        pa = _pa
    except Exception:
        pa = None

    def _default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()

        if pd_timestamp is not None and isinstance(obj, pd_timestamp):  # type: ignore[arg-type]
            return obj.isoformat()

        if hasattr(obj, "to_pydatetime"):
            try:
                dt = obj.to_pydatetime()
                if isinstance(dt, datetime):
                    return dt.isoformat()
            except Exception:
                pass

        if pa is not None:
            try:
                if isinstance(obj, pa.Scalar):  # type: ignore[attr-defined]
                    return obj.as_py()
            except Exception:
                pass

        if np is not None:
            try:
                if isinstance(obj, (np.integer, np.floating, np.bool_)):  # type: ignore[attr-defined]
                    return obj.item()
                if isinstance(obj, np.datetime64):  # type: ignore[attr-defined]
                    return str(obj)
            except Exception:
                pass

        if isinstance(obj, Path):
            return obj.as_posix()

        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except Exception:
                return obj.decode("utf-8", errors="replace")

        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass

        return str(obj)

    return _default


def write_jsonl(path: Path, batches: Iterable[dict[str, list[Any]]]) -> tuple[int, list[str]]:
    num_rows = 0
    columns: list[str] = []
    default = json_default()
    with path.open("w", encoding="utf-8") as f:
        for batch in batches:
            if not columns:
                columns = list(batch.keys())
            for i in range(len(next(iter(batch.values())))):
                record = {k: v[i] for k, v in batch.items()}
                f.write(json.dumps(record, ensure_ascii=False, default=default) + "\n")
                num_rows += 1
    return num_rows, columns


def write_parquet_shards(
    output_dir: Path,
    batches: Iterable[dict[str, list[Any]]],
    max_rows_per_file: int,
) -> tuple[int, list[str], list[Path]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency: pyarrow. Install with: pip install pyarrow"
        ) from e

    ensure_dir(output_dir)
    files: list[Path] = []
    num_rows = 0
    columns: list[str] = []
    writer: pq.ParquetWriter | None = None
    current_rows = 0
    file_index = 0

    def new_writer(table: pa.Table) -> pq.ParquetWriter:
        nonlocal file_index
        path = output_dir / f"part-{file_index:05d}.parquet"
        file_index += 1
        files.append(path)
        return pq.ParquetWriter(path.as_posix(), table.schema, compression="zstd")

    try:
        for batch in batches:
            if not columns:
                columns = list(batch.keys())
            table = pa.Table.from_pydict(batch)
            if writer is None:
                writer = new_writer(table)
                current_rows = 0

            if current_rows + table.num_rows > max_rows_per_file and current_rows > 0:
                writer.close()
                writer = new_writer(table)
                current_rows = 0

            writer.write_table(table)
            current_rows += table.num_rows
            num_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    return num_rows, columns, files


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_metadata(output_dir: Path, result: DownloadResult) -> Path:
    ensure_dir(output_dir)
    path = output_dir / "_metadata.json"
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="jniimi/tripadvisor-review-rating",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-dir",
        default=str(Path("data") / "raw" / "tripadvisor"),
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "jsonl"],
        default="parquet",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "--sample-1000",
        action="store_true",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Optional HuggingFace Hub endpoint (e.g. https://hf-mirror.com).",
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Optional HF cache root (defaults to ./ .cache/huggingface).",
    )
    return parser.parse_args()


def load_hf_dataset(dataset_name: str, split: str, sample_1000: bool) -> Any:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency: datasets. Install with: pip install datasets"
        ) from e

    ds = load_dataset(dataset_name, split=split)
    if sample_1000:
        ds = ds.select(range(min(1000, len(ds))))
    return ds


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    elif "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = (Path.cwd() / ".cache" / "huggingface").as_posix()

    try:
        ds = load_hf_dataset(args.dataset, args.split, args.sample_1000)
    except Exception as e:
        msg = str(e)
        if "Network is unreachable" in msg or "ConnectionError" in msg:
            raise RuntimeError(
                "Failed to reach HuggingFace Hub. If outbound access to huggingface.co is blocked, "
                "try a mirror endpoint, e.g. add: --hf-endpoint https://hf-mirror.com"
            ) from e
        raise
    batches = iter_batches(ds, batch_size=args.batch_size)

    created_at = utc_now_iso()
    out_files: list[Path]
    columns: list[str]
    num_rows: int

    if args.format == "jsonl":
        out_path = output_dir / "data.jsonl"
        num_rows, columns = write_jsonl(out_path, batches)
        out_files = [out_path]
    else:
        num_rows, columns, out_files = write_parquet_shards(
            output_dir=output_dir,
            batches=batches,
            max_rows_per_file=args.max_rows_per_file,
        )

    result = DownloadResult(
        dataset_name=args.dataset,
        split=args.split,
        output_dir=output_dir.as_posix(),
        output_format=args.format,
        files=[p.as_posix() for p in out_files],
        num_rows=int(num_rows),
        columns=columns,
        created_at_utc=created_at,
    )
    write_metadata(output_dir, result)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
