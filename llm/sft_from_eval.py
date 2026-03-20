from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_eval_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def _to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    q = str(row.get("question") or "").strip()
    result = row.get("result") or {}
    if not isinstance(result, dict):
        result = {}
    answer = str(result.get("answer") or "").strip()
    if not q or not answer:
        return []
    system = "You are a helpful assistant for hotel review Q&A. Answer in English and cite evidence as [id]."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": q},
        {"role": "assistant", "content": answer},
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=str, required=True)
    p.add_argument("--out", dest="out_path", type=str, default="artifacts/sft/sft.jsonl")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    in_path = Path(args.in_path)
    rows = _load_eval_jsonl(in_path)
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        msgs = _to_messages(r)
        if not msgs:
            continue
        out_rows.append({"messages": msgs})
        if args.limit and len(out_rows) >= args.limit:
            break

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(out_rows), "out": out_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

