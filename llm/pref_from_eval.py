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


def _make_rejected(result: dict[str, Any]) -> str:
    refs = result.get("references") or []
    if isinstance(refs, list) and refs:
        first = refs[0]
        if isinstance(first, dict):
            t = str(first.get("chunk_text") or "").strip()
            if t:
                return t
    return "I don't have enough evidence from reviews to answer this question."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=str, required=True)
    p.add_argument("--out", dest="out_path", type=str, default="artifacts/pref/dpo.jsonl")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    rows = _load_eval_jsonl(Path(args.in_path))
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        q = str(r.get("question") or "").strip()
        result = r.get("result") or {}
        if not isinstance(result, dict):
            continue
        chosen = str(result.get("answer") or "").strip()
        if not q or not chosen:
            continue
        rejected = _make_rejected(result)
        out_rows.append({"prompt": q, "chosen": chosen, "rejected": rejected})
        if args.limit and len(out_rows) >= args.limit:
            break

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(out_rows), "out": out_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

