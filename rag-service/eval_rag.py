from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str


def _default_questions() -> list[EvalItem]:
    raw = [
        ("q_location_1", "How is the hotel location recently?"),
        ("q_location_2", "Is the location convenient for public transportation and walking?"),
        ("q_location_3", "Are there restaurants and shops near the hotel?"),
        ("q_location_4", "Is the area around the hotel safe and quiet at night?"),
        ("q_location_5", "How close is the hotel to major attractions?"),
        ("q_rooms_1", "How are the rooms in recent reviews?"),
        ("q_rooms_2", "Are the rooms spacious and comfortable?"),
        ("q_rooms_3", "Any recurring issues with room maintenance or outdated furniture?"),
        ("q_clean_1", "How is cleanliness in recent guest feedback?"),
        ("q_clean_2", "Do reviews mention dirty bathrooms or bad housekeeping?"),
        ("q_sleep_1", "How is the sleep quality and noise level?"),
        ("q_sleep_2", "Do guests complain about street noise or thin walls?"),
        ("q_value_1", "Is this hotel considered good value for money?"),
        ("q_value_2", "Do guests feel the price matches the quality and location?"),
        ("q_overall_1", "Would guests recommend this hotel overall?"),
        ("q_overall_2", "What are the most common pros and cons mentioned?"),
        ("q_recent_1", "In the most recent reviews, what are the biggest issues?"),
        ("q_recent_2", "Have there been improvements over the last year?"),
        ("q_business_1", "Is this hotel good for business trips based on reviews?"),
        ("q_family_1", "Is this hotel family-friendly according to reviews?"),
        ("q_breakfast_1", "How is the breakfast experience described in reviews?"),
        ("q_staff_1", "What do guests say about staff service and responsiveness?"),
        ("q_wifi_1", "Do guests mention Wi‑Fi quality and reliability?"),
        ("q_parking_1", "Is parking convenient and reasonably priced?"),
        ("q_checkin_1", "Are there frequent complaints about check-in/check-out?"),
        ("q_amenities_1", "What amenities are commonly praised or criticized?"),
        ("q_location_recency", "How is the hotel location in the latest stays?"),
        ("q_clean_recency", "Any recent cleanliness complaints worth noting?"),
        ("q_noise_recency", "Do recent guests mention noise problems?"),
        ("q_value_recency", "Are recent guests satisfied with value for money?"),
    ]
    return [EvalItem(id=qid, question=q) for qid, q in raw]


def _load_questions(path: Path | None) -> list[EvalItem]:
    if path is None:
        return _default_questions()
    items: list[EvalItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        qid = str(obj.get("id") or "").strip() or f"q_{len(items)+1}"
        q = str(obj.get("question") or "").strip()
        if not q:
            continue
        items.append(EvalItem(id=qid, question=q))
    return items


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(raw)
    if isinstance(obj, dict):
        return obj
    return {"raw": obj}


def _extract_route_stats(references: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in references:
        sources = r.get("sources")
        if not isinstance(sources, dict):
            src = r.get("source")
            if isinstance(src, str) and src:
                counts[src] = counts.get(src, 0) + 1
            continue
        for k in sources.keys():
            if isinstance(k, str) and k:
                counts[k] = counts.get(k, 0) + 1
    return counts


def _claim_metrics(claims: Any, references: list[dict[str, Any]]) -> dict[str, Any]:
    ref_n = len(references)
    if not isinstance(claims, list) or not claims:
        return {
            "claims_count": 0,
            "claims_refs_avg": 0.0,
            "claims_refs_max": 0,
            "claims_ref_ids_all_valid": True,
        }
    per_claim_counts: list[int] = []
    all_valid = True
    for c in claims:
        if not isinstance(c, dict):
            continue
        ref_ids = c.get("ref_ids") or []
        if not isinstance(ref_ids, list):
            continue
        ids: list[int] = []
        for rid in ref_ids:
            if isinstance(rid, int):
                ids.append(rid)
        per_claim_counts.append(len(ids))
        for rid in ids:
            if rid < 0 or rid >= ref_n:
                all_valid = False
    avg = (sum(per_claim_counts) / len(per_claim_counts)) if per_claim_counts else 0.0
    mx = max(per_claim_counts) if per_claim_counts else 0
    return {
        "claims_count": len(per_claim_counts),
        "claims_refs_avg": avg,
        "claims_refs_max": mx,
        "claims_ref_ids_all_valid": all_valid,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=str, default="", help="Path to questions jsonl; default uses built-in set.")
    p.add_argument("--outdir", type=str, default="artifacts/eval", help="Output directory.")
    p.add_argument("--endpoint", type=str, default="", help="If set, call HTTP endpoint instead of in-process RagSystem.")
    p.add_argument("--timeout_s", type=float, default=120.0)
    args = p.parse_args()

    q_path = Path(args.questions) if args.questions.strip() else None
    items = _load_questions(q_path)
    if not items:
        raise SystemExit("No questions loaded")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_jsonl = outdir / f"run_{run_id}.jsonl"
    out_summary = outdir / f"summary_{run_id}.json"

    use_http = bool(args.endpoint.strip())
    if use_http:
        endpoint = args.endpoint.strip()
    else:
        service_dir = Path(__file__).resolve().parent
        sys.path.insert(0, service_dir.as_posix())
        from rag_service.modules.rag_system import RagSystem  # type: ignore

        rag = RagSystem()
        endpoint = ""

    agg = {
        "n": 0,
        "intent_source": {},
        "routes": {},
        "claims_ref_ids_all_valid_fail": 0,
        "claims_refs_max": 0,
        "avg_claims_per_q_sum": 0.0,
    }

    with out_jsonl.open("w", encoding="utf-8") as f:
        for it in items:
            payload = {"question": it.question}
            if use_http:
                result = _post_json(endpoint, payload, timeout_s=float(args.timeout_s))
            else:
                result = rag.chat(payload)  # type: ignore[name-defined]

            intent = result.get("intent") if isinstance(result, dict) else {}
            references = result.get("references") if isinstance(result, dict) else []
            claims = result.get("claims") if isinstance(result, dict) else []

            intent_source = None
            if isinstance(intent, dict):
                intent_source = intent.get("intent_source")
            if not isinstance(intent_source, str) or not intent_source:
                intent_source = "unknown"

            if not isinstance(references, list):
                references = []
            route_stats = _extract_route_stats([r for r in references if isinstance(r, dict)])
            cm = _claim_metrics(claims, [r for r in references if isinstance(r, dict)])

            row = {
                "id": it.id,
                "question": it.question,
                "intent_source": intent_source,
                "route_stats": route_stats,
                "claim_metrics": cm,
                "result": result,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            agg["n"] += 1
            agg["intent_source"][intent_source] = agg["intent_source"].get(intent_source, 0) + 1
            for k, v in route_stats.items():
                agg["routes"][k] = agg["routes"].get(k, 0) + int(v)
            if not bool(cm.get("claims_ref_ids_all_valid", True)):
                agg["claims_ref_ids_all_valid_fail"] += 1
            mx = int(cm.get("claims_refs_max") or 0)
            if mx > agg["claims_refs_max"]:
                agg["claims_refs_max"] = mx
            agg["avg_claims_per_q_sum"] += float(cm.get("claims_count") or 0)

    avg_claims = agg["avg_claims_per_q_sum"] / agg["n"] if agg["n"] else 0.0
    summary = {
        "run_id": run_id,
        "n": agg["n"],
        "intent_source": agg["intent_source"],
        "routes_total_mentions": agg["routes"],
        "avg_claims_per_question": avg_claims,
        "claims_ref_ids_all_valid_fail": agg["claims_ref_ids_all_valid_fail"],
        "claims_refs_max": agg["claims_refs_max"],
        "out_jsonl": out_jsonl.as_posix(),
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

