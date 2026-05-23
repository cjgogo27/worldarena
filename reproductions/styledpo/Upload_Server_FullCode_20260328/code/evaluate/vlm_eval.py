#!/usr/bin/env python3
"""Aggregate VLM metrics from preference-pair benchmark records."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM metric aggregation for preference pairs")
    parser.add_argument("--pairs_file", type=str, default="results/pair_benchmark_build/preference_pairs.json")
    parser.add_argument("--output_json", type=str, default="results/evaluations/vlm_results.json")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3.5-9B")
    return parser.parse_args()


def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def main() -> int:
    args = parse_args()
    with open(args.pairs_file, "r", encoding="utf-8") as f:
        groups = json.load(f)

    chosen_scores: List[float] = []
    rejected_scores: List[float] = []
    margins: List[float] = []
    per_split: Dict[str, Dict[str, float]] = {}

    for g in groups:
        split = g.get("split", "unknown")
        chosen = float(g.get("chosen", {}).get("vlm_score", g.get("chosen", {}).get("score", 0.0)))
        rejected_list = g.get("rejected_list", [])
        if not rejected_list:
            continue
        local_rej = [float(r.get("vlm_score", r.get("score", 0.0))) for r in rejected_list]
        chosen_scores.append(chosen)
        rejected_scores.extend(local_rej)
        margins.extend([chosen - x for x in local_rej])

        rec = per_split.setdefault(split, {"chosen_sum": 0.0, "chosen_n": 0, "rejected_sum": 0.0, "rejected_n": 0, "wins": 0, "pairs": 0})
        rec["chosen_sum"] += chosen
        rec["chosen_n"] += 1
        rec["rejected_sum"] += sum(local_rej)
        rec["rejected_n"] += len(local_rej)
        rec["wins"] += sum(1 for x in local_rej if chosen > x)
        rec["pairs"] += len(local_rej)

    result = {
        "generated_at": now_iso(),
        "vlm_model": args.vlm_model,
        "pairs_file": args.pairs_file,
        "group_count": len(groups),
        "chosen_vlm_mean": mean(chosen_scores),
        "rejected_vlm_mean": mean(rejected_scores),
        "vlm_margin_mean": mean(margins),
        "vlm_win_rate": float(sum(1 for x in margins if x > 0) / len(margins)) if margins else 0.0,
        "per_split": {},
    }

    for split, rec in per_split.items():
        result["per_split"][split] = {
            "chosen_vlm_mean": rec["chosen_sum"] / rec["chosen_n"] if rec["chosen_n"] else 0.0,
            "rejected_vlm_mean": rec["rejected_sum"] / rec["rejected_n"] if rec["rejected_n"] else 0.0,
            "vlm_win_rate": rec["wins"] / rec["pairs"] if rec["pairs"] else 0.0,
            "pair_count": rec["pairs"],
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {output_json}")
    print(f"VLM_MARGIN_MEAN {result['vlm_margin_mean']:.6f}")
    print(f"VLM_WIN_RATE {result['vlm_win_rate']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
