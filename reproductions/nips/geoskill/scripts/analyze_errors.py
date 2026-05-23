# pyright: reportImplicitRelativeImport=false, reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator import hallucination_flag, haversine_km
from src.skill_parser import COUNTRY_TO_REGION


DEFAULT_METHODS = [
    "direct_vlm",
    "cot_vlm",
    "skill_conditioned",
    "georeasoner",
    "geocot",
    "gre_multistage",
    "img2loc_rag",
]

REGION_ORDER = ["africa", "asia", "europe", "north_america", "south_america", "oceania"]
CONFIDENCE_BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.000001)]
DISTANCE_BINS_KM = [0, 25, 100, 250, 500, 1000, 2500, 5000, 10000, 20000]
INVALID_DISTANCE_PENALTY_KM = 20037.0


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _clamp_confidence(x: Any) -> float:
    val = _safe_float(x)
    if val is None:
        return 0.0
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def _valid_coords(lat: Any, lng: Any) -> bool:
    latf = _safe_float(lat)
    lngf = _safe_float(lng)
    return latf is not None and lngf is not None and -90 <= latf <= 90 and -180 <= lngf <= 180


def _distance_error_km(record: dict[str, Any]) -> float | None:
    pred = record.get("prediction", {})
    if not _valid_coords(pred.get("predicted_lat"), pred.get("predicted_lng")):
        return None
    return haversine_km(
        float(pred["predicted_lat"]),
        float(pred["predicted_lng"]),
        float(record["ground_truth_lat"]),
        float(record["ground_truth_lng"]),
    )


def _distance_with_penalty(record: dict[str, Any]) -> float:
    d = _distance_error_km(record)
    return d if d is not None else INVALID_DISTANCE_PENALTY_KM


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _load_records(results_dir: Path, method: str) -> list[dict[str, Any]]:
    method_path = results_dir / method / "latest_predictions.json"
    if method_path.exists():
        return json.loads(method_path.read_text(encoding="utf-8"))

    combined_path = results_dir / "latest_predictions.json"
    if combined_path.exists():
        combined = json.loads(combined_path.read_text(encoding="utf-8"))
        if isinstance(combined, dict) and method in combined and isinstance(combined[method], list):
            return combined[method]

    return []


def _country_confusions(records: list[dict[str, Any]], top_k: int = 10) -> dict[str, Any]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    mistakes = Counter()

    for rec in records:
        gt = rec.get("ground_truth_country", "unknown")
        pred = rec.get("prediction", {}).get("predicted_country", "unknown")
        matrix[pred][gt] += 1
        if gt != pred:
            mistakes[(gt, pred)] += 1

    top_pairs = [
        {"ground_truth_country": gt, "predicted_country": pred, "count": cnt}
        for (gt, pred), cnt in mistakes.most_common(top_k)
    ]
    return {
        "matrix_pred_x_gt": {p: dict(cols) for p, cols in matrix.items()},
        "top_confused_pairs": top_pairs,
    }


def _region_confusions(records: list[dict[str, Any]], top_k: int = 10) -> dict[str, Any]:
    matrix: dict[str, dict[str, int]] = {pr: {gt: 0 for gt in REGION_ORDER} for pr in REGION_ORDER}
    mistakes = Counter()

    for rec in records:
        gt_country = rec.get("ground_truth_country", "unknown")
        gt_region = COUNTRY_TO_REGION.get(gt_country, "unknown")
        pred_region = rec.get("prediction", {}).get("predicted_region", "unknown")
        if gt_region not in matrix:
            matrix[gt_region] = {g: 0 for g in REGION_ORDER}
        if pred_region not in matrix:
            matrix[pred_region] = {g: 0 for g in REGION_ORDER}
        if gt_region not in matrix[pred_region]:
            matrix[pred_region][gt_region] = 0
        matrix[pred_region][gt_region] += 1
        if gt_region != pred_region:
            mistakes[(gt_region, pred_region)] += 1

    top_pairs = [
        {"ground_truth_region": gt, "predicted_region": pred, "count": cnt}
        for (gt, pred), cnt in mistakes.most_common(top_k)
    ]
    return {
        "region_order": REGION_ORDER,
        "matrix_pred_x_gt": matrix,
        "top_confused_pairs": top_pairs,
    }


def _distance_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    distances = sorted(d for d in (_distance_error_km(r) for r in records) if d is not None)

    hist: list[dict[str, Any]] = []
    if distances:
        for lo, hi in zip(DISTANCE_BINS_KM[:-1], DISTANCE_BINS_KM[1:]):
            count = sum(1 for d in distances if lo <= d < hi)
            hist.append({"range_km": f"[{lo},{hi})", "count": count, "rate": count / len(distances)})
        ge_last = sum(1 for d in distances if d >= DISTANCE_BINS_KM[-1])
        hist.append(
            {
                "range_km": f"[{DISTANCE_BINS_KM[-1]},inf)",
                "count": ge_last,
                "rate": ge_last / len(distances),
            }
        )

    return {
        "num_valid_distances": len(distances),
        "histogram": hist,
        "percentiles_km": {
            "p25": _percentile(distances, 0.25),
            "p50": _percentile(distances, 0.50),
            "p75": _percentile(distances, 0.75),
            "p90": _percentile(distances, 0.90),
            "p95": _percentile(distances, 0.95),
        },
    }


def _compact_case(rec: dict[str, Any], include_retrieval: bool = True) -> dict[str, Any]:
    pred = rec.get("prediction", {})
    case = {
        "game_id": rec.get("game_id"),
        "round": rec.get("round"),
        "ground_truth": {
            "country": rec.get("ground_truth_country"),
            "region": COUNTRY_TO_REGION.get(rec.get("ground_truth_country", "unknown"), "unknown"),
            "lat": rec.get("ground_truth_lat"),
            "lng": rec.get("ground_truth_lng"),
        },
        "prediction": {
            "country": pred.get("predicted_country", "unknown"),
            "region": pred.get("predicted_region", "unknown"),
            "lat": pred.get("predicted_lat"),
            "lng": pred.get("predicted_lng"),
            "confidence": pred.get("confidence"),
        },
        "distance_error_km": _distance_error_km(rec),
        "reasoning": pred.get("reasoning_text", ""),
    }
    if include_retrieval:
        if "retrieved_skills" in pred:
            case["retrieved_skills"] = pred.get("retrieved_skills", [])
        if "retrieved_refs" in pred:
            case["retrieved_refs"] = pred.get("retrieved_refs", [])
    return case


def _failure_case_studies(method_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    skill_records = method_records.get("skill_conditioned", [])
    direct_records = method_records.get("direct_vlm", [])

    skill_with_d = [(r, _distance_error_km(r)) for r in skill_records]
    skill_with_d_valid = [(r, d) for r, d in skill_with_d if d is not None]

    worst = sorted(skill_with_d_valid, key=lambda x: x[1], reverse=True)[:5]
    best = [
        (r, d)
        for r, d in sorted(skill_with_d_valid, key=lambda x: x[1])
        if r.get("prediction", {}).get("predicted_country", "unknown") == r.get("ground_truth_country", "unknown")
    ][:5]

    direct_by_key = {(r.get("game_id"), r.get("round", 1)): r for r in direct_records}
    swing_rows = []
    for srec in skill_records:
        key = (srec.get("game_id"), srec.get("round", 1))
        if key not in direct_by_key:
            continue
        drec = direct_by_key[key]
        d_direct = _distance_with_penalty(drec)
        d_skill = _distance_with_penalty(srec)
        swing_rows.append((srec, d_direct - d_skill, d_direct, d_skill))

    biggest_wins = sorted(swing_rows, key=lambda x: x[1], reverse=True)[:5]
    biggest_losses = sorted(swing_rows, key=lambda x: x[1])[:5]

    return {
        "worst_skill_conditioned": [
            {**_compact_case(r), "distance_error_km": d} for r, d in worst
        ],
        "best_skill_conditioned_correct_country": [
            {**_compact_case(r), "distance_error_km": d} for r, d in best
        ],
        "skill_conditioned_biggest_wins_vs_direct_vlm": [
            {
                **_compact_case(r),
                "direct_vlm_distance_km": direct_d,
                "skill_conditioned_distance_km": skill_d,
                "distance_improvement_km": margin,
            }
            for r, margin, direct_d, skill_d in biggest_wins
        ],
        "skill_conditioned_biggest_losses_vs_direct_vlm": [
            {
                **_compact_case(r),
                "direct_vlm_distance_km": direct_d,
                "skill_conditioned_distance_km": skill_d,
                "distance_improvement_km": margin,
            }
            for r, margin, direct_d, skill_d in biggest_losses
        ],
    }


def _retrieval_analysis_for_method(records: list[dict[str, Any]], retrieval_key: str) -> dict[str, Any]:
    total_items = 0
    correct_region_items = 0
    scores: list[float] = []
    labels: list[float] = []
    misleading_cases = []

    for rec in records:
        pred = rec.get("prediction", {})
        retrieved = pred.get(retrieval_key, [])
        if not isinstance(retrieved, list) or not retrieved:
            continue

        gt_country = rec.get("ground_truth_country", "unknown")
        gt_region = COUNTRY_TO_REGION.get(gt_country, "unknown")
        pred_country = pred.get("predicted_country", "unknown")
        pred_region = pred.get("predicted_region", "unknown")
        is_correct = 1.0 if pred_country == gt_country else 0.0

        for item in retrieved:
            if not isinstance(item, dict):
                continue
            hint = item.get("region_hint", "unknown")
            if hint == gt_region:
                correct_region_items += 1
            total_items += 1

            score = _safe_float(item.get("score"))
            if score is not None:
                scores.append(score)
                labels.append(is_correct)

        top = retrieved[0] if retrieved and isinstance(retrieved[0], dict) else {}
        top_score = _safe_float(top.get("score"))
        top_hint = top.get("region_hint", "unknown")
        if (
            pred_country != gt_country
            and top_hint != "unknown"
            and top_hint != gt_region
            and top_hint == pred_region
        ):
            misleading_cases.append(
                {
                    **_compact_case(rec),
                    "top_retrieved_item": top,
                    "top_retrieved_score": top_score,
                    "gt_region": gt_region,
                }
            )

    misleading_cases.sort(key=lambda x: x.get("top_retrieved_score") or -1.0, reverse=True)
    corr = _pearson_corr(scores, labels)

    return {
        "num_samples_with_retrieval": sum(
            1 for r in records if isinstance(r.get("prediction", {}).get(retrieval_key, None), list)
        ),
        "retrieved_items_total": total_items,
        "retrieved_region_hint_match_rate": (correct_region_items / total_items) if total_items else None,
        "score_accuracy_correlation": corr,
        "score_accuracy_correlation_points": len(scores),
        "misleading_cases_top5": misleading_cases[:5],
    }


def _hallucination_analysis(records: list[dict[str, Any]], max_examples: int = 5) -> dict[str, Any]:
    flags: list[tuple[dict[str, Any], bool]] = []
    for rec in records:
        pred = rec.get("prediction", {})
        reasoning = pred.get("reasoning_text", "")
        expert = rec.get("expert_chain", "")
        flags.append((rec, hallucination_flag(reasoning, expert)))

    positives = [r for r, f in flags if f]
    examples = []
    for rec in positives[:max_examples]:
        pred = rec.get("prediction", {})
        examples.append(
            {
                "game_id": rec.get("game_id"),
                "round": rec.get("round"),
                "ground_truth_country": rec.get("ground_truth_country"),
                "predicted_country": pred.get("predicted_country", "unknown"),
                "reasoning_excerpt": pred.get("reasoning_text", "")[:500],
            }
        )

    return {
        "heuristic_hallucination_rate": (len(positives) / len(records)) if records else 0.0,
        "num_hallucinations": len(positives),
        "num_samples": len(records),
        "examples": examples,
    }


def _confidence_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for lo, hi in CONFIDENCE_BINS:
        in_bin = []
        for rec in records:
            pred = rec.get("prediction", {})
            conf = _clamp_confidence(pred.get("confidence", 0.0))
            if lo <= conf < hi:
                hit = 1.0 if pred.get("predicted_country", "unknown") == rec.get("ground_truth_country", "unknown") else 0.0
                in_bin.append(hit)
        rows.append(
            {
                "bin": f"[{lo:.2f},{min(hi, 1.0):.2f}{')' if hi < 1.0 else ']'}",
                "count": len(in_bin),
                "accuracy": (sum(in_bin) / len(in_bin)) if in_bin else None,
            }
        )
    return {"bins": rows}


def _method_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "country_accuracy": None,
            "continent_accuracy": None,
            "median_distance_km": None,
            "mean_distance_km": None,
        }

    c_hits = []
    r_hits = []
    dists = []
    for rec in records:
        pred = rec.get("prediction", {})
        gt_country = rec.get("ground_truth_country", "unknown")
        gt_region = COUNTRY_TO_REGION.get(gt_country, "unknown")
        c_hits.append(1.0 if pred.get("predicted_country", "unknown") == gt_country else 0.0)
        r_hits.append(1.0 if pred.get("predicted_region", "unknown") == gt_region else 0.0)
        d = _distance_error_km(rec)
        if d is not None:
            dists.append(d)

    dists.sort()
    return {
        "num_samples": len(records),
        "country_accuracy": sum(c_hits) / len(c_hits),
        "continent_accuracy": sum(r_hits) / len(r_hits),
        "median_distance_km": _percentile(dists, 0.5) if dists else None,
        "mean_distance_km": (sum(dists) / len(dists)) if dists else None,
    }


def _print_findings(analysis: dict[str, Any], methods: list[str]) -> None:
    print("\n" + "=" * 88)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 88)
    print(f"Results dir: {analysis['results_dir']}")
    print("\nMethod-level snapshot:")
    print(f"{'Method':<18} {'CountryAcc':>10} {'RegionAcc':>10} {'MedDist(km)':>12} {'HallucRate':>11}")
    print("-" * 66)
    for m in methods:
        s = analysis["per_method"].get(m, {}).get("summary", {})
        h = analysis["per_method"].get(m, {}).get("hallucination_analysis", {})
        cacc = s.get("country_accuracy")
        racc = s.get("continent_accuracy")
        med = s.get("median_distance_km")
        hr = h.get("heuristic_hallucination_rate")
        cacc_s = f"{cacc:.3f}" if isinstance(cacc, (float, int)) else "N/A"
        racc_s = f"{racc:.3f}" if isinstance(racc, (float, int)) else "N/A"
        med_s = f"{med:.1f}" if isinstance(med, (float, int)) else "N/A"
        hr_s = f"{hr:.3f}" if isinstance(hr, (float, int)) else "N/A"
        print(f"{m:<18} {cacc_s:>10} {racc_s:>10} {med_s:>12} {hr_s:>11}")

    print("\nTop country confusions (top-3 each method):")
    for m in methods:
        conf = analysis["per_method"].get(m, {}).get("country_confusion", {}).get("top_confused_pairs", [])[:3]
        if not conf:
            print(f"  - {m}: none")
            continue
        pairs = ", ".join([f"{x['ground_truth_country']}→{x['predicted_country']} ({x['count']})" for x in conf])
        print(f"  - {m}: {pairs}")

    cs = analysis.get("case_studies", {})
    print("\nCase studies (skill_conditioned):")
    print(f"  Worst cases: {len(cs.get('worst_skill_conditioned', []))}")
    print(f"  Best correct-country cases: {len(cs.get('best_skill_conditioned_correct_country', []))}")
    print(f"  Biggest wins vs direct_vlm: {len(cs.get('skill_conditioned_biggest_wins_vs_direct_vlm', []))}")
    print(f"  Biggest losses vs direct_vlm: {len(cs.get('skill_conditioned_biggest_losses_vs_direct_vlm', []))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed error analysis and case-study generation for GeoSkill experiments")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/full_100",
        help="Experiment results directory (contains method subfolders with latest_predictions.json)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS,
        help="Methods to analyze",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir

    method_records: dict[str, list[dict[str, Any]]] = {}
    for method in args.methods:
        method_records[method] = _load_records(results_dir, method)

    analysis: dict[str, Any] = {
        "results_dir": str(results_dir),
        "methods": args.methods,
        "per_method": {},
        "case_studies": {},
        "skill_retrieval_analysis": {},
    }

    for method in args.methods:
        records = method_records.get(method, [])
        analysis["per_method"][method] = {
            "summary": _method_summary(records),
            "country_confusion": _country_confusions(records),
            "region_confusion": _region_confusions(records),
            "distance_error_distribution": _distance_distribution(records),
            "hallucination_analysis": _hallucination_analysis(records),
            "confidence_calibration": _confidence_calibration(records),
        }

    analysis["case_studies"] = _failure_case_studies(method_records)

    if "skill_conditioned" in method_records:
        analysis["skill_retrieval_analysis"]["skill_conditioned"] = _retrieval_analysis_for_method(
            method_records["skill_conditioned"], "retrieved_skills"
        )
    if "img2loc_rag" in method_records:
        analysis["skill_retrieval_analysis"]["img2loc_rag"] = _retrieval_analysis_for_method(
            method_records["img2loc_rag"], "retrieved_refs"
        )

    output_path = results_dir / "error_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_findings(analysis, args.methods)
    print(f"\nSaved analysis JSON: {output_path}")


if __name__ == "__main__":
    main()
