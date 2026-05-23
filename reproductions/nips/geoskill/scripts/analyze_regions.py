# pyright: reportImplicitRelativeImport=false, reportExplicitAny=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator import evaluate_predictions
from src.skill_parser import COUNTRY_TO_REGION


REGIONS = ["europe", "asia", "north_america", "south_america", "africa", "oceania"]
DEFAULT_METHODS = [
    "direct_vlm",
    "cot_vlm",
    "geocot",
    "georeasoner",
    "gre_multistage",
    "skill_conditioned",
    "img2loc_rag",
]
METRIC_KEYS = [
    "country_accuracy",
    "continent_accuracy",
    "distance_error_km_mean_valid_only",
    "distance_error_km_median",
    "valid_coordinate_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cross-region generalization from experiment predictions")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/full_100",
        help="Directory containing one subdirectory per method with latest_predictions.json",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS,
        help="Methods to analyze",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return math.nan


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def pretty_metric(value: Any, metric: str) -> str:
    if value is None:
        return "N/A"
    if is_nan(value):
        return "N/A"
    if metric in {"country_accuracy", "continent_accuracy", "valid_coordinate_rate"}:
        return f"{float(value):.3f}"
    if metric in {"distance_error_km_mean_valid_only", "distance_error_km_median"}:
        return f"{float(value):.1f}"
    if metric == "sample_count":
        return str(int(value))
    return f"{value}"


def load_method_predictions(results_dir: Path, method: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    pred_path = results_dir / method / "latest_predictions.json"
    if not pred_path.exists():
        return None, f"missing file: {pred_path}"

    try:
        raw = json.loads(pred_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"failed to parse {pred_path}: {exc}"

    if not isinstance(raw, list):
        return None, f"unexpected JSON format in {pred_path} (expected list)"

    return raw, None


def region_of_record(record: dict[str, Any]) -> str:
    gt_country = str(record.get("ground_truth_country", "")).lower()
    return COUNTRY_TO_REGION.get(gt_country, "unknown")


def subset_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "country_accuracy": 0.0,
            "continent_accuracy": 0.0,
            "distance_error_km_mean_valid_only": math.nan,
            "distance_error_km_median": math.nan,
            "valid_coordinate_rate": 0.0,
            "sample_count": 0,
        }

    m = evaluate_predictions(records)
    return {
        "country_accuracy": float(m.get("country_accuracy", 0.0)),
        "continent_accuracy": float(m.get("continent_accuracy", 0.0)),
        "distance_error_km_mean_valid_only": safe_float(m.get("distance_error_km_mean_valid_only", math.nan)),
        "distance_error_km_median": safe_float(m.get("distance_error_km_median", math.nan)),
        "valid_coordinate_rate": float(m.get("valid_coordinate_rate", 0.0)),
        "sample_count": len(records),
    }


def build_metric_tables(per_region_metrics: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    for metric in METRIC_KEYS + ["sample_count"]:
        tables[metric] = {}
        for method, method_data in per_region_metrics.items():
            tables[metric][method] = {
                region: method_data.get(region, {}).get(metric, math.nan if "distance" in metric else 0.0)
                for region in REGIONS
            }
    return tables


def method_ranking_for_region(
    per_region_metrics: dict[str, dict[str, dict[str, Any]]],
    region: str,
    metric: str,
) -> dict[str, Any]:
    values: list[tuple[str, float]] = []
    for method, rdata in per_region_metrics.items():
        if region not in rdata:
            continue
        metric_value = rdata[region].get(metric)
        if metric == "sample_count":
            continue
        if metric_value is None or is_nan(metric_value):
            continue
        values.append((method, float(metric_value)))

    if not values:
        return {"best": [], "worst": [], "criterion": "none", "values": {}}

    higher_is_better = metric not in {"distance_error_km_mean_valid_only", "distance_error_km_median"}
    best_value = max(v for _, v in values) if higher_is_better else min(v for _, v in values)
    worst_value = min(v for _, v in values) if higher_is_better else max(v for _, v in values)

    best_methods = [m for m, v in values if v == best_value]
    worst_methods = [m for m, v in values if v == worst_value]

    return {
        "best": best_methods,
        "worst": worst_methods,
        "criterion": "higher_is_better" if higher_is_better else "lower_is_better",
        "values": {m: v for m, v in values},
    }


def compute_generalization_gap(per_region_metrics: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method, region_data in per_region_metrics.items():
        values = [
            float(region_data[region]["continent_accuracy"])
            for region in REGIONS
            if region_data.get(region, {}).get("sample_count", 0) > 0
        ]
        if not values:
            output[method] = {
                "generalization_gap": math.nan,
                "max_continent_accuracy": math.nan,
                "min_continent_accuracy": math.nan,
            }
            continue
        output[method] = {
            "generalization_gap": max(values) - min(values),
            "max_continent_accuracy": max(values),
            "min_continent_accuracy": min(values),
        }
    return output


def to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def print_metric_table(metric: str, table: dict[str, dict[str, Any]]) -> None:
    print(f"\n=== {metric} (method x region) ===")
    col_w = max(16, max((len(m) for m in table.keys()), default=6) + 2)
    region_w = 14
    header = f"{'method':<{col_w}}" + "".join(f"{r:>{region_w}}" for r in REGIONS)
    print(header)
    print("-" * len(header))
    for method in sorted(table.keys()):
        row = f"{method:<{col_w}}"
        for region in REGIONS:
            row += f"{pretty_metric(table[method].get(region), metric):>{region_w}}"
        print(row)


def print_rankings(best_worst_by_region: dict[str, Any]) -> None:
    print("\n=== Best / Worst methods per region ===")
    for region in REGIONS:
        print(f"\n[{region}]")
        region_data = best_worst_by_region.get(region, {})
        for metric in METRIC_KEYS:
            item = region_data.get(metric, {})
            best = ", ".join(item.get("best", [])) or "N/A"
            worst = ", ".join(item.get("worst", [])) or "N/A"
            print(f"  {metric:<26} best: {best} | worst: {worst}")


def print_generalization_gap(gaps: dict[str, Any]) -> None:
    print("\n=== Generalization gap by method (max continent_acc - min continent_acc, lower is better) ===")
    rows = []
    for method, stats in gaps.items():
        gap = stats.get("generalization_gap", math.nan)
        rows.append((method, gap))
    rows.sort(key=lambda x: (math.inf if is_nan(x[1]) else x[1], x[0]))

    print(f"{'method':<24}{'gap':>10}{'min_acc':>12}{'max_acc':>12}")
    print("-" * 58)
    for method, _ in rows:
        stats = gaps[method]
        gap_str = pretty_metric(stats.get("generalization_gap", math.nan), "continent_accuracy")
        min_str = pretty_metric(stats.get("min_continent_accuracy", math.nan), "continent_accuracy")
        max_str = pretty_metric(stats.get("max_continent_accuracy", math.nan), "continent_accuracy")
        print(f"{method:<24}{gap_str:>10}{min_str:>12}{max_str:>12}")


def print_leave_one_region_out(loo_metrics: dict[str, dict[str, dict[str, Any]]]) -> None:
    print("\n=== Leave-one-region-out continent_accuracy ===")
    col_w = max(16, max((len(m) for m in loo_metrics.keys()), default=6) + 2)
    region_w = 14
    header = f"{'method':<{col_w}}" + "".join(f"exclude:{r[:10]:>{region_w-8}}" for r in REGIONS)
    print(header)
    print("-" * len(header))
    for method in sorted(loo_metrics.keys()):
        row = f"{method:<{col_w}}"
        for region in REGIONS:
            value = loo_metrics[method][region]["continent_accuracy"]
            row += f"{pretty_metric(value, 'continent_accuracy'):>{region_w}}"
        print(row)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir

    per_region_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    leave_one_region_out: dict[str, dict[str, dict[str, Any]]] = {}
    missing_methods: dict[str, str] = {}
    available_methods: list[str] = []

    for method in args.methods:
        records, err = load_method_predictions(results_dir, method)
        if err is not None or records is None:
            missing_methods[method] = err or "unknown error"
            continue

        available_methods.append(method)
        region_buckets = {region: [] for region in REGIONS}
        for rec in records:
            region = region_of_record(rec)
            if region in region_buckets:
                region_buckets[region].append(rec)

        per_region_metrics[method] = {
            region: subset_metrics(region_buckets[region])
            for region in REGIONS
        }

        leave_one_region_out[method] = {}
        for excluded in REGIONS:
            loo_subset = [rec for rec in records if region_of_record(rec) != excluded]
            leave_one_region_out[method][excluded] = subset_metrics(loo_subset)

    metric_tables = build_metric_tables(per_region_metrics)
    best_worst_by_region = {
        region: {
            metric: method_ranking_for_region(per_region_metrics, region, metric)
            for metric in METRIC_KEYS
        }
        for region in REGIONS
    }
    generalization_gap = compute_generalization_gap(per_region_metrics)

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "methods_requested": args.methods,
        "methods_available": available_methods,
        "missing_methods": missing_methods,
        "regions": REGIONS,
        "per_region_metrics": per_region_metrics,
        "metric_tables": metric_tables,
        "best_worst_by_region": best_worst_by_region,
        "generalization_gap": generalization_gap,
        "leave_one_region_out": leave_one_region_out,
    }

    out_path = results_dir / "region_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_json_safe(output), ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Cross-region analysis ===")
    print(f"results_dir: {results_dir}")
    if missing_methods:
        print("\nMissing/unreadable methods:")
        for method, msg in missing_methods.items():
            print(f"  - {method}: {msg}")

    for metric in METRIC_KEYS + ["sample_count"]:
        print_metric_table(metric, metric_tables[metric])
    print_rankings(best_worst_by_region)
    print_generalization_gap(generalization_gap)
    print_leave_one_region_out(leave_one_region_out)

    print(f"\nSaved analysis JSON to: {out_path}")


if __name__ == "__main__":
    main()
