"""Recompute all experiment metrics using evaluator v2 (penalty for missing coords).

Reads existing latest_predictions.json for every method under experiments/full_100/
and writes latest_metrics_v2.json alongside it.  Also writes summary_metrics_v2.json
to experiments/full_100/.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluator import evaluate_predictions  # noqa: E402 (after sys.path fix)

RESULTS_DIR = ROOT / "experiments" / "full_100"

METHODS = [
    "direct_vlm",
    "cot_vlm",
    "geocot",
    "georeasoner",
    "gre_multistage",
    "skill_conditioned",
    "skill_conditioned_v2",
    "skill_conditioned_v3",
    "skill_conditioned_v4",
    "img2loc_rag",
]


def main() -> None:
    summary: dict[str, dict] = {}

    for method in METHODS:
        pred_path = RESULTS_DIR / method / "latest_predictions.json"
        if not pred_path.exists():
            print(f"SKIP {method}: {pred_path} not found")
            continue

        with open(pred_path) as f:
            records = json.load(f)

        metrics = evaluate_predictions(records)
        metrics["n_samples"] = len(records)
        metrics["n_errors"] = sum(
            1 for r in records
            if r.get("error") is not None or not _has_valid_coords(r)
        )

        out_path = RESULTS_DIR / method / "latest_metrics_v2.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

        summary[method] = metrics
        print(
            f"{method:<22} country={metrics['country_accuracy']:.3f}  "
            f"continent={metrics['continent_accuracy']:.3f}  "
            f"dist_median={metrics['distance_error_km_median']:.1f}  "
            f"dist_mean_valid={metrics['distance_error_km_mean_valid_only']:.1f}  "
            f"dist_mean_penalized={metrics['distance_error_km_mean_penalized']:.1f}  "
            f"valid={metrics['valid_coordinate_rate']:.2%}"
        )

    summary_path = RESULTS_DIR / "summary_metrics_v2.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {summary_path}")


def _has_valid_coords(rec: dict) -> bool:
    import math

    pred = rec.get("prediction", {})
    plat = pred.get("predicted_lat", math.nan)
    plng = pred.get("predicted_lng", math.nan)
    try:
        return (
            isinstance(plat, (float, int))
            and isinstance(plng, (float, int))
            and not math.isnan(float(plat))
            and not math.isnan(float(plng))
            and -90 <= float(plat) <= 90
            and -180 <= float(plng) <= 180
        )
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
