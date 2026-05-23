import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .skill_parser import COUNTRY_TO_REGION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geo_localization_metrics import GeoLocalizationMetrics


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\-]+", text.lower())


HALLUCINATION_SUSPICIOUS_PATTERNS = [
    "i can read the exact",
    "the sign clearly reads",
    "license plate reads",
    "the exact address is",
    "phone number is",
    "i can see the street name",
    "the gps coordinates show",
]


def hallucination_flag(model_reasoning: str, expert_chain: str) -> bool:
    model_lower = model_reasoning.lower()
    expert_lower = expert_chain.lower()
    for p in HALLUCINATION_SUSPICIOUS_PATTERNS:
        if p in model_lower and p not in expert_lower:
            return True
    model_tokens = set(tokenize(model_reasoning))
    expert_tokens = set(tokenize(expert_chain))
    fabricated_specifics = {"billboard", "storefront", "phone", "advertisement", "banner", "exact-address"}
    rare_claims = model_tokens & fabricated_specifics
    if rare_claims and not (rare_claims & expert_tokens):
        return True
    return False


def expert_chain_token_f1(model_reasoning: str, expert_chain: str) -> float:
    mt = tokenize(model_reasoning)
    et = tokenize(expert_chain)
    if not mt or not et:
        return 0.0
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "with", "this", "that", "it", "be", "has", "have",
        "from", "as", "by", "not", "can", "which", "their", "its", "more", "most",
        "very", "also", "likely", "common", "found", "could", "would", "may",
    }
    mset = set(mt) - stop_words
    eset = set(et) - stop_words
    if not mset or not eset:
        return 0.0
    tp = len(mset & eset)
    if tp == 0:
        return 0.0
    precision = tp / len(mset)
    recall = tp / len(eset)
    return 2 * precision * recall / (precision + recall)


# Requested threshold set for Acc@ metrics.
DISTANCE_THRESHOLDS_KM = [10, 25, 200, 750, 2000]

# Maximum possible distance on Earth (half circumference), used as penalty
# for predictions that fail to produce valid coordinates.
MAX_PENALTY_KM = 20_037.0


def _as_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def evaluate_predictions(records: list[dict[str, Any]]) -> dict[str, float]:
    country_hits_known = []
    continent_hits_known = []
    distances_penalized = []
    valid_distances = []
    valid_coords_count = 0

    for rec in records:
        gt_country = str(rec["ground_truth_country"]).strip().lower()
        pred_country = str(rec["prediction"].get("predicted_country", "unknown")).strip().lower()
        if gt_country != "unknown":
            country_hits_known.append(1.0 if gt_country == pred_country else 0.0)

        gt_region = COUNTRY_TO_REGION.get(gt_country, "unknown")
        pred_region = COUNTRY_TO_REGION.get(pred_country, "unknown")
        if gt_region != "unknown":
            continent_hits_known.append(1.0 if gt_region == pred_region else 0.0)

        plat = rec["prediction"].get("predicted_lat", math.nan)
        plng = rec["prediction"].get("predicted_lng", math.nan)
        has_coords = (
            isinstance(plat, (float, int))
            and isinstance(plng, (float, int))
            and not math.isnan(float(plat))
            and not math.isnan(float(plng))
            and -90 <= float(plat) <= 90
            and -180 <= float(plng) <= 180
        )
        if has_coords:
            valid_coords_count += 1
            d = GeoLocalizationMetrics.haversine_distance(
                rec["ground_truth_lat"],
                rec["ground_truth_lng"],
                float(plat),
                float(plng),
            )
            distances_penalized.append(d)
            valid_distances.append(d)
        else:
            distances_penalized.append(MAX_PENALTY_KM)

    n = len(records)
    if n == 0:
        raise ValueError("Cannot evaluate empty list of predictions.")

    penalized_report = GeoLocalizationMetrics.comprehensive_metrics(
        distances_km=distances_penalized,
        valid_count=valid_coords_count,
        total_count=n,
    )
    acc_at = GeoLocalizationMetrics.compute_accuracy_across_thresholds(
        distances_km=distances_penalized,
        thresholds_km=DISTANCE_THRESHOLDS_KM,
    )

    mean_valid_only = GeoLocalizationMetrics.mean_error(valid_distances)

    metrics: dict[str, float] = {
        "country_accuracy": float(np.mean(country_hits_known)) if country_hits_known else float("nan"),
        "continent_accuracy": float(np.mean(continent_hits_known)) if continent_hits_known else float("nan"),
        "country_known_rate": (len(country_hits_known) / n) if n > 0 else float("nan"),
        "continent_known_rate": (len(continent_hits_known) / n) if n > 0 else float("nan"),
        "distance_error_km_median": _as_float_or_nan(penalized_report.get("median_error_km", float("nan"))),
        "distance_error_km_mean_valid_only": float(mean_valid_only) if mean_valid_only is not None else float("nan"),
        "distance_error_km_mean_penalized": _as_float_or_nan(penalized_report.get("mean_error_km", float("nan"))),
        "distance_error_km_std_penalized": _as_float_or_nan(penalized_report.get("std_error_km", float("nan"))),
        "valid_coordinate_rate": valid_coords_count / n,
        "coverage_rate": float(penalized_report.get("coverage_rate", valid_coords_count / n)),
        "total_samples": float(penalized_report.get("total_samples", n)),
        "valid_predictions": float(penalized_report.get("valid_predictions", valid_coords_count)),
    }

    metrics.update({k: float(v) for k, v in acc_at.items()})

    return metrics
