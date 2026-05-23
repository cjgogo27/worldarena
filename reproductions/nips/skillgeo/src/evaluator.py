import math

import re

from typing import Any

import numpy as np

from .skill_parser import COUNTRY_TO_REGION



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





DISTANCE_THRESHOLDS_KM = [1, 25, 150, 750, 2500]







MAX_PENALTY_KM = 20_037.0





def evaluate_predictions(records: list[dict[str, Any]]) -> dict[str, float]:

    country_hits = []

    continent_hits = []

    distances = []

    valid_distances = []

    hallucinations = []

    evidence_scores = []

    valid_coords_count = 0



    for rec in records:

        gt_country = rec["ground_truth_country"]

        pred_country = rec["prediction"].get("predicted_country", "unknown")

        country_hits.append(1.0 if gt_country == pred_country else 0.0)



        gt_region = COUNTRY_TO_REGION.get(gt_country, "unknown")

        pred_region = COUNTRY_TO_REGION.get(pred_country, "unknown")

        continent_hits.append(1.0 if gt_region == pred_region else 0.0)



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

            d = haversine_km(float(plat), float(plng), rec["ground_truth_lat"], rec["ground_truth_lng"])

            distances.append(d)

            valid_distances.append(d)

        else:

            distances.append(MAX_PENALTY_KM)



        model_reasoning = rec["prediction"].get("reasoning_text", "")

        expert = rec.get("expert_chain", "")

        hallucinations.append(1.0 if hallucination_flag(model_reasoning, expert) else 0.0)

        evidence_scores.append(expert_chain_token_f1(model_reasoning, expert))



    n = len(records)

    if n == 0:

        raise ValueError("Cannot evaluate empty list of predictions.")



    metrics: dict[str, float] = {

        "country_accuracy": float(np.mean(country_hits)) if country_hits else 0.0,

        "continent_accuracy": float(np.mean(continent_hits)) if continent_hits else 0.0,

        "distance_error_km_median": float(np.median(distances)) if distances else float("nan"),

        "distance_error_km_mean_valid_only": float(np.mean(valid_distances)) if valid_distances else float("nan"),

        "distance_error_km_mean_penalized": float(np.mean(distances)) if distances else float("nan"),

        "valid_coordinate_rate": valid_coords_count / n,

        "heuristic_hallucination_rate": float(np.mean(hallucinations)) if hallucinations else 0.0,

        "expert_chain_token_f1": float(np.mean(evidence_scores)) if evidence_scores else 0.0,

    }



    for threshold in DISTANCE_THRESHOLDS_KM:

        within = sum(1 for d in distances if d <= threshold)

        metrics[f"Acc@{threshold}km"] = within / n



    return metrics
