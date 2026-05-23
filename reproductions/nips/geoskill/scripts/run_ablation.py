# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false, reportMissingTypeArgument=false

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from tqdm import tqdm

from src.baselines import (
    _GEOLOCATION_SYSTEM,
    _JSON_SCHEMA_INSTRUCTION,
    _parse_json_prediction,
)
from src.evaluator import evaluate_predictions
from src.skill_library import SkillLibrary
from src.skill_parser import Skill, parse_expert_chain
from src.vlm_client import VLMClient, VLMConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = PROJECT_ROOT / "experiments" / "ablation"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_human_expert_file(game_dir: Path) -> Path:
    matches = sorted(game_dir.glob("Human_Expert_*.txt"))
    if not matches:
        raise FileNotFoundError(f"No Human_Expert_*.txt found in {game_dir}")
    return matches[0]


def build_skill_library(data_root: Path, game_ids: list[str], embedding_model: str) -> SkillLibrary:
    library = SkillLibrary(embedding_model_name=embedding_model)
    all_skills: list[Skill] = []
    for gid in game_ids:
        gdir = data_root / gid
        expert_path = gdir / "Human_Expert_1.txt"
        if not expert_path.exists():
            try:
                expert_path = find_human_expert_file(gdir)
            except FileNotFoundError:
                continue
        expert_text = expert_path.read_text(encoding="utf-8")
        all_skills.extend(parse_expert_chain(expert_text, source_game_id=gid))
    library.add_skills(all_skills)
    print(f"Skill library built: {len(all_skills)} skills from {len(game_ids)} games")
    return library


def build_filtered_skill_library(
    skills: list[Skill],
    embedding_model: str,
    predicate: Callable[[Skill], bool],
) -> SkillLibrary:
    filtered = [s for s in skills if predicate(s)]
    library = SkillLibrary(embedding_model_name=embedding_model)
    library.add_skills(filtered)
    return library


def make_sample(data_root: Path, gid: str, round_idx: int = 1) -> dict[str, Any] | None:
    gdir = data_root / gid
    metadata_path = gdir / f"{gid}_rounds_metadata.json"
    image_path = gdir / f"{gid}_{round_idx}.png"

    if not metadata_path.exists() or not image_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if round_idx - 1 >= len(metadata):
        return None

    gt = metadata[round_idx - 1]
    expert_chain = ""
    expert_path = gdir / "Human_Expert_1.txt"
    if expert_path.exists():
        expert_chain = expert_path.read_text(encoding="utf-8")

    return {
        "game_id": gid,
        "round": round_idx,
        "image_path": str(image_path),
        "ground_truth_country": gt["streakLocationCode"].lower(),
        "ground_truth_lat": float(gt["lat"]),
        "ground_truth_lng": float(gt["lng"]),
        "expert_chain": expert_chain,
    }


def _describe_scene(vlm: VLMClient, image_path: str) -> str:
    describe_system = "You are a visual scene analyst. Describe concrete visual details only — no speculation about location."
    describe_user = (
        "Describe this street view image in 8-12 bullet points. Focus ONLY on:\n"
        "- Road surface, markings, driving side\n"
        "- Signs, text, scripts visible\n"
        "- Vegetation types, terrain, soil color\n"
        "- Architecture, building materials, roof types\n"
        "- Utility poles, power lines, bollards\n"
        "- Vehicles, license plates\n"
        "- Climate/weather indicators"
    )
    return vlm.query(
        image_path=image_path,
        system_prompt=describe_system,
        user_prompt=describe_user,
        temperature=0.1,
    )


def _format_skills(skills: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            f"- Skill {i+1} (region_hint={s.get('region_hint', 'unknown')}, confidence={float(s.get('confidence', 0.0)):.2f}): {s.get('skill_text', '')}"
            for i, s in enumerate(skills)
        ]
    )


def no_skill_predict(vlm: VLMClient, skill_library: SkillLibrary, image_path: str) -> dict[str, Any]:
    _ = skill_library
    scene_desc = _describe_scene(vlm, image_path)
    user = (
        "You are given a scene description generated from this same image. Use it as supporting context, "
        "but ground your final answer in the image evidence.\n\n"
        f"Scene Description:\n{scene_desc}\n\n"
        "Now provide your best geolocation prediction.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    out = vlm.query(image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=user, temperature=0.2)
    pred = _parse_json_prediction(out)
    pred["scene_description"] = scene_desc
    pred["retrieved_skills"] = []
    return pred


def random_skill_predict(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    top_k: int = 5,
    min_score: float = 0.15,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    _ = min_score
    scene_desc = _describe_scene(vlm, image_path)
    r = rng if rng is not None else random

    if not skill_library.skills:
        relevant: list[dict[str, Any]] = []
    else:
        selected = r.sample(skill_library.skills, k=min(top_k, len(skill_library.skills)))
        relevant = []
        for s in selected:
            payload = asdict(s)
            payload["score"] = 0.0
            payload["bm25_score"] = 0.0
            payload["semantic_score"] = 0.0
            relevant.append(payload)

    skills_text = _format_skills(relevant)
    user = (
        "Below are randomly selected expert-derived geographic reasoning skills. "
        "Use them cautiously and rely on actual image evidence when deciding location.\n\n"
        f"Expert Skills:\n{skills_text}\n\n"
        "Now analyze the image using your observations and any applicable skills.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    out = vlm.query(image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=user, temperature=0.2)
    pred = _parse_json_prediction(out)
    pred["retrieved_skills"] = relevant
    pred["scene_description"] = scene_desc
    return pred


def shuffled_order_predict(
    vlm: VLMClient,
    skill_library: SkillLibrary,
    image_path: str,
    top_k: int = 5,
    min_score: float = 0.15,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    scene_desc = _describe_scene(vlm, image_path)
    retrieved = skill_library.retrieve(scene_desc, top_k=top_k, alpha=0.5)
    relevant = [s for s in retrieved if s["score"] >= min_score]
    if not relevant:
        relevant = retrieved[:2]

    r = rng if rng is not None else random
    shuffled = list(relevant)
    r.shuffle(shuffled)

    skills_text = _format_skills(shuffled)
    user = (
        "Below are expert-derived geographic reasoning skills that MAY be relevant to this image. "
        "IMPORTANT: Only apply skills that match what you actually see in the image. "
        "If a skill contradicts the visual evidence, IGNORE it and rely on the image.\n\n"
        f"Expert Skills:\n{skills_text}\n\n"
        "Now analyze the image using both your own observations and any applicable skills.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    out = vlm.query(image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=user, temperature=0.2)
    pred = _parse_json_prediction(out)
    pred["retrieved_skills"] = shuffled
    pred["scene_description"] = scene_desc
    return pred


def filtered_skill_predict(
    vlm: VLMClient,
    filtered_library: SkillLibrary,
    image_path: str,
    top_k: int = 5,
    min_score: float = 0.15,
    label: str = "filtered",
) -> dict[str, Any]:
    scene_desc = _describe_scene(vlm, image_path)
    retrieved = filtered_library.retrieve(scene_desc, top_k=top_k, alpha=0.5)
    relevant = [s for s in retrieved if s["score"] >= min_score]
    if not relevant:
        relevant = retrieved[:2]

    skills_text = _format_skills(relevant)
    user = (
        f"Below are expert-derived geographic reasoning skills from the {label} subset. "
        "IMPORTANT: Only apply skills that match what you actually see in the image. "
        "If a skill contradicts the visual evidence, IGNORE it and rely on the image.\n\n"
        f"Expert Skills:\n{skills_text}\n\n"
        "Now analyze the image using both your own observations and any applicable skills.\n\n"
        + _JSON_SCHEMA_INSTRUCTION
    )
    out = vlm.query(image_path=image_path, system_prompt=_GEOLOCATION_SYSTEM, user_prompt=user, temperature=0.2)
    pred = _parse_json_prediction(out)
    pred["retrieved_skills"] = relevant
    pred["scene_description"] = scene_desc
    return pred


def run_method_on_sample(method_fn: Callable[[str], dict[str, Any]], sample: dict[str, Any]) -> dict[str, Any]:
    try:
        pred = method_fn(sample["image_path"])
        return {
            "game_id": sample["game_id"],
            "round": sample["round"],
            "ground_truth_country": sample["ground_truth_country"],
            "ground_truth_lat": sample["ground_truth_lat"],
            "ground_truth_lng": sample["ground_truth_lng"],
            "expert_chain": sample["expert_chain"],
            "prediction": pred,
            "error": None,
        }
    except Exception as exc:
        return {
            "game_id": sample["game_id"],
            "round": sample["round"],
            "ground_truth_country": sample["ground_truth_country"],
            "ground_truth_lat": sample["ground_truth_lat"],
            "ground_truth_lng": sample["ground_truth_lng"],
            "expert_chain": sample["expert_chain"],
            "prediction": {
                "predicted_country": "unknown",
                "predicted_region": "unknown",
                "predicted_lat": float("nan"),
                "predicted_lng": float("nan"),
                "reasoning_text": f"ERROR: {exc}",
                "evidence_spans": [],
                "confidence": 0.0,
            },
            "error": str(exc),
        }


def save_results(exp_dir: Path, method_name: str, records: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    method_dir = exp_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    (method_dir / f"predictions_{timestamp}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (method_dir / f"metrics_{timestamp}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (method_dir / "latest_predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (method_dir / "latest_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full.yaml")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Subset of ablation methods to run",
    )
    parser.add_argument("--max-games", type=int, default=None, help="Limit number of games")
    parser.add_argument("--round", type=int, default=1, help="Which round to evaluate (1-5)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers per method")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random/shuffled variants")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    data_root = PROJECT_ROOT / cfg["paths"]["data_root"]
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    game_ids = cfg["dataset"].get("game_ids", [])
    if args.max_games:
        game_ids = game_ids[: args.max_games]

    samples = []
    for gid in game_ids:
        s = make_sample(data_root, gid, round_idx=args.round)
        if s:
            samples.append(s)
    print(f"Loaded {len(samples)}/{len(game_ids)} samples for round {args.round}")

    if not samples:
        print("ERROR: No valid samples found. Check data directory.")
        sys.exit(1)

    vlm = VLMClient(
        VLMConfig(
            base_url=cfg["vlm"]["base_url"],
            api_key=os.getenv("VLM_API_KEY", cfg["vlm"].get("api_key", "")),
            model=cfg["vlm"]["model"],
            max_tokens=cfg["vlm"].get("max_tokens", 1500),
            max_image_side=cfg["vlm"].get("max_image_side", 1024),
            retries=cfg["vlm"].get("retries", 3),
            backoff_seconds=cfg["vlm"].get("backoff_seconds", 1.0),
        )
    )

    skill_cfg = cfg.get("skills", {})
    top_k = int(skill_cfg.get("top_k", 5))
    min_score = float(skill_cfg.get("min_score", 0.15))
    embedding_model = skill_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    skill_library = build_skill_library(
        data_root=data_root,
        game_ids=game_ids,
        embedding_model=embedding_model,
    )

    atomic_lib = build_filtered_skill_library(
        skill_library.skills,
        embedding_model=embedding_model,
        predicate=lambda s: len(s.skill_text.split()) < 25,
    )
    composed_lib = build_filtered_skill_library(
        skill_library.skills,
        embedding_model=embedding_model,
        predicate=lambda s: len(s.skill_text.split()) >= 25,
    )
    print(f"Atomic skills: {len(atomic_lib.skills)} | Composed skills: {len(composed_lib.skills)}")

    rng_random = random.Random(args.seed)
    rng_shuffle = random.Random(args.seed)

    all_methods: dict[str, Callable[[str], dict[str, Any]]] = {
        "no_skill": lambda img: no_skill_predict(vlm, skill_library, img),
        "random_skill": lambda img: random_skill_predict(
            vlm, skill_library, img, top_k=top_k, min_score=min_score, rng=rng_random
        ),
        "shuffled_order": lambda img: shuffled_order_predict(
            vlm, skill_library, img, top_k=top_k, min_score=min_score, rng=rng_shuffle
        ),
        "atomic_only": lambda img: filtered_skill_predict(
            vlm, atomic_lib, img, top_k=top_k, min_score=min_score, label="atomic_only"
        ),
        "composed_only": lambda img: filtered_skill_predict(
            vlm, composed_lib, img, top_k=top_k, min_score=min_score, label="composed_only"
        ),
    }

    methods_to_run = args.methods if args.methods else list(all_methods.keys())
    print(f"Methods to run: {methods_to_run}")

    all_metrics: dict[str, dict[str, Any]] = {}

    for method_name in methods_to_run:
        if method_name not in all_methods:
            print(f"WARNING: Unknown method '{method_name}', skipping")
            continue

        fn = all_methods[method_name]
        print(f"\n{'=' * 60}")
        print(f"Running: {method_name} ({len(samples)} samples)")
        print(f"{'=' * 60}")

        records: list[dict[str, Any]] = []
        errors = 0
        start_time = time.time()

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(run_method_on_sample, fn, s): s["game_id"]
                    for s in samples
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=method_name):
                    result = future.result()
                    records.append(result)
                    if result.get("error"):
                        errors += 1
        else:
            for sample in tqdm(samples, desc=method_name):
                result = run_method_on_sample(fn, sample)
                records.append(result)
                if result.get("error"):
                    errors += 1

        elapsed = time.time() - start_time
        records.sort(key=lambda r: r["game_id"])

        metrics = evaluate_predictions(records)
        metrics["elapsed_seconds"] = elapsed
        metrics["num_samples"] = len(records)
        metrics["num_errors"] = errors

        save_results(ABLATION_DIR, method_name, records, metrics)
        all_metrics[method_name] = metrics

        print(f"\n{method_name} results ({elapsed:.1f}s, {errors} errors):")
        print(f"  Country Acc: {metrics['country_accuracy']:.3f}")
        print(f"  Continent Acc: {metrics['continent_accuracy']:.3f}")
        print(f"  Distance (median): {metrics['distance_error_km_median']:.1f} km")
        print(f"  Distance (mean valid-only): {metrics['distance_error_km_mean_valid_only']:.1f} km")
        print(f"  Distance (mean penalized): {metrics['distance_error_km_mean_penalized']:.1f} km")
        print(f"  Distance (std penalized): {metrics['distance_error_km_std_penalized']:.1f} km")
        print(f"  Valid Coords: {metrics['valid_coordinate_rate']:.2%}")
        print(f"  Acc@1km: {metrics['Acc@1km']:.3f}")
        print(f"  Acc@10km: {metrics['Acc@10km']:.3f}")
        print(f"  Acc@25km: {metrics['Acc@25km']:.3f}")
        print(f"  Acc@100km: {metrics['Acc@100km']:.3f}")
        print(f"  Acc@200km: {metrics['Acc@200km']:.3f}")
        print(f"  Acc@750km: {metrics['Acc@750km']:.3f}")
        print(f"  Acc@2500km: {metrics['Acc@2500km']:.3f}")

    summary_path = ABLATION_DIR / "summary_metrics.json"
    summary_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    header = (
        f"{'Method':<18} {'Country':>8} {'Continent':>10} {'DistMed':>8} {'Valid%':>8} "
        f"{'A@1':>6} {'A@10':>6} {'A@25':>6} {'A@100':>7} {'A@200':>7} {'A@750':>7} {'A@2500':>8}"
    )
    print(header)
    print("-" * len(header))
    for method, m in all_metrics.items():
        dist_str = f"{m['distance_error_km_median']:.0f}" if not (m['distance_error_km_median'] != m['distance_error_km_median']) else "N/A"
        print(
            f"{method:<18} {m['country_accuracy']:>8.3f} {m['continent_accuracy']:>8.3f} "
            f"{dist_str:>10} {m['valid_coordinate_rate']:>7.1%} {m['Acc@1km']:>6.3f} {m['Acc@10km']:>6.3f} {m['Acc@25km']:>6.3f} "
            f"{m['Acc@100km']:>7.3f} {m['Acc@200km']:>7.3f} {m['Acc@750km']:>7.3f} {m['Acc@2500km']:>8.3f}"
        )

    print(f"\nAll ablation results saved to: {ABLATION_DIR}")


if __name__ == "__main__":
    main()
