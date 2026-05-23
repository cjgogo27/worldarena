import json

import os

import sys

import time

import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime, timezone

from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import yaml

from tqdm import tqdm



from src.baselines import (

    cot_vlm_predict,

    direct_vlm_predict,

    geocomp_external_predict,

    geocot_predict,

    geovista_external_predict,

    georeasoner_predict,

    gre_multistage_predict,

    img2loc_rag_predict,

    safa_external_proxy_predict,

    skill_conditioned_predict,

    skill_conditioned_v2_predict,

    skill_conditioned_v3_predict,

    skill_conditioned_v4_predict,

)

from src.evaluator import evaluate_predictions

from src.skill_library import SkillLibrary

from src.skill_optimizer import dump_skills_jsonl, fuse_atomic_skills, synthesize_failure_skills

from src.skill_parser import parse_candidate_chain, parse_expert_chain

from src.vlm_client import VLMClient, VLMConfig



PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWN_EXTERNAL_BASELINES = ["GeoComp", "GeoReasoner", "GeoVista", "SAFA"]

EXTERNAL_BASELINE_METHODS = {

    "GeoComp": "external_geocomp",

    "GeoReasoner": "external_georeasoner",

    "GeoVista": "external_geovista",

    "SAFA": "external_safa",

}



ALL_100_GAME_IDS = [

    "1NJsXTxIF9GGMDxC", "2xnQdwiCve2rHWVt", "3I4ZtihbhZy5qZzQ", "3OuVMcpGjmm0tVfG", "3uP6lYo9pzx5Q0km",

    "4UvmdTHySo6AXW4M", "56Q4T4rpv9O9sCpP", "5GS2RPgTrf85UZM5", "5bPlIRT7eGN79a2E", "5l0GTCFZI877KxkV",

    "67gLC5CcGgkQIWEW", "6fGwHxCTvCbaK77Q", "6ypQOh9cOoE7WaWH", "74bPHM081cMUaNKT", "8G5DpHP2KCtVKrk9",

    "8Uo6ejwXYqmp9av3", "9NNxopqtafH8pTbN", "9lNwy1vjD53PTSwt", "9oZfZYQEl9GWjZPu", "AVKTblAzBqaYrcKe",

    "DKGAIVKGYnLWmMy9", "EEI3fwu4iZvPT2zP", "G3aNW5xo5JUCnAhB", "GbfNQanBbRoPIoek", "HF1MKwIFNCNb4FdM",

    "HLDekf7z49ovS9Y8", "HVBiwFr2gEcM1dze", "IozkbMt8zbdH9XCv", "JXpmlobJL7nhnp3E", "JfPkjboMSjCsG1Qu",

    "JnTw9kl2nWPFaoUg", "KTKKKeKESyntpGnS", "KWr0ySMIaUQRvikA", "KXkXw08ZCDqL1e2q", "KYuWftPRQfRJ5DKL",

    "KZ2f6LqzJRyChcg8", "KqjXykSvjIJYNXY8", "Lz6IDBZmUTr5oPdV", "MjBPBTn3iUXdebSu", "N2IFsQAIIcXPuNKn",

    "N4EN1CWspaIlcCYq", "No9HbFZyTOwfaU8s", "NqhvwHTYOTtGmZgm", "QTmrnxW99iiyS2xS", "QhbT3Koc9Mxo6EdL",

    "STQGgl6Uh9muExNb", "SkFHwlV4Z2q9fUje", "SwSJB5DW8YXOSLn2", "UYeUpGpQlvb7ExfK", "VGLhN3OEa5t3YAFj",

    "VpFDohrcFkP6vSIB", "X7L1cT0kKi0gYjlg", "XFWA7xovGlFChF2o", "XdOtqbtqAbSvRHij", "Xf8Xjx82sajsyJdw",

    "XwCZ4C6astQYennV", "Y2QhKx7sks9MExvw", "YlXQJ8JIXvcRXwF5", "ZMSyLdPt9G7MhcHC", "ZxTL8aae9sCgxLlS",

    "aKDWzyoV4cSA6Da2", "amHvOfYODNHwXiJg", "bwyAndeFczBo5Gzj", "cObaVm83g5wtLkhH", "cV9n1YpMOmaTJpG7",

    "d8lnc9Ex03KQUJEL", "e1zn5HV7y0bMjzQL", "e3PS9nBwbU94sIyP", "f49AGnnCYZbZVLbW", "f7TDBGoaR3hv0dVw",

    "iF7CiuoojlFChTQ9", "k4PVd80ZGfSSesQF", "kMEhvQ4rbv1USOGs", "kkJz0TJOX5PkNDIu", "kkQArBREMnrdoW0R",

    "lCqI2T7mSJAgSbAi", "lHnt4Veq3Tu10VKJ", "lesmx8gOI1f44LS4", "mMZ8js56Rj2JtLlc", "nnxZ3V6PpY9PoPLX",

    "oIqmyQzBGtaIeI84", "oUaCVxfbGe5Aj0AA", "qDVJNPKVIYAK5Irg", "qKKfsK89gK6yZii9", "qOTM2e7p8dcC02Fz",

    "qPJtxO4pQJtlqpth", "qREuLz0JK8CfSFMC", "qRoLM1xS7DE5KEjE", "sFHfIZ209HMmIVTP", "sMPhUAFKpkB21pnD",

    "sf2wEErB2IDY3qJ5", "sq3HZiEly0mZ6jNJ", "tFoKseRvMTUy6eCq", "uluG1Pbks9IwTjsA", "va4YOTJhmCbe6lWr",

    "w3MFlsmvpeCTNUcG", "xYdYhcgnMuyj35eG", "xnoF4a1DE73F1kiV", "z2mhsiTu4DYWixQf", "z5NxwG2Ul6PVWSaj",

]





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

    all_skills = []

    for gid in game_ids:

        gdir = data_root / gid

        expert_path = gdir / "Human_Expert_1.txt"

        if not expert_path.exists():

            try:

                expert_path = find_human_expert_file(gdir)

            except FileNotFoundError:

                expert_path = None

        if expert_path and expert_path.exists():

            expert_text = expert_path.read_text(encoding="utf-8")

            all_skills.extend(parse_expert_chain(expert_text, source_game_id=gid))

        for round_idx in range(1, 6):

            cand_path = gdir / f"candidate_reasoning_chain_gpt4_{round_idx}.txt"

            if cand_path.exists():

                all_skills.extend(

                    parse_candidate_chain(cand_path.read_text(encoding="utf-8"), source_game_id=gid, round_num=round_idx)

                )

    library.add_skills(all_skills)

    print(f"Skill library built: {len(all_skills)} skills from {len(game_ids)} games")

    return library





def _retry_failed_with_updated_library(method_fn, records: list[dict]) -> tuple[list[dict], int]:

    improved = 0

    fixed_records = []

    for rec in records:

        if rec.get("error") is None:

            fixed_records.append(rec)

            continue

        retry_input = {

            "game_id": rec["game_id"],

            "round": rec["round"],

            "image_path": rec.get("image_path") or "",

            "ground_truth_country": rec["ground_truth_country"],

            "ground_truth_lat": rec["ground_truth_lat"],

            "ground_truth_lng": rec["ground_truth_lng"],

            "expert_chain": rec.get("expert_chain", ""),

        }

        retried = run_method_on_sample(method_fn, retry_input)

        if retried.get("error") is None:

            improved += 1

            fixed_records.append(retried)

        else:

            fixed_records.append(rec)

    return fixed_records, improved





def make_sample(data_root: Path, gid: str, round_idx: int = 1) -> dict | None:

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





def run_method_on_sample(method_fn, sample: dict) -> dict:

    try:

        pred = method_fn(sample["image_path"])

        return {

            "game_id": sample["game_id"],

            "round": sample["round"],

            "image_path": sample["image_path"],

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

            "image_path": sample["image_path"],

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





def save_results(exp_dir: Path, method_name: str, records: list[dict], metrics: dict) -> None:

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





def _load_existing_summary(summary_path: Path) -> dict:

    if not summary_path.exists():

        return {}

    try:

        return json.loads(summary_path.read_text(encoding="utf-8"))

    except json.JSONDecodeError:

        return {}





def _scan_external_baseline_status(project_root: Path, exp_dir: Path) -> dict[str, dict]:

    status: dict[str, dict] = {}

    ext_root = project_root / "external_baselines"

    artifact_suffixes = {

        ".jsonl", ".log", ".ckpt", ".pt", ".pth", ".bin", ".safetensors", ".csv",

    }



    for name in KNOWN_EXTERNAL_BASELINES:

        baseline_dir = ext_root / name

        if not baseline_dir.exists():

            status[name] = {

                "present": False,

                "has_run_or_eval_script": False,

                "local_results_found": False,

                "artifact_file_count": 0,

                "artifact_examples": [],

            }

            continue



        candidate_scripts = list(baseline_dir.rglob("*.py")) + list(baseline_dir.rglob("*.sh"))

        run_eval_script_found = any(

            token in p.name.lower()

            for p in candidate_scripts

            for token in ("run", "infer", "eval", "test", "train")

        )



        artifact_files = []

        for p in baseline_dir.rglob("*"):

            if not p.is_file():

                continue

            if ".git" in p.parts or "node_modules" in p.parts:

                continue

            if p.suffix.lower() in artifact_suffixes:

                artifact_files.append(p)



        method_alias = EXTERNAL_BASELINE_METHODS.get(name)

        local_result_hits = list(exp_dir.glob(f"{name}/**/latest_metrics*.json"))

        if method_alias:

            local_result_hits += list(exp_dir.glob(f"{method_alias}/**/latest_metrics*.json"))



        status[name] = {

            "present": True,

            "has_run_or_eval_script": bool(run_eval_script_found),

            "local_results_found": len(local_result_hits) > 0,

            "local_result_files": [str(p.relative_to(project_root)) for p in local_result_hits[:10]],

            "artifact_file_count": len(artifact_files),

            "artifact_examples": [str(p.relative_to(project_root)) for p in artifact_files[:10]],

        }



    return status





def _write_audit_report(

    exp_dir: Path,

    config_path: Path,

    game_ids: list[str],

    samples: list[dict],

    methods_requested: list[str],

    method_metrics: dict[str, dict],

) -> None:

    loaded_ids = {s["game_id"] for s in samples}

    missing_ids = [gid for gid in game_ids if gid not in loaded_ids]



    method_status = {}

    for method_name, metrics in method_metrics.items():

        n_samples = int(metrics.get("num_samples", 0))

        n_errors = int(metrics.get("num_errors", 0))

        method_status[method_name] = {

            "num_samples": n_samples,

            "num_errors": n_errors,

            "num_success": max(0, n_samples - n_errors),

            "error_rate": (n_errors / n_samples) if n_samples > 0 else 0.0,

        }



    audit = {

        "timestamp_utc": datetime.now(timezone.utc).isoformat(),

        "project_root": str(PROJECT_ROOT),

        "config": str(config_path.relative_to(PROJECT_ROOT)),

        "dataset": {

            "expected_game_count": len(game_ids),

            "loaded_sample_count": len(samples),

            "missing_game_count": len(missing_ids),

            "missing_game_ids": missing_ids,

        },

        "run": {

            "methods_requested": methods_requested,

            "methods_executed": list(method_metrics.keys()),

        },

        "method_status": method_status,

        "external_baselines": _scan_external_baseline_status(PROJECT_ROOT, exp_dir),

    }



    (exp_dir / "run_audit.json").write_text(

        json.dumps(audit, ensure_ascii=False, indent=2, default=str),

        encoding="utf-8",

    )





def main() -> None:

    import argparse



    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/full.yaml")

    parser.add_argument("--methods", type=str, nargs="+", default=None,

                        help="Subset of methods to run (default: all)")

    parser.add_argument("--max-games", type=int, default=None,

                        help="Limit number of games (for testing)")

    parser.add_argument("--round", type=int, default=1,

                        help="Which round to evaluate (1-5)")

    parser.add_argument("--workers", type=int, default=1,

                        help="Parallel workers per method")

    args = parser.parse_args()



    cfg = load_config(PROJECT_ROOT / args.config)

    data_root = PROJECT_ROOT / cfg["paths"]["data_root"]

    exp_dir = PROJECT_ROOT / cfg["paths"]["experiment_dir"]

    exp_dir.mkdir(parents=True, exist_ok=True)



    game_ids = cfg["dataset"].get("game_ids", ALL_100_GAME_IDS)

    if args.max_games:

        game_ids = game_ids[:args.max_games]



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

            request_timeout_seconds=cfg["vlm"].get("request_timeout_seconds", 45.0),

        )

    )



    skill_library = build_skill_library(

        data_root=data_root,

        game_ids=game_ids,

        embedding_model=cfg["skills"]["embedding_model"],

    )



    skill_cfg = cfg.get("skills", {})

    top_k = skill_cfg.get("top_k", 5)

    min_score = skill_cfg.get("min_score", 0.15)

    retrieval_mode = skill_cfg.get("retrieval_mode", "hybrid")



    training_free_cfg = cfg.get("training_free", {})

    enable_failure_recovery = bool(training_free_cfg.get("enable_failure_recovery", True))

    enable_skill_fusion = bool(training_free_cfg.get("enable_skill_fusion", True))

    recovery_max_records = int(training_free_cfg.get("recovery_max_records", 20))

    skill_update_dir = exp_dir / "skill_updates"

    skill_update_dir.mkdir(parents=True, exist_ok=True)



    all_methods = {

        "direct_vlm": lambda img: direct_vlm_predict(vlm, img),

        "cot_vlm": lambda img: cot_vlm_predict(vlm, img),

        "external_geocomp": lambda img: geocomp_external_predict(vlm, img),

        "external_georeasoner": lambda img: georeasoner_predict(vlm, img),

        "external_geovista": lambda img: geovista_external_predict(vlm, img),

        "external_safa": lambda img: safa_external_proxy_predict(vlm, skill_library, img, retrieval_mode=retrieval_mode),

        "skill_conditioned": lambda img: skill_conditioned_predict(

            vlm, skill_library, img, top_k=top_k, min_score=min_score, retrieval_mode=retrieval_mode

        ),

        "skill_conditioned_v2": lambda img: skill_conditioned_v2_predict(

            vlm, skill_library, img, top_k=top_k, retrieval_mode=retrieval_mode

        ),

        "skill_conditioned_v3": lambda img: skill_conditioned_v3_predict(

            vlm, skill_library, img, top_k=10, retrieval_mode=retrieval_mode

        ),

        "skill_conditioned_v4": lambda img: skill_conditioned_v4_predict(

            vlm, skill_library, img, top_k=12, retrieval_mode=retrieval_mode

        ),

        "georeasoner": lambda img: georeasoner_predict(vlm, img),

        "geocot": lambda img: geocot_predict(vlm, img),

        "gre_multistage": lambda img: gre_multistage_predict(vlm, img),

        "img2loc_rag": lambda img: img2loc_rag_predict(vlm, skill_library, img, top_k=8, retrieval_mode=retrieval_mode),

    }



    methods_to_run = args.methods if args.methods else list(all_methods.keys())

    print(f"Methods to run: {methods_to_run}")



    all_metrics = {}



    for method_name in methods_to_run:

        if method_name not in all_methods:

            print(f"WARNING: Unknown method '{method_name}', skipping")

            continue



        fn = all_methods[method_name]

        print(f"\n{'='*60}")

        print(f"Running: {method_name} ({len(samples)} samples)")

        print(f"{'='*60}")



        records = []

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



        if method_name.startswith("skill_conditioned") and enable_failure_recovery:

            failed = [r for r in records if r.get("error") is not None]

            if failed:

                try:

                    recovered_skills = synthesize_failure_skills(

                        vlm=vlm,

                        failed_records=failed,

                        max_records=recovery_max_records,

                    )

                    if enable_skill_fusion:

                        fused_skills = fuse_atomic_skills(recovered_skills)

                    else:

                        fused_skills = []



                    if recovered_skills or fused_skills:

                        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

                        if recovered_skills:

                            dump_skills_jsonl(recovered_skills, skill_update_dir / f"recovered_{method_name}_{ts}.jsonl")

                            skill_library.add_skills(recovered_skills)

                        if fused_skills:

                            dump_skills_jsonl(fused_skills, skill_update_dir / f"fused_{method_name}_{ts}.jsonl")

                            skill_library.add_skills(fused_skills)



                        records, improved_count = _retry_failed_with_updated_library(fn, records)

                        metrics = evaluate_predictions(records)

                        metrics["elapsed_seconds"] = elapsed

                        metrics["num_samples"] = len(records)

                        metrics["num_errors"] = sum(1 for r in records if r.get("error") is not None)

                        metrics["recovered_failures"] = improved_count

                        metrics["generated_skill_count"] = len(recovered_skills)

                        metrics["fused_skill_count"] = len(fused_skills)

                except Exception as recovery_exc:

                    metrics["recovery_error"] = str(recovery_exc)



        save_results(exp_dir, method_name, records, metrics)

        all_metrics[method_name] = metrics



        print(f"\n{method_name} results ({elapsed:.1f}s, {errors} errors):")

        print(f"  Country Acc: {metrics['country_accuracy']:.3f}")

        print(f"  Continent Acc: {metrics['continent_accuracy']:.3f}")

        print(f"  Distance (median): {metrics['distance_error_km_median']:.1f} km")

        print(f"  Distance (mean valid-only): {metrics['distance_error_km_mean_valid_only']:.1f} km")

        print(f"  Distance (mean penalized): {metrics['distance_error_km_mean_penalized']:.1f} km")

        print(f"  Valid Coords: {metrics['valid_coordinate_rate']:.2%}")

        print(f"  Acc@1km: {metrics['Acc@1km']:.3f}")

        print(f"  Acc@25km: {metrics['Acc@25km']:.3f}")

        print(f"  Acc@150km: {metrics['Acc@150km']:.3f}")

        print(f"  Acc@750km: {metrics['Acc@750km']:.3f}")

        print(f"  Acc@2500km: {metrics['Acc@2500km']:.3f}")

        print(f"  Heuristic Hallucination Rate: {metrics['heuristic_hallucination_rate']:.3f}")

        print(f"  Expert Chain Token F1: {metrics['expert_chain_token_f1']:.3f}")



    summary_path = exp_dir / "summary_metrics.json"

    prior_summary = _load_existing_summary(summary_path)

    if isinstance(prior_summary, dict):

        prior_summary.update(all_metrics)

        merged_summary = prior_summary

    else:

        merged_summary = dict(all_metrics)

    merged_summary["_meta"] = {

        "last_updated_utc": datetime.now(timezone.utc).isoformat(),

        "methods_in_last_run": methods_to_run,

        "num_methods_total_in_summary": len([k for k in merged_summary.keys() if k != "_meta"]),

    }

    summary_path.write_text(json.dumps(merged_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")



    _write_audit_report(

        exp_dir=exp_dir,

        config_path=PROJECT_ROOT / args.config,

        game_ids=game_ids,

        samples=samples,

        methods_requested=methods_to_run,

        method_metrics=all_metrics,

    )



    print(f"\n{'='*80}")

    print("FINAL SUMMARY")

    print(f"{'='*80}")

    header = (

        f"{'Method':<22} {'Country':>8} {'Continent':>10} {'DistMed':>8} {'Valid%':>8} "

        f"{'A@1':>6} {'A@25':>6} {'A@150':>7} {'A@750':>7} {'A@2500':>8} {'TokF1':>7} {'HallucH':>8}"

    )

    print(header)

    print("-" * len(header))

    for method, m in all_metrics.items():

        dist_str = f"{m['distance_error_km_median']:.0f}" if not (m['distance_error_km_median'] != m['distance_error_km_median']) else "N/A"

        print(

            f"{method:<22} {m['country_accuracy']:>8.3f} {m['continent_accuracy']:>8.3f} "

            f"{dist_str:>10} {m['valid_coordinate_rate']:>7.1%} {m['Acc@1km']:>6.3f} {m['Acc@25km']:>6.3f} "

            f"{m['Acc@150km']:>7.3f} {m['Acc@750km']:>7.3f} {m['Acc@2500km']:>8.3f} "

            f"{m['expert_chain_token_f1']:>7.3f} {m['heuristic_hallucination_rate']:>8.3f}"

        )

        n_samples = int(m.get("num_samples", 0))

        n_errors = int(m.get("num_errors", 0))

        print(f"  -> success={max(0, n_samples - n_errors)}/{n_samples}, fail={n_errors}/{n_samples}")



    print(f"\nAll results saved to: {exp_dir}")





if __name__ == "__main__":

    main()
