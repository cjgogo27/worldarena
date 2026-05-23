import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from tqdm import tqdm
from geo_localization_metrics import GeoLocalizationMetrics

from src.baselines import (
    cot_vlm_predict,
    direct_vlm_predict,
    ep_bev_external_predict,
    geocomp_external_predict,
    geocot_predict,
    georeasoner_external_skill_boost_predict,
    geovista_external_skill_boost_predict,
    geovista_external_predict,
    georeasoner_predict,
    georeasoner_external_predict,
    gre_multistage_predict,
    img2loc_rag_predict,
    safa_external_predict,
    sample4geo_external_predict,
    skill_conditioned_predict,
    skill_conditioned_v2_predict,
    skill_conditioned_v3_predict,
    skill_conditioned_v4_predict,
)
from src.evaluator import evaluate_predictions
from src.GeoVista.skill_graph_runtime import summarize_rollout_skill_edges
from src.skill_library import SkillLibrary
from src.skill_optimizer import dump_skills_jsonl, fuse_atomic_skills, synthesize_failure_skills
from src.skill_parser import COUNTRY_NAME_TO_ISO2, parse_candidate_chain, parse_expert_chain
from src.vlm_client import VLMClient, VLMConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWN_EXTERNAL_BASELINES = ["GeoComp", "GeoReasoner", "GeoVista", "SAFA", "EP-BEV", "Sample4Geo"]
EXTERNAL_BASELINE_METHODS = {
    "GeoComp": "external_geocomp",
    "GeoReasoner": "external_georeasoner",
    "GeoVista": "external_geovista",
    "SAFA": "external_safa",
    "EP-BEV": "external_ep_bev",
    "Sample4Geo": "external_sample4geo",
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

_GEO_METRICS = GeoLocalizationMetrics()


def _sorted_acc_keys(metrics: dict[str, Any]) -> list[str]:
    keys: list[tuple[float, str]] = []
    for k in metrics.keys():
        if not isinstance(k, str) or not k.startswith("Acc@") or not k.endswith("km"):
            continue
        num = k[4:-2]
        try:
            keys.append((float(num), k))
        except ValueError:
            continue
    keys.sort(key=lambda x: x[0])
    return [k for _, k in keys]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_vlm_block(cfg: dict, keys: list[str], default_key: str | None = "vlm") -> dict[str, Any] | None:
    for key in keys:
        block = cfg.get(key)
        if not isinstance(block, dict):
            continue
        if "base_url" in block and "model" in block:
            return block
        nested = block.get("vlm")
        if isinstance(nested, dict) and "base_url" in nested and "model" in nested:
            return nested
    if default_key:
        fallback = cfg.get(default_key)
        if isinstance(fallback, dict) and "base_url" in fallback and "model" in fallback:
            return fallback
    return None


def _build_vlm_client(vlm_cfg: dict[str, Any], env_keys: list[str]) -> VLMClient:
    api_key = ""
    for env_key in env_keys:
        env_val = os.getenv(env_key, "")
        if env_val:
            api_key = env_val
            break
    if not api_key:
        api_key = str(vlm_cfg.get("api_key", ""))

    return VLMClient(
        VLMConfig(
            base_url=vlm_cfg["base_url"],
            api_key=api_key,
            model=vlm_cfg["model"],
            max_tokens=vlm_cfg.get("max_tokens", 1500),
            max_image_side=vlm_cfg.get("max_image_side", 1024),
            retries=vlm_cfg.get("retries", 3),
            backoff_seconds=vlm_cfg.get("backoff_seconds", 1.0),
            request_timeout_seconds=vlm_cfg.get("request_timeout_seconds", 45.0),
            enable_thinking=vlm_cfg.get("enable_thinking"),
        )
    )


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


def _as_valid_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def _is_failed_record(record: dict, failure_distance_km: float) -> bool:
    if record.get("error") is not None:
        return True

    pred = record.get("prediction") or {}
    gt_country = str(record.get("ground_truth_country", "unknown")).strip().lower()
    pred_country = str(pred.get("predicted_country", "unknown")).strip().lower()

    if not pred_country or pred_country == "unknown" or pred_country != gt_country:
        return True

    gt_lat = _as_valid_float(record.get("ground_truth_lat"))
    gt_lng = _as_valid_float(record.get("ground_truth_lng"))
    pred_lat = _as_valid_float(pred.get("predicted_lat"))
    pred_lng = _as_valid_float(pred.get("predicted_lng"))
    if gt_lat is None or gt_lng is None or pred_lat is None or pred_lng is None:
        return True

    dist_km = _GEO_METRICS.haversine_distance(gt_lat, gt_lng, pred_lat, pred_lng)
    if dist_km != dist_km:
        return True
    return dist_km > failure_distance_km


def _retry_failed_with_updated_library(
    method_fn,
    records: list[dict],
    failure_distance_km: float,
) -> tuple[list[dict], int]:
    improved = 0
    fixed_records = []
    for rec in records:
        if not _is_failed_record(rec, failure_distance_km):
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
            "sample_id": rec.get("sample_id"),
            "dataset_name": rec.get("dataset_name"),
            "dataset_version": rec.get("dataset_version"),
        }
        retried = run_method_on_sample(method_fn, retry_input)
        if not _is_failed_record(retried, failure_distance_km):
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
        "sample_id": f"georc::{gid}::round{round_idx}",
        "dataset_name": "georc",
        "dataset_version": "georc_v1",
        "round": round_idx,
        "image_path": str(image_path),
        "ground_truth_country": gt["streakLocationCode"].lower(),
        "ground_truth_lat": float(gt["lat"]),
        "ground_truth_lng": float(gt["lng"]),
        "expert_chain": expert_chain,
    }


def _infer_country_from_record(record: dict) -> str:
    invalid_tokens = {"", "unknown", "na", "n/a", "none", "null", "nan"}
    for key in ["country_code", "country", "gt_country", "label_country", "streakLocationCode"]:
        val = record.get(key)
        if not isinstance(val, str):
            continue
        raw = val.strip().lower()
        if raw in invalid_tokens:
            continue

        # Prefer ISO-2 labels when available.
        if len(raw) == 2 and raw.isalpha():
            return raw

        # Normalize country names into ISO-2 for evaluator consistency.
        iso = COUNTRY_NAME_TO_ISO2.get(raw)
        if iso:
            return iso

        # Fallback: keep unknown mappings as unknown instead of introducing noisy labels.
        if key == "country_code" and len(raw) in {2, 3}:
            return raw[:2]

    return "unknown"


def _infer_lat_lng_from_record(record: dict) -> tuple[float, float] | None:
    lat_keys = ["lat", "latitude", "gt_lat", "label_lat"]
    lng_keys = ["lng", "lon", "longitude", "gt_lng", "label_lng", "label_lon"]
    lat = None
    lng = None
    for k in lat_keys:
        if k in record:
            try:
                lat = float(record[k])
                break
            except Exception:
                pass
    for k in lng_keys:
        if k in record:
            try:
                lng = float(record[k])
                break
            except Exception:
                pass
    if lat is None or lng is None:
        return None
    return lat, lng


def _resolve_image_path(dataset_root: Path, record: dict) -> str | None:
    for key in ["image_path", "image", "img", "filepath", "file_path", "image_file"]:
        p = record.get(key)
        if not isinstance(p, str) or not p.strip():
            continue
        candidate = Path(p)
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        if candidate.exists():
            return str(candidate)
    return None


def _load_manifest_dataset_samples(
    dataset_name: str,
    dataset_root: Path,
    manifest_jsonl: Path,
    limit: int | None = None,
    dataset_version: str = "v1",
) -> list[dict]:
    if not manifest_jsonl.exists():
        raise FileNotFoundError(f"{dataset_name} manifest not found: {manifest_jsonl}")

    samples: list[dict] = []
    with manifest_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_path = _resolve_image_path(dataset_root, obj)
            lat_lng = _infer_lat_lng_from_record(obj)
            if image_path is None or lat_lng is None:
                continue
            gt_country = _infer_country_from_record(obj)
            sample_id = str(obj.get("sample_id") or obj.get("id") or f"{dataset_name}_{i}")
            expert_chain = str(obj.get("expert_chain") or obj.get("reasoning") or "")
            lat, lng = lat_lng
            samples.append(
                {
                    "game_id": sample_id,
                    "sample_id": f"{dataset_name}::{sample_id}",
                    "dataset_name": dataset_name,
                    "dataset_version": dataset_version,
                    "round": 1,
                    "image_path": image_path,
                    "ground_truth_country": gt_country,
                    "ground_truth_lat": lat,
                    "ground_truth_lng": lng,
                    "expert_chain": expert_chain,
                }
            )
            if limit and len(samples) >= limit:
                break
    return samples


def _load_samples_from_cfg(project_root: Path, cfg: dict, args) -> tuple[list[dict], list[str]]:
    # Backward-compatible single-dataset config
    if "datasets" not in cfg:
        data_root = project_root / cfg["paths"]["data_root"]
        game_ids = cfg["dataset"].get("game_ids", ALL_100_GAME_IDS)
        if args.max_games:
            game_ids = game_ids[:args.max_games]
        samples = []
        expected_ids = []
        for gid in game_ids:
            expected_ids.append(f"georc::{gid}::round{args.round}")
            s = make_sample(data_root, gid, round_idx=args.round)
            if s:
                samples.append(s)
        return samples, expected_ids

    # Multi-dataset config
    all_samples: list[dict] = []
    expected_ids: list[str] = []
    for ds in cfg.get("datasets", []):
        ds_name = str(ds.get("name", "")).lower()
        if ds_name == "georc":
            ds_root = project_root / ds.get("data_root", cfg["paths"]["data_root"])
            game_ids = ds.get("game_ids", ALL_100_GAME_IDS)
            if args.max_games:
                game_ids = game_ids[:args.max_games]
            ds_round = int(ds.get("round", args.round))
            for gid in game_ids:
                s = make_sample(ds_root, gid, round_idx=ds_round)
                if s:
                    s["dataset_version"] = str(ds.get("dataset_version", "georc_v1"))
                    all_samples.append(s)
                    expected_ids.append(s["sample_id"])
        elif ds_name == "earthwhere":
            ds_root = project_root / str(ds.get("data_root", "data/earthwhere"))
            manifest = project_root / str(ds.get("manifest_jsonl", "data/earthwhere/manifest.jsonl"))
            ds_samples = _load_manifest_dataset_samples(
                dataset_name="earthwhere",
                dataset_root=ds_root,
                manifest_jsonl=manifest,
                limit=args.max_games,
                dataset_version=str(ds.get("dataset_version", "earthwhere_v1")),
            )
            all_samples.extend(ds_samples)
            expected_ids.extend([s["sample_id"] for s in ds_samples])
        elif ds_name == "im2gps3k":
            ds_root = project_root / str(ds.get("data_root", "data/im2gps3k"))
            manifest = project_root / str(ds.get("manifest_jsonl", "data/im2gps3k/manifest.jsonl"))
            ds_samples = _load_manifest_dataset_samples(
                dataset_name="im2gps3k",
                dataset_root=ds_root,
                manifest_jsonl=manifest,
                limit=args.max_games,
                dataset_version=str(ds.get("dataset_version", "im2gps3k_v1")),
            )
            all_samples.extend(ds_samples)
            expected_ids.extend([s["sample_id"] for s in ds_samples])

    return all_samples, expected_ids


def run_method_on_sample(
    method_fn,
    sample: dict,
    max_attempts: int = 1,
    retry_backoff_seconds: float = 1.5,
) -> dict:
    image_path = str(sample["image_path"])
    if not Path(image_path).exists():
        gid = str(sample.get("game_id", "")).strip()
        rnd = int(sample.get("round", 1))
        fallback_candidates: list[Path] = []
        if gid:
            fallback_candidates.extend(
                [
                    PROJECT_ROOT / "data" / "georc" / gid / f"{gid}_{rnd}.png",
                    PROJECT_ROOT.parent / "skillgeo" / "data" / "georc" / gid / f"{gid}_{rnd}.png",
                ]
            )
        fallback_candidates.append(PROJECT_ROOT / "data" / "georc" / Path(image_path).name)

        for cand in fallback_candidates:
            if cand.exists():
                image_path = str(cand)
                break

    max_attempts = max(1, int(max_attempts))
    retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            pred = method_fn(image_path)
            result = {
                "game_id": sample["game_id"],
                "round": sample["round"],
                "image_path": image_path,
                "ground_truth_country": sample["ground_truth_country"],
                "ground_truth_lat": sample["ground_truth_lat"],
                "ground_truth_lng": sample["ground_truth_lng"],
                "expert_chain": sample["expert_chain"],
                "sample_id": sample.get("sample_id"),
                "dataset_name": sample.get("dataset_name"),
                "dataset_version": sample.get("dataset_version"),
                "prediction": pred,
                "error": None,
            }
            if attempt > 1:
                result["retry_attempts"] = attempt
            return result
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_timeout_like = any(
                token in msg
                for token in [
                    "timed out",
                    "read timeout",
                    "connection reset",
                    "temporarily unavailable",
                    "internal server error",
                    "too many requests",
                    "rate limit",
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                ]
            )
            is_network_like = any(
                token in msg
                for token in [
                    "proxyerror",
                    "unable to connect to proxy",
                    "connection refused",
                    "failed to establish a new connection",
                    "max retries exceeded",
                    "connection aborted",
                    "name or service not known",
                    "temporary failure in name resolution",
                ]
            )
            is_auth_like = any(
                token in msg
                for token in [
                    "unauthorized",
                    "invalid token",
                    "401",
                ]
            )
            if attempt < max_attempts and (is_timeout_like or is_auth_like or is_network_like):
                if retry_backoff_seconds > 0:
                    backoff = retry_backoff_seconds * attempt
                    # Some providers intermittently return 401 during hot periods.
                    # Use a longer cool-down so the next sample retry can recover.
                    if is_auth_like:
                        backoff = max(backoff, 20.0 * attempt)
                    # Proxy / network failures often need a bit longer to recover.
                    if is_network_like:
                        backoff = max(backoff, 10.0 * attempt)
                    time.sleep(backoff)
                continue
            break

    return {
        "game_id": sample["game_id"],
        "round": sample["round"],
        "image_path": image_path,
        "ground_truth_country": sample["ground_truth_country"],
        "ground_truth_lat": sample["ground_truth_lat"],
        "ground_truth_lng": sample["ground_truth_lng"],
        "expert_chain": sample["expert_chain"],
        "sample_id": sample.get("sample_id"),
        "dataset_name": sample.get("dataset_name"),
        "dataset_version": sample.get("dataset_version"),
        "prediction": {
            "predicted_country": "unknown",
            "predicted_region": "unknown",
            "predicted_lat": float("nan"),
            "predicted_lng": float("nan"),
            "reasoning_text": f"ERROR: {last_exc}",
            "evidence_spans": [],
            "confidence": 0.0,
        },
        "error": str(last_exc),
        "error_type": type(last_exc).__name__ if last_exc else "RuntimeError",
        "retry_attempts": max_attempts,
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


def save_rollout_snapshot(
    exp_dir: Path,
    method_name: str,
    stage_name: str,
    records: list[dict],
    metrics: dict,
) -> None:
    method_dir = exp_dir / method_name / "rollout_trace"
    method_dir.mkdir(parents=True, exist_ok=True)

    safe_stage = (
        stage_name.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        if stage_name
        else "unknown_stage"
    )
    payload = {
        "stage": stage_name,
        "metrics": metrics,
        "num_records": len(records),
        "num_errors": sum(1 for r in records if r.get("error") is not None),
    }

    (method_dir / f"{safe_stage}_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (method_dir / f"{safe_stage}_predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def save_skill_dataset_assets(exp_dir: Path, method_name: str, records: list[dict]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    assets_dir = exp_dir / "skill_dataset_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path = assets_dir / f"{method_name}_{timestamp}.jsonl"
    latest_path = assets_dir / f"{method_name}_latest.jsonl"

    lines: list[str] = []
    for rec in records:
        pred = rec.get("prediction", {})
        skills = pred.get("retrieved_skills", [])
        if not isinstance(skills, list):
            skills = []
        item = {
            "task": "geolocation",
            "sample_id": rec.get("sample_id") or f"{rec.get('game_id', 'unknown')}::{rec.get('round', 1)}",
            "dataset_name": rec.get("dataset_name", "georc"),
            "dataset_version": rec.get("dataset_version", "unknown"),
            "reasoning_trajectory": {
                "model_reasoning": pred.get("reasoning_text", ""),
                "expert_reasoning": rec.get("expert_chain", ""),
                "stage1": pred.get("stage1_raw", ""),
                "vote_summary": pred.get("vote_summary", ""),
            },
            "skills": [
                {
                    "skill_text": str(s.get("skill_text", "")),
                    "region_hint": str(s.get("region_hint", "unknown")),
                    "confidence": float(s.get("confidence", 0.0) or 0.0),
                    "visual_cues": [str(v) for v in s.get("visual_cues", [])] if isinstance(s, dict) else [],
                    "source_game_id": str(s.get("source_game_id", "")) if isinstance(s, dict) else "",
                    "source_round": int(s.get("source_round", 0)) if isinstance(s, dict) else 0,
                }
                for s in skills if isinstance(s, dict)
            ],
            "quality_tags": {
                "has_error": rec.get("error") is not None,
                "valid_coordinates": (
                    isinstance(pred.get("predicted_lat"), (float, int))
                    and isinstance(pred.get("predicted_lng"), (float, int))
                ),
                "method": method_name,
            },
        }
        lines.append(json.dumps(item, ensure_ascii=False))

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    latest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
    expected_sample_ids: list[str],
    samples: list[dict],
    methods_requested: list[str],
    method_metrics: dict[str, dict],
) -> None:
    loaded_ids = {s.get("sample_id", s.get("game_id", "")) for s in samples}
    missing_ids = [sid for sid in expected_sample_ids if sid not in loaded_ids]
    datasets_loaded = sorted({str(s.get("dataset_name", "unknown")) for s in samples})

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
            "expected_sample_count": len(expected_sample_ids),
            "expected_game_count": len(expected_sample_ids),
            "loaded_sample_count": len(samples),
            "missing_game_count": len(missing_ids),
            "missing_sample_ids": missing_ids,
            "missing_game_ids": missing_ids,
            "datasets_loaded": datasets_loaded,
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

    samples, expected_sample_ids = _load_samples_from_cfg(PROJECT_ROOT, cfg, args)
    print(f"Loaded {len(samples)}/{len(expected_sample_ids)} samples for round {args.round}")

    if not samples:
        print("ERROR: No valid samples found. Check data directory.")
        sys.exit(1)

    main_vlm_cfg = _extract_vlm_block(
        cfg,
        keys=["main_agent", "vlm_small", "small_model", "vlm"],
        default_key="vlm",
    )
    if not main_vlm_cfg:
        raise KeyError("No valid main-agent VLM config found. Expected one of: main_agent/vlm_small/small_model/vlm")

    skill_update_vlm_cfg = _extract_vlm_block(
        cfg,
        keys=["llm_large", "vlm_large", "large_model"],
        default_key=None,
    )
    if not skill_update_vlm_cfg:
        skill_update_vlm_cfg = main_vlm_cfg

    main_vlm = _build_vlm_client(main_vlm_cfg, env_keys=["VLM_SMALL_API_KEY", "VLM_LARGE_API_KEY", "VLM_API_KEY"])
    skill_update_vlm = _build_vlm_client(skill_update_vlm_cfg, env_keys=["VLM_LARGE_API_KEY", "VLM_API_KEY"])

    print(f"Main-agent model (small): {main_vlm_cfg.get('model', 'unknown')}")
    print(f"Skill-update model (large): {skill_update_vlm_cfg.get('model', 'unknown')}")

    skill_library = build_skill_library(
        data_root=data_root,
        game_ids=[s["game_id"] for s in samples if s.get("dataset_name") == "georc"],
        embedding_model=cfg["skills"]["embedding_model"],
    )

    skill_cfg = cfg.get("skills", {})
    top_k = skill_cfg.get("top_k", 5)
    min_score = skill_cfg.get("min_score", 0.15)
    retrieval_mode = skill_cfg.get("retrieval_mode", "hybrid")
    skill_graph_top_k = int(skill_cfg.get("skill_graph_top_k", 7))
    skill_graph_top_k = max(6, min(8, skill_graph_top_k))
    candidate_vote_count = int(skill_cfg.get("candidate_vote_count", skill_cfg.get("rollout_candidate_count", 3)))
    candidate_vote_count = max(1, candidate_vote_count)
    candidate_vote_base_temperature = float(skill_cfg.get("candidate_vote_base_temperature", 0.10))
    print(f"Skill graph top_k (clamped): {skill_graph_top_k}")
    print(f"Skill graph candidate vote count: {candidate_vote_count}")

    training_free_cfg = cfg.get("training_free", {})
    enable_failure_recovery = bool(training_free_cfg.get("enable_failure_recovery", True))
    enable_skill_fusion = bool(training_free_cfg.get("enable_skill_fusion", True))
    recovery_max_records = int(training_free_cfg.get("recovery_max_records", 20))
    failure_distance_km = float(training_free_cfg.get("failure_distance_km", 200.0))
    rollout_rounds = int(training_free_cfg.get("rollout_rounds", 2))

    search_evo_cfg = training_free_cfg.get("search_evolution", {}) if isinstance(training_free_cfg, dict) else {}
    search_evolution_enabled = bool(search_evo_cfg.get("enabled", False))
    search_evolution_provider = str(search_evo_cfg.get("provider", "brave")).strip().lower()
    search_evolution_max_cases = int(search_evo_cfg.get("max_cases", 4))
    search_evolution_queries_per_case = int(search_evo_cfg.get("queries_per_case", 2))
    search_evolution_results_per_query = int(search_evo_cfg.get("results_per_query", 3))
    search_evolution_timeout_seconds = float(search_evo_cfg.get("timeout_seconds", 8.0))

    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    sample_retry_attempts = int(runtime_cfg.get("sample_retry_attempts", 2))
    sample_retry_backoff_seconds = float(runtime_cfg.get("sample_retry_backoff_seconds", 1.5))
    inter_sample_delay_seconds = float(runtime_cfg.get("inter_sample_delay_seconds", 0.0))
    search_evolution_context_char_budget = int(search_evo_cfg.get("context_char_budget", 3000))
    search_provider_needs_key = search_evolution_provider == "brave"

    search_evolution_api_key = str(search_evo_cfg.get("api_key", "")).strip()
    if not search_evolution_api_key:
        search_evolution_api_key = (
            os.getenv("BRAVE_SEARCH_API_KEY", "")
            or os.getenv("BRAVE_API_KEY", "")
            or os.getenv("WEB_SEARCH_API_KEY", "")
        )

    if search_evolution_enabled and search_provider_needs_key and not search_evolution_api_key:
        print("Search-evolution enabled but no API key found; fallback to offline evolution only.")

    search_evolution_effective = bool(
        search_evolution_enabled and (not search_provider_needs_key or search_evolution_api_key)
    )

    external_cfg = cfg.get("external_baselines", {})
    skill_update_dir = exp_dir / "skill_updates"
    skill_update_dir.mkdir(parents=True, exist_ok=True)

    all_methods = {
        "direct_vlm": lambda img: direct_vlm_predict(main_vlm, img),
        "cot_vlm": lambda img: cot_vlm_predict(main_vlm, img),
        "external_geocomp": lambda img: geocomp_external_predict(main_vlm, img, official_cfg=external_cfg.get("GeoComp")),
        "external_georeasoner": lambda img: georeasoner_external_predict(main_vlm, img, official_cfg=external_cfg.get("GeoReasoner")),
        "external_georeasoner_skill_boost": lambda img: georeasoner_external_skill_boost_predict(
            main_vlm,
            skill_library,
            img,
            retrieval_mode=retrieval_mode,
            top_k=max(8, int(top_k)),
            official_cfg=external_cfg.get("GeoReasoner"),
        ),
        "external_geovista": lambda img: geovista_external_predict(main_vlm, img, official_cfg=external_cfg.get("GeoVista")),
        "external_geovista_skill_graph": lambda img: geovista_external_skill_boost_predict(
            main_vlm,
            skill_library,
            img,
            retrieval_mode=retrieval_mode,
            top_k=skill_graph_top_k,
            candidate_vote_count=candidate_vote_count,
            candidate_vote_base_temperature=candidate_vote_base_temperature,
            official_cfg=external_cfg.get("GeoVista"),
        ),
        "external_geovista_skill_boost": lambda img: geovista_external_skill_boost_predict(
            main_vlm,
            skill_library,
            img,
            retrieval_mode=retrieval_mode,
            top_k=max(10, int(top_k)),
            candidate_vote_count=candidate_vote_count,
            candidate_vote_base_temperature=candidate_vote_base_temperature,
            official_cfg=external_cfg.get("GeoVista"),
        ),
        "external_safa": lambda img: safa_external_predict(
            main_vlm, skill_library, img, retrieval_mode=retrieval_mode, official_cfg=external_cfg.get("SAFA")
        ),
        "external_ep_bev": lambda img: ep_bev_external_predict(main_vlm, img, official_cfg=external_cfg.get("EP-BEV")),
        "external_sample4geo": lambda img: sample4geo_external_predict(main_vlm, img, official_cfg=external_cfg.get("Sample4Geo")),
        "skill_conditioned": lambda img: skill_conditioned_predict(
            main_vlm, skill_library, img, top_k=top_k, min_score=min_score, retrieval_mode=retrieval_mode
        ),
        "skill_conditioned_v2": lambda img: skill_conditioned_v2_predict(
            main_vlm, skill_library, img, top_k=top_k, retrieval_mode=retrieval_mode
        ),
        "skill_conditioned_v3": lambda img: skill_conditioned_v3_predict(
            main_vlm, skill_library, img, top_k=10, retrieval_mode=retrieval_mode
        ),
        "skill_conditioned_v4": lambda img: skill_conditioned_v4_predict(
            main_vlm, skill_library, img, top_k=12, retrieval_mode=retrieval_mode
        ),
        "georeasoner": lambda img: georeasoner_predict(main_vlm, img),
        "geocot": lambda img: geocot_predict(main_vlm, img),
        "gre_multistage": lambda img: gre_multistage_predict(main_vlm, img),
        "img2loc_rag": lambda img: img2loc_rag_predict(main_vlm, skill_library, img, top_k=8, retrieval_mode=retrieval_mode),
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
        method_start_time = time.time()

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        run_method_on_sample,
                        fn,
                        s,
                        sample_retry_attempts,
                        sample_retry_backoff_seconds,
                    ): s["game_id"]
                    for s in samples
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=method_name):
                    result = future.result()
                    records.append(result)
                    if result.get("error"):
                        errors += 1
        else:
            for idx, sample in enumerate(tqdm(samples, desc=method_name), start=1):
                result = run_method_on_sample(
                    fn,
                    sample,
                    sample_retry_attempts,
                    sample_retry_backoff_seconds,
                )
                records.append(result)
                if result.get("error"):
                    errors += 1
                if inter_sample_delay_seconds > 0 and idx < len(samples):
                    # Gentle pacing helps reduce transient upstream API failures.
                    time.sleep(inter_sample_delay_seconds)

        initial_pass_elapsed = time.time() - method_start_time
        records.sort(key=lambda r: r["game_id"])

        metrics = evaluate_predictions(records)
        metrics["initial_pass_elapsed_seconds"] = initial_pass_elapsed
        metrics["rollout_elapsed_seconds"] = 0.0
        metrics["elapsed_seconds"] = initial_pass_elapsed
        metrics["num_samples"] = len(records)
        metrics["num_errors"] = errors
        save_rollout_snapshot(exp_dir, method_name, "round_0_initial", records, metrics)

        support_offline_update = method_name.startswith("skill_conditioned") or method_name in {
            "external_geovista_skill_graph",
            "external_geovista_skill_boost",
        }

        if support_offline_update and enable_failure_recovery:
            total_improved = 0
            total_recovered_skills = 0
            total_fused_skills = 0
            rounds_executed = 0
            graph_versions_written = 0
            rollout_start_time = time.time()

            try:
                for rollout_idx in range(1, max(1, rollout_rounds) + 1):
                    failed = [r for r in records if _is_failed_record(r, failure_distance_km)]
                    if not failed:
                        break

                    rounds_executed += 1
                    synthesis_records = records if method_name.startswith("external_geovista") else failed
                    recovered_skills = synthesize_failure_skills(
                        vlm=skill_update_vlm,
                        failed_records=synthesis_records,
                        max_records=recovery_max_records,
                        skill_search_enabled=search_evolution_effective,
                        skill_search_provider=search_evolution_provider,
                        skill_search_api_key=search_evolution_api_key,
                        skill_search_max_cases=search_evolution_max_cases,
                        skill_search_queries_per_case=search_evolution_queries_per_case,
                        skill_search_results_per_query=search_evolution_results_per_query,
                        skill_search_timeout_seconds=search_evolution_timeout_seconds,
                        skill_search_context_char_budget=search_evolution_context_char_budget,
                    )
                    fused_skills = fuse_atomic_skills(recovered_skills) if enable_skill_fusion else []

                    if not recovered_skills and not fused_skills:
                        continue

                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    if recovered_skills:
                        dump_skills_jsonl(
                            recovered_skills,
                            skill_update_dir / f"recovered_{method_name}_r{rollout_idx}_{ts}.jsonl",
                        )
                        skill_library.add_skills(recovered_skills)
                        total_recovered_skills += len(recovered_skills)
                    if fused_skills:
                        dump_skills_jsonl(
                            fused_skills,
                            skill_update_dir / f"fused_{method_name}_r{rollout_idx}_{ts}.jsonl",
                        )
                        skill_library.add_skills(fused_skills)
                        total_fused_skills += len(fused_skills)

                    if method_name.startswith("external_geovista"):
                        graph_edges = summarize_rollout_skill_edges(records, top_k=80)
                        graph_payload = {
                            "method": method_name,
                            "rollout_idx": rollout_idx,
                            "timestamp": ts,
                            "num_records": len(records),
                            "num_failed_records": len(failed),
                            "edge_count": len(graph_edges),
                            "edges": graph_edges,
                        }
                        graph_path = skill_update_dir / f"graph_{method_name}_r{rollout_idx}_{ts}.json"
                        graph_path.write_text(
                            json.dumps(graph_payload, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        (skill_update_dir / f"graph_{method_name}_latest.json").write_text(
                            json.dumps(graph_payload, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        graph_versions_written += 1

                    records, improved_count = _retry_failed_with_updated_library(
                        fn,
                        records,
                        failure_distance_km=failure_distance_km,
                    )
                    total_improved += improved_count

                    round_metrics = evaluate_predictions(records)
                    round_metrics["initial_pass_elapsed_seconds"] = initial_pass_elapsed
                    round_metrics["elapsed_seconds"] = time.time() - method_start_time
                    round_metrics["num_samples"] = len(records)
                    round_metrics["num_errors"] = sum(1 for r in records if r.get("error") is not None)
                    round_metrics["recovered_failures_so_far"] = total_improved
                    round_metrics["generated_skill_count_so_far"] = total_recovered_skills
                    round_metrics["fused_skill_count_so_far"] = total_fused_skills
                    round_metrics["rollout_round"] = rollout_idx
                    round_metrics["rollout_failure_distance_km"] = failure_distance_km
                    round_metrics["remaining_failed_cases"] = sum(
                        1 for r in records if _is_failed_record(r, failure_distance_km)
                    )
                    save_rollout_snapshot(
                        exp_dir,
                        method_name,
                        f"round_{rollout_idx}_post_update",
                        records,
                        round_metrics,
                    )

                metrics = evaluate_predictions(records)
                rollout_elapsed = time.time() - rollout_start_time
                total_elapsed = time.time() - method_start_time
                metrics["initial_pass_elapsed_seconds"] = initial_pass_elapsed
                metrics["rollout_elapsed_seconds"] = rollout_elapsed
                metrics["elapsed_seconds"] = total_elapsed
                metrics["num_samples"] = len(records)
                metrics["num_errors"] = sum(1 for r in records if r.get("error") is not None)
                metrics["recovered_failures"] = total_improved
                metrics["generated_skill_count"] = total_recovered_skills
                metrics["fused_skill_count"] = total_fused_skills
                metrics["rollout_rounds"] = rounds_executed
                metrics["rollout_failure_distance_km"] = failure_distance_km
                metrics["search_evolution_enabled"] = search_evolution_effective
                metrics["search_evolution_provider"] = search_evolution_provider
                metrics["skill_graph_versions"] = graph_versions_written
                metrics["remaining_failed_cases"] = sum(
                    1 for r in records if _is_failed_record(r, failure_distance_km)
                )
            except Exception as recovery_exc:
                metrics["recovery_error"] = str(recovery_exc)
                metrics["initial_pass_elapsed_seconds"] = initial_pass_elapsed
                metrics["rollout_elapsed_seconds"] = time.time() - rollout_start_time
                metrics["elapsed_seconds"] = time.time() - method_start_time

        total_elapsed = float(metrics.get("elapsed_seconds", time.time() - method_start_time) or 0.0)
        final_errors = int(metrics.get("num_errors", errors) or 0)
        initial_elapsed_report = float(metrics.get("initial_pass_elapsed_seconds", initial_pass_elapsed) or 0.0)
        rollout_elapsed_report = float(metrics.get("rollout_elapsed_seconds", 0.0) or 0.0)

        save_results(exp_dir, method_name, records, metrics)
        save_skill_dataset_assets(exp_dir, method_name, records)
        all_metrics[method_name] = metrics

        print(f"\n{method_name} results ({total_elapsed:.1f}s total, {final_errors} errors):")
        if rollout_elapsed_report > 0:
            print(
                f"  Timing: initial={initial_elapsed_report:.1f}s + "
                f"rollout={rollout_elapsed_report:.1f}s"
            )
        print(f"  Country Acc: {metrics['country_accuracy']:.3f}")
        print(f"  Continent Acc: {metrics['continent_accuracy']:.3f}")
        print(f"  Distance (median): {metrics['distance_error_km_median']:.1f} km")
        print(f"  Distance (mean valid-only): {metrics['distance_error_km_mean_valid_only']:.1f} km")
        print(f"  Distance (mean penalized): {metrics['distance_error_km_mean_penalized']:.1f} km")
        print(f"  Distance (std penalized): {metrics['distance_error_km_std_penalized']:.1f} km")
        print(f"  Valid Coords: {metrics['valid_coordinate_rate']:.2%}")
        country_known_rate = metrics.get("country_known_rate")
        continent_known_rate = metrics.get("continent_known_rate")
        if isinstance(country_known_rate, (int, float)) and country_known_rate == country_known_rate:
            print(f"  Country Known Rate: {country_known_rate:.2%}")
        if isinstance(continent_known_rate, (int, float)) and continent_known_rate == continent_known_rate:
            print(f"  Continent Known Rate: {continent_known_rate:.2%}")
        if isinstance(country_known_rate, (int, float)) and country_known_rate == country_known_rate and country_known_rate < 0.5:
            print("  NOTE: Many samples have unknown country labels; country/continent accuracy is not directly comparable.")
        for acc_key in _sorted_acc_keys(metrics):
            print(f"  {acc_key}: {metrics[acc_key]:.3f}")

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
        "main_agent_model": main_vlm_cfg.get("model", "unknown"),
        "skill_update_model": skill_update_vlm_cfg.get("model", "unknown"),
    }
    summary_path.write_text(json.dumps(merged_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    _write_audit_report(
        exp_dir=exp_dir,
        config_path=PROJECT_ROOT / args.config,
        expected_sample_ids=expected_sample_ids,
        samples=samples,
        methods_requested=methods_to_run,
        method_metrics=all_metrics,
    )

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    summary_acc_keys = _sorted_acc_keys(next(iter(all_metrics.values()))) if all_metrics else []
    header = (
        f"{'Method':<22} {'Country':>8} {'Continent':>10} {'DistMed':>8} {'Valid%':>8} "
        f"{'Time(s)':>8} {'Rollout(s)':>10} {'s/sample':>9}"
    )
    for acc_key in summary_acc_keys:
        threshold = acc_key[4:-2]
        header += f" {'A@' + threshold:>7}"
    print(header)
    print("-" * len(header))
    for method, m in all_metrics.items():
        n_samples = int(m.get("num_samples", 0) or 0)
        total_time = float(m.get("elapsed_seconds", float("nan")) or float("nan"))
        rollout_time = float(m.get("rollout_elapsed_seconds", 0.0) or 0.0)
        per_sample = total_time / n_samples if n_samples > 0 and total_time == total_time else float("nan")
        total_time_str = "N/A" if total_time != total_time else f"{total_time:.1f}"
        rollout_time_str = "N/A" if rollout_time != rollout_time else f"{rollout_time:.1f}"
        per_sample_str = "N/A" if per_sample != per_sample else f"{per_sample:.2f}"
        dist_str = f"{m['distance_error_km_median']:.0f}" if not (m['distance_error_km_median'] != m['distance_error_km_median']) else "N/A"
        row = (
            f"{method:<22} {m['country_accuracy']:>8.3f} {m['continent_accuracy']:>8.3f} "
            f"{dist_str:>10} {m['valid_coordinate_rate']:>7.1%} "
            f"{total_time_str:>8} {rollout_time_str:>10} {per_sample_str:>9}"
        )
        for acc_key in summary_acc_keys:
            row += f" {float(m.get(acc_key, 0.0)):>7.3f}"
        print(row)
        n_errors = int(m.get("num_errors", 0))
        print(f"  -> success={max(0, n_samples - n_errors)}/{n_samples}, fail={n_errors}/{n_samples}")

    print(f"\nAll results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
