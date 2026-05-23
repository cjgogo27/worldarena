import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINES = __import__("src.baselines", fromlist=["*"])
EVALUATOR = __import__("src.evaluator", fromlist=["*"])
SKILL_LIBRARY_MOD = __import__("src.skill_library", fromlist=["*"])
SKILL_PARSER_MOD = __import__("src.skill_parser", fromlist=["*"])
VLM_MOD = __import__("src.vlm_client", fromlist=["*"])


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_vlm_block(cfg: dict, keys: list[str], default_key: str | None = "vlm") -> dict | None:
    for key in keys:
        block = cfg.get(key)
        if not isinstance(block, dict):
            continue
        if "base_url" in block and "model" in block:
            return block
        nested = block.get("vlm")
        if isinstance(nested, dict) and "base_url" in nested and "model" in nested:
            return nested
    if default_key and isinstance(cfg.get(default_key), dict):
        fallback = cfg.get(default_key)
        if "base_url" in fallback and "model" in fallback:
            return fallback
    return None


def get_game_ids(cfg) -> list[str]:
    return list(cfg["dataset"]["game_ids"])


def find_human_expert_file(game_dir: Path) -> Path:
    matches = sorted(game_dir.glob("Human_Expert_*.txt"))
    if not matches:
        raise FileNotFoundError(f"No Human_Expert_*.txt found in {game_dir}")
    return matches[0]


def build_skill_library(data_root: Path, game_ids: list[str], embedding_model: str):
    library = SKILL_LIBRARY_MOD.SkillLibrary(embedding_model_name=embedding_model)
    all_skills = []
    for gid in game_ids:
        gdir = data_root / gid
        expert_text = find_human_expert_file(gdir).read_text(encoding="utf-8")
        all_skills.extend(SKILL_PARSER_MOD.parse_expert_chain(expert_text, source_game_id=gid))
    library.add_skills(all_skills)
    return library


def make_sample(data_root: Path, gid: str, round_idx: int = 1) -> dict[str, object]:
    gdir = data_root / gid
    metadata = json.loads((gdir / f"{gid}_rounds_metadata.json").read_text(encoding="utf-8"))
    gt = metadata[round_idx - 1]
    expert_chain = find_human_expert_file(gdir).read_text(encoding="utf-8")
    return {
        "game_id": gid,
        "round": round_idx,
        "image_path": str(gdir / f"{gid}_{round_idx}.png"),
        "ground_truth_country": gt["streakLocationCode"].lower(),
        "ground_truth_lat": float(gt["lat"]),
        "ground_truth_lng": float(gt["lng"]),
        "expert_chain": expert_chain,
    }


def main() -> None:
    project_root = PROJECT_ROOT
    cfg = load_config(project_root / "configs" / "pilot.yaml")

    data_root = project_root / cfg["paths"]["data_root"]
    exp_dir = project_root / cfg["paths"]["experiment_dir"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    game_ids = get_game_ids(cfg)
    samples = [make_sample(data_root, gid, round_idx=1) for gid in game_ids]

    small_vlm_cfg = extract_vlm_block(
        cfg,
        keys=["main_agent", "vlm_small", "small_model", "vlm"],
        default_key="vlm",
    )
    if not small_vlm_cfg:
        raise KeyError("No valid small-model config found. Expected one of: main_agent/vlm_small/small_model/vlm")

    vlm = VLM_MOD.VLMClient(
        VLM_MOD.VLMConfig(
            base_url=small_vlm_cfg["base_url"],
            api_key=(
                os.getenv("VLM_SMALL_API_KEY")
                or os.getenv("VLM_API_KEY")
                or small_vlm_cfg.get("api_key", "")
            ),
            model=small_vlm_cfg["model"],
            max_tokens=small_vlm_cfg.get("max_tokens", 1024),
            max_image_side=small_vlm_cfg.get("max_image_side", 1024),
            retries=small_vlm_cfg.get("retries", 3),
            backoff_seconds=small_vlm_cfg.get("backoff_seconds", 1.0),
            request_timeout_seconds=small_vlm_cfg.get("request_timeout_seconds", 45.0),
        )
    )

    skill_library = build_skill_library(
        data_root=data_root,
        game_ids=game_ids,
        embedding_model=cfg["skills"]["embedding_model"],
    )

    methods = {
        "direct_vlm": lambda img: BASELINES.direct_vlm_predict(vlm, img),
        "cot_vlm": lambda img: BASELINES.cot_vlm_predict(vlm, img),
        "skill_conditioned_vlm": lambda img: BASELINES.skill_conditioned_predict(vlm, skill_library, img, top_k=cfg["skills"].get("top_k", 5)),
    }

    all_outputs = {}
    metrics = {}

    for method_name, fn in methods.items():
        records = []
        for sample in tqdm(samples, desc=f"Running {method_name}"):
            pred = fn(sample["image_path"])
            records.append(
                {
                    "game_id": sample["game_id"],
                    "round": sample["round"],
                    "ground_truth_country": sample["ground_truth_country"],
                    "ground_truth_lat": sample["ground_truth_lat"],
                    "ground_truth_lng": sample["ground_truth_lng"],
                    "expert_chain": sample["expert_chain"],
                    "prediction": pred,
                }
            )
        all_outputs[method_name] = records
        metrics[method_name] = EVALUATOR.evaluate_predictions(records)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (exp_dir / f"predictions_{timestamp}.json").write_text(json.dumps(all_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp_dir / f"metrics_{timestamp}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp_dir / "latest_predictions.json").write_text(json.dumps(all_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp_dir / "latest_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== GeoSkill Pilot (20 samples, round 1) ===")
    print(
        "method\tcountry_accuracy\tcontinent_accuracy\t"
        "distance_error_km_median\tvalid_coordinate_rate\t"
        "Acc@1km\tAcc@10km\tAcc@25km\tAcc@100km\tAcc@200km\tAcc@750km\tAcc@2500km"
    )
    for method, m in metrics.items():
        print(
            f"{method}\t{m['country_accuracy']:.3f}\t{m['continent_accuracy']:.3f}\t"
            f"{m['distance_error_km_median']:.1f}\t{m['valid_coordinate_rate']:.3f}\t"
            f"{m['Acc@1km']:.3f}\t{m['Acc@10km']:.3f}\t{m['Acc@25km']:.3f}\t"
            f"{m['Acc@100km']:.3f}\t{m['Acc@200km']:.3f}\t{m['Acc@750km']:.3f}\t{m['Acc@2500km']:.3f}"
        )


if __name__ == "__main__":
    main()
