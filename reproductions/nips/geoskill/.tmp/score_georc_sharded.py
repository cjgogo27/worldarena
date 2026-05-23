import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/data/alice/cjtest/NIPS/geoskill")
GEORC_ROOT = ROOT / "external_baselines" / "GeoRC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GEORC_ROOT))

from scripts.export_predictions_to_georc import export_predictions
from src.llm.parse_chains import parse_reasoning_chains


def _resolve_gt_path(challenge_dir: Path) -> Path | None:
    for filename in ("Human_Chain_3.txt", "Human_Expert_3.txt", "Human_Expert_2.txt", "Human_Expert_1.txt"):
        path = challenge_dir / filename
        if path.exists():
            return path
    return None


def _expected_rows_for_shard(challenge_root: Path, pattern: str, rounds: int, modulo: int, remainder: int) -> int:
    dirs = sorted([d for d in challenge_root.iterdir() if d.is_dir()])
    if modulo > 1:
        dirs = [d for idx, d in enumerate(dirs) if idx % modulo == remainder]

    total = 0
    for challenge_dir in dirs:
        gt_path = _resolve_gt_path(challenge_dir)
        if gt_path is None:
            continue
        try:
            gt_chains = parse_reasoning_chains(str(gt_path))
        except Exception:
            gt_chains = []
        if not gt_chains:
            continue
        for round_idx in range(1, rounds + 1):
            if round_idx > len(gt_chains):
                continue
            cand_path = challenge_dir / f"{pattern}{round_idx}.txt"
            if cand_path.exists():
                total += 1
    return total


def _load_results_len(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return len(data) if isinstance(data, list) else 0


def _backup_existing(path: Path) -> None:
    if not path.exists():
        return
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    backup = path.with_name(f"{path.name}.bak.{ts}")
    path.rename(backup)


def _run_score_shard(
    score_python: Path,
    georc_root: Path,
    challenge_root: Path,
    pattern: str,
    rounds: int,
    llm_model: str,
    cache_dir: str,
    suffix: str,
    modulo: int,
    remainder: int,
    max_attempts: int,
) -> tuple[Path, int, int]:
    shard_suffix = f"{suffix}_shard{remainder}"
    output_path = georc_root / f"vlm_scores_key_points_{shard_suffix}.json"
    _backup_existing(output_path)

    expected_rows = _expected_rows_for_shard(challenge_root, pattern, rounds, modulo, remainder)
    for attempt in range(1, max_attempts + 1):
        cmd = [
            str(score_python),
            str(georc_root / "score.py"),
            "--mode",
            "key_points",
            "--llm_model",
            llm_model,
            "--challenge_path",
            str(challenge_root),
            "--pattern",
            pattern,
            "--rounds",
            str(rounds),
            "--cache_dir",
            cache_dir,
            "--suffix",
            shard_suffix,
            "--modulo",
            str(modulo),
            "--remainder",
            str(remainder),
        ]
        proc = subprocess.run(cmd, cwd=str(georc_root), text=True, capture_output=True, check=False)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        observed_rows = _load_results_len(output_path)
        print(
            f"[score shard {remainder}] attempt={attempt} observed_rows={observed_rows} expected_rows={expected_rows} rc={proc.returncode}"
        )
        if observed_rows >= expected_rows and expected_rows >= 0:
            return output_path, observed_rows, expected_rows
        time.sleep(2.0)

    return output_path, _load_results_len(output_path), expected_rows


def _combine_shards(shard_paths: list[Path], combined_path: Path) -> int:
    combined = []
    for path in shard_paths:
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            rows = []
        if isinstance(rows, list):
            combined.extend(rows)
    combined.sort(key=lambda row: (str(row.get("challenge", "")), int(row.get("round", 0) or 0)))
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    return len(combined)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--challenge-root", default=str(ROOT / "data" / "georc"))
    parser.add_argument("--score-python", default="/data2/miniconda3/envs/vgllm/bin/python")
    parser.add_argument("--llm-model", default="qwen_tiny_llm")
    parser.add_argument("--cache-dir", default=str(ROOT / ".cache" / "georc_score"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    predictions_path = Path(args.predictions).resolve()
    challenge_root = Path(args.challenge_root).resolve()
    score_python = Path(args.score_python).resolve()

    export_summary = export_predictions(
        predictions_path=predictions_path,
        challenge_root=challenge_root,
        pattern=args.pattern,
        force=True,
        write_prediction_json=True,
    )
    print(json.dumps(export_summary, ensure_ascii=False, indent=2))

    shard_paths = []
    total_observed = 0
    total_expected = 0
    for remainder in range(args.shards):
        output_path, observed_rows, expected_rows = _run_score_shard(
            score_python=score_python,
            georc_root=GEORC_ROOT,
            challenge_root=challenge_root,
            pattern=args.pattern,
            rounds=args.rounds,
            llm_model=args.llm_model,
            cache_dir=args.cache_dir,
            suffix=args.suffix,
            modulo=args.shards,
            remainder=remainder,
            max_attempts=args.max_attempts,
        )
        shard_paths.append(output_path)
        total_observed += observed_rows
        total_expected += expected_rows

    combined_path = GEORC_ROOT / f"vlm_scores_key_points_{args.suffix}.json"
    _backup_existing(combined_path)
    combined_rows = _combine_shards(shard_paths, combined_path)
    print(
        json.dumps(
            {
                "combined_path": str(combined_path),
                "combined_rows": combined_rows,
                "expected_rows": total_expected,
                "shards": [str(path) for path in shard_paths],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if combined_rows >= total_expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
