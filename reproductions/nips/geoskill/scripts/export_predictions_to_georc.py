import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEORC_ROOT = PROJECT_ROOT / "external_baselines" / "GeoRC"
DEFAULT_SCORE_PYTHON = Path("/data2/miniconda3/envs/vgllm/bin/python")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    candidates = [text]
    if "```" in text:
        parts = text.split("```")
        candidates.extend(part.strip() for part in parts if part.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        candidate = candidate.strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def _normalize_reasoning_lines(prediction: dict[str, Any]) -> list[str]:
    raw_text = str(prediction.get("reasoning_text", "") or "").strip()
    payload = _extract_json_blob(raw_text)

    if payload:
        evidence = payload.get("evidence")
        if not isinstance(evidence, list):
            evidence = prediction.get("evidence_spans", [])
        lines = [str(item).strip() for item in evidence if str(item).strip()]

        reasoning = str(payload.get("reasoning", "") or "").strip()
        if reasoning:
            for part in reasoning.replace(". ", ".\n").splitlines():
                part = part.strip()
                if part:
                    lines.append(part)

        address_bits = [
            str(payload.get("city", "") or "").strip(),
            str(payload.get("province_or_state", "") or "").strip(),
            str(payload.get("country", "") or "").strip(),
        ]
        address_bits = [x for x in address_bits if x]
        if address_bits:
            lines.append(f"Conclusion: likely location is {', '.join(address_bits)}")

        if lines:
            return lines

    fallback_lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line:
            fallback_lines.append(line)
    if fallback_lines:
        return fallback_lines

    evidence_spans = prediction.get("evidence_spans", [])
    lines = [str(item).strip() for item in evidence_spans if str(item).strip()]
    if lines:
        return lines

    pred_country = str(prediction.get("predicted_country", "unknown") or "unknown").strip()
    pred_region = str(prediction.get("predicted_region", "unknown") or "unknown").strip()
    return [f"Predicted country: {pred_country}", f"Predicted region: {pred_region}"]


def _reasoning_chain_text(prediction: dict[str, Any]) -> str:
    lines = _normalize_reasoning_lines(prediction)
    return "\n".join(lines).strip() + "\n"


def export_predictions(
    predictions_path: Path,
    challenge_root: Path,
    pattern: str,
    force: bool = False,
    write_prediction_json: bool = True,
) -> dict[str, Any]:
    records = _load_json(predictions_path)
    if not isinstance(records, list):
        raise ValueError(f"Predictions file must contain a list: {predictions_path}")

    exported = 0
    skipped_missing_dirs = 0
    rounds_found: set[int] = set()

    for record in records:
        if not isinstance(record, dict):
            continue
        challenge_id = str(record.get("game_id", "")).strip()
        if not challenge_id:
            continue
        round_idx = int(record.get("round", 1) or 1)
        rounds_found.add(round_idx)
        prediction = record.get("prediction", {})
        if not isinstance(prediction, dict):
            prediction = {}

        challenge_dir = challenge_root / challenge_id
        if not challenge_dir.exists():
            skipped_missing_dirs += 1
            continue

        reasoning_path = challenge_dir / f"{pattern}{round_idx}.txt"
        prediction_path = challenge_dir / f"candidate_prediction_{pattern.removeprefix('candidate_reasoning_chain_')}{round_idx}.json"

        if force or not reasoning_path.exists():
            reasoning_path.write_text(_reasoning_chain_text(prediction), encoding="utf-8")

        if write_prediction_json and (force or not prediction_path.exists()):
            prediction_path.write_text(
                json.dumps(
                    {
                        "game_id": challenge_id,
                        "round": round_idx,
                        "sample_id": record.get("sample_id"),
                        "dataset_name": record.get("dataset_name"),
                        "dataset_version": record.get("dataset_version"),
                        "prediction": prediction,
                        "error": record.get("error"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        exported += 1

    return {
        "predictions_path": str(predictions_path),
        "challenge_root": str(challenge_root),
        "pattern": pattern,
        "num_records": len(records),
        "num_exported": exported,
        "skipped_missing_dirs": skipped_missing_dirs,
        "rounds_found": sorted(rounds_found),
    }


def maybe_run_score(
    georc_root: Path,
    score_python: Path,
    challenge_root: Path,
    pattern: str,
    rounds: int,
    mode: str,
    llm_model: str,
    cache_dir: str,
    suffix: str,
) -> subprocess.CompletedProcess[str]:
    score_script = georc_root / "score.py"
    cmd = [
        str(score_python),
        str(score_script),
        "--mode",
        mode,
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
        suffix,
    ]
    return subprocess.run(cmd, cwd=str(georc_root), text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export geoskill experiment predictions into GeoRC benchmark files")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON from run_experiment.py")
    parser.add_argument("--challenge-root", default=str(PROJECT_ROOT / "data" / "georc"))
    parser.add_argument("--pattern", default="candidate_reasoning_chain_geovista_skill_graph_")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-prediction-json", action="store_true")
    parser.add_argument("--score", action="store_true", help="Run GeoRC score.py after export")
    parser.add_argument("--score-mode", default="key_points", choices=["key_points", "bipartite", "vlm_judge"])
    parser.add_argument("--score-rounds", type=int, default=1)
    parser.add_argument("--llm-model", default="qwen_llm")
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / ".cache" / "georc_score"))
    parser.add_argument("--score-suffix", default="geovista_skill_graph")
    parser.add_argument("--georc-root", default=str(DEFAULT_GEORC_ROOT))
    parser.add_argument("--score-python", default=str(DEFAULT_SCORE_PYTHON))
    args = parser.parse_args()

    predictions_path = Path(args.predictions).resolve()
    challenge_root = Path(args.challenge_root).resolve()
    georc_root = Path(args.georc_root).resolve()
    score_python = Path(args.score_python).resolve()

    summary = export_predictions(
        predictions_path=predictions_path,
        challenge_root=challenge_root,
        pattern=args.pattern,
        force=bool(args.force),
        write_prediction_json=not bool(args.no_prediction_json),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.score:
        return 0

    result = maybe_run_score(
        georc_root=georc_root,
        score_python=score_python,
        challenge_root=challenge_root,
        pattern=args.pattern,
        rounds=int(args.score_rounds),
        mode=args.score_mode,
        llm_model=args.llm_model,
        cache_dir=args.cache_dir,
        suffix=args.score_suffix,
    )

    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
