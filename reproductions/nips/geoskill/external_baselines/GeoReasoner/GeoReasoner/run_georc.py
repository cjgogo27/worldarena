import argparse
import json
from pathlib import Path

from infer import _load_model, predict_single


def _preferred_gt_path(challenge_dir: Path) -> Path | None:
    for name in ("Human_Expert_3.txt", "Human_Expert_2.txt", "Human_Expert_1.txt", "Human_Chain_3.txt"):
        path = challenge_dir / name
        if path.exists():
            return path
    return None


def _read_truth_country(challenge_dir: Path, round_idx: int) -> str:
    metadata_path = challenge_dir / f"{challenge_dir.name}_rounds_metadata.json"
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        if 0 <= round_idx - 1 < len(data):
            return str(data[round_idx - 1].get("streakLocationCode", "")).lower()
    return ""


def _summarize_country_acc(records: list[dict]) -> dict:
    valid = [r for r in records if r.get("target_country_code")]
    correct = [
        r
        for r in valid
        if r.get("pred_country_code", "").lower() == r.get("target_country_code", "").lower()
    ]
    total = len(valid)
    return {
        "country_acc": (len(correct) / total) if total else 0.0,
        "correct": len(correct),
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge_path", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N challenge directories when > 0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    challenge_root = Path(args.challenge_path).resolve()
    challenge_dirs = sorted([p for p in challenge_root.iterdir() if p.is_dir()])
    if args.limit > 0:
        challenge_dirs = challenge_dirs[: args.limit]

    model, processor = _load_model(
        model_name=args.model,
        cache_dir=args.cache_dir,
        load_in_4bit=not args.no_4bit,
    )

    run_records = []
    for challenge_dir in challenge_dirs:
        gt_path = _preferred_gt_path(challenge_dir)
        if gt_path is None:
            continue

        for round_idx in range(1, 6):
            image_path = challenge_dir / f"{challenge_dir.name}_{round_idx}.png"
            reasoning_path = challenge_dir / f"candidate_reasoning_chain_georeasoner_{round_idx}.txt"
            prediction_path = challenge_dir / f"candidate_prediction_georeasoner_{round_idx}.json"

            if not image_path.exists():
                continue
            if reasoning_path.exists() and prediction_path.exists() and not args.force:
                prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            else:
                prediction = predict_single(
                    image_path=str(image_path),
                    model=model,
                    processor=processor,
                    max_new_tokens=args.max_new_tokens,
                )
                reasoning_text = "\n".join(prediction.get("reasoning_chain", [])) or prediction.get("reasoning", "")
                reasoning_path.write_text(reasoning_text.strip() + "\n", encoding="utf-8")
                prediction_path.write_text(json.dumps(prediction, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            run_records.append(
                {
                    "challenge": challenge_dir.name,
                    "round": round_idx,
                    "pred_country_code": str(prediction.get("country_code", "")).lower(),
                    "pred_country": prediction.get("country", ""),
                    "target_country_code": _read_truth_country(challenge_dir, round_idx),
                    "reasoning_file": str(reasoning_path),
                    "prediction_file": str(prediction_path),
                }
            )

    summary = {
        "model": args.model,
        "challenge_path": str(challenge_root),
        **_summarize_country_acc(run_records),
        "records": run_records,
    }
    output_path = challenge_root / "georeasoner_run_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
