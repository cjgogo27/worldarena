from __future__ import annotations

import json
from pathlib import Path


def load_stage_payloads(rollout_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(rollout_dir.glob("round_*_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        stage = str(payload.get("stage", path.stem))
        rows.append(
            {
                "stage": stage,
                "country_accuracy": metrics.get("country_accuracy"),
                "continent_accuracy": metrics.get("continent_accuracy"),
                "valid_coordinate_rate": metrics.get("valid_coordinate_rate"),
                "Acc@25km": metrics.get("Acc@25km"),
                "Acc@200km": metrics.get("Acc@200km"),
                "Acc@750km": metrics.get("Acc@750km"),
                "num_errors": metrics.get("num_errors"),
                "generated_skill_count_so_far": metrics.get(
                    "generated_skill_count_so_far",
                    metrics.get("generated_skill_count"),
                ),
                "fused_skill_count_so_far": metrics.get(
                    "fused_skill_count_so_far",
                    metrics.get("fused_skill_count"),
                ),
                "recovered_failures_so_far": metrics.get(
                    "recovered_failures_so_far",
                    metrics.get("recovered_failures"),
                ),
                "remaining_failed_cases": metrics.get("remaining_failed_cases"),
            }
        )
    return rows


def fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--method", default="skill_conditioned_v3")
    args = parser.parse_args()

    rollout_dir = args.experiment_dir / args.method / "rollout_trace"
    if not rollout_dir.exists():
        raise SystemExit(f"rollout trace not found: {rollout_dir}")

    rows = load_stage_payloads(rollout_dir)
    out_json = args.experiment_dir / f"{args.method}_rollout_summary.json"
    out_md = args.experiment_dir / f"{args.method}_rollout_summary.md"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| Stage | Country Acc | Continent Acc | Valid Rate | Acc@25km | Acc@200km | Acc@750km | Generated Skills | Fused Skills | Recovered Failures | Remaining Failed | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {country_accuracy} | {continent_accuracy} | {valid_coordinate_rate} | {Acc@25km} | {Acc@200km} | {Acc@750km} | {generated_skill_count_so_far} | {fused_skill_count_so_far} | {recovered_failures_so_far} | {remaining_failed_cases} | {num_errors} |".format(
                **{k: fmt(v) for k, v in row.items()}
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
