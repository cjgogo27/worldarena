import json
import sys
from pathlib import Path
from statistics import mean


ROOT = Path("/data/alice/cjtest/NIPS/geoskill")
GEORC_ROOT = ROOT / "external_baselines" / "GeoRC"
OUT_DIR = ROOT / "experiments" / "ablation_geovista_true_rollout50"
FORMAL_EXPECTED_SAMPLES = 50
sys.path.insert(0, str(GEORC_ROOT))

from src.llm.parse_chains import parse_reasoning_chains


CASES = [
    {
        "label": "w.o.skill",
        "method": "direct_vlm",
        "exp_dir": ROOT / "experiments" / "ablation_geovista_true_mtl_rollout50_woskill",
        "score_file": GEORC_ROOT / "vlm_scores_key_points_ablation_true_woskill_50.json",
    },
    {
        "label": "roll out 次数1",
        "method": "external_geovista_skill_graph",
        "exp_dir": ROOT / "experiments" / "ablation_geovista_true_mtl_rollout50_vote1",
        "score_file": GEORC_ROOT / "vlm_scores_key_points_ablation_true_vote1_50.json",
    },
    {
        "label": "roll out 次数3",
        "method": "external_geovista_skill_graph",
        "exp_dir": ROOT / "experiments" / "ablation_geovista_true_mtl_rollout50_vote3",
        "score_file": GEORC_ROOT / "vlm_scores_key_points_ablation_true_vote3_50.json",
    },
    {
        "label": "roll out 次数5",
        "method": "external_geovista_skill_graph",
        "exp_dir": ROOT / "experiments" / "ablation_geovista_true_mtl_rollout50_vote5",
        "score_file": GEORC_ROOT / "vlm_scores_key_points_ablation_true_vote5_50.json",
    },
    {
        "label": "Ours（完整状态，不消融）",
        "method": "external_geovista_skill_graph",
        "exp_dir": ROOT / "experiments" / "ablation_geovista_true_mtl_rollout50_vote1",
        "score_file": GEORC_ROOT / "vlm_scores_key_points_ablation_true_vote1_50.json",
        "alias_of": "roll out 次数1",
    },
]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _resolve_gt_path(challenge: str) -> Path | None:
    challenge_dir = ROOT / "data" / "georc" / challenge
    for filename in ("Human_Chain_3.txt", "Human_Expert_3.txt", "Human_Expert_2.txt", "Human_Expert_1.txt"):
        path = challenge_dir / filename
        if path.exists():
            return path
    return None


def _expected_score_rows(predictions_path: Path) -> int:
    if not predictions_path.exists():
        return 0

    records = json.loads(predictions_path.read_text(encoding="utf-8"))
    expected = 0
    for record in records:
        challenge = str(record.get("game_id", "")).strip()
        round_num = int(record.get("round", 1) or 1)
        if not challenge:
            continue
        gt_path = _resolve_gt_path(challenge)
        if gt_path is None:
            continue
        try:
            gt_chains = parse_reasoning_chains(str(gt_path))
        except Exception:
            gt_chains = []
        if round_num <= len(gt_chains):
            expected += 1
    return expected


def _summarize_scores(score_path: Path, expected_rows: int) -> dict[str, object]:
    if not score_path.exists():
        return {
            "count": 0,
            "score_rows": 0,
            "expected_rows": expected_rows,
            "complete": False,
            "avg_precision": None,
            "avg_recall": None,
            "avg_f1": None,
        }

    rows = json.loads(score_path.read_text(encoding="utf-8"))
    valid = [
        {
            "challenge": str(row.get("challenge", "")).strip(),
            "precision": _clip01(row.get("precision", 0.0)),
            "recall": _clip01(row.get("recall", 0.0)),
            "f1": _clip01(row.get("f1", 0.0)),
        }
        for row in rows
        if str(row.get("challenge", "")).strip()
    ]

    complete = len(valid) >= expected_rows > 0

    if not valid or not complete:
        return {
            "count": len(valid),
            "score_rows": len(valid),
            "expected_rows": expected_rows,
            "complete": complete,
            "avg_precision": None,
            "avg_recall": None,
            "avg_f1": None,
        }

    return {
        "count": len(valid),
        "score_rows": len(valid),
        "expected_rows": expected_rows,
        "complete": True,
        "avg_precision": mean(item["precision"] for item in valid),
        "avg_recall": mean(item["recall"] for item in valid),
        "avg_f1": mean(item["f1"] for item in valid),
    }


def _load_metrics(metrics_path: Path) -> dict[str, object]:
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _fmt_metric(metrics: dict[str, object], key: str, expected_samples: int) -> str:
    if int(metrics.get("num_samples", 0) or 0) < expected_samples:
        return "N/A"
    value = metrics.get(key)
    if value is None:
        return "N/A"
    return f"{float(value):.3f}"


def _fmt_score(summary: dict[str, object], key: str) -> str:
    value = summary.get(key)
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    details = []
    for case in CASES:
        metrics_path = case["exp_dir"] / case["method"] / "latest_metrics.json"
        predictions_path = case["exp_dir"] / case["method"] / "latest_predictions.json"
        metrics = _load_metrics(metrics_path)
        expected_samples = FORMAL_EXPECTED_SAMPLES
        expected_score_rows = _expected_score_rows(predictions_path)
        score_summary = _summarize_scores(case["score_file"], expected_score_rows)

        row = {
            "消融": case["label"],
            "10km": _fmt_metric(metrics, "Acc@10km", expected_samples),
            "25km": _fmt_metric(metrics, "Acc@25km", expected_samples),
            "200km": _fmt_metric(metrics, "Acc@200km", expected_samples),
            "750km": _fmt_metric(metrics, "Acc@750km", expected_samples),
            "2000km": _fmt_metric(metrics, "Acc@2000km", expected_samples),
            "Precision": _fmt_score(score_summary, "avg_precision"),
            "Recall": _fmt_score(score_summary, "avg_recall"),
            "F1": _fmt_score(score_summary, "avg_f1"),
            "ValidScoreCount": score_summary["count"],
            "ExpectedScoreCount": score_summary["expected_rows"],
            "ScoreComplete": score_summary["complete"],
        }
        rows.append(row)
        details.append(
            {
                "label": case["label"],
                "alias_of": case.get("alias_of"),
                "metrics_path": str(metrics_path),
                "predictions_path": str(predictions_path),
                "score_file": str(case["score_file"]),
                "num_samples_expected": expected_samples,
                "num_samples_observed": int(metrics.get("num_samples", 0) or 0),
                "score_summary": score_summary,
            }
        )

    (OUT_DIR / "summary_true_rollout_50.json").write_text(
        json.dumps({"rows": rows, "details": details}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_header = [
        "消融",
        "10km",
        "25km",
        "200km",
        "750km",
        "2000km",
        "Precision",
        "Recall",
        "F1",
        "ValidScoreCount",
        "ExpectedScoreCount",
        "ScoreComplete",
    ]
    csv_lines = [",".join(csv_header)]
    for row in rows:
        csv_lines.append(",".join(str(row[h]) for h in csv_header))
    (OUT_DIR / "summary_true_rollout_50.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md_lines = [
        "| 消融 | 10km | 25km | 200km | 750km | 2000km | Precision | Recall | F1 | 评分条数 | 期望条数 | 评分完整 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['消融']} | {row['10km']} | {row['25km']} | {row['200km']} | {row['750km']} | {row['2000km']} | "
            f"{row['Precision']} | {row['Recall']} | {row['F1']} | {row['ValidScoreCount']} | {row['ExpectedScoreCount']} | {row['ScoreComplete']} |"
        )
    (OUT_DIR / "summary_true_rollout_50.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(OUT_DIR / "summary_true_rollout_50.md")


if __name__ == "__main__":
    main()
