#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="/data2/miniconda3/envs/vgllm/bin/python"
MAX_GAMES="${1:-50}"

cd "$ROOT_DIR"

AB_DIR="experiments/ablation_geovista_skill_graph_georc${MAX_GAMES}"
mkdir -p "$AB_DIR"

run_case() {
  local name="$1"
  local config="$2"
  local method="$3"
  local log_file="$AB_DIR/${name}.log"

  echo "[$(date '+%F %T')] START ${name} method=${method} max_games=${MAX_GAMES}" | tee -a "$AB_DIR/pipeline.log"
  "$PY" scripts/run_experiment.py \
    --config "$config" \
    --methods "$method" \
    --max-games "$MAX_GAMES" \
    --workers 1 \
    > "$log_file" 2>&1
  echo "[$(date '+%F %T')] DONE  ${name}" | tee -a "$AB_DIR/pipeline.log"
}

run_case "woskill" "configs/ablation_geovista_skill_graph_woskill_georc50.yaml" "direct_vlm"
run_case "rollout1" "configs/ablation_geovista_skill_graph_rollout1_georc50.yaml" "external_geovista_skill_graph"
run_case "rollout3" "configs/ablation_geovista_skill_graph_rollout3_georc50.yaml" "external_geovista_skill_graph"
run_case "rollout5" "configs/ablation_geovista_skill_graph_rollout5_georc50.yaml" "external_geovista_skill_graph"

"$PY" - <<'PY'
import json
import re
from datetime import datetime, timezone
from pathlib import Path

cases = [
    ("woskill", "w.o.skill（direct_vlm）", Path("experiments/ablation_geovista_skill_graph_georc50_woskill/direct_vlm/latest_metrics.json")),
    ("rollout1", "roll out 次数1", Path("experiments/ablation_geovista_skill_graph_georc50_rollout1/external_geovista_skill_graph/latest_metrics.json")),
    ("rollout3", "roll out 次数3", Path("experiments/ablation_geovista_skill_graph_georc50_rollout3/external_geovista_skill_graph/latest_metrics.json")),
    ("rollout5", "roll out 次数5", Path("experiments/ablation_geovista_skill_graph_georc50_rollout5/external_geovista_skill_graph/latest_metrics.json")),
]

pipeline_log = Path("experiments/ablation_geovista_skill_graph_georc50/pipeline.log")


def parse_ts(raw: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S UTC"):
        try:
            dt = datetime.strptime(raw, fmt)
            if fmt.endswith("UTC"):
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def fmt_float(val: float | None, ndigits: int = 1) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float) and val != val:
        return "N/A"
    return f"{val:.{ndigits}f}"


starts: dict[str, datetime] = {}
dones: dict[str, datetime] = {}
if pipeline_log.exists():
    pattern = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<evt>START|DONE)\s+(?P<name>[a-zA-Z0-9_]+)")
    for line in pipeline_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        ts = parse_ts(m.group("ts"))
        if ts is None:
            continue
        name = m.group("name")
        if m.group("evt") == "START":
            starts[name] = ts
        else:
            dones[name] = ts


header = [
    "消融",
    "10km",
    "25km",
    "200km",
    "750km",
    "2000km",
    "ReportedTime(s)",
    "WallTime(s)",
    "SecPerSample",
    "OverheadVsR1",
]
rows = []
rollout1_wall = None
rollout1_reported = None

for case_key, name, p in cases:
    wall_elapsed = None
    if case_key in starts and case_key in dones:
        wall_elapsed = (dones[case_key] - starts[case_key]).total_seconds()

    if not p.exists():
        rows.append([name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", fmt_float(wall_elapsed), "N/A", "N/A"])
        continue

    m = json.loads(p.read_text(encoding="utf-8"))
    reported_elapsed = float(m.get("elapsed_seconds", float("nan")))
    num_samples = int(m.get("num_samples", 0) or 0)
    sec_per_sample = None
    if reported_elapsed == reported_elapsed and num_samples > 0:
        sec_per_sample = reported_elapsed / num_samples

    if case_key == "rollout1":
        rollout1_wall = wall_elapsed
        rollout1_reported = reported_elapsed if reported_elapsed == reported_elapsed else None

    baseline = rollout1_wall if rollout1_wall is not None else rollout1_reported
    current = wall_elapsed if wall_elapsed is not None else (reported_elapsed if reported_elapsed == reported_elapsed else None)
    overhead = None
    if baseline is not None and baseline > 0 and current is not None:
        overhead = current / baseline

    rows.append([
        name,
        f"{float(m.get('Acc@10km', 0.0)):.3f}",
        f"{float(m.get('Acc@25km', 0.0)):.3f}",
        f"{float(m.get('Acc@200km', 0.0)):.3f}",
        f"{float(m.get('Acc@750km', 0.0)):.3f}",
        f"{float(m.get('Acc@2000km', 0.0)):.3f}",
        fmt_float(reported_elapsed),
        fmt_float(wall_elapsed),
        fmt_float(sec_per_sample, 3),
        "N/A" if overhead is None else f"{overhead:.2f}x",
    ])

out_dir = Path("experiments/ablation_geovista_skill_graph_georc50")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "summary.csv"
csv_lines = [",".join(header)] + [",".join(r) for r in rows]
csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

md_path = out_dir / "summary.md"
md = [
    "| 消融 | 10km | 25km | 200km | 750km | 2000km | Reported(s) | Wall(s) | Sec/Sample | Overhead vs R1 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    md.append(
        f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} | {r[9]} |"
    )
md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

print(f"Wrote {csv_path}")
print(f"Wrote {md_path}")
PY

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$AB_DIR/pipeline.log"
