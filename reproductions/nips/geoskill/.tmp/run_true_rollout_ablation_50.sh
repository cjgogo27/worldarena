#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/alice/cjtest/NIPS/geoskill"
PY="/data2/miniconda3/envs/vgllm/bin/python"
LOG_DIR="$ROOT/experiments/ablation_geovista_true_rollout50"
mkdir -p "$LOG_DIR"

cd "$ROOT"

run_case() {
  local name="$1"
  local config="$2"
  local method="$3"
  local pattern="$4"
  local suffix="$5"
  local exp_dir="$6"
  local run_log="$LOG_DIR/${name}.log"
  local score_log="$LOG_DIR/${name}_score.log"

  echo "[$(date -u '+%F %T')] START ${name}" | tee -a "$LOG_DIR/pipeline.log"
  "$PY" scripts/run_experiment.py \
    --config "$config" \
    --methods "$method" \
    --max-games 50 \
    --workers 1 \
    > "$run_log" 2>&1

  "$PY" scripts/export_predictions_to_georc.py \
    --predictions "$exp_dir/$method/latest_predictions.json" \
    --challenge-root "$ROOT/data/georc" \
    --pattern "$pattern" \
    --force \
    --score \
    --score-mode key_points \
    --score-rounds 1 \
    --llm-model qwen_tiny_llm \
    --score-suffix "$suffix" \
    > "$score_log" 2>&1

  "$PY" .tmp/summarize_true_rollout_ablation.py > "$LOG_DIR/summary_refresh.log" 2>&1 || true
  echo "[$(date -u '+%F %T')] DONE ${name}" | tee -a "$LOG_DIR/pipeline.log"
}

run_case \
  "woskill" \
  ".tmp/ablation_geovista_true_mtl_woskill_50.yaml" \
  "direct_vlm" \
  "candidate_reasoning_chain_ablation_true_woskill_" \
  "ablation_true_woskill_50" \
  "$ROOT/experiments/ablation_geovista_true_mtl_rollout50_woskill"

run_case \
  "vote1" \
  ".tmp/ablation_geovista_true_mtl_vote1_50.yaml" \
  "external_geovista_skill_graph" \
  "candidate_reasoning_chain_ablation_true_vote1_" \
  "ablation_true_vote1_50" \
  "$ROOT/experiments/ablation_geovista_true_mtl_rollout50_vote1"

run_case \
  "vote3" \
  ".tmp/ablation_geovista_true_mtl_vote3_50.yaml" \
  "external_geovista_skill_graph" \
  "candidate_reasoning_chain_ablation_true_vote3_" \
  "ablation_true_vote3_50" \
  "$ROOT/experiments/ablation_geovista_true_mtl_rollout50_vote3"

run_case \
  "vote5" \
  ".tmp/ablation_geovista_true_mtl_vote5_50.yaml" \
  "external_geovista_skill_graph" \
  "candidate_reasoning_chain_ablation_true_vote5_" \
  "ablation_true_vote5_50" \
  "$ROOT/experiments/ablation_geovista_true_mtl_rollout50_vote5"

run_case \
  "ours" \
  ".tmp/ablation_geovista_true_mtl_ours_50.yaml" \
  "external_geovista_skill_graph" \
  "candidate_reasoning_chain_ablation_true_ours_" \
  "ablation_true_ours_50" \
  "$ROOT/experiments/ablation_geovista_true_mtl_rollout50_ours"

"$PY" .tmp/summarize_true_rollout_ablation.py | tee -a "$LOG_DIR/pipeline.log"
echo "[$(date -u '+%F %T')] ALL DONE" | tee -a "$LOG_DIR/pipeline.log"
