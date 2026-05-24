#!/usr/bin/env bash
set -euo pipefail

usage() { echo "usage: $0 <variant> <raw_dir> <manifest_dir> <manifest_parts> [do_dpo]"; exit 1; }
VARIANT="${1:?$(usage)}"
RAW_DIR="${2:?$(usage)}"
MANIFEST_DIR="${3:?$(usage)}"
MANIFEST_PARTS="${4:?$(usage)}"   # space-separated filenames, e.g. "part_00.json part_01.json ..."
DO_DPO="${5:-false}"

ROOT="/data/alice/cjtest/VideoX-Fun"
SEEDVR_DIR="$ROOT/eval_ckpt_${VARIANT}_seedvr"
RAW_FLAT="$ROOT/eval_ckpt_${VARIANT}_raw_flat"
SEEDVR_FLAT="$ROOT/eval_ckpt_${VARIANT}_seedvr_flat"
COMBINED_MANIFEST="$ROOT/eval_ckpt_${VARIANT}_manifest.json"
RAW_SUMMARY="$ROOT/eval_ckpt_${VARIANT}_raw_summary_vlm.json"
SEEDVR_SUMMARY="$ROOT/eval_ckpt_${VARIANT}_seedvr_summary_vlm.json"
RAW_DATASET="$ROOT/eval_ckpt_${VARIANT}_raw_generated_dataset"
SEEDVR_DATASET="$ROOT/eval_ckpt_${VARIANT}_seedvr_generated_dataset"
RAW_CONFIG="$ROOT/eval_ckpt_${VARIANT}_raw_metrics.yaml"
SEEDVR_CONFIG="$ROOT/eval_ckpt_${VARIANT}_seedvr_metrics.yaml"
RAW_METRICS="$ROOT/metrics_output_ckpt_${VARIANT}_raw"
SEEDVR_METRICS="$ROOT/metrics_output_ckpt_${VARIANT}_seedvr"
RAW_VLM_OUT="$ROOT/output_VLM_ckpt_${VARIANT}_raw"
SEEDVR_VLM_OUT="$ROOT/output_VLM_ckpt_${VARIANT}_seedvr"
RAW_VLM_TMP="$ROOT/tmp_VLM_ckpt_${VARIANT}_raw"
SEEDVR_VLM_TMP="$ROOT/tmp_VLM_ckpt_${VARIANT}_seedvr"

LOG="$ROOT/eval_ckpt_${VARIANT}_raw_seedvr_eval.log"
PROGRESS="$ROOT/eval_ckpt_${VARIANT}_raw_seedvr_eval_progress.log"

WORLDARENA_ROOT="/data/alice/cjtest/model_repros/WorldArena/video_quality"
PREP_SCRIPT="/data/alice/cjtest/model_repros/worldarena_gigaworld_public/scripts/prepare_worldarena_generated_dataset.py"

# DPO paths (only used if DO_DPO=true)
DPO_OUT="/data/alice/cjtest/datasets/worldarena_dpo_pairs_ckpt2200_${VARIANT}"
DPO_FLAT="/data/alice/cjtest/datasets/worldarena_dpo_eval_flat_${VARIANT}"
DPO_MANIFEST="/data/alice/cjtest/datasets/worldarena_dpo_eval_manifest_${VARIANT}.json"
DPO_SUMMARY="/data/alice/cjtest/datasets/worldarena_dpo_eval_summary_vlm_${VARIANT}.json"
DPO_INDEX="/data/alice/cjtest/datasets/worldarena_dpo_eval_index_${VARIANT}.json"
DPO_DATASET="/data/alice/cjtest/datasets/worldarena_dpo_eval_generated_dataset_${VARIANT}"
DPO_CONFIG="/data/alice/cjtest/datasets/worldarena_dpo_eval_metrics_${VARIANT}.yaml"
DPO_METRICS="/data/alice/cjtest/datasets/worldarena_dpo_eval_metrics_output_${VARIANT}"
DPO_VLM_OUT="/data/alice/cjtest/datasets/worldarena_dpo_eval_output_VLM_${VARIANT}"
DPO_VLM_TMP="/data/alice/cjtest/datasets/worldarena_dpo_eval_tmp_VLM_${VARIANT}"

ts(){ date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
progress(){ echo "[$(ts)] $*" | tee -a "$PROGRESS"; }
count_mp4(){ find "$1" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l | tr -d ' '; }

mkdir -p "$(dirname "$LOG")" "$(dirname "$PROGRESS")"
if [ "$DO_DPO" = true ]; then
  mkdir -p "$DPO_OUT"
fi
log "pipeline_start variant=$VARIANT raw=$RAW_DIR seedvr=$SEEDVR_DIR manifest=$MANIFEST_DIR"

# ---- wait for raw + seedvr videos ----
wait_for_count() {
  local dir=$1 label=$2 target=${3:-1000}
  while true; do
    local count
    count=$(count_mp4 "$dir")
    progress "$label count=$count/$target"
    if [ "$count" -ge "$target" ]; then
      break
    fi
    sleep 300
  done
}

wait_for_count "$RAW_DIR" raw 1000
wait_for_count "$SEEDVR_DIR" seedvr 1000

# ---- build combined manifest + VLM summaries ----
build_manifest_and_summaries() {
  log "build_manifest_and_summaries"
  python3 - "$MANIFEST_DIR" "$COMBINED_MANIFEST" "$RAW_SUMMARY" "$SEEDVR_SUMMARY" "$MANIFEST_PARTS" <<'PY' >> "$LOG" 2>&1
import json, sys
from pathlib import Path
manifest_dir = Path(sys.argv[1])
combined_manifest = Path(sys.argv[2])
raw_summary = Path(sys.argv[3])
seedvr_summary = Path(sys.argv[4])
manifest_parts = sys.argv[5].split()

seen = {}
for name in manifest_parts:
    path = manifest_dir / name
    if not path.exists():
        continue
    for item in json.loads(path.read_text(encoding='utf-8')):
        out = item['output_video']
        item = dict(item)
        item.setdefault('task_name', 'fixed_scene_task')
        if 'episode_id' not in item:
            stem = Path(out).stem
            item['episode_id'] = stem[len('fixed_scene_task_'):] if stem.startswith('fixed_scene_task_') else stem
        seen[out] = item

parts = list(seen.values())
parts.sort(key=lambda x: x['output_video'])
combined_manifest.write_text(json.dumps(parts, ensure_ascii=False, indent=2), encoding='utf-8')

summary = [{'gt_path': f"/synthetic/worldarena/fixed_scene_task/{item['output_video']}", 'prompt': item['prompt']} for item in parts]
raw_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
seedvr_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print({'manifest_items': len(parts)})
PY
}

# ---- flatten: only copy manifest-listed videos ----
flatten_selected() {
  local src=$1 dst=$2 label=$3
  log "flatten_selected $label"
  rm -rf "$dst"
  mkdir -p "$dst"
  python3 - "$src" "$dst" "$COMBINED_MANIFEST" <<'PY' >> "$LOG" 2>&1
import json, shutil, sys
from pathlib import Path
src=Path(sys.argv[1]); dst=Path(sys.argv[2]); manifest=Path(sys.argv[3])
copied=0; missing=[]
for item in json.loads(manifest.read_text(encoding='utf-8')):
    name=item['output_video']
    source=src/name
    if not source.exists():
        missing.append(str(source)); continue
    shutil.copy2(source, dst/name)
    copied += 1
print({'copied': copied, 'missing': missing[:5], 'missing_count': len(missing), 'dst': str(dst)})
if missing:
    raise SystemExit('missing selected videos')
PY
}

# ---- write config yaml ----
write_config() {
  local config=$1 model_name=$2 dataset=$3 metrics_out=$4 action_out=$5
  cat > "$config" <<EOF
model_name: $model_name
data:
  gt_path: /data/alice/cjtest/VideoX-Fun/private_gt_unavailable
  val_base: $dataset
data_action_following:
  gt_path: /data/alice/cjtest/VideoX-Fun/private_gt_unavailable
  val_base: $dataset
save_path: $metrics_out
save_path_action_following: $action_out
ckpt:
  aesthetic_quality:
    clip: ViT-L/14
    aesthetic_head: /data/alice/cjtest/model_repros/WorldArena/video_quality/models/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth
  background_consistency:
    clip: ViT-B/32
    raft: /data/alice/cjtest/model_repros/WorldArena/video_quality/models/raft_model/models/raft-things.pth
  dynamic_degree:
    raft: /data/alice/cjtest/model_repros/WorldArena/video_quality/models/raft_model/models/raft-things.pth
  flow_score:
    raft: /data/alice/cjtest/model_repros/WorldArena/video_quality/models/raft_model/models/raft-things.pth
  image_quality:
    musiq: /data/alice/cjtest/model_repros/WorldArena/video_quality/models/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth
  subject_consistency:
    repo: /data/alice/cjtest/model_repros/WorldArena/video_quality/models/dino_model/facebookresearch_dino_main
    weight: /data/alice/cjtest/model_repros/WorldArena/video_quality/models/dino_model/dino_vitbase16_pretrain.pth
    model: dino_vitb16
    raft: /data/alice/cjtest/model_repros/WorldArena/video_quality/models/raft_model/models/raft-things.pth
  vlm_model: Qwen/Qwen3-VL-8B-Instruct
EOF
}

# ---- prepare dataset for eval ----
prepare_dataset() {
  local flat=$1 out=$2 label=$3
  log "prepare_dataset $label"
  conda run -n WorldArena python "$PREP_SCRIPT" \
    --manifest "$COMBINED_MANIFEST" \
    --video-dir "$flat" \
    --output-base "$out" \
    --overwrite >> "$LOG" 2>&1
}

# ---- standard eval (image/video quality metrics) ----
run_standard_eval() {
  local config=$1 label=$2 out=$3
  local result="$out/${label}_generated_results.json"
  if [ -f "$result" ]; then
    log "standard_eval_skip_existing $label result=$result"
    return
  fi
  log "standard_eval_start $label"
  MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29691 + RANDOM % 100)) RANK=0 WORLD_SIZE=1 PYTHONNOUSERSITE=1 \
    LD_LIBRARY_PATH="/data2/miniconda3/envs/WorldArena/lib:${LD_LIBRARY_PATH:-}" \
    /data2/miniconda3/envs/WorldArena/bin/python "$WORLDARENA_ROOT/evaluate.py" \
    --dimension image_quality aesthetic_quality background_consistency dynamic_degree flow_score subject_consistency \
    --config "$config" \
    --overwrite >> "$LOG" 2>&1
  log "standard_eval_done $label"
}

# ---- VLM eval ----
run_vlm_eval() {
  local label=$1 flat=$2 summary=$3 out_root=$4 tmp_root=$5 config=$6
  local result="$out_root/$label/${label}_summary_val_all_intern.json"
  if [ -f "$result" ]; then
    log "vlm_eval_skip_existing $label result=$result"
    return
  fi
  log "vlm_eval_start $label"
  PYTHONNOUSERSITE=1 /data2/miniconda3/envs/WorldArena_VLM/bin/python "$WORLDARENA_ROOT/VLM_judge.py" \
    --model_name "$label" \
    --video_dir "$flat" \
    --summary_json "$summary" \
    --metrics all \
    --num_frames 16 \
    --output_root "$out_root" \
    --tmp_root "$tmp_root" \
    --config_path "$config" >> "$LOG" 2>&1
  log "vlm_eval_done $label"
}

# ============================================================
# MAIN
# ============================================================

build_manifest_and_summaries

flatten_selected "$RAW_DIR" "$RAW_FLAT" raw
flatten_selected "$SEEDVR_DIR" "$SEEDVR_FLAT" seedvr

prepare_dataset "$RAW_FLAT" "$RAW_DATASET" raw
prepare_dataset "$SEEDVR_FLAT" "$SEEDVR_DATASET" seedvr

write_config "$RAW_CONFIG" "${VARIANT}_raw" "$RAW_DATASET" "$RAW_METRICS" "$ROOT/metrics_output_action_following_ckpt_${VARIANT}_raw"
write_config "$SEEDVR_CONFIG" "${VARIANT}_seedvr" "$SEEDVR_DATASET" "$SEEDVR_METRICS" "$ROOT/metrics_output_action_following_ckpt_${VARIANT}_seedvr"

run_standard_eval "$RAW_CONFIG" "eval_ckpt_${VARIANT}_raw" "$RAW_METRICS"
run_standard_eval "$SEEDVR_CONFIG" "eval_ckpt_${VARIANT}_seedvr" "$SEEDVR_METRICS"

run_vlm_eval "eval_ckpt_${VARIANT}_raw" "$RAW_FLAT" "$RAW_SUMMARY" "$RAW_VLM_OUT" "$RAW_VLM_TMP" "$RAW_CONFIG"
run_vlm_eval "eval_ckpt_${VARIANT}_seedvr" "$SEEDVR_FLAT" "$SEEDVR_SUMMARY" "$SEEDVR_VLM_OUT" "$SEEDVR_VLM_TMP" "$SEEDVR_CONFIG"

# ---- optional DPO eval + pair building ----
if [ "$DO_DPO" = true ]; then
  log "prepare_dpo_eval_start"
  python3 "$ROOT/scripts/worldarena/prepare_dpo_data_for_worldarena_eval.py" \
    --dpo-root /data/alice/cjtest/datasets/dpo_data/dpo_data \
    --flat-dir "$DPO_FLAT" \
    --manifest "$DPO_MANIFEST" \
    --summary-json "$DPO_SUMMARY" \
    --index-json "$DPO_INDEX" \
    --overwrite >> "$LOG" 2>&1
  prepare_dataset "$DPO_FLAT" "$DPO_DATASET" dpo
  write_config "$DPO_CONFIG" "${VARIANT}_dpo_eval" "$DPO_DATASET" "$DPO_METRICS" "/data/alice/cjtest/datasets/worldarena_dpo_eval_metrics_action_output_${VARIANT}"
  run_standard_eval "$DPO_CONFIG" "${VARIANT}_dpo_eval" "$DPO_METRICS"
  run_vlm_eval "${VARIANT}_dpo_eval" "$DPO_FLAT" "$DPO_SUMMARY" "$DPO_VLM_OUT" "$DPO_VLM_TMP" "$DPO_CONFIG"

  log "build_dpo_pairs_start"
  python3 "$ROOT/scripts/worldarena/build_worldarena_dpo_pairs.py" \
    --dpo-root /data/alice/cjtest/datasets/dpo_data/dpo_data \
    --raw-video-dir "$RAW_FLAT" \
    --seedvr-video-dir "$SEEDVR_FLAT" \
    --raw-metrics "$RAW_METRICS/eval_ckpt_${VARIANT}_raw_generated_results.json" \
    --seedvr-metrics "$SEEDVR_METRICS/eval_ckpt_${VARIANT}_seedvr_generated_results.json" \
    --raw-vlm "$RAW_VLM_OUT/eval_ckpt_${VARIANT}_raw/eval_ckpt_${VARIANT}_raw_summary_val_all_intern.json" \
    --seedvr-vlm "$SEEDVR_VLM_OUT/eval_ckpt_${VARIANT}_seedvr/eval_ckpt_${VARIANT}_seedvr_summary_val_all_intern.json" \
    --dpo-metrics "$DPO_METRICS/${VARIANT}_dpo_eval_generated_results.json" \
    --dpo-vlm "$DPO_VLM_OUT/${VARIANT}_dpo_eval/${VARIANT}_dpo_eval_summary_val_all_intern.json" \
    --dpo-index "$DPO_INDEX" \
    --raw-manifest "$COMBINED_MANIFEST" \
    --output-dir "$DPO_OUT" \
    --min-margin 5.0 >> "$LOG" 2>&1
  log "build_dpo_pairs_done output=$DPO_OUT"
fi

log "pipeline_done variant=$VARIANT"
progress "DONE variant=$VARIANT raw_metrics=$RAW_METRICS seedvr_metrics=$SEEDVR_METRICS"
