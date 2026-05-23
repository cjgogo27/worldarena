#!/usr/bin/env bash
set -euo pipefail

BASE="/data/alice/cjtest/model_repros/worldarena_wan_public"
MANIFEST="$BASE/manifests/test_main_manifest.json"
RAW_DIR="$BASE/outputs/test10_t1_seedvr"
GEN_DATASET="$BASE/generated_dataset_test10_t1_seedvr"
SUMMARY_JSON="$BASE/manifests/test10_t1_seedvr_summary_vlm.json"
CONFIG_PATH="$BASE/config/worldarena_public_metrics_test10_t1_seedvr.yaml"
LOG="$BASE/logs/test10_t1_seedvr_eval.log"

WORLDARENA_ROOT="/data/alice/cjtest/model_repros/WorldArena/video_quality"

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

count_outputs() {
  python3 - <<'PY'
from pathlib import Path
p = Path('/data/alice/cjtest/model_repros/worldarena_wan_public/outputs/test10_t1_seedvr')
print(len(list(p.glob('*.mp4'))) if p.exists() else 0)
PY
}

echo "[$(timestamp)] waiting for Wan SeedVR refined outputs" >> "$LOG"
while true; do
  COUNT="$(count_outputs)"
  echo "[$(timestamp)] mp4_count=$COUNT" >> "$LOG"
  if [ "$COUNT" -ge 10 ]; then
    break
  fi
  sleep 60
done

echo "[$(timestamp)] preparing generated_dataset" >> "$LOG"
conda run -n WorldArena python "/data/alice/cjtest/model_repros/worldarena_gigaworld_public/scripts/prepare_worldarena_generated_dataset.py" \
  --manifest "$MANIFEST" \
  --video-dir "$RAW_DIR" \
  --output-base "$GEN_DATASET" \
  --overwrite >> "$LOG" 2>&1

echo "[$(timestamp)] building VLM summary" >> "$LOG"
python3 - <<'PY' >> "$LOG" 2>&1
import json
from pathlib import Path
manifest = json.loads(Path('/data/alice/cjtest/model_repros/worldarena_wan_public/manifests/test_main_manifest.json').read_text())
out = [{'gt_path': f"/synthetic/worldarena/fixed_scene_task/{item['output_video']}", 'prompt': item['prompt']} for item in manifest]
path = Path('/data/alice/cjtest/model_repros/worldarena_wan_public/manifests/test10_t1_seedvr_summary_vlm.json')
path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(path)
print(len(out))
PY

echo "[$(timestamp)] writing metric config" >> "$LOG"
cat > "$CONFIG_PATH" <<'EOF'
model_name: wan_public_test10_t1_seedvr
data:
  gt_path: /data/alice/cjtest/model_repros/worldarena_wan_public/private_gt_unavailable
  val_base: /data/alice/cjtest/model_repros/worldarena_wan_public/generated_dataset_test10_t1_seedvr
data_action_following:
  gt_path: /data/alice/cjtest/model_repros/worldarena_wan_public/private_gt_unavailable
  val_base: /data/alice/cjtest/model_repros/worldarena_wan_public/generated_dataset_test10_t1_seedvr
save_path: /data/alice/cjtest/model_repros/worldarena_wan_public/metrics_output_test10_t1_seedvr
save_path_action_following: /data/alice/cjtest/model_repros/worldarena_wan_public/metrics_output_action_following_test10_t1_seedvr
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

echo "[$(timestamp)] running standard metrics subset" >> "$LOG"
LD_LIBRARY_PATH="/data2/miniconda3/envs/WorldArena/lib:${LD_LIBRARY_PATH:-}" PYTHONNOUSERSITE=1 \
  /data2/miniconda3/envs/WorldArena/bin/python "$WORLDARENA_ROOT/evaluate.py" \
  --dimension image_quality aesthetic_quality background_consistency dynamic_degree flow_score subject_consistency \
  --config "$CONFIG_PATH" \
  --overwrite >> "$LOG" 2>&1

echo "[$(timestamp)] running VLM metrics" >> "$LOG"
PYTHONNOUSERSITE=1 /data2/miniconda3/envs/WorldArena_VLM/bin/python "$WORLDARENA_ROOT/VLM_judge.py" \
  --model_name wan_public_test10_t1_seedvr \
  --video_dir "$RAW_DIR" \
  --summary_json "$SUMMARY_JSON" \
  --metrics all \
  --num_frames 16 \
  --output_root "$BASE/output_VLM" \
  --tmp_root "$BASE/tmp_VLM" \
  --config_path "$CONFIG_PATH" >> "$LOG" 2>&1

echo "[$(timestamp)] done" >> "$LOG"
