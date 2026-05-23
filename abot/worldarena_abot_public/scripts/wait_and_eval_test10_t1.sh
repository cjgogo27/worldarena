#!/usr/bin/env bash
set -euo pipefail

BASE="/data/alice/cjtest/model_repros/worldarena_abot_public"
MANIFEST="$BASE/manifests/test_main_manifest.json"
RAW_DIR="$BASE/outputs/test10_t1_raw"
FLAT_DIR="$BASE/outputs/test10_t1_flat"
GEN_DATASET="$BASE/generated_dataset_test10_t1"
SUMMARY_JSON="$BASE/manifests/test10_t1_summary_vlm.json"
CONFIG_PATH="$BASE/config/worldarena_public_metrics_test10_t1.yaml"
LOG="$BASE/logs/test10_t1_eval.log"
RESULT_JSON="$RAW_DIR/results.json"

WORLDARENA_ROOT="/data/alice/cjtest/model_repros/WorldArena/video_quality"
ABOT_ROOT="/data/alice/cjtest/model_repros/ABot-PhysWorld"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

count_success() {
  python3 - <<'PY'
import json
from pathlib import Path
p = Path('/data/alice/cjtest/model_repros/worldarena_abot_public/outputs/test10_t1_raw/results.json')
mp4_dir = Path('/data/alice/cjtest/model_repros/worldarena_abot_public/outputs/test10_t1_raw')
if p.exists():
    rows = json.loads(p.read_text())
    print(sum(1 for r in rows if r.get('status') == 'success'))
elif mp4_dir.exists():
    print(len(list(mp4_dir.glob('*.mp4'))))
else:
    print(0)
PY
}

echo "[$(timestamp)] waiting for Track1-formal ABot test10 results" >> "$LOG"
while true; do
  SUCCESS_COUNT="$(count_success)"
  echo "[$(timestamp)] success_count=$SUCCESS_COUNT" >> "$LOG"
  if [ "$SUCCESS_COUNT" -ge 10 ]; then
    break
  fi
  sleep 180
done

while [ ! -f "$RESULT_JSON" ]; do
  echo "[$(timestamp)] waiting for results.json flush" >> "$LOG"
  sleep 30
done

echo "[$(timestamp)] remapping outputs" >> "$LOG"
python3 "$BASE/scripts/remap_abot_results.py" \
  --manifest "$MANIFEST" \
  --results-json "$RESULT_JSON" \
  --target-dir "$FLAT_DIR" \
  --copy >> "$LOG" 2>&1

echo "[$(timestamp)] preparing generated_dataset" >> "$LOG"
conda run -n WorldArena python "$BASE/../worldarena_gigaworld_public/scripts/prepare_worldarena_generated_dataset.py" \
  --manifest "$MANIFEST" \
  --video-dir "$FLAT_DIR" \
  --output-base "$GEN_DATASET" \
  --overwrite >> "$LOG" 2>&1

echo "[$(timestamp)] building VLM summary" >> "$LOG"
python3 - <<'PY' >> "$LOG" 2>&1
import json
from pathlib import Path
manifest = json.loads(Path('/data/alice/cjtest/model_repros/worldarena_abot_public/manifests/test_main_manifest.json').read_text())
rows = json.loads(Path('/data/alice/cjtest/model_repros/worldarena_abot_public/outputs/test10_t1_raw/results.json').read_text())
keep = {(r['video'], r['prompt']) for r in rows if r.get('status') == 'success'}
out = []
for item in manifest:
    if (item['image'], item['prompt']) in keep:
        out.append({'gt_path': f"/synthetic/worldarena/fixed_scene_task/{item['output_video']}", 'prompt': item['prompt']})
path = Path('/data/alice/cjtest/model_repros/worldarena_abot_public/manifests/test10_t1_summary_vlm.json')
path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(path)
print(len(out))
PY

echo "[$(timestamp)] writing metric config" >> "$LOG"
cat > "$CONFIG_PATH" <<'EOF'
model_name: abot_public_test10_t1
data:
  gt_path: /data/alice/cjtest/model_repros/worldarena_abot_public/private_gt_unavailable
  val_base: /data/alice/cjtest/model_repros/worldarena_abot_public/generated_dataset_test10_t1
data_action_following:
  gt_path: /data/alice/cjtest/model_repros/worldarena_abot_public/private_gt_unavailable
  val_base: /data/alice/cjtest/model_repros/worldarena_abot_public/generated_dataset_test10_t1
save_path: /data/alice/cjtest/model_repros/worldarena_abot_public/metrics_output_test10_t1
save_path_action_following: /data/alice/cjtest/model_repros/worldarena_abot_public/metrics_output_action_following_test10_t1
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
  --model_name abot_public_test10_t1 \
  --video_dir "$FLAT_DIR" \
  --summary_json "$SUMMARY_JSON" \
  --metrics all \
  --num_frames 16 \
  --output_root "$BASE/output_VLM" \
  --tmp_root "$BASE/tmp_VLM" \
  --config_path "$CONFIG_PATH" >> "$LOG" 2>&1

echo "[$(timestamp)] done" >> "$LOG"
