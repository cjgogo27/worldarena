#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_raw"
FLAT_DIR="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_flat"
MANIFEST_DIR="/data/alice/cjtest/VideoX-Fun/test_dataset/manifests_1000_gpu0567_fastresume"
SUMMARY_JSON="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_summary_vlm.json"
COMBINED_MANIFEST="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_manifest.json"
CONFIG_YAML="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_metrics.yaml"
GEN_DATASET="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_generated_dataset"
LOG="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_finalize.log"
TRAIN_SCRIPT="/data/alice/cjtest/VideoX-Fun/scripts/wan2.1/wait_and_train_worldarena_wan_i2v_lora.sh"
WORLDARENA_ROOT="/data/alice/cjtest/model_repros/WorldArena/video_quality"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

echo "[$(ts)] finalize start" >> "$LOG"

# Launch overflow workers when current ones finish but total < 1000
launch_overflow() {
  local gpu=$1 manifest=$2 log=$3
  local pid_file="/tmp/overflow_${gpu}.pid"
  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 nohup bash -c "
    cd /data/alice/cjtest/VideoX-Fun
    /data/envs/videox_fun_wan/bin/python /data/alice/cjtest/VideoX-Fun/scripts/wan2.1/batch_predict_i2v_worldarena.py \
      --manifest $manifest \
      --dataset-root /data/alice/cjtest/VideoX-Fun/test_dataset \
      --base-model /data/alice/cjtest/model_repros/ABot-PhysWorld/models/Wan-AI/Wan2.1-I2V-14B-480P \
      --lora-path /data/alice/cjtest/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora/checkpoint-2200.safetensors \
      --output-dir $RAW_DIR \
      --enable-teacache --teacache-threshold 0.20 --gpu-memory-mode none
  " > "$log" 2>&1 &
  echo $! > "$pid_file"
  echo "[$(ts)] overflow_worker launched: GPU=$gpu manifest=$manifest" >> "$LOG"
}

while true; do
  count=$(find "$RAW_DIR" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')
  echo "[$(ts)] mp4_count=$count" >> "$LOG"
  if [ "$count" -ge 1000 ]; then
    break
  fi

  # Auto-launch overflow workers if current inference workers finished
  for gpu in 5 6 7; do
    of_manifest="$MANIFEST_DIR/part_00_overflow_gpu${gpu}.json"
    of_log="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_logs/overflow_gpu${gpu}.log"
    # Check if overflow manifest has pending videos
    pending=$(python3 -c "
import json
from pathlib import Path
m=json.loads(Path('$of_manifest').read_text())
raw=Path('$RAW_DIR')
remain=[i for i in m if not (raw/i['output_video']).exists()]
print(len(remain))
" 2>/dev/null || echo 0)
    if [ "$pending" -gt 0 ]; then
      # Check if current worker on this GPU has finished (GPU memory almost free)
      mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' || echo "999999")
      if [ "$mem_used" -lt 50000 ] 2>/dev/null; then
        launch_overflow "$gpu" "$of_manifest" "$of_log"
      fi
    fi
  done

  sleep 300
done

echo "[$(ts)] all_videos_ready" >> "$LOG"

mkdir -p "$FLAT_DIR"
python3 - <<'PY' >> "$LOG" 2>&1
import json, shutil
from pathlib import Path
raw = Path('/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_raw')
flat = Path('/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_flat')
for src in raw.glob('*.mp4'):
    dst = flat / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
print('copied', len(list(flat.glob('*.mp4'))))
PY

python3 - <<'PY' >> "$LOG" 2>&1
import json
from pathlib import Path
parts=[]
manifest_dir='/data/alice/cjtest/VideoX-Fun/test_dataset/manifests_1000_gpu0567_fastresume'
for name in ['part_00.json','part_01.json','part_02.json','part_03.json','part_00_overflow_gpu5.json','part_00_overflow_gpu6.json','part_00_overflow_gpu7.json']:
    p=Path(manifest_dir)/name
    if p.exists():
        parts.extend(json.loads(p.read_text()))
manifest_path=Path('/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_manifest.json')
manifest_path.write_text(json.dumps(parts, ensure_ascii=False, indent=2), encoding='utf-8')
summary=[{'gt_path': f"/synthetic/worldarena/fixed_scene_task/{item['output_video']}", 'prompt': item['prompt']} for item in parts]
path=Path('/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_summary_vlm.json')
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(path, len(summary))
print(manifest_path, len(parts))
PY

conda run -n WorldArena python /data/alice/cjtest/model_repros/worldarena_gigaworld_public/scripts/prepare_worldarena_generated_dataset.py \
  --manifest "$COMBINED_MANIFEST" \
  --video-dir "$FLAT_DIR" \
  --output-base "$GEN_DATASET" \
  --overwrite >> "$LOG" 2>&1

cat > "$CONFIG_YAML" <<'EOF'
model_name: videoxfun_ckpt_latest_test1000
data:
  gt_path: /data/alice/cjtest/VideoX-Fun/private_gt_unavailable
  val_base: /data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_generated_dataset
data_action_following:
  gt_path: /data/alice/cjtest/VideoX-Fun/private_gt_unavailable
  val_base: /data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_generated_dataset
save_path: /data/alice/cjtest/VideoX-Fun/metrics_output_ckpt_latest_test1000
save_path_action_following: /data/alice/cjtest/VideoX-Fun/metrics_output_action_following_ckpt_latest_test1000
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

MASTER_ADDR=127.0.0.1 MASTER_PORT=29671 RANK=0 WORLD_SIZE=1 PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="/data2/miniconda3/envs/WorldArena/lib:${LD_LIBRARY_PATH:-}" /data2/miniconda3/envs/WorldArena/bin/python "$WORLDARENA_ROOT/evaluate.py" \
  --dimension image_quality aesthetic_quality background_consistency dynamic_degree flow_score subject_consistency \
  --config "$CONFIG_YAML" \
  --overwrite >> "$LOG" 2>&1

PYTHONNOUSERSITE=1 /data2/miniconda3/envs/WorldArena_VLM/bin/python "$WORLDARENA_ROOT/VLM_judge.py" \
  --model_name videoxfun_ckpt_latest_test1000 \
  --video_dir "$FLAT_DIR" \
  --summary_json "$SUMMARY_JSON" \
  --metrics all \
  --num_frames 16 \
  --output_root /data/alice/cjtest/VideoX-Fun/output_VLM_ckpt_latest_test1000 \
  --tmp_root /data/alice/cjtest/VideoX-Fun/tmp_VLM_ckpt_latest_test1000 \
  --config_path "$CONFIG_YAML" >> "$LOG" 2>&1

echo "[$(ts)] restarting training" >> "$LOG"
cd /data/alice/cjtest/VideoX-Fun && export CUDA_VISIBLE_DEVICES=6,7 PYTHONNOUSERSITE=1; bash "$TRAIN_SCRIPT" >> "$LOG" 2>&1
