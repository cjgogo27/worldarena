#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_raw"
SEEDVR_IN="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr_in"
SEEDVR_OUT="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr"
LOG="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr.log"
SEEDVR_ROOT="/data/alice/cjtest/model_repros/FlowWAM_WorldArena/inference/refiner/SeedVR"

mkdir -p "$SEEDVR_IN" "$SEEDVR_OUT"

ts(){ date -u +"%Y-%m-%dT%H:%M:%SZ"; }

pick_gpu() {
  python3 -c "
import subprocess, json
out = subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], capture_output=True, text=True)
best = (-1, 0)
for line in out.stdout.strip().splitlines():
    idx, free = line.split(', ')
    idx, free = int(idx), int(free)
    if free > best[1]:
        best = (idx, free)
print(best[0])
"
}

while true; do
  # link any new raw mp4s into input dir
  python3 - <<'PY' >> "$LOG" 2>&1
from pathlib import Path
raw=Path('/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_raw')
inp=Path('/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr_in')
out=Path('/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr')
added=0
for src in raw.glob('*.mp4'):
    dst=inp/src.name
    out_file=out/src.name
    if out_file.exists() or dst.exists():
        continue
    dst.symlink_to(src)
    added += 1
print('added_inputs', added)
PY

  count_in=$(find "$SEEDVR_IN" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')
  if [ "$count_in" -gt 0 ]; then
    TARGET_GPU=$(pick_gpu)
    echo "[$(ts)] refining pending_videos=$count_in on GPU=$TARGET_GPU" >> "$LOG"
    cd "$SEEDVR_ROOT" && PATH=/data2/envs/seedvr/bin:$PATH MASTER_ADDR=127.0.0.1 MASTER_PORT=29681 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 PYTHONPATH="$SEEDVR_ROOT" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$TARGET_GPU /data2/envs/seedvr/bin/python "$SEEDVR_ROOT/projects/inference_seedvr2_3b.py" \
      --video_path "$SEEDVR_IN" \
      --output_dir "$SEEDVR_OUT" \
      --seed 42 \
      --res_h 480 \
      --res_w 832 \
      --sp_size 1 \
      --out_fps 24 >> "$LOG" 2>&1 || true
    # clear symlinks already processed successfully
    python3 - <<'PY' >> "$LOG" 2>&1
from pathlib import Path
inp=Path('/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr_in')
out=Path('/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr')
removed=0
for src in list(inp.glob('*.mp4')):
    if (out/src.name).exists():
        src.unlink()
        removed += 1
print('removed_inputs', removed)
PY
  fi

  raw_count=$(find "$RAW_DIR" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')
  out_count=$(find "$SEEDVR_OUT" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')
  echo "[$(ts)] raw_count=$raw_count seedvr_count=$out_count" >> "$LOG"
  if [ "$raw_count" -ge 1000 ] && [ "$out_count" -ge 1000 ]; then
    echo "[$(ts)] seedvr_done" >> "$LOG"
    break
  fi
  sleep 300
done
