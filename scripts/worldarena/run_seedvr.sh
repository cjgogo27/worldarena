#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:?usage: $0 raw_dir seedvr_in seedvr_out log [target_count] [gpu]}"
SEEDVR_IN="${2:?}"
SEEDVR_OUT="${3:?}"
LOG="${4:?}"
TARGET="${5:-1000}"
PIN_GPU="${6:-}"   # optional: pin to a specific GPU index
# Set SEEDVR_ROOT env var to override; default assumes FlowWAM_WorldArena repo
SEEDVR_ROOT="${SEEDVR_ROOT:-/data/alice/cjtest/model_repros/FlowWAM_WorldArena/inference/refiner/SeedVR}"

mkdir -p "$SEEDVR_IN" "$SEEDVR_OUT" "$(dirname "$LOG")"

ts(){ date -u +"%Y-%m-%dT%H:%M:%SZ"; }

pick_gpu() {
  python3 -c "
import subprocess, os
# Exclude GPUs already running a seedvr inference process
used = set()
try:
    out = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=10)
    for line in out.stdout.splitlines():
        if 'inference_seedvr' in line:
            for part in line.split():
                if part.startswith('CUDA_VISIBLE_DEVICES='):
                    used.add(int(part.split('=')[1]))
except: pass
out = subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], capture_output=True, text=True)
best = (-1, 0)
for line in out.stdout.strip().splitlines():
    idx, free = line.split(', ')
    idx, free = int(idx), int(free)
    if idx in used:
        continue
    if free > best[1]:
        best = (idx, free)
print(best[0])
"
}

if [ -n "$PIN_GPU" ]; then
  TARGET_GPU="$PIN_GPU"
else
  TARGET_GPU=$(pick_gpu)
fi
echo "[$(ts)] seedvr_daemon_start raw=$RAW_DIR in=$SEEDVR_IN out=$SEEDVR_OUT target=$TARGET gpu=$TARGET_GPU" >> "$LOG"

while true; do
  python3 - "$RAW_DIR" "$SEEDVR_IN" "$SEEDVR_OUT" <<'PY' >> "$LOG" 2>&1
import sys
from pathlib import Path
raw = Path(sys.argv[1])
inp = Path(sys.argv[2])
out = Path(sys.argv[3])
added = 0
for src in raw.glob('*.mp4'):
    dst = inp / src.name
    out_file = out / src.name
    if out_file.exists() or dst.exists():
        continue
    dst.symlink_to(src)
    added += 1
print('added_inputs', added)
PY

  count_in=$(find "$SEEDVR_IN" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')
  if [ "$count_in" -gt 0 ]; then
    echo "[$(ts)] refining pending_videos=$count_in on GPU=$TARGET_GPU" >> "$LOG"
    cd "$SEEDVR_ROOT" && \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29681 + RANDOM % 100)) \
      RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
      PYTHONPATH="$SEEDVR_ROOT" PYTHONNOUSERSITE=1 \
      CUDA_VISIBLE_DEVICES=$TARGET_GPU \
      ${SEEDVR_PYTHON:-python} "$SEEDVR_ROOT/projects/inference_seedvr2_3b.py" \
        --video_path "$SEEDVR_IN" \
        --output_dir "$SEEDVR_OUT" \
        --seed 42 \
        --res_h 480 \
        --res_w 832 \
        --sp_size 1 \
        --out_fps 24 >> "$LOG" 2>&1 || true

    python3 - "$SEEDVR_IN" "$SEEDVR_OUT" <<'PY' >> "$LOG" 2>&1
import sys
from pathlib import Path
inp = Path(sys.argv[1])
out = Path(sys.argv[2])
removed = 0
for src in list(inp.glob('*.mp4')):
    if (out / src.name).exists():
        src.unlink()
        removed += 1
print('removed_inputs', removed)
PY
  fi

  raw_count=$(find "$RAW_DIR" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')
  out_count=$(find "$SEEDVR_OUT" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')
  echo "[$(ts)] raw_count=$raw_count seedvr_count=$out_count" >> "$LOG"
  if [ "$raw_count" -ge "$TARGET" ] && [ "$out_count" -ge "$TARGET" ]; then
    echo "[$(ts)] seedvr_done target=$TARGET" >> "$LOG"
    break
  fi
  sleep 300
done
