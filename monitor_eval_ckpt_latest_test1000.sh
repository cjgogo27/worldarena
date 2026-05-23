#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_raw"
LOG_P1="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_logs/part_00.log"
LOG_P2="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_logs/part_01.log"
MON_LOG="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_monitor.log"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

while true; do
  count=$(find "$OUT_DIR" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')
  p1=$(pgrep -f 'manifests_1000_gpu67/part_00.json' || true)
  p2=$(pgrep -f 'manifests_1000_gpu67/part_01.json' || true)
  echo "[$(ts)] mp4_count=$count p1=${p1:-none} p2=${p2:-none}" >> "$MON_LOG"
  [ -f "$LOG_P1" ] && tail -n 2 "$LOG_P1" >> "$MON_LOG" || true
  [ -f "$LOG_P2" ] && tail -n 2 "$LOG_P2" >> "$MON_LOG" || true
  if [ "$count" -ge 1000 ]; then
    echo "[$(ts)] all_videos_done" >> "$MON_LOG"
    break
  fi
  sleep 300
done
