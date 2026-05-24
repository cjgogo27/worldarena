#!/usr/bin/env bash
# Launcher: start SeedVR for instructions_2 raw videos
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="/data/alice/cjtest/VideoX-Fun"

setsid -f bash "$SCRIPT_DIR/run_seedvr.sh" \
  "$ROOT/eval_instr_instructions2_test1000_raw" \
  "$ROOT/eval_ckpt_instructions2_seedvr_in" \
  "$ROOT/eval_ckpt_instructions2_seedvr" \
  "$ROOT/eval_ckpt_instructions2_seedvr.log" \
  1000 3
echo "SeedVR launched for instructions_2"
echo "Monitor: tail -f $ROOT/eval_ckpt_instructions2_seedvr.log"
