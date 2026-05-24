#!/usr/bin/env bash
# Launcher: start SeedVR for instructions_1 raw videos
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="/data/alice/cjtest/VideoX-Fun"

setsid -f bash "$SCRIPT_DIR/run_seedvr.sh" \
  "$ROOT/eval_instr_instructions1_test1000_raw" \
  "$ROOT/eval_ckpt_instructions1_seedvr_in" \
  "$ROOT/eval_ckpt_instructions1_seedvr" \
  "$ROOT/eval_ckpt_instructions1_seedvr.log" \
  1000 2
echo "SeedVR launched for instructions_1"
echo "Monitor: tail -f $ROOT/eval_ckpt_instructions1_seedvr.log"
