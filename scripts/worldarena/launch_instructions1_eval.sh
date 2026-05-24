#!/usr/bin/env bash
# Launcher: run eval pipeline for instructions_1 (after SeedVR completes)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="/data/alice/cjtest/VideoX-Fun"

nohup bash "$SCRIPT_DIR/run_eval_pipeline.sh" \
  instructions1 \
  "$ROOT/eval_instr_instructions1_test1000_raw" \
  "$ROOT/test_dataset/manifests_instructions_1" \
  "part_00.json part_01.json part_02.json part_03.json" \
  false &
echo "Eval pipeline launched for instructions_1 (PID=$!)"
echo "Monitor: tail -f $ROOT/eval_ckpt_instructions1_raw_seedvr_eval.log"
echo "IMPORTANT: SeedVR must have completed before this pipeline proceeds."
echo "  SeedVR output dir: $ROOT/eval_ckpt_instructions1_seedvr"
echo "  Pipeline waits for 1000 files in both raw and seedvr dirs."
