#!/usr/bin/env bash
# Launcher: run eval pipeline for instructions_2 (after SeedVR completes)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="/data/alice/cjtest/VideoX-Fun"

nohup bash "$SCRIPT_DIR/run_eval_pipeline.sh" \
  instructions2 \
  "$ROOT/eval_instr_instructions2_test1000_raw" \
  "$ROOT/test_dataset/manifests_instructions_2_8parts" \
  "part_00.json part_01.json part_02.json part_03.json part_04.json part_05.json part_06.json part_07.json" \
  false &
echo "Eval pipeline launched for instructions_2 (PID=$!)"
echo "Monitor: tail -f $ROOT/eval_ckpt_instructions2_raw_seedvr_eval.log"
echo "IMPORTANT: SeedVR must have completed before this pipeline proceeds."
echo "  SeedVR output dir: $ROOT/eval_ckpt_instructions2_seedvr"
echo "  Pipeline waits for 1000 files in both raw and seedvr dirs."
