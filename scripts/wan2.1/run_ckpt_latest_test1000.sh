#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="/data/alice/cjtest/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora"
LATEST=$(ls "$CKPT_DIR"/checkpoint-*.safetensors 2>/dev/null | grep -v 'compatible_with_comfyui' | sort -V | tail -n 1)
BASE_MODEL="/data/alice/cjtest/model_repros/ABot-PhysWorld/models/Wan-AI/Wan2.1-I2V-14B-480P"
OUT_ROOT="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000"
LOG_ROOT="/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_logs"
MANIFEST_ROOT="/data/alice/cjtest/VideoX-Fun/test_dataset/manifests_1000"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
if [ -z "$LATEST" ]; then
  echo "No checkpoint found" >&2
  exit 1
fi
echo "Using checkpoint: $LATEST"
