#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="/data/envs/videox_fun_wan"
TRAIN_SCRIPT="/data/alice/cjtest/VideoX-Fun/scripts/wan2.1/train_worldarena_wan_i2v_lora.sh"
WAIT_LOG="/data/alice/cjtest/VideoX-Fun/wait_and_train_worldarena_wan_i2v_lora.log"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] wait-and-train start" | tee -a "$WAIT_LOG"

until [ -x "$ENV_PREFIX/bin/python" ]; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for env python" | tee -a "$WAIT_LOG"
  sleep 60
done

until PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" - <<'PY'
import deepspeed, yunchang, xfuser, modelscope, openpyxl, omegaconf, datasets
print('env-ready')
PY
do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for package readiness" | tee -a "$WAIT_LOG"
  sleep 120
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching training" | tee -a "$WAIT_LOG"
exec bash "$TRAIN_SCRIPT"
