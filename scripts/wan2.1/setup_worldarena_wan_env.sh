#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="/data/envs/videox_fun_wan"
LOG_FILE="/data/alice/cjtest/VideoX-Fun/setup_videox_fun_wan.log"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] setup start" | tee -a "$LOG_FILE"

if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.10 pip 2>&1 | tee -a "$LOG_FILE"
fi

PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/pip" install --upgrade pip setuptools wheel 2>&1 | tee -a "$LOG_FILE"
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/pip" install -r /data/alice/cjtest/VideoX-Fun/requirements.txt 2>&1 | tee -a "$LOG_FILE"
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/pip" install deepspeed==0.17.0 yunchang xfuser modelscope openpyxl 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] setup done" | tee -a "$LOG_FILE"
