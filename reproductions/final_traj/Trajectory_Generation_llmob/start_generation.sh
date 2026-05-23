#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data/alice/cjtest/FinalTraj/Trajectory_Generation_llmob"
LOG_DIR="/data/alice/cjtest/FinalTraj/logs"
DATASET="${1:-2019}"
MODE="${2:-1}"
MAX_USERS="${3:-}"
SESSION="llmob_${DATASET}_mode${MODE}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/llmob_generate_${DATASET}_mode${MODE}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${BASE_DIR}/result" "${BASE_DIR}/output_trajectories"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it before running generation." >&2
  echo "Example: export OPENAI_API_KEY='sk-...'" >&2
  exit 1
fi

cd "${BASE_DIR}"
GENERATE_CMD="python generate.py --dataset '${DATASET}' --mode '${MODE}'"
if [[ -n "${MAX_USERS}" ]]; then
  GENERATE_CMD="${GENERATE_CMD} --max_users '${MAX_USERS}'"
fi

screen -dmS "${SESSION}" bash -lc "${GENERATE_CMD} > '${LOG_FILE}' 2>&1; python evaluate.py --dataset '${DATASET}' --mode '${MODE}' >> '${LOG_FILE}' 2>&1"

echo "Started LLMob generation in screen session: ${SESSION}"
echo "Log file: ${LOG_FILE}"
echo "Attach with: screen -r ${SESSION}"
