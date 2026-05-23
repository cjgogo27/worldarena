#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data/alice/cjtest/FinalTraj/Trajectory_Generation_bhandari24"
LOG_DIR="/data/alice/cjtest/FinalTraj/logs"
LOCATION="${1:-sf}"
MODEL="${2:-gpt3}"
NUM_SAMPLES="${3:-1}"
SESSION="bhandari24_${LOCATION}_${MODEL}_${NUM_SAMPLES}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BASE_DIR}/outputs/${LOCATION}_${MODEL}_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/bhandari24_generate_${LOCATION}_${MODEL}_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${BASE_DIR}/output_trajectories"

case "${MODEL}" in
  gpt3)
    if [[ -z "${OPENAI_API_KEY:-}" && ! -f "${BASE_DIR}/openai_key_new" ]]; then
      echo "OPENAI_API_KEY is not set and openai_key_new is missing." >&2
      exit 1
    fi
    ;;
  gpt4)
    if [[ -z "${AZURE_OPENAI_API_KEY:-}" && ! -f "${BASE_DIR}/openai_key_azure" ]]; then
      echo "AZURE_OPENAI_API_KEY is not set and openai_key_azure is missing." >&2
      exit 1
    fi
    ;;
  gemini)
    if [[ -z "${GOOGLE_API_KEY:-}" && ! -f "${BASE_DIR}/palm_api_key" && ! -f "${BASE_DIR}/palm_api_key2" ]]; then
      echo "GOOGLE_API_KEY is not set and Gemini key files are missing." >&2
      exit 1
    fi
    ;;
  llama|llama2-70b|llama-2-trained)
    if [[ ! -f "${BASE_DIR}/access_token" && -z "${HF_TOKEN:-}" ]]; then
      echo "HuggingFace Llama-2 access token is required for ${MODEL}." >&2
      exit 1
    fi
    ;;
  qwen-local)
    QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/data/alice/cjtest/FinalTraj/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B}"
    if [[ ! -d "${QWEN_MODEL_PATH}" ]]; then
      echo "Qwen model path does not exist: ${QWEN_MODEL_PATH}" >&2
      exit 1
    fi
    ;;
esac

cd "${BASE_DIR}"
if [[ "${MODEL}" == "qwen-local" ]]; then
  screen -dmS "${SESSION}" bash -lc "python main.py --type completion --location '${LOCATION}' --use_model '${MODEL}' --qwen_model_path '${QWEN_MODEL_PATH}' --year 2017 --num_samples '${NUM_SAMPLES}' --out_folder '${OUT_DIR}' > '${LOG_FILE}' 2>&1"
else
  screen -dmS "${SESSION}" bash -lc "python main.py --type completion --location '${LOCATION}' --use_model '${MODEL}' --year 2017 --num_samples '${NUM_SAMPLES}' --out_folder '${OUT_DIR}' > '${LOG_FILE}' 2>&1"
fi

echo "Started Bhandari24 generation in screen session: ${SESSION}"
echo "Output directory: ${OUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Attach with: screen -r ${SESSION}"
