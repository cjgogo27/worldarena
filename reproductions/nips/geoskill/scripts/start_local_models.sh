#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/alice/cjtest/NIPS/geoskill"
LOG_DIR="$ROOT/experiments/full_100/local_server_logs"
mkdir -p "$LOG_DIR"
TRANSFORMERS_BIN="/data2/miniconda3/envs/vgllm/bin/transformers"

MAIN_MODEL="/data/alice/cjtest/styledpo/Style-DPO-v2.8-fullcode-20260328_extracted/Upload_Server_FullCode_20260328/models/Qwen3.5-9B"
SMALL_MODEL="/data/alice/cjtest/styledpo/Style-DPO-v2.8-fullcode-20260328_extracted/Upload_Server_FullCode_20260328/models/Qwen3.5-9B"

# Stop stale local serving processes on the target ports.
pkill -f "transformers serve --force-model $MAIN_MODEL --port 8000" || true
pkill -f "transformers serve --force-model $SMALL_MODEL --port 8001" || true

# Main multimodal model for image geolocation.
nohup "$TRANSFORMERS_BIN" serve --force-model "$MAIN_MODEL" --port 8000 --continuous-batching \
  > "$LOG_DIR/main_qwen35_9b.log" 2>&1 &
echo $! > "$LOG_DIR/main_qwen35_9b.pid"

# Recovery model server (same model as main, separate port).
nohup "$TRANSFORMERS_BIN" serve --force-model "$SMALL_MODEL" --port 8001 --continuous-batching \
  > "$LOG_DIR/small_qwen35_9b.log" 2>&1 &
echo $! > "$LOG_DIR/small_qwen35_9b.pid"

echo "Started local model servers."
echo "  main pid=$(cat "$LOG_DIR/main_qwen35_9b.pid") port=8000"
echo "  small pid=$(cat "$LOG_DIR/small_qwen35_9b.pid") port=8001"
echo "Logs: $LOG_DIR"
