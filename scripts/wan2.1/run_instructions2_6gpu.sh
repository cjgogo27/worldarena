#!/usr/bin/env bash
# Launch each GPU inference in its own session (setsid) so the bash tool timeout cannot kill them
set -euo pipefail

CKPT_PATH="/data/alice/cjtest/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora/checkpoint-2200.safetensors"
BASE_MODEL="/data/alice/cjtest/model_repros/ABot-PhysWorld/models/Wan-AI/Wan2.1-I2V-14B-480P"
PREDICT_SCRIPT="/data/alice/cjtest/VideoX-Fun/scripts/wan2.1/batch_predict_i2v_worldarena.py"
MANIFEST_DIR="/data/alice/cjtest/VideoX-Fun/test_dataset/manifests_instructions_2_6parts"
OUT_DIR="/data/alice/cjtest/VideoX-Fun/eval_instr_instructions2_test1000_raw"
LOG_DIR="/data/alice/cjtest/VideoX-Fun/infer_instructions_logs"
PYTHON="/data/envs/videox_fun_wan/bin/python"
GPU_LIST="2 3 4 5 6 7"

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Launching instructions_2 inference on GPUs $GPU_LIST"

idx=0
for gpu in $GPU_LIST; do
    manifest="$MANIFEST_DIR/part_$(printf '%02d' $idx).json"
    log="$LOG_DIR/instr_instructions2_gpu${gpu}.log"
    echo "Starting GPU $gpu (manifest part_$(printf '%02d' $idx)) -> log: $log"

    setsid -f env CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 \
        $PYTHON "$PREDICT_SCRIPT" \
        --manifest "$manifest" \
        --base-model "$BASE_MODEL" \
        --lora-path "$CKPT_PATH" \
        --output-dir "$OUT_DIR" \
        --height 480 --width 832 \
        --video-length 121 --fps 24 \
        --steps 50 \
        --guidance-scale 6.0 \
        --seed 43 \
        --lora-weight 0.55 \
        --gpu-memory-mode auto \
        --shift 3.0 \
        --enable-teacache \
        --teacache-threshold 0.20 \
        >> "$log" 2>&1

    idx=$((idx + 1))
done

echo ""
echo "All processes launched with setsid."
echo "Monitor: tail -f $LOG_DIR/instr_instructions2_gpu*.log"
echo "Videos: $OUT_DIR"
echo "Check running: ps aux | grep batch_predict | grep -v grep"
