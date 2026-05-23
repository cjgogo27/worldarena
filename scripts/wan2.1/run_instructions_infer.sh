#!/usr/bin/env bash
# Run inference for instructions and instructions_1 (each 1000 videos)
# Uses the same SFT LoRA checkpoint (ckpt-2200) as the original instructions_2 inference
set -euo pipefail

CKPT_PATH="/data/alice/cjtest/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora/checkpoint-2200.safetensors"
BASE_MODEL="/data/alice/cjtest/model_repros/ABot-PhysWorld/models/Wan-AI/Wan2.1-I2V-14B-480P"
PREDICT_SCRIPT="/data/alice/cjtest/VideoX-Fun/scripts/wan2.1/batch_predict_i2v_worldarena.py"
MANIFEST_BASE="/data/alice/cjtest/VideoX-Fun/test_dataset"
OUT_BASE="/data/alice/cjtest/VideoX-Fun"
LOG_DIR="$OUT_BASE/infer_instructions_logs"
GPU_MEM_MODE="auto"

mkdir -p "$LOG_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Inferences configs:
#   instructions  → 4 parts × 250, run on GPU 0-3
#   instructions_1 → 4 parts × 250, run on GPU 4-7
# Each part runs in background, we wait for all 4 to complete per instruction set.

run_part() {
    local gpu=$1 manifest=$2 out_dir=$3 label=$4
    local log="$LOG_DIR/${label}_gpu${gpu}.log"
    echo "[$(ts)] Starting $label on GPU $gpu" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 \
        /data/envs/videox_fun_wan/bin/python "$PREDICT_SCRIPT" \
        --manifest "$manifest" \
        --base-model "$BASE_MODEL" \
        --lora-path "$CKPT_PATH" \
        --output-dir "$out_dir" \
        --height 480 --width 832 \
        --video-length 121 --fps 24 \
        --steps 50 \
        --guidance-scale 6.0 \
        --seed 43 \
        --lora-weight 0.55 \
        --gpu-memory-mode "$GPU_MEM_MODE" \
        --shift 3.0 \
        --enable-teacache \
        --teacache-threshold 0.20 \
        2>&1 | tee -a "$log"
    echo "[$(ts)] Done $label on GPU $gpu (exit=$?)" | tee -a "$log"
}

# ===== instruction set: instructions (old) =====
echo "========================================"
echo "[$(ts)] Launching inference: instructions (1000 videos across GPU 0-3)"
echo "========================================"

PID_FILE=""
for i in 0 1 2 3; do
    part_file="$MANIFEST_BASE/manifests_instructions/part_$(printf '%02d' $i).json"
    out_dir="$OUT_BASE/eval_instr_instructions_test1000_raw"
    mkdir -p "$out_dir"
    run_part $i "$part_file" "$out_dir" "instr_instructions_part${i}" &
    PID[$i]=$!
done

# Wait for all 4 parts to complete
for i in 0 1 2 3; do
    wait ${PID[$i]}
    echo "[$(ts)] instructions part_0${i} finished"
done

echo "[$(ts)] === instructions inference complete ==="

# ===== instruction set: instructions_1 =====
echo "========================================"
echo "[$(ts)] Launching inference: instructions_1 (1000 videos across GPU 4-7)"
echo "========================================"

for i in 0 1 2 3; do
    gpu=$((i + 4))
    part_file="$MANIFEST_BASE/manifests_instructions_1/part_$(printf '%02d' $i).json"
    out_dir="$OUT_BASE/eval_instr_instructions1_test1000_raw"
    mkdir -p "$out_dir"
    run_part $gpu "$part_file" "$out_dir" "instr_instructions1_part${i}" &
    PID[$i]=$!
done

for i in 0 1 2 3; do
    wait ${PID[$i]}
    echo "[$(ts)] instructions_1 part_0${i} finished"
done

echo "[$(ts)] === instructions_1 inference complete ==="
echo "[$(ts)] === ALL DONE ==="
echo "[$(ts)] Output dirs:"
echo "  instructions:  $OUT_BASE/eval_instr_instructions_test1000_raw/"
echo "  instructions_1: $OUT_BASE/eval_instr_instructions1_test1000_raw/"
