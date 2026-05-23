#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="/data/alice/cjtest/model_repros/ABot-PhysWorld/models/Wan-AI/Wan2.1-I2V-14B-480P"
export DATASET_NAME="/data/alice/cjtest/datasets/worldarena_wan_i2v_clean50_fast5"
export DATASET_META_NAME="/data/alice/cjtest/datasets/worldarena_wan_i2v_clean50_fast5/metadata.json"
export OUTPUT_DIR="/data/alice/cjtest/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora_fast5"
export LOG_DIR="$OUTPUT_DIR/logs"
export TRAIN_LOG="$OUTPUT_DIR/train.log"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
RESUME_ARG=()
if ls "$OUTPUT_DIR"/checkpoint-* >/dev/null 2>&1; then
  RESUME_ARG=(--resume_from_checkpoint latest)
fi
PYTHONNOUSERSITE=1 /data/envs/videox_fun_wan/bin/accelerate launch --num_processes=2 --mixed_precision="bf16" /data/alice/cjtest/VideoX-Fun/scripts/wan2.1/train_lora.py \
  --config_path="/data/alice/cjtest/VideoX-Fun/config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATASET_NAME" \
  --train_data_meta="$DATASET_META_NAME" \
  --image_sample_size=640 \
  --video_sample_size=640 \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=1 \
  --checkpointing_steps=50 \
  --validation_steps=50 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=20 \
  --seed=42 \
  --output_dir="$OUTPUT_DIR" \
  --logging_dir="logs" \
  --report_to="tensorboard" \
  --tracker_project_name="worldarena_wan_i2v_lora_fast5" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --rank=16 \
  --network_alpha=8 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --low_vram \
  --train_mode="i2v" \
  --validation_paths "/data/alice/cjtest/datasets/worldarena_wan_i2v_clean50_fast5/validation_images/val_000.png" \
  --validation_prompts "In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. The arm maintains high stability with no deformation and enters the frame to Take the hammer for nails and smash the block." \
  "${RESUME_ARG[@]}" 2>&1 | tee -a "$TRAIN_LOG"
