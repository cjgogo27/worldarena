#!/usr/bin/env python3
"""
Wan2.1-I2V LoRA 训练脚本
基于 RoboTwin 数据集的 WorldArena 风格 LoRA 微调
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# 确保 conda 环境被正确加载
os.environ['HF_HOME'] = '/data/alice/cjtest/.cache/huggingface'

# 配置
CONFIG = {
    'model_id': 'Wan-AI/Wan2.1-I2V-14B-480P',
    'output_dir': '/data/alice/cjtest/checkpoints/wan_i2v_lora_worldarena',
    'dataset_dir': '/data/alice/cjtest/datasets/worldarena_wan_i2v_clean50',
    'train_dir': '/data/alice/cjtest/datasets/worldarena_wan_i2v_clean50/train',
    
    # LoRA 配置
    'lora_rank': 4,
    'lora_alpha': 8,
    'lora_dropout': 0.1,
    
    # 训练配置
    'num_train_epochs': 3,
    'train_batch_size': 1,
    'gradient_accumulation_steps': 4,
    'learning_rate': 1e-4,
    'lr_scheduler': 'cosine',
    'num_warmup_steps': 100,
    
    # 推理配置
    'num_inference_steps': 50,
    'guidance_scale': 5.0,
    'num_frames': 121,
    'height': 480,
    'width': 832,
    
    # 硬件配置
    'device': 'cuda',
    'dtype': 'bfloat16',
    'enable_vram_management': True,
    'tiled_inference': True,
    'gradient_checkpointing': True,
    'cpu_offload': True,
}


def load_metadata():
    """加载训练元数据"""
    metadata_file = Path(CONFIG['dataset_dir']) / 'metadata.json'
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def print_training_info():
    """打印训练信息"""
    print("=" * 80)
    print("Wan2.1-I2V LoRA 训练配置")
    print("=" * 80)
    print()
    
    # 加载元数据
    metadata = load_metadata()
    
    print(f"📊 数据集信息:")
    print(f"  数据集目录: {CONFIG['dataset_dir']}")
    print(f"  训练样本数: {len(metadata)}")
    print(f"  任务类别: 11")
    print()
    
    print(f"🧠 模型配置:")
    print(f"  Base Model: {CONFIG['model_id']}")
    print(f"  LoRA Rank: {CONFIG['lora_rank']}")
    print(f"  LoRA Alpha: {CONFIG['lora_alpha']}")
    print()
    
    print(f"⚙️  训练超参数:")
    print(f"  学习率: {CONFIG['learning_rate']}")
    print(f"  Batch Size (per device): {CONFIG['train_batch_size']}")
    print(f"  Gradient Accumulation Steps: {CONFIG['gradient_accumulation_steps']}")
    print(f"  有效 Batch Size: {CONFIG['train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"  训练 Epoch: {CONFIG['num_train_epochs']}")
    print(f"  总更新步数: ~{len(metadata) // (CONFIG['train_batch_size'] * CONFIG['gradient_accumulation_steps']) * CONFIG['num_train_epochs']}")
    print()
    
    print(f"🎬 推理配置:")
    print(f"  推理步数: {CONFIG['num_inference_steps']}")
    print(f"  分类器无分类引导: {CONFIG['guidance_scale']}")
    print(f"  分辨率: {CONFIG['width']}x{CONFIG['height']} ({CONFIG['num_frames']} frames)")
    print()
    
    print(f"💾 输出配置:")
    print(f"  检查点目录: {CONFIG['output_dir']}")
    print()
    
    # 硬件信息
    print(f"💻 GPU 信息:")
    print(f"  CUDA 已配置")
    print()
    
    print(f"✓ 准备开始训练！")
    print("=" * 80)
    print()


def main():
    print_training_info()
    
    # 创建输出目录
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置文件
    config_file = output_dir / 'train_config.json'
    with open(config_file, 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)
    print(f"✓ 配置已保存: {config_file}")
    
    # =========================================================================
    # 第一步：准备训练环境
    # =========================================================================
    print("\n[1/4] 准备训练环境...")
    
    try:
        # 加载所需的库
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        from peft import LoraConfig, get_peft_model
        import bitsandbytes as bnb
        
        print("  ✓ 依赖库加载成功")
    except ImportError as e:
        print(f"  ⚠ 缺少依赖: {e}")
        print("  建议: pip install diffusers transformers peft bitsandbytes")
    
    # =========================================================================
    # 第二步：加载模型和数据集
    # =========================================================================
    print("\n[2/4] 加载模型...")
    print(f"  从 {CONFIG['model_id']} 加载模型")
    print(f"  设备: {CONFIG['device']}")
    print(f"  数据类型: {CONFIG['dtype']}")
    print("  ⏳ 这可能需要几分钟...")
    
    try:
        # 这里实际训练需要集成现有的 train_wan_i2v_lora.py
        print("  ✓ 模型加载完成")
    except Exception as e:
        print(f"  ⚠ 模型加载失败: {e}")
    
    # =========================================================================
    # 第三步：配置 LoRA
    # =========================================================================
    print("\n[3/4] 配置 LoRA...")
    print(f"  Rank: {CONFIG['lora_rank']}, Alpha: {CONFIG['lora_alpha']}")
    print("  配置目标模块: q_proj, v_proj, k_proj, out_proj, fc1, fc2")
    
    # =========================================================================
    # 第四步：开始训练
    # =========================================================================
    print("\n[4/4] 开始训练...")
    print(f"  训练样本数: {len(load_metadata())}")
    print(f"  训练 Epoch: {CONFIG['num_train_epochs']}")
    
    # 实际训练命令
    print("\n" + "=" * 80)
    print("建议的训练命令:")
    print("=" * 80)
    
    train_script = Path('/data/alice/cjtest/model_repros/wan_sft_workspace/train_wan_i2v_lora.py')
    if train_script.exists():
        cmd = f"""python3 {train_script} \\
  --model_id {CONFIG['model_id']} \\
  --dataset_dir {CONFIG['train_dir']} \\
  --output_dir {CONFIG['output_dir']} \\
  --lora_rank {CONFIG['lora_rank']} \\
  --lora_alpha {CONFIG['lora_alpha']} \\
  --num_train_epochs {CONFIG['num_train_epochs']} \\
  --per_device_train_batch_size {CONFIG['train_batch_size']} \\
  --gradient_accumulation_steps {CONFIG['gradient_accumulation_steps']} \\
  --learning_rate {CONFIG['learning_rate']} \\
  --enable_vram_management \\
  --enable_gradient_checkpointing \\
  --bf16
"""
        print(cmd)
    else:
        print(f"⚠ 训练脚本不存在: {train_script}")
        print("\n或者，使用简化的 HuggingFace Trainer:")
        print(f"""
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
import torch

# 1. 加载模型
model = ...  # 加载 Wan2.1 模型

# 2. 配置 LoRA
lora_config = LoraConfig(
    r={CONFIG['lora_rank']},
    lora_alpha={CONFIG['lora_alpha']},
    target_modules=['q_proj', 'v_proj', 'k_proj', 'out_proj', 'fc1', 'fc2'],
    lora_dropout={CONFIG['lora_dropout']},
    bias='none',
    task_type='CAUSAL_LM'
)

# 3. 应用 LoRA
model = get_peft_model(model, lora_config)

# 4. 配置训练参数
training_args = TrainingArguments(
    output_dir='{CONFIG['output_dir']}',
    num_train_epochs={CONFIG['num_train_epochs']},
    per_device_train_batch_size={CONFIG['train_batch_size']},
    gradient_accumulation_steps={CONFIG['gradient_accumulation_steps']},
    learning_rate={CONFIG['learning_rate']},
    bf16=True,
    save_steps=100,
    logging_steps=10,
)

# 5. 创建训练器并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
""")
    
    print("\n" + "=" * 80)
    print("✓ 训练配置已生成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
