#!/usr/bin/env python3
"""
批量推理脚本：使用训练好的 LoRA checkpoint 生成 test_dataset 的 1000 个视频
用于验证 WorldArena Wan2.1 I2V LoRA 模型的泛化能力
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from PIL import Image

# Add project roots to path
current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path))
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                               WanTransformer3DModel)
from videox_fun.pipeline import WanFunInpaintPipeline
from videox_fun.utils import register_auto_device_hook, safe_enable_group_offload
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import save_videos_grid


class TestDatasetInferencer:
    def __init__(
        self,
        model_path="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP",
        lora_ckpt_path=None,
        config_path="config/wan2.1/wan_civitai.yaml",
        output_dir="test_dataset_inference_output",
        gpu_memory_mode="sequential_cpu_offload",
        device_id=0,
    ):
        """初始化推理器"""
        self.model_path = model_path
        self.lora_ckpt_path = lora_ckpt_path
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_memory_mode = gpu_memory_mode
        self.device_id = device_id
        
        # 设备
        torch.cuda.set_device(device_id)
        self.device = torch.device(f"cuda:{device_id}")
        
        print(f"🔧 初始化 I2V 推理器...")
        print(f"   模型路径: {model_path}")
        print(f"   LoRA checkpoint: {lora_ckpt_path}")
        print(f"   GPU: cuda:{device_id}")
        
        # 加载配置
        self.config = OmegaConf.load(config_path)
        
        # 加载模型
        self._load_models()
        
        print(f"✓ 推理器初始化完成")

    def _load_models(self):
        """加载所有必要的模型组件"""
        print("  加载模型组件...")
        
        # Tokenizer
        tokenizer_path = os.path.join(self.model_path, 
                                     self.config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Text Encoder
        text_encoder_path = os.path.join(self.model_path,
                                        self.config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder'))
        self.text_encoder = WanT5EncoderModel.from_pretrained(
            text_encoder_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # VAE
        vae_path = os.path.join(self.model_path,
                               self.config['vae_kwargs'].get('vae_subpath', 'vae'))
        self.vae = AutoencoderKLWan.from_pretrained(
            vae_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # Image Encoder (CLIP)
        image_encoder_path = os.path.join(self.model_path,
                                         self.config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder'))
        self.image_encoder = CLIPModel.from_pretrained(
            image_encoder_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # Transformer (DIT)
        config_file = os.path.join(self.model_path, 'transformer_config.json')
        transformer_config = OmegaConf.load(config_file)
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # 应用 LoRA 如果提供
        if self.lora_ckpt_path and os.path.exists(self.lora_ckpt_path):
            print(f"  应用 LoRA checkpoint: {self.lora_ckpt_path}")
            merge_lora(self.transformer, self.lora_ckpt_path)
        
        # 创建 Pipeline
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.pipeline = WanFunInpaintPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.scheduler,
            image_encoder=self.image_encoder,
        ).to(self.device)
        
        print(f"  ✓ 所有模型加载完成")

    def infer_single(self, image_path, prompt, output_video_path, **kwargs):
        """推理单个视频"""
        try:
            # 加载输入图像
            image = Image.open(image_path).convert("RGB")
            
            # 生成视频
            with torch.no_grad():
                video = self.pipeline(
                    prompt=prompt,
                    image=image,
                    height=self.config.get('height', 480),
                    width=self.config.get('width', 832),
                    num_frames=81,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    use_dynamic_cfg=False,
                ).videos
            
            # 保存视频
            save_videos_grid(video, str(output_video_path), fps=30)
            return True
            
        except Exception as e:
            print(f"  ✗ 推理失败: {e}")
            return False

    def batch_infer(self, validation_config_path, max_samples=None):
        """批量推理"""
        print(f"\n🎥 开始批量推理...")
        
        # 加载验证配置
        with open(validation_config_path) as f:
            validation_examples = json.load(f)
        
        if max_samples:
            validation_examples = validation_examples[:max_samples]
        
        print(f"  总推理任务: {len(validation_examples)}")
        
        # 统计结果
        results = []
        success_count = 0
        fail_count = 0
        
        # 批量推理
        for example in tqdm(validation_examples, desc="推理进度"):
            episode_id = example['episode']
            image_path = Path('/data/alice/cjtest/VideoX-Fun/test_dataset') / example['image']
            prompt = example['prompt']
            output_video = self.output_dir / f"episode_{episode_id:04d}.mp4"
            
            success = self.infer_single(str(image_path), prompt, str(output_video))
            
            result = {
                "episode_id": episode_id,
                "image": str(image_path),
                "prompt": prompt,
                "output_video": str(output_video),
                "status": "success" if success else "failed"
            }
            results.append(result)
            
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        # 保存推理结果
        results_file = self.output_dir / "inference_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 打印统计信息
        print(f"\n✓ 推理完成")
        print(f"  成功: {success_count}/{len(validation_examples)}")
        print(f"  失败: {fail_count}/{len(validation_examples)}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  结果文件: {results_file}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="批量推理 test_dataset")
    parser.add_argument("--lora_ckpt", type=str, 
                       default="/data/alice/cjtest/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora/checkpoint-1700/checkpoint-1700.safetensors",
                       help="LoRA checkpoint 路径")
    parser.add_argument("--output_dir", type=str,
                       default="/data/alice/cjtest/VideoX-Fun/test_dataset_inference_1000",
                       help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大推理样本数（用于测试）")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID")
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = TestDatasetInferencer(
        model_path="/data/alice/cjtest/model_repros/ABot-PhysWorld/models/Wan-AI/Wan2.1-I2V-14B-480P",
        lora_ckpt_path=args.lora_ckpt,
        output_dir=args.output_dir,
        device_id=args.gpu_id,
    )
    
    # 批量推理
    results = inferencer.batch_infer(
        validation_config_path="/data/alice/cjtest/VideoX-Fun/test_dataset/validation/validation_config.json",
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
