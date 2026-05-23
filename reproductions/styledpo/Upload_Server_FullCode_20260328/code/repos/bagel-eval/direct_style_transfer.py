#!/usr/bin/env python3
"""
直接使用BAGEL风格转换 - 完整版本
==================================

直接复制app.py中的模型初始化逻辑来进行风格转换
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import random
from PIL import Image

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入BAGEL模块
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer

def init_bagel_model(model_path, mode=2):
    """
    初始化BAGEL模型 - 从app.py复制的代码
    
    Args:
        model_path: 模型路径
        mode: 1=全精度, 2=NF4量化, 3=INT8量化
    
    Returns:
        (model, tokenizer, vae_model, vae_transform, vit_transform, new_token_ids, inferencer)
    """
    
    print(f"📦 初始化BAGEL模型 (模式 {mode})...")
    
    # 加载配置
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    # 加载VAE
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    # 确保VAE模型在正确的设备上
    if torch.cuda.is_available():
        vae_model = vae_model.cuda()

    # 创建BAGEL配置
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    # 加载tokenizer  
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # 初始化空模型
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # 图像变换
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # 设备映射
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    # 同设备模块
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # 根据模式加载模型
    if mode == 1:
        print("   使用全精度模式...")
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    elif mode == 2:
        print("   使用NF4量化模式...")
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=False, 
            bnb_4bit_quant_type="nf4"
        )
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif mode == 3:
        print("   使用INT8量化模式...")
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=True, 
            torch_dtype=torch.float32
        )
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    else:
        raise ValueError(f"不支持的模式: {mode}")

    # 创建推理器
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    print("✅ BAGEL模型初始化完成")
    
    return model, tokenizer, vae_model, vae_transform, vit_transform, new_token_ids, inferencer

def set_seed(seed):
    """设置随机种子"""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

def style_transfer(
    input_image_path,
    style_prompt,
    output_path,
    model_path="models/BAGEL-7B-MoT",
    mode=2,
    cfg_text_scale=4.0,
    cfg_img_scale=1.5,
    num_timesteps=50,
    seed=42
):
    """
    执行风格转换
    
    Args:
        input_image_path: 输入图像路径
        style_prompt: 风格提示词
        output_path: 输出路径
        model_path: 模型路径
        mode: 模型模式
        cfg_text_scale: 文本引导强度
        cfg_img_scale: 图像保真度
        num_timesteps: 推理步数
        seed: 随机种子
    """
    
    print("🎨 BAGEL风格转换")
    print("=" * 50)
    print(f"输入图像: {input_image_path}")
    print(f"风格提示: {style_prompt}")
    print(f"输出路径: {output_path}")
    print(f"模型路径: {model_path}")
    print(f"模型模式: {mode}")
    print("=" * 50)
    
    # 检查文件
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"输入图像不存在: {input_image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    # 设置随机种子
    set_seed(seed)
    print(f"🎲 设置随机种子: {seed}")
    
    # 初始化模型
    try:
        model, tokenizer, vae_model, vae_transform, vit_transform, new_token_ids, inferencer = init_bagel_model(
            model_path, mode
        )
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        raise
    
    # 加载输入图像
    print("🖼️ 加载输入图像...")
    try:
        input_image = Image.open(input_image_path).convert('RGB')
        print(f"   图像尺寸: {input_image.size}")
    except Exception as e:
        print(f"❌ 图像加载失败: {e}")
        raise
    
    # 执行风格转换
    print("🚀 执行风格转换...")
    print(f"   文本引导: {cfg_text_scale}")
    print(f"   图像保真: {cfg_img_scale}")
    print(f"   推理步数: {num_timesteps}")
    
    try:
        # 使用推理器进行风格转换
        result = inferencer(
            image=input_image,
            text=style_prompt,
            understanding_output=False,    # 图像生成模式
            cfg_text_scale=cfg_text_scale, # 文本引导强度
            cfg_img_scale=cfg_img_scale,   # 图像保真度
            num_timesteps=num_timesteps,   # 推理步数
            think=False,                   # 不启用思考模式
            image_shapes=(1024, 1024)      # 输出图像尺寸
        )
        
        # 检查结果
        if result['image'] is not None:
            output_image = result['image']
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存图像
            output_image.save(output_path, quality=95)
            
            print("✅ 风格转换成功!")
            print(f"   输出: {output_path}")
            print(f"   尺寸: {output_image.size}")
            print(f"   大小: {os.path.getsize(output_path) / 1024:.1f} KB")
            
            return True
        else:
            print("❌ 未能生成图像")
            return False
            
    except Exception as e:
        print(f"❌ 风格转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BAGEL风格转换工具")
    parser.add_argument("--input_image", "-i", required=True, help="输入图像路径")
    parser.add_argument("--style_prompt", "-s", required=True, help="风格描述提示词")
    parser.add_argument("--output", "-o", default="style_output.png", help="输出图像路径")
    parser.add_argument("--model_path", "-m", default="models/BAGEL-7B-MoT", help="模型路径")
    parser.add_argument("--mode", type=int, default=2, choices=[1, 2, 3], help="模型模式 (1=全精度, 2=NF4, 3=INT8)")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0, help="文本引导强度")
    parser.add_argument("--cfg_img_scale", type=float, default=1.5, help="图像保真度")
    parser.add_argument("--num_timesteps", type=int, default=50, help="推理步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    try:
        success = style_transfer(
            input_image_path=args.input_image,
            style_prompt=args.style_prompt,
            output_path=args.output,
            model_path=args.model_path,
            mode=args.mode,
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=args.cfg_img_scale,
            num_timesteps=args.num_timesteps,
            seed=args.seed
        )
        
        if success:
            print("\n🎉 风格转换完成!")
            return 0
        else:
            print("\n❌ 风格转换失败!")
            return 1
            
    except Exception as e:
        print(f"\n💥 程序执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main())