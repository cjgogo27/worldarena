#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bagel模型命令行界面工具

这个脚本提供了一个命令行界面，让用户能够使用Bagel模型的三种主要功能：
1. 文生图(Text to Image)
2. 图像编辑(Image Edit)
3. 图像理解(Image Understanding)
并将结果保存到本地文件系统。
"""

import os
import argparse
import torch
import random
import numpy as np
from PIL import Image
from datetime import datetime

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer


def set_seed(seed):
    """设置随机种子以确保结果可重现"""
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


def init_model(model_path, mode=1):
    """初始化模型"""
    print(f"正在加载模型: {model_path}")
    
    # 加载配置
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

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

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # 模型加载和多GPU推理准备
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

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
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    elif mode == 2:  # NF4
        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4")
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif mode == 3:  # INT8
        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, torch_dtype=torch.float32)
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    else:
        raise NotImplementedError(f"不支持的模型加载模式: {mode}")

    # 初始化推理器
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    
    print("模型加载完成")
    return inferencer


def get_image_ratio_shape(ratio):
    """根据比例获取图像尺寸"""
    if ratio == "1:1":
        return (1024, 1024)
    elif ratio == "4:3":
        return (768, 1024)
    elif ratio == "3:4":
        return (1024, 768)
    elif ratio == "16:9":
        return (576, 1024)
    elif ratio == "9:16":
        return (1024, 576)
    else:
        raise ValueError(f"不支持的图像比例: {ratio}")


def text_to_image(inferencer, prompt, output_dir, seed=0, image_ratio="1:1", **kwargs):
    """文生图功能"""
    print(f"正在生成图像: {prompt}")
    
    # 设置随机种子
    set_seed(seed)
    
    # 获取图像尺寸
    image_shapes = get_image_ratio_shape(image_ratio)
    
    # 设置超参数
    inference_hyper = dict(
        max_think_token_n=kwargs.get("max_think_token_n", 1024),
        do_sample=kwargs.get("do_sample", False),
        text_temperature=kwargs.get("text_temperature", 0.3),
        cfg_text_scale=kwargs.get("cfg_text_scale", 4.0),
        cfg_interval=kwargs.get("cfg_interval", [0.4, 1.0]),
        timestep_shift=kwargs.get("timestep_shift", 3.0),
        num_timesteps=kwargs.get("num_timesteps", 50),
        cfg_renorm_min=kwargs.get("cfg_renorm_min", 0.0),
        cfg_renorm_type=kwargs.get("cfg_renorm_type", "global"),
        image_shapes=image_shapes,
    )
    
    # 调用推理器
    result = inferencer(text=prompt, think=kwargs.get("show_thinking", False), **inference_hyper)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图像
    if result["image"] is not None:
        img_path = os.path.join(output_dir, f"text2img_{timestamp}.png")
        result["image"].save(img_path)
        print(f"图像已保存至: {img_path}")
    
    # 保存思考文本（如果有）
    if result.get("text"):
        text_path = os.path.join(output_dir, f"text2img_{timestamp}_thinking.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"思考过程已保存至: {text_path}")
    
    return result


def edit_image(inferencer, image_path, prompt, output_dir, seed=0, **kwargs):
    """图像编辑功能"""
    print(f"正在编辑图像: {image_path}，提示词: {prompt}")
    
    # 设置随机种子
    set_seed(seed)
    
    # 加载图像
    try:
        image = Image.open(image_path)
        image = pil_img2rgb(image)
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None
    
    # 设置超参数
    inference_hyper = dict(
        max_think_token_n=kwargs.get("max_think_token_n", 1024),
        do_sample=kwargs.get("do_sample", False),
        text_temperature=kwargs.get("text_temperature", 0.3),
        cfg_text_scale=kwargs.get("cfg_text_scale", 4.0),
        cfg_img_scale=kwargs.get("cfg_img_scale", 2.0),
        cfg_interval=kwargs.get("cfg_interval", [0.0, 1.0]),
        timestep_shift=kwargs.get("timestep_shift", 3.0),
        num_timesteps=kwargs.get("num_timesteps", 50),
        cfg_renorm_min=kwargs.get("cfg_renorm_min", 0.0),
        cfg_renorm_type=kwargs.get("cfg_renorm_type", "text_channel"),
    )
    
    # 调用推理器
    result = inferencer(image=image, text=prompt, think=kwargs.get("show_thinking", False), **inference_hyper)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存编辑后的图像
    if result["image"] is not None:
        img_path = os.path.join(output_dir, f"image_edit_{timestamp}.png")
        result["image"].save(img_path)
        print(f"编辑后的图像已保存至: {img_path}")
    
    # 保存思考文本（如果有）
    if result.get("text"):
        text_path = os.path.join(output_dir, f"image_edit_{timestamp}_thinking.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"思考过程已保存至: {text_path}")
    
    return result


def image_understanding(inferencer, image_path, prompt, output_dir, **kwargs):
    """图像理解功能"""
    print(f"正在理解图像: {image_path}，问题: {prompt}")
    
    # 加载图像
    try:
        image = Image.open(image_path)
        image = pil_img2rgb(image)
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None
    
    # 设置超参数
    inference_hyper = dict(
        do_sample=kwargs.get("do_sample", False),
        text_temperature=kwargs.get("text_temperature", 0.3),
        max_think_token_n=kwargs.get("max_new_tokens", 512),
    )
    
    # 调用推理器
    result = inferencer(image=image, text=prompt, think=kwargs.get("show_thinking", False), 
                       understanding_output=True, **inference_hyper)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存理解结果
    if result["text"]:
        text_path = os.path.join(output_dir, f"image_understanding_{timestamp}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"图像理解结果已保存至: {text_path}")
    
    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Bagel模型命令行工具")
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT", help="模型路径")
    parser.add_argument("--mode", type=int, default=1, help="模型加载模式 (1=正常, 2=NF4量化, 3=INT8量化)")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 文生图命令
    t2i_parser = subparsers.add_parser("text2img", help="文本生成图像")
    t2i_parser.add_argument("--prompt", type=str, required=True, help="文本提示词")
    t2i_parser.add_argument("--output_dir", type=str, default="outputs/text2img", help="输出目录")
    t2i_parser.add_argument("--seed", type=int, default=0, help="随机种子 (0=随机)")
    t2i_parser.add_argument("--image_ratio", type=str, default="1:1", choices=["1:1", "4:3", "3:4", "16:9", "9:16"], help="图像比例")
    t2i_parser.add_argument("--cfg_text_scale", type=float, default=4.0, help="文本CFG强度")
    t2i_parser.add_argument("--show_thinking", action="store_true", help="启用思考模式")
    
    # 图像编辑命令
    edit_parser = subparsers.add_parser("edit", help="图像编辑")
    edit_parser.add_argument("--image_path", type=str, required=True, help="输入图像路径")
    edit_parser.add_argument("--prompt", type=str, required=True, help="编辑提示词")
    edit_parser.add_argument("--output_dir", type=str, default="outputs/edit", help="输出目录")
    edit_parser.add_argument("--seed", type=int, default=0, help="随机种子 (0=随机)")
    edit_parser.add_argument("--cfg_text_scale", type=float, default=4.0, help="文本CFG强度")
    edit_parser.add_argument("--cfg_img_scale", type=float, default=2.0, help="图像CFG强度")
    edit_parser.add_argument("--show_thinking", action="store_true", help="启用思考模式")
    
    # 图像理解命令
    und_parser = subparsers.add_parser("understand", help="图像理解")
    und_parser.add_argument("--image_path", type=str, required=True, help="输入图像路径")
    und_parser.add_argument("--prompt", type=str, required=True, help="问题提示词")
    und_parser.add_argument("--output_dir", type=str, default="outputs/understand", help="输出目录")
    und_parser.add_argument("--show_thinking", action="store_true", help="启用思考模式")
    
    args = parser.parse_args()
    
    # 加载模型
    inferencer = init_model(args.model_path, args.mode)
    
    # 执行相应命令
    if args.command == "text2img":
        # 收集所有相关参数
        t2i_kwargs = {
            "show_thinking": args.show_thinking,
            "cfg_text_scale": args.cfg_text_scale,
        }
        text_to_image(inferencer, args.prompt, args.output_dir, args.seed, args.image_ratio, **t2i_kwargs)
        
    elif args.command == "edit":
        # 收集所有相关参数
        edit_kwargs = {
            "show_thinking": args.show_thinking,
            "cfg_text_scale": args.cfg_text_scale,
            "cfg_img_scale": args.cfg_img_scale,
        }
        edit_image(inferencer, args.image_path, args.prompt, args.output_dir, args.seed, **edit_kwargs)
        
    elif args.command == "understand":
        # 收集所有相关参数
        und_kwargs = {
            "show_thinking": args.show_thinking,
        }
        image_understanding(inferencer, args.image_path, args.prompt, args.output_dir, **und_kwargs)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()