#!/usr/bin/env python3
"""
BAGEL 风格迁移脚本
输入：风格参考图 + 内容参考图
输出：风格化后的内容图像
无需文本提示词，自动进行风格迁移
"""

import numpy as np
import os
import torch
import random
import datetime
from pathlib import Path

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image

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

import argparse
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model


def setup_parser():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="BAGEL风格迁移脚本")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT",
                       help="模型路径")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3],
                       help="模型加载模式: 1=常规, 2=NF4量化, 3=INT8量化")
    
    # 输入图像参数
    parser.add_argument("--style_image", type=str, required=True,
                       help="风格参考图像路径")
    parser.add_argument("--content_image", type=str, required=True,
                       help="内容参考图像路径")
    
    # 输出相关参数
    parser.add_argument("--output_dir", type=str, default="/data/mayue/cjy/BAGEL/output",
                       help="图片保存目录")
    parser.add_argument("--output_name", type=str, default=None,
                       help="输出文件名前缀（不包含扩展名）")
    
    # 推理参数
    parser.add_argument("--seed", type=int, default=414307,
                       help="随机种子，0表示随机")
    parser.add_argument("--use_content_size", action="store_true",
                       help="使用内容图像的尺寸作为输出尺寸")
    
    # 高级推理参数
    parser.add_argument("--cfg_text_scale", type=float, default=4.0,
                       help="CFG文本缩放比例")
    parser.add_argument("--cfg_img_scale", type=float, default=2.0,
                       help="CFG图像缩放比例")
    parser.add_argument("--cfg_interval", type=float, default=0,
                       help="CFG应用间隔开始值")
    parser.add_argument("--timestep_shift", type=float, default=3.0,
                       help="时间步偏移")
    parser.add_argument("--num_timesteps", type=int, default=50,
                       help="去噪步数")
    parser.add_argument("--cfg_renorm_min", type=float, default=1.0,
                       help="CFG重归一化最小值")
    parser.add_argument("--cfg_renorm_type", type=str, default="global",
                       choices=["global", "local", "text_channel"],
                       help="CFG重归一化类型")
    
    # 风格迁移策略
    parser.add_argument("--style_strength", type=float, default=0.8,
                       help="风格强度 (0.0-1.0)")
    parser.add_argument("--auto_style_prompt", action="store_true",
                       help="自动生成风格描述提示词")
    parser.add_argument("--style_prompt_template", type=str, 
                       default="Apply the artistic style from this reference image to the content image",
                       help="风格迁移提示词模板")
    
    # 思考模式相关参数
    parser.add_argument("--show_thinking", action="store_true",
                       help="启用思考模式")
    parser.add_argument("--max_think_token_n", type=int, default=1024,
                       help="最大思考token数")
    parser.add_argument("--do_sample", action="store_true",
                       help="启用采样")
    parser.add_argument("--text_temperature", type=float, default=0.3,
                       help="文本生成温度")
    
    return parser


def set_seed(seed):
    """设置随机种子以保证结果可重现"""
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


def initialize_model(args):
    """初始化BAGEL模型"""
    print(f"正在加载模型: {args.model_path}")
    
    # 加载配置
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

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

    # 初始化空模型
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # 加载tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # 图像变换
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # 设备映射
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

    # 加载模型权重
    print(f"使用模式 {args.mode} 加载模型权重...")
    if args.mode == 1:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(args.model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    elif args.mode == 2:  # NF4
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_use_double_quant=False, 
            bnb_4bit_quant_type="nf4"
        )
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(args.model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    elif args.mode == 3:  # INT8
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_8bit=True, 
            torch_dtype=torch.float32
        )
        model = load_and_quantize_model(
            model, 
            weights_location=os.path.join(args.model_path, "ema.safetensors"), 
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
            offload_folder="offload",
        ).eval()
    else:
        raise NotImplementedError(f"不支持的模式: {args.mode}")

    # 初始化推理器
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    
    print("模型初始化完成!")
    return inferencer


def analyze_style_image(style_image_path, inferencer):
    """分析风格图像并生成风格描述"""
    print(f"分析风格图像: {style_image_path}")
    
    style_image = Image.open(style_image_path)
    style_image = pil_img2rgb(style_image)
    
    # 使用模型理解风格图像
    style_analysis_prompt = "Describe the artistic style, color palette, texture, and visual characteristics of this image in detail."
    
    result = inferencer(
        image=style_image, 
        text=style_analysis_prompt, 
        understanding_output=True,
        do_sample=False,
        text_temperature=0.3,
        max_think_token_n=512
    )
    
    style_description = result.get("text", "")
    print(f"风格分析: {style_description[:200]}...")
    
    return style_description


def generate_style_transfer_prompt(style_description, template):
    """基于风格描述生成风格迁移提示词"""
    if style_description:
        prompt = f"Apply the following artistic style to the content image: {style_description}. Maintain the content structure while adopting the style characteristics."
    else:
        prompt = template
    
    return prompt


def perform_style_transfer(inferencer, args):
    """执行风格迁移"""
    print("开始风格迁移...")
    print(f"风格图像: {args.style_image}")
    print(f"内容图像: {args.content_image}")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.style_image):
        raise FileNotFoundError(f"风格图像不存在: {args.style_image}")
    
    if not os.path.exists(args.content_image):
        raise FileNotFoundError(f"内容图像不存在: {args.content_image}")
    
    # 加载图像
    style_image = Image.open(args.style_image)
    content_image = Image.open(args.content_image)
    
    style_image = pil_img2rgb(style_image)
    content_image = pil_img2rgb(content_image)
    
    # 设置随机种子
    if args.seed > 0:
        set_seed(args.seed)
        print(f"使用种子: {args.seed}")
    else:
        seed = random.randint(1, 1000000)
        set_seed(seed)
        print(f"随机种子: {seed}")
    
    # 生成风格描述
    style_prompt = args.style_prompt_template
    if args.auto_style_prompt:
        try:
            style_description = analyze_style_image(args.style_image, inferencer)
            style_prompt = generate_style_transfer_prompt(style_description, args.style_prompt_template)
        except Exception as e:
            print(f"自动风格分析失败，使用默认模板: {e}")
            style_prompt = args.style_prompt_template
    
    print(f"风格迁移提示词: {style_prompt}")
    
    # 方法1: 使用风格图像+内容图像+描述性提示词
    # 构建输入序列：风格图像 -> 内容图像 -> 提示词
    input_list = [style_image, content_image, style_prompt]
    
    # 设置推理参数
    inference_hyper = dict(
        max_think_token_n=args.max_think_token_n if args.show_thinking else 1024,
        do_sample=args.do_sample if args.show_thinking else False,
        text_temperature=args.text_temperature if args.show_thinking else 0.3,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[args.cfg_interval, 1.0],
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
        image_shapes=content_image.size[::-1] if args.use_content_size else (1024, 1024),
    )
    
    # 执行推理
    print("正在生成风格迁移图像...")
    output_list = inferencer.interleave_inference(input_list, think=args.show_thinking, **inference_hyper)
    
    # 提取结果
    result_image = None
    thinking_text = None
    
    for output in output_list:
        if isinstance(output, Image.Image):
            result_image = output
        elif isinstance(output, str):
            thinking_text = output
    
    return result_image, thinking_text, style_prompt


def save_style_transfer_result(image, style_image_path, content_image_path, output_dir, 
                             output_name=None, style_prompt=None, thinking_text=None):
    """保存风格迁移结果和相关信息"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    if output_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"style_transfer_{timestamp}"
    
    # 保存图像
    image_path = os.path.join(output_dir, f"{output_name}.png")
    image.save(image_path, "PNG")
    print(f"风格迁移图像已保存到: {image_path}")
    
    # 保存详细信息
    info_path = os.path.join(output_dir, f"{output_name}.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"风格迁移信息\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"风格图像: {style_image_path}\n")
        f.write(f"内容图像: {content_image_path}\n")
        f.write(f"风格提示词: {style_prompt}\n")
        if thinking_text:
            f.write(f"\n思考过程:\n{thinking_text}\n")
    
    print(f"信息已保存到: {info_path}")
    return image_path, info_path


def main():
    """主函数"""
    parser = setup_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("BAGEL 风格迁移脚本")
    print("=" * 60)
    
    try:
        # 初始化模型
        inferencer = initialize_model(args)
        
        # 执行风格迁移
        result_image, thinking_text, style_prompt = perform_style_transfer(inferencer, args)
        
        if result_image is None:
            raise RuntimeError("风格迁移失败，未生成图像")
        
        # 保存结果
        image_path, info_path = save_style_transfer_result(
            result_image,
            args.style_image,
            args.content_image,
            args.output_dir,
            args.output_name,
            style_prompt,
            thinking_text
        )
        
        print("\n" + "=" * 60)
        print("风格迁移完成!")
        print(f"风格图像: {args.style_image}")
        print(f"内容图像: {args.content_image}")
        print(f"输出图像: {image_path}")
        print(f"信息文件: {info_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())