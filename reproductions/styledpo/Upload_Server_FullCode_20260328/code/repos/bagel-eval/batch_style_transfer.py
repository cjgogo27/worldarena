#!/usr/bin/env python3
"""
BAGEL 批处理风格迁移脚本
支持文件夹对文件夹的1对1风格迁移，并自动记录实验日志
"""

import numpy as np
import os
import torch
import random
import datetime
import json
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse
from collections import defaultdict

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
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建当前实验的时间戳
        self.experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 日志文件路径
        self.csv_log_path = self.log_dir / f"experiment_log_{self.experiment_timestamp}.csv"
        self.json_log_path = self.log_dir / f"experiment_details_{self.experiment_timestamp}.json"
        self.summary_log_path = self.log_dir / f"experiment_summary_{self.experiment_timestamp}.txt"
        
        # 实验数据存储
        self.experiment_data = []
        self.summary_stats = defaultdict(list)
        
        # 初始化CSV文件
        self.init_csv_log()
        
    def init_csv_log(self):
        """初始化CSV日志文件"""
        fieldnames = [
            'experiment_id', 'timestamp', 'style_image', 'content_image', 
            'output_image', 'seed', 'cfg_text_scale', 'cfg_img_scale',
            'num_timesteps', 'timestep_shift', 'cfg_interval', 'cfg_renorm_type',
            'cfg_renorm_min', 'auto_style_prompt', 'generation_time_seconds',
            'style_description', 'thinking_process', 'image_width', 'image_height',
            'success', 'error_message'
        ]
        
        with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def log_experiment(self, experiment_data: Dict[str, Any]):
        """记录单次实验数据"""
        # 添加到内存存储
        self.experiment_data.append(experiment_data)
        
        # 写入CSV文件
        with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'experiment_id', 'timestamp', 'style_image', 'content_image', 
                'output_image', 'seed', 'cfg_text_scale', 'cfg_img_scale',
                'num_timesteps', 'timestep_shift', 'cfg_interval', 'cfg_renorm_type',
                'cfg_renorm_min', 'auto_style_prompt', 'generation_time_seconds',
                'style_description', 'thinking_process', 'image_width', 'image_height',
                'success', 'error_message'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(experiment_data)
        
        # 更新统计信息
        self.update_stats(experiment_data)
    
    def update_stats(self, data: Dict[str, Any]):
        """更新统计信息"""
        if data['success']:
            self.summary_stats['successful_generations'].append(data['experiment_id'])
            self.summary_stats['generation_times'].append(data['generation_time_seconds'])
        else:
            self.summary_stats['failed_generations'].append(data['experiment_id'])
    
    def save_detailed_log(self):
        """保存详细的JSON日志"""
        detailed_data = {
            'experiment_metadata': {
                'timestamp': self.experiment_timestamp,
                'total_experiments': len(self.experiment_data),
                'successful': len(self.summary_stats['successful_generations']),
                'failed': len(self.summary_stats['failed_generations']),
            },
            'experiments': self.experiment_data,
            'summary_statistics': {
                'average_generation_time': np.mean(self.summary_stats['generation_times']) if self.summary_stats['generation_times'] else 0,
                'total_generation_time': sum(self.summary_stats['generation_times']),
                'min_generation_time': min(self.summary_stats['generation_times']) if self.summary_stats['generation_times'] else 0,
                'max_generation_time': max(self.summary_stats['generation_times']) if self.summary_stats['generation_times'] else 0,
            }
        }
        
        with open(self.json_log_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
    
    def save_summary(self):
        """保存实验总结"""
        with open(self.summary_log_path, 'w', encoding='utf-8') as f:
            f.write(f"BAGEL 批处理风格迁移实验总结\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"实验时间: {self.experiment_timestamp}\n")
            f.write(f"总实验数: {len(self.experiment_data)}\n")
            f.write(f"成功数: {len(self.summary_stats['successful_generations'])}\n")
            f.write(f"失败数: {len(self.summary_stats['failed_generations'])}\n")
            
            if self.summary_stats['generation_times']:
                f.write(f"平均生成时间: {np.mean(self.summary_stats['generation_times']):.2f} 秒\n")
                f.write(f"总生成时间: {sum(self.summary_stats['generation_times']):.2f} 秒\n")
                f.write(f"最短生成时间: {min(self.summary_stats['generation_times']):.2f} 秒\n")
                f.write(f"最长生成时间: {max(self.summary_stats['generation_times']):.2f} 秒\n")
            
            f.write(f"\n实验详细信息:\n")
            f.write(f"CSV日志: {self.csv_log_path}\n")
            f.write(f"JSON日志: {self.json_log_path}\n")


def setup_parser():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="BAGEL批处理风格迁移脚本")
    
    # 模型相关参数
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT",
                       help="模型路径")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3],
                       help="模型加载模式: 1=常规, 2=NF4量化, 3=INT8量化")
    
    # 输入输出参数
    parser.add_argument("--style_dir", type=str, default="/data/mayue/cjy/BAGEL/data_evl/style",
                       help="风格图像文件夹路径")
    parser.add_argument("--content_dir", type=str, default="/data/mayue/cjy/BAGEL/data_evl/content",
                       help="内容图像文件夹路径")
    parser.add_argument("--output_dir", type=str, default="/data/mayue/cjy/BAGEL/output/batch_results",
                       help="输出结果文件夹路径")
    parser.add_argument("--log_dir", type=str, default="/data/mayue/cjy/BAGEL/output/experiment_logs",
                       help="实验日志文件夹路径")
    
    # 匹配策略
    parser.add_argument("--matching_strategy", type=str, default="alphabetical", 
                       choices=["filename", "alphabetical", "random"],
                       help="图像匹配策略: filename=按文件名匹配, alphabetical=按字母顺序, random=随机匹配")
    
    # 推理参数
    parser.add_argument("--seed_start", type=int, default=1000,
                       help="起始随机种子")
    parser.add_argument("--seed_increment", type=int, default=1,
                       help="种子递增步长")
    parser.add_argument("--use_content_size", action="store_true",
                       help="使用内容图像的尺寸作为输出尺寸")
    
    # 高级推理参数
    parser.add_argument("--cfg_text_scale", type=float, default=3.5,
                       help="CFG文本缩放比例")
    parser.add_argument("--cfg_img_scale", type=float, default=2.0,
                       help="CFG图像缩放比例")
    parser.add_argument("--cfg_interval", type=float, default=0,
                       help="CFG应用间隔开始值")
    parser.add_argument("--timestep_shift", type=float, default=3.0,
                       help="时间步偏移")
    parser.add_argument("--num_timesteps", type=int, default=50,
                       help="去噪步数")
    parser.add_argument("--cfg_renorm_min", type=float, default=0,
                       help="CFG重归一化最小值")
    parser.add_argument("--cfg_renorm_type", type=str, default="global",
                       choices=["global", "local", "text_channel"],
                       help="CFG重归一化类型")
    
    # 风格迁移策略
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
    
    # 批处理控制
    parser.add_argument("--max_pairs", type=int, default=None,
                       help="最大处理图像对数量（用于测试）")
    parser.add_argument("--skip_existing", action="store_true",
                       help="跳过已存在的输出文件")
    parser.add_argument("--continue_on_error", action="store_true",
                       help="遇到错误时继续处理其他图像")
    
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


def get_image_files(directory: str) -> List[str]:
    """获取目录中的所有图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    image_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return sorted(image_files)


def match_image_pairs(style_files: List[str], content_files: List[str], 
                     strategy: str = "filename") -> List[Tuple[str, str]]:
    """根据策略匹配风格图像和内容图像"""
    if strategy == "filename":
        # 按文件名匹配（去除扩展名）
        style_dict = {Path(f).stem: f for f in style_files}
        content_dict = {Path(f).stem: f for f in content_files}
        
        pairs = []
        for name in style_dict:
            if name in content_dict:
                pairs.append((style_dict[name], content_dict[name]))
        
        # 如果没有直接匹配，尝试智能匹配
        if not pairs:
            print("  警告: 按文件名匹配失败，尝试智能匹配...")
            # 尝试去除常见后缀进行匹配
            style_simplified = {}
            content_simplified = {}
            
            for f in style_files:
                stem = Path(f).stem
                # 移除常见的风格标识符
                simplified = stem.replace('_style', '').replace('_02', '').replace('_03', '').replace('_04', '').replace('_05', '')
                simplified = simplified.replace('style_', '').replace('style1', '1').replace('style2', '2')
                if simplified not in style_simplified:
                    style_simplified[simplified] = f
            
            for f in content_files:
                stem = Path(f).stem
                # 移除常见的内容标识符
                simplified = stem.replace('_content', '').replace('_02', '').replace('_03', '').replace('_04', '').replace('_05', '')
                simplified = simplified.replace('content_', '').replace('content1', '1').replace('content2', '2')
                if simplified not in content_simplified:
                    content_simplified[simplified] = f
            
            for name in style_simplified:
                if name in content_simplified:
                    pairs.append((style_simplified[name], content_simplified[name]))
            
            if pairs:
                print(f"  智能匹配成功，找到 {len(pairs)} 对")
        
    elif strategy == "alphabetical":
        # 按字母顺序匹配
        style_files_sorted = sorted(style_files)
        content_files_sorted = sorted(content_files)
        
        min_length = min(len(style_files_sorted), len(content_files_sorted))
        pairs = list(zip(style_files_sorted[:min_length], content_files_sorted[:min_length]))
        
    elif strategy == "random":
        # 随机匹配
        min_length = min(len(style_files), len(content_files))
        style_sample = random.sample(style_files, min_length)
        content_sample = random.sample(content_files, min_length)
        pairs = list(zip(style_sample, content_sample))
        
    else:
        raise ValueError(f"不支持的匹配策略: {strategy}")
    
    return pairs


def initialize_model(args):
    """初始化BAGEL模型（复用之前的代码）"""
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


def analyze_style_image(style_image_path: str, inferencer) -> str:
    """分析风格图像并生成风格描述"""
    try:
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
        
        return result.get("text", "")
    except Exception as e:
        print(f"风格分析失败: {e}")
        return ""


def perform_single_style_transfer(inferencer, style_image_path: str, content_image_path: str, 
                                 output_path: str, args, seed: int) -> Dict[str, Any]:
    """执行单次风格迁移并返回详细信息"""
    start_time = datetime.datetime.now()
    
    try:
        # 设置随机种子
        set_seed(seed)
        
        # 加载图像
        style_image = Image.open(style_image_path)
        content_image = Image.open(content_image_path)
        
        style_image = pil_img2rgb(style_image)
        content_image = pil_img2rgb(content_image)
        
        # 生成风格描述
        style_description = ""
        if args.auto_style_prompt:
            style_description = analyze_style_image(style_image_path, inferencer)
            style_prompt = f"Apply the following artistic style to the content image: {style_description}. Maintain the content structure while adopting the style characteristics."
        else:
            style_prompt = args.style_prompt_template
        
        # 构建输入序列
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
        output_list = inferencer.interleave_inference(input_list, think=args.show_thinking, **inference_hyper)
        
        # 提取结果
        result_image = None
        thinking_text = ""
        
        for output in output_list:
            if isinstance(output, Image.Image):
                result_image = output
            elif isinstance(output, str):
                thinking_text = output
        
        if result_image is None:
            raise RuntimeError("未生成图像")
        
        # 保存图像
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_image.save(output_path, "PNG")
        
        end_time = datetime.datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # 返回详细信息
        return {
            'success': True,
            'generation_time_seconds': generation_time,
            'style_description': style_description,
            'thinking_process': thinking_text,
            'image_width': result_image.width,
            'image_height': result_image.height,
            'error_message': ""
        }
        
    except Exception as e:
        end_time = datetime.datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        return {
            'success': False,
            'generation_time_seconds': generation_time,
            'style_description': "",
            'thinking_process': "",
            'image_width': 0,
            'image_height': 0,
            'error_message': str(e)
        }


def main():
    """主函数"""
    parser = setup_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("BAGEL 批处理风格迁移脚本")
    print("=" * 60)
    
    try:
        # 初始化实验记录器
        logger = ExperimentLogger(args.log_dir)
        print(f"实验日志将保存到: {args.log_dir}")
        
        # 获取图像文件列表
        print(f"扫描风格图像目录: {args.style_dir}")
        style_files = get_image_files(args.style_dir)
        print(f"找到 {len(style_files)} 个风格图像")
        
        print(f"扫描内容图像目录: {args.content_dir}")
        content_files = get_image_files(args.content_dir)
        print(f"找到 {len(content_files)} 个内容图像")
        
        if not style_files:
            raise ValueError(f"风格图像目录为空: {args.style_dir}")
        if not content_files:
            raise ValueError(f"内容图像目录为空: {args.content_dir}")
        
        # 匹配图像对
        print(f"使用 {args.matching_strategy} 策略匹配图像对...")
        image_pairs = match_image_pairs(style_files, content_files, args.matching_strategy)
        print(f"成功匹配 {len(image_pairs)} 个图像对")
        
        if args.max_pairs:
            image_pairs = image_pairs[:args.max_pairs]
            print(f"限制处理数量为 {len(image_pairs)} 对")
        
        # 初始化模型
        inferencer = initialize_model(args)
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 批处理
        print(f"\n开始批处理风格迁移...")
        successful_count = 0
        failed_count = 0
        
        for i, (style_path, content_path) in enumerate(image_pairs, 1):
            style_name = Path(style_path).stem
            content_name = Path(content_path).stem
            
            # 生成输出文件名
            output_name = f"{style_name}_to_{content_name}_{args.seed_start + i * args.seed_increment}"
            output_path = os.path.join(args.output_dir, f"{output_name}.png")
            
            print(f"\n[{i}/{len(image_pairs)}] 处理图像对:")
            print(f"  风格: {Path(style_path).name}")
            print(f"  内容: {Path(content_path).name}")
            print(f"  输出: {output_name}.png")
            
            # 检查是否跳过已存在的文件
            if args.skip_existing and os.path.exists(output_path):
                print(f"  跳过（文件已存在）")
                continue
            
            # 计算当前种子
            current_seed = args.seed_start + i * args.seed_increment
            
            try:
                # 执行风格迁移
                result = perform_single_style_transfer(
                    inferencer, style_path, content_path, output_path, args, current_seed
                )
                
                if result['success']:
                    successful_count += 1
                    print(f"  ✅ 成功 ({result['generation_time_seconds']:.2f}s)")
                else:
                    failed_count += 1
                    print(f"  ❌ 失败: {result['error_message']}")
                
                # 记录实验数据
                experiment_data = {
                    'experiment_id': i,
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'style_image': style_path,
                    'content_image': content_path,
                    'output_image': output_path,
                    'seed': current_seed,
                    'cfg_text_scale': args.cfg_text_scale,
                    'cfg_img_scale': args.cfg_img_scale,
                    'num_timesteps': args.num_timesteps,
                    'timestep_shift': args.timestep_shift,
                    'cfg_interval': args.cfg_interval,
                    'cfg_renorm_type': args.cfg_renorm_type,
                    'cfg_renorm_min': args.cfg_renorm_min,
                    'auto_style_prompt': args.auto_style_prompt,
                    **result
                }
                
                logger.log_experiment(experiment_data)
                
            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                print(f"  ❌ 异常: {error_msg}")
                
                if not args.continue_on_error:
                    raise
                
                # 记录失败的实验
                experiment_data = {
                    'experiment_id': i,
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'style_image': style_path,
                    'content_image': content_path,
                    'output_image': output_path,
                    'seed': current_seed,
                    'cfg_text_scale': args.cfg_text_scale,
                    'cfg_img_scale': args.cfg_img_scale,
                    'num_timesteps': args.num_timesteps,
                    'timestep_shift': args.timestep_shift,
                    'cfg_interval': args.cfg_interval,
                    'cfg_renorm_type': args.cfg_renorm_type,
                    'cfg_renorm_min': args.cfg_renorm_min,
                    'auto_style_prompt': args.auto_style_prompt,
                    'success': False,
                    'generation_time_seconds': 0,
                    'style_description': "",
                    'thinking_process': "",
                    'image_width': 0,
                    'image_height': 0,
                    'error_message': error_msg
                }
                
                logger.log_experiment(experiment_data)
        
        # 保存最终日志
        logger.save_detailed_log()
        logger.save_summary()
        
        print("\n" + "=" * 60)
        print("批处理完成!")
        print(f"总处理数: {len(image_pairs)}")
        print(f"成功数: {successful_count}")
        print(f"失败数: {failed_count}")
        print(f"成功率: {successful_count/len(image_pairs)*100:.1f}%")
        print(f"\n结果保存在: {args.output_dir}")
        print(f"实验日志: {logger.csv_log_path}")
        print(f"详细日志: {logger.json_log_path}")
        print(f"总结报告: {logger.summary_log_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"批处理错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())