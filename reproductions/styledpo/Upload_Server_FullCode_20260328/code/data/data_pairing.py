#!/usr/bin/env python3
"""
数据样本对自动匹配脚本
自动为 BAGEL 生成的图像选择正/负样本对用于 DPO 训练

项目：Style-DPO
版本：v2.0
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np

# ===== 配置区域 =====

@dataclass
class PreferencePair:
    """DPO 训练的偏好数据对"""
    chosen: Dict  # {"image_path": "...", "prompt": "...", "score": 0.9}
    rejected: Dict  # {"image_path": "...", "prompt": "...", "score": 0.3}
    style_category: str
    reasoning: str = ""  # 为什么选择这个对

# ===== 设置日志 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== 核心函数 =====

def load_style_prompt_map(style_prompt_file: str = None) -> Dict[str, str]:
    """加载风格提示词映射文件。

    期望格式：
    {
      "prompts": {
        "Style Name": "prompt text"
      }
    }
    或直接是 {"Style Name": "prompt text"}
    """
    if not style_prompt_file:
        return {}

    prompt_path = Path(style_prompt_file)
    if not prompt_path.exists():
        logger.warning(f"Style prompt file not found: {style_prompt_file}")
        return {}

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and isinstance(data.get("prompts"), dict):
            prompt_map = data["prompts"]
        elif isinstance(data, dict):
            prompt_map = data
        else:
            logger.warning(f"Invalid style prompt format: {style_prompt_file}")
            return {}

        # 只保留字符串值，避免坏数据污染
        prompt_map = {k: v for k, v in prompt_map.items() if isinstance(v, str)}
        logger.info(f"Loaded {len(prompt_map)} style prompts from {style_prompt_file}")
        return prompt_map

    except Exception as e:
        logger.error(f"Failed to load style prompt file {style_prompt_file}: {e}")
        return {}

def load_vlm_model(model_name: str = "Qwen/Qwen3.5-9B"):
    """
    加载 VLM 模型用于评分
    
    Args:
        model_name: 模型名称或路径
        
    Returns:
        模型和处理器
        
    Note:
        如果 GPU 显存不足，可以使用：
        - 4-bit 量化
        - 较小的模型版本
        - 本地推理 API
    """
    logger.info(f"Loading VLM model: {model_name}")
    
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # ⚠️ 注意：这里需要根据实际的 VLM 模型调整
        # 当前代码是框架，需要根据选用的 VLM 模型（默认 Qwen/Qwen3.5-9B）调整
        
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True
        )
        model.eval()
        
        return model, processor, device
        
    except Exception as e:
        logger.error(f"Failed to load VLM model: {e}")
        logger.warning("Falling back to CLIP-based evaluation")
        return None, None, None

def score_image_with_vlm(
    image_path: str,
    style_description: str,
    vlm_model,
    processor,
    device: str
) -> float:
    """
    使用 VLM 评分图像与风格描述的匹配度
    
    Args:
        image_path: 图像路径
        style_description: 风格描述文本
        vlm_model: VLM 模型
        processor: 处理器
        device: 计算设备
        
    Returns:
        匹配分数 (0-1)
        
    Note:
        目前返回模拟分数，需要根据实际的 VLM 模型实现
    """
    if vlm_model is None:
        logger.warning("VLM model not available, returning random score")
        return np.random.rand()
    
    try:
        from PIL import Image
        import torch
        
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # 构建评估提示词
        # ⭐ 核心：使用与项目相同的评估提示词
        prompt = f"""You are a style transfer evaluation expert.
Given an image and a target style description, rate how well the image matches the style.

Target Style: {style_description}

Rate the match on a scale of 0 to 1, where:
- 0 = No match at all
- 0.5 = Partial match
- 1 = Perfect match

Respond with ONLY a number between 0 and 1.
"""
        
        # ⚠️ 这部分需要根据实际使用的 VLM 模型调整
        # 以下是 Qwen/Qwen3.5-9B 的伪代码示例
        
        # inputs = processor(
        #     text=prompt,
        #     images=[image],
        #     return_tensors="pt"
        # ).to(device)
        
        # with torch.no_grad():
        #     outputs = vlm_model(**inputs)
        #     score = outputs.score[0].item()
        
        # return min(max(score, 0), 1)  # Clamp to [0, 1]
        
        # 暂时返回模拟分数（需要实际实现）
        logger.warning(f"VLM evaluation not fully implemented, returning mock score")
        return np.random.rand()
        
    except Exception as e:
        logger.error(f"Error evaluating {image_path}: {e}")
        return 0.5  # 默认中间值

def score_image_with_clip(
    image_path: str,
    style_description: str,
    clip_model=None,
    processor=None
) -> float:
    """
    使用 CLIP 评分（备选方案，速度更快）
    
    Args:
        image_path: 图像路径
        style_description: 风格描述
        clip_model: CLIP 模型（如果已加载）
        processor: 处理器
        
    Returns:
        匹配分数 (0-1)
    """
    if clip_model is None:
        # 动态加载 CLIP
        try:
            import clip
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model, preprocess = clip.load("ViT-B/32", device=device)
            return score_image_with_clip(image_path, style_description, clip_model, preprocess)
        except ImportError:
            logger.error("CLIP not available, returning random score")
            return np.random.rand()
    
    try:
        import clip
        import torch
        from PIL import Image
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载和处理图像
        image = Image.open(image_path).convert("RGB")
        image_input = processor(image).unsqueeze(0).to(device)
        
        # 处理文本
        text_input = clip.tokenize([
            f"a {style_description} style image",
            "an unrelated image"
        ]).to(device)
        
        # 计算相似度
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度（使用 sigmoid 转为 0-1）
            logits = (100.0 * image_features @ text_features.T)
            similarity = torch.softmax(logits, dim=-1)[0, 0].item()
        
        return similarity
        
    except Exception as e:
        logger.error(f"CLIP evaluation error: {e}")
        return 0.5

def load_generated_images(
    image_dir: str,
    style_prompt_file: str = None
) -> Dict[str, List[Dict]]:
    """
    加载 BAGEL 生成的图像
    
    Args:
        image_dir: 图像目录
        style_prompt_file: 风格提示词 JSON 文件路径
        
    Returns:
        {style_name: [{"path": "...", "prompt": "..."}, ...]}
    """
    logger.info(f"Loading images from {image_dir}")
    
    images_by_style = {}
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    style_prompt_map = load_style_prompt_map(style_prompt_file)
    fallback_prompt_count = 0
    
    image_dir = Path(image_dir)
    
    # 简单加载：按目录结构组织
    for style_dir in image_dir.iterdir():
        if not style_dir.is_dir():
            continue
            
        style_name = style_dir.name
        images = []
        
        for image_file in style_dir.iterdir():
            if image_file.suffix.lower() in image_extensions:
                prompt_text = style_prompt_map.get(style_name)
                if prompt_text is None:
                    prompt_text = f"in {style_name} style"
                    fallback_prompt_count += 1

                images.append({
                    "path": str(image_file),
                    "prompt": prompt_text,
                    "style": style_name
                })
        
        if images:
            images_by_style[style_name] = images
            logger.info(f"  {style_name}: {len(images)} images")
    
    logger.info(f"Total styles: {len(images_by_style)}")
    if fallback_prompt_count > 0:
        logger.warning(f"Fallback prompt used {fallback_prompt_count} times (style prompt file coverage incomplete)")
    return images_by_style

def create_preference_pairs(
    images_by_style: Dict[str, List[Dict]],
    scoring_fn,
    min_score_diff: float = 0.3,
    max_pairs_per_style: int = None
) -> List[PreferencePair]:
    """
    从生成的图像创建偏好对
    
    策略：
    1. 对每个风格的所有图像评分
    2. 选择得分最高的为 chosen
    3. 选择得分最低的为 rejected
    4. 确保分数差异足够大（min_score_diff）
    
    Args:
        images_by_style: 按风格组织的图像列表
        scoring_fn: 评分函数
        min_score_diff: 最小分数差异阈值
        max_pairs_per_style: 每个风格最多生成多少对
        
    Returns:
        偏好对列表
    """
    logger.info("Creating preference pairs")
    
    pairs = []
    
    for style_name, images in tqdm(images_by_style.items(), desc="Processing styles"):
        
        # ===== 第 1 步：评分所有图像 =====
        scored_images = []
        for img_info in tqdm(images, desc=f"Scoring {style_name}", leave=False):
            score = scoring_fn(
                img_info["path"],
                img_info["prompt"]
            )
            scored_images.append({
                **img_info,
                "score": score
            })
        
        # ===== 第 2 步：按分数排序 =====
        scored_images.sort(key=lambda x: x["score"], reverse=True)
        
        # ===== 第 3 步：创建对 =====
        num_pairs = 0
        max_pairs = max_pairs_per_style or len(scored_images) // 4
        
        for i in range(0, len(scored_images) // 2):
            chosen = scored_images[i]
            rejected = scored_images[-(i+1)]  # 从后往前取
            
            score_diff = chosen["score"] - rejected["score"]
            
            if score_diff >= min_score_diff:
                pair = PreferencePair(
                    chosen={
                        "image_path": chosen["path"],
                        "prompt": chosen["prompt"],
                        "score": float(chosen["score"])
                    },
                    rejected={
                        "image_path": rejected["path"],
                        "prompt": rejected["prompt"],
                        "score": float(rejected["score"])
                    },
                    style_category=style_name,
                    reasoning=f"Score diff: {score_diff:.3f}"
                )
                pairs.append(pair)
                num_pairs += 1
                
                if num_pairs >= max_pairs:
                    break
        
        logger.info(f"  {style_name}: created {num_pairs} pairs")
    
    logger.info(f"Total pairs created: {len(pairs)}")
    return pairs

def save_preference_pairs(pairs: List[PreferencePair], output_path: str):
    """保存偏好对到 JSON 文件"""
    
    logger.info(f"Saving {len(pairs)} pairs to {output_path}")
    
    # 转换为可序列化的字典
    pairs_dict = [asdict(pair) for pair in pairs]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved to {output_path}")

# ===== 主函数 =====

def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate DPO preference pairs from BAGEL images"
    )
    parser.add_argument("--image_dir", required=True, help="Directory containing generated images")
    parser.add_argument("--output_file", required=True, help="Output JSON file path")
    parser.add_argument("--scoring_method", choices=["vlm", "clip"], default="clip",
                       help="Scoring method (VLM is more accurate but slower)")
    parser.add_argument("--vlm_model", default="Qwen/Qwen3.5-9B", help="VLM model name")
    parser.add_argument(
        "--style_prompt_file",
        default="resources/styles/style_prompts_40_v1.json",
        help="JSON file containing style_name -> prompt mapping",
    )
    parser.add_argument("--min_score_diff", type=float, default=0.3, help="Minimum score difference")
    parser.add_argument("--max_pairs_per_style", type=int, default=None, help="Max pairs per style")
    parser.add_argument("--dry_run", action="store_true", help="Test without saving")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Style-DPO Preference Pair Generation")
    logger.info("="*60)
    logger.info(f"Config: {args}")
    logger.info("="*60)
    
    # ===== 第 1 步：加载图像 =====
    images_by_style = load_generated_images(args.image_dir, args.style_prompt_file)
    
    # ===== 第 2 步：初始化评分函数 =====
    if args.scoring_method == "vlm":
        logger.info("Using VLM for scoring")
        vlm_model, processor, device = load_vlm_model(args.vlm_model)
        scoring_fn = lambda img, prompt: score_image_with_vlm(img, prompt, vlm_model, processor, device)
    else:
        logger.info("Using CLIP for scoring")
        scoring_fn = score_image_with_clip
    
    # ===== 第 3 步：创建偏好对 =====
    pairs = create_preference_pairs(
        images_by_style,
        scoring_fn,
        min_score_diff=args.min_score_diff,
        max_pairs_per_style=args.max_pairs_per_style
    )
    
    # ===== 第 4 步：保存结果 =====
    if not args.dry_run:
        save_preference_pairs(pairs, args.output_file)
        logger.info("✓ Done!")
    else:
        logger.info(f"DRY RUN: Would save {len(pairs)} pairs to {args.output_file}")

if __name__ == "__main__":
    main()
