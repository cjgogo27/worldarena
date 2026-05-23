#!/usr/bin/env python3
"""
将 RoboTwin aloha-agilex_clean_50 数据转换为 Wan2.1 LoRA 训练格式
输出: datasets/worldarena_wan_i2v_clean50/
  ├── train/*.mp4
  ├── metadata.json
  ├── validation_images/*.png
  ├── validation_examples.json
  └── quality_report.json
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict
import random

# 配置
ROBOTWIN_DATASET = Path("/data/alice/cjtest/lara-wm/data/robotwin/dataset")
OUTPUT_DIR = Path("/data/alice/cjtest/datasets/worldarena_wan_i2v_clean50")
TRAIN_DIR = OUTPUT_DIR / "train"
VALIDATION_IMG_DIR = OUTPUT_DIR / "validation_images"

# 每个任务有 50 个 episode，总共 50 任务 = 2500 个视频
# 划分: 80% 训练 (2000), 20% 验证 (500)
TRAIN_SPLIT = 0.8


def get_episode_instruction(task_dir: Path, episode_id: int) -> str:
    """从 JSON 文件获取 episode 的 instruction"""
    # 指令前缀 - WorldArena 要求的格式
    INSTRUCTION_PREFIX = "In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. The arm maintains high stability with no deformation and enters the frame to "
    
    # 尝试 episodeX.json 格式
    instruction_file = task_dir / "instructions" / f"episode{episode_id}.json"
    
    if not instruction_file.exists():
        # 尝试 0X.json 格式
        instruction_file = task_dir / "instructions" / f"{episode_id:02d}.json"
    
    if instruction_file.exists():
        try:
            with open(instruction_file, 'r') as f:
                data = json.load(f)
                
                # 首先尝试 'seen' 字段（列表）
                if 'seen' in data and isinstance(data['seen'], list) and len(data['seen']) > 0:
                    raw_instruction = data['seen'][0]
                    return INSTRUCTION_PREFIX + raw_instruction
                
                # 然后尝试 'instruction' 字段
                elif 'instruction' in data:
                    raw_instruction = data['instruction']
                    if isinstance(raw_instruction, list):
                        raw_instruction = raw_instruction[0] if raw_instruction else ""
                    return INSTRUCTION_PREFIX + raw_instruction
                
                # 最后尝试 'text' 字段
                elif 'text' in data:
                    raw_instruction = data['text']
                    if isinstance(raw_instruction, list):
                        raw_instruction = raw_instruction[0] if raw_instruction else ""
                    return INSTRUCTION_PREFIX + raw_instruction
                
                # 获取第一个非空 value
                else:
                    for value in data.values():
                        if isinstance(value, list) and len(value) > 0:
                            return INSTRUCTION_PREFIX + str(value[0])
                        elif value:
                            return INSTRUCTION_PREFIX + str(value)
        except Exception as e:
            pass
    
    # 如果没有找到指令，返回默认前缀 + 任务名
    return INSTRUCTION_PREFIX + "Perform the task"


def extract_first_frame(video_file: Path, output_image: Path) -> bool:
    """从视频中提取第一帧作为预处理图像"""
    try:
        cmd = [
            "ffmpeg",
            "-i", str(video_file),
            "-vf", "select=eq(n\\,0)",
            "-q:v", "2",
            "-y",
            str(output_image)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and output_image.exists()
    except:
        return False


def process_robotwin_data():
    """处理 RoboTwin 数据为训练格式"""
    print("=" * 80)
    print("RoboTwin → Wan2.1 LoRA 训练数据转换器")
    print("=" * 80)
    
    # 创建输出目录
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 收集所有 episode
    all_episodes = []
    
    for task_dir in sorted(ROBOTWIN_DATASET.iterdir()):
        if not task_dir.is_dir():
            continue
        
        # 查找 aloha-agilex_clean_50 目录
        extracted_dir = task_dir / "extracted" / "aloha-agilex_clean_50"
        if not extracted_dir.exists():
            print(f"⚠ 任务 {task_dir.name}: 未解压数据")
            continue
        
        video_dir = extracted_dir / "video"
        if not video_dir.exists():
            print(f"⚠ 任务 {task_dir.name}: 无 video 目录")
            continue
        
        # 收集这个任务的所有 episode
        for video_file in sorted(video_dir.glob("*.mp4")):
            # 从 "episode0.mp4" 或 "0.mp4" 中提取 episode_id
            try:
                if video_file.stem.startswith("episode"):
                    episode_id = int(video_file.stem.replace("episode", ""))
                else:
                    episode_id = int(video_file.stem)
            except ValueError:
                # 如果无法解析，使用文件名的哈希作为 ID
                episode_id = hash(video_file.name) % 100
            
            instruction = get_episode_instruction(extracted_dir, episode_id)
            
            all_episodes.append({
                'task': task_dir.name,
                'episode_id': episode_id,
                'video_path': video_file,
                'instruction': instruction or f"{task_dir.name} episode {episode_id}"
            })
    
    print(f"\n📊 数据统计:")
    print(f"  收集到的 episode: {len(all_episodes)}")
    print(f"  目标: 2500 个 (50 tasks × 50 episodes)")
    
    if len(all_episodes) == 0:
        print("\n❌ 未找到任何数据！请先解压 zip 文件")
        return False
    
    # 随机划分训练/验证
    random.shuffle(all_episodes)
    split_idx = int(len(all_episodes) * TRAIN_SPLIT)
    train_episodes = all_episodes[:split_idx]
    val_episodes = all_episodes[split_idx:]
    
    print(f"\n📂 数据划分:")
    print(f"  训练集: {len(train_episodes)} episodes ({len(train_episodes)/len(all_episodes)*100:.1f}%)")
    print(f"  验证集: {len(val_episodes)} episodes ({len(val_episodes)/len(all_episodes)*100:.1f}%)")
    
    # 复制训练视频
    print(f"\n📹 复制训练视频:")
    train_metadata = []
    success_count = 0
    
    for i, ep in enumerate(train_episodes, 1):
        output_video = TRAIN_DIR / f"episode_{success_count:05d}.mp4"
        
        try:
            shutil.copy2(ep['video_path'], output_video)
            train_metadata.append({
                'video_file': f"episode_{success_count:05d}.mp4",
                'instruction': ep['instruction'],
                'task': ep['task'],
                'episode_id': ep['episode_id']
            })
            success_count += 1
            
            if i % 500 == 0:
                print(f"  进度: {i}/{len(train_episodes)}")
        except Exception as e:
            print(f"  ⚠ 复制失败: {ep['video_path'].name} - {e}")
    
    print(f"  ✓ 复制完成: {success_count}/{len(train_episodes)}")
    
    # 提取验证集的第一帧
    print(f"\n🖼️  提取验证集图像:")
    val_metadata = []
    val_success_count = 0
    
    for i, ep in enumerate(val_episodes, 1):
        image_path = VALIDATION_IMG_DIR / f"validation_{val_success_count:05d}.png"
        
        if extract_first_frame(ep['video_path'], image_path):
            val_metadata.append({
                'image_file': f"validation_{val_success_count:05d}.png",
                'instruction': ep['instruction'],
                'task': ep['task'],
                'episode_id': ep['episode_id']
            })
            val_success_count += 1
            
            if i % 100 == 0:
                print(f"  进度: {i}/{len(val_episodes)}")
        else:
            print(f"  ⚠ 提取失败: {ep['video_path'].name}")
    
    print(f"  ✓ 提取完成: {val_success_count}/{len(val_episodes)}")
    
    # 保存元数据
    print(f"\n💾 保存元数据:")
    
    # 训练元数据 (metadata.json)
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(train_metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 训练元数据: {len(train_metadata)} 个样本")
    
    # 验证元数据 (validation_examples.json)
    with open(OUTPUT_DIR / "validation_examples.json", 'w') as f:
        json.dump(val_metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 验证元数据: {len(val_metadata)} 个样本")
    
    # 质量报告
    quality_report = {
        'dataset_name': 'worldarena_wan_i2v_clean50',
        'total_episodes': len(all_episodes),
        'train_episodes': len(train_metadata),
        'validation_episodes': len(val_metadata),
        'train_videos_dir': str(TRAIN_DIR),
        'validation_images_dir': str(VALIDATION_IMG_DIR),
        'video_resolution': 'variable',  # RoboTwin resolution
        'fps': 30,  # RoboTwin typically 30 fps
        'task_categories': list(set([ep['task'] for ep in all_episodes])),
        'total_unique_tasks': len(set([ep['task'] for ep in all_episodes]))
    }
    
    with open(OUTPUT_DIR / "quality_report.json", 'w') as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)
    
    # 最终统计
    print(f"\n" + "=" * 80)
    print(f"✓ 数据转换完成！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  训练视频数: {len(train_metadata)}")
    print(f"  验证图像数: {len(val_metadata)}")
    print(f"  任务类别: {quality_report['total_unique_tasks']}")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    # 检查是否需要先等待解压
    print("⏳ 检查数据准备状态...")
    
    # 计算已解压的任务
    extracted_count = 0
    for task_dir in ROBOTWIN_DATASET.iterdir():
        if task_dir.is_dir():
            if (task_dir / "extracted" / "aloha-agilex_clean_50").exists():
                extracted_count += 1
    
    print(f"✓ 已解压: {extracted_count}/50 个任务")
    
    if extracted_count < 5:
        print("\n⚠ 数据解压不足，请先运行数据解压脚本")
        print("  或等待下载完成后运行本脚本")
    else:
        process_robotwin_data()
