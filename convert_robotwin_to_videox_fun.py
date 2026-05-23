#!/usr/bin/env python3
"""
将 RoboTwin 数据转换为 VideoX-Fun Wan2.1 LoRA 训练格式
输出格式符合 VideoX-Fun 标准

输出结构:
  datasets/worldarena_wan_i2v_clean50/
  ├── train/                          # 训练视频
  │   ├── video_00000.mp4
  │   ├── video_00001.mp4
  │   └── ...
  └── metadata.json                   # VideoX-Fun 标准格式
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

# 指令前缀
INSTRUCTION_PREFIX = "In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. The arm maintains high stability with no deformation and enters the frame to "


def get_episode_instruction(task_dir: Path, episode_id: int) -> str:
    """从 JSON 文件获取 episode 的 instruction"""
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
    
    return INSTRUCTION_PREFIX + "Perform the task"


def get_video_dimensions(video_path: Path) -> Dict[str, int]:
    """使用 ffprobe 获取视频分辨率"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            width, height = map(int, result.stdout.strip().split(','))
            return {"width": width, "height": height}
    except Exception as e:
        pass
    
    # 如果获取失败，返回默认值
    return {"width": 832, "height": 480}


def process_robotwin_data():
    """处理 RoboTwin 数据为 VideoX-Fun 训练格式"""
    print("=" * 80)
    print("RoboTwin → VideoX-Fun Wan2.1 LoRA 数据转换器")
    print("=" * 80)
    
    # 创建输出目录
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
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
            # 从 "episodeX.mp4" 中提取 episode_id
            try:
                if video_file.stem.startswith("episode"):
                    episode_id = int(video_file.stem.replace("episode", ""))
                else:
                    episode_id = int(video_file.stem)
            except ValueError:
                episode_id = hash(video_file.name) % 100
            
            instruction = get_episode_instruction(extracted_dir, episode_id)
            
            all_episodes.append({
                'task': task_dir.name,
                'episode_id': episode_id,
                'video_path': video_file,
                'instruction': instruction
            })
    
    print(f"\n📊 数据统计:")
    print(f"  收集到的 episode: {len(all_episodes)}")
    
    if len(all_episodes) == 0:
        print("\n❌ 未找到任何数据！请先解压 zip 文件")
        return False
    
    # 随机打乱并复制视频
    print(f"\n📹 复制视频到训练目录:")
    random.shuffle(all_episodes)
    
    metadata = []
    success_count = 0
    
    for i, ep in enumerate(all_episodes, 1):
        # 新文件名
        output_video = TRAIN_DIR / f"video_{success_count:05d}.mp4"
        
        try:
            # 复制视频
            shutil.copy2(ep['video_path'], output_video)
            
            # 获取视频尺寸
            dims = get_video_dimensions(output_video)
            
            # 创建元数据条目（符合 VideoX-Fun 格式）
            metadata_entry = {
                "file_path": f"train/video_{success_count:05d}.mp4",  # 相对路径
                "text": ep['instruction'],  # 指令（英文提示词）
                "type": "video",  # 固定为 "video"
                "width": dims["width"],
                "height": dims["height"],
                "task": ep['task'],  # 额外信息（任务名）
                "episode_id": ep['episode_id']  # 额外信息（episode ID）
            }
            metadata.append(metadata_entry)
            success_count += 1
            
            if i % 100 == 0:
                print(f"  进度: {i}/{len(all_episodes)}")
        except Exception as e:
            print(f"  ⚠ 复制失败: {ep['video_path'].name} - {e}")
    
    print(f"  ✓ 复制完成: {success_count}/{len(all_episodes)}")
    
    # 保存元数据
    print(f"\n💾 保存元数据:")
    metadata_file = OUTPUT_DIR / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 元数据已保存: {len(metadata)} 个样本")
    
    # 最终统计
    print(f"\n" + "=" * 80)
    print(f"✓ 数据转换完成！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  训练视频数: {len(metadata)}")
    print(f"  任务类别: {len(set(ep['task'] for ep in all_episodes))}")
    print(f"  格式: VideoX-Fun Wan2.1 LoRA 标准格式")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    print("⏳ 检查数据准备状态...")
    
    # 计算已解压的任务
    extracted_count = 0
    total_videos = 0
    for task_dir in ROBOTWIN_DATASET.iterdir():
        if task_dir.is_dir():
            extracted = task_dir / "extracted" / "aloha-agilex_clean_50"
            if extracted.exists():
                video_count = len(list((extracted / "video").glob("*.mp4"))) if (extracted / "video").exists() else 0
                if video_count > 0:
                    extracted_count += 1
                    total_videos += video_count
    
    print(f"✓ 已解压: {extracted_count} 个任务, {total_videos} 个视频")
    
    if extracted_count > 0:
        process_robotwin_data()
    else:
        print("\n⚠ 数据解压不足，请先运行数据解压脚本")
