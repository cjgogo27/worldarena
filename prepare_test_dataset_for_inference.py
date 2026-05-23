#!/usr/bin/env python3
"""
将 test_dataset 转换为推理验证集格式
支持用 WorldArena LoRA ckpt 生成 1000 个推理视频
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

def prepare_test_dataset_validation():
    """将 test_dataset 转换为 validation_examples.json 格式"""
    
    test_dataset_path = Path('/data/alice/cjtest/VideoX-Fun')
    instructions_dir = test_dataset_path / 'instructions_2' / 'fixed_scene_task'
    first_frame_dir = test_dataset_path / 'first_frame' / 'fixed_scene_task'
    
    output_dir = test_dataset_path / 'test_dataset_validation'
    output_dir.mkdir(exist_ok=True)
    
    # 拷贝第一帧到验证集目录
    validation_images_dir = output_dir / 'validation_images'
    validation_images_dir.mkdir(exist_ok=True)
    
    # 收集所有指令
    validation_examples = []
    
    # 获取所有 episode 编号
    instruction_files = sorted(instructions_dir.glob('episode*.json'))
    print(f"找到 {len(instruction_files)} 个指令文件")
    
    for idx, instr_file in enumerate(tqdm(instruction_files, desc="准备验证集")):
        # 提取 episode 编号
        episode_num = instr_file.stem.replace('episode', '')
        
        # 加载指令
        with open(instr_file) as f:
            data = json.load(f)
            instruction = data.get('instruction', '')
        
        # 对应的第一帧
        first_frame_path = first_frame_dir / f'episode{episode_num}.png'
        
        if first_frame_path.exists():
            # 生成验证例子
            example = {
                "episode": int(episode_num),
                "image": f"validation_images/episode{episode_num}.png",
                "prompt": instruction,
                "source": "test_dataset_from_huggingface"
            }
            validation_examples.append(example)
    
    # 排序并生成元数据
    validation_examples.sort(key=lambda x: x['episode'])
    
    # 保存验证集配置
    with open(output_dir / 'validation_config.json', 'w') as f:
        json.dump(validation_examples, f, indent=2, ensure_ascii=False)
    
    # 保存统计信息
    stats = {
        "total_validation_examples": len(validation_examples),
        "source": "WorldArena test_dataset from HuggingFace",
        "instruction_field": "prompt",
        "image_field": "image",
        "format": "Ready for I2V inference with VideoX-Fun Wan2.1"
    }
    
    with open(output_dir / 'validation_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ 验证集准备完成")
    print(f"  总数据量: {len(validation_examples)} examples")
    print(f"  输出目录: {output_dir}")
    print(f"  配置文件: {output_dir / 'validation_config.json'}")
    
    return validation_examples


if __name__ == '__main__':
    validation_examples = prepare_test_dataset_validation()
    print(f"\n第一个例子:")
    print(json.dumps(validation_examples[0], indent=2, ensure_ascii=False))
