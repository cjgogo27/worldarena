#!/usr/bin/env python3
"""
RoboTwin 2.0 aloha-agilex_clean_50 数据集下载和验证脚本
下载所有 50 个任务的完整数据集（每个任务 ~500MB）
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict

# 所有 50 个 RoboTwin 2.0 任务
ALL_TASKS = [
    "adjust_bottle", "beat_block_hammer", "bend_rope", "click_bell", "cut_meat",
    "dump_bin_bigbin", "grab_ball", "grab_roller", "handover_block", "insert_pen",
    "insert_screw", "insert_stick", "knock_door", "knock_door_with_handle",
    "move_box", "move_clothes", "move_sandal", "open_dishwasher",
    "open_drawer_horizontal", "open_drawer_vertical", "open_laptop", "open_trash_can",
    "open_window", "pick_apple", "pick_cup", "pick_glass", "pick_jewelry",
    "pick_knife", "pick_orange", "pick_small_ball", "pick_towel", "place_a2b_left",
    "place_a2b_right", "place_cup", "place_napkin", "place_toothbrush", "play_game",
    "press_stapler", "push_door", "push_drawer", "push_pull_handle", "put_food_in_bowl",
    "put_item_on_shelf", "put_item_on_table", "reach_cabinet", "reach_table",
    "roll_rope", "stack_blocks_two", "turn_paper", "twist_bottle"
]

DATASET_DIR = Path("/data/alice/cjtest/lara-wm/data/robotwin/dataset")
HF_REPO = "TianxingChen/RoboTwin2.0"


def get_local_tasks() -> List[str]:
    """获取本地已存在的任务目录"""
    if not DATASET_DIR.exists():
        return []
    return [d.name for d in DATASET_DIR.iterdir() if d.is_dir()]


def get_missing_tasks() -> List[str]:
    """获取需要下载的任务"""
    local = set(get_local_tasks())
    all_tasks = set(ALL_TASKS)
    return sorted(list(all_tasks - local))


def check_zip_exists(task_name: str) -> bool:
    """检查任务的 zip 文件是否存在"""
    task_dir = DATASET_DIR / task_name
    zip_file = task_dir / "aloha-agilex_clean_50.zip"
    return zip_file.exists() if task_dir.exists() else False


def download_task(task_name: str) -> bool:
    """下载单个任务的数据"""
    task_dir = DATASET_DIR / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    zip_file = task_dir / "aloha-agilex_clean_50.zip"
    
    # 检查是否已存在
    if zip_file.exists():
        print(f"  ✓ {task_name}: zip 文件已存在 ({zip_file.stat().st_size / 1e9:.1f}GB)")
        return True
    
    print(f"  ⏳ 下载 {task_name}...")
    
    try:
        # 使用 huggingface-hub 下载
        cmd = f"""
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='{HF_REPO}',
    filename='dataset/{task_name}/aloha-agilex_clean_50.zip',
    repo_type='dataset',
    local_dir='{task_dir}',
    local_dir_use_symlinks=False
)
print('✓ 下载完成')
"
        """
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0 and zip_file.exists():
            size_gb = zip_file.stat().st_size / 1e9
            print(f"  ✓ {task_name}: 下载成功 ({size_gb:.1f}GB)")
            return True
        else:
            print(f"  ✗ {task_name}: 下载失败")
            if result.stderr:
                print(f"    错误: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ {task_name}: 下载超时")
        return False
    except Exception as e:
        print(f"  ✗ {task_name}: 下载异常 - {e}")
        return False


def extract_task(task_name: str) -> bool:
    """解压单个任务的数据"""
    task_dir = DATASET_DIR / task_name
    zip_file = task_dir / "aloha-agilex_clean_50.zip"
    extracted_dir = task_dir / "extracted"
    
    if not zip_file.exists():
        print(f"  ✗ {task_name}: zip 文件不存在")
        return False
    
    # 检查是否已解压
    if (extracted_dir / "aloha-agilex_clean_50").exists():
        print(f"  ✓ {task_name}: 已解压")
        return True
    
    print(f"  ⏳ 解压 {task_name}...")
    
    try:
        extracted_dir.mkdir(parents=True, exist_ok=True)
        cmd = f"cd {extracted_dir} && unzip -q {zip_file} -d . && echo '✓ 解压完成'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"  ✓ {task_name}: 解压成功")
            return True
        else:
            print(f"  ✗ {task_name}: 解压失败 - {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ {task_name}: 解压超时")
        return False
    except Exception as e:
        print(f"  ✗ {task_name}: 解压异常 - {e}")
        return False


def verify_task_structure(task_name: str) -> Dict[str, bool]:
    """验证任务的数据结构"""
    task_dir = DATASET_DIR / task_name / "extracted" / "aloha-agilex_clean_50"
    
    checks = {
        "task_exists": task_dir.exists(),
        "video_dir": (task_dir / "video").exists(),
        "instructions_dir": (task_dir / "instructions").exists(),
        "scene_info": (task_dir / "scene_info.json").exists(),
    }
    
    # 检查文件数量
    if checks["video_dir"]:
        video_files = list((task_dir / "video").glob("*.mp4"))
        checks["50_videos"] = len(video_files) == 50
        checks["video_count"] = len(video_files)
    
    if checks["instructions_dir"]:
        instruction_files = list((task_dir / "instructions").glob("*.json"))
        checks["50_instructions"] = len(instruction_files) == 50
        checks["instruction_count"] = len(instruction_files)
    
    return checks


def main():
    print("=" * 80)
    print("RoboTwin 2.0 aloha-agilex_clean_50 数据集下载器")
    print("=" * 80)
    
    # 创建数据集目录
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取本地和缺失的任务
    local_tasks = get_local_tasks()
    missing_tasks = get_missing_tasks()
    
    print(f"\n📊 状态概览:")
    print(f"  本地已有: {len(local_tasks)}/50 个任务")
    print(f"  需要下载: {len(missing_tasks)} 个任务")
    
    if missing_tasks:
        print(f"  缺失任务: {', '.join(missing_tasks[:5])}..." if len(missing_tasks) > 5 else f"  缺失任务: {', '.join(missing_tasks)}")
    
    # 检查是否安装了 huggingface-hub
    try:
        import huggingface_hub
        print(f"\n✓ huggingface-hub 已安装 (版本: {huggingface_hub.__version__})")
    except ImportError:
        print(f"\n⚠ huggingface-hub 未安装，尝试安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub", "-q"], 
                       capture_output=True)
    
    # 下载缺失的任务
    if missing_tasks:
        print(f"\n📥 下载缺失的任务:")
        downloaded = 0
        for i, task in enumerate(missing_tasks, 1):
            print(f"\n[{i}/{len(missing_tasks)}] {task}")
            if download_task(task):
                downloaded += 1
        
        print(f"\n✓ 下载完成: {downloaded}/{len(missing_tasks)} 个任务")
    
    # 解压所有任务
    print(f"\n📦 解压所有任务:")
    all_tasks = get_local_tasks()
    extracted = 0
    for i, task in enumerate(all_tasks, 1):
        print(f"[{i}/{len(all_tasks)}] {task}", end=" ")
        if extract_task(task):
            extracted += 1
        else:
            print()
    
    # 验证数据完整性
    print(f"\n✔️  验证数据结构:")
    all_valid = True
    for task in sorted(all_tasks):
        checks = verify_task_structure(task)
        
        if checks.get("50_videos") and checks.get("50_instructions"):
            print(f"  ✓ {task}: {checks['video_count']} 视频, {checks['instruction_count']} 指令")
        else:
            print(f"  ✗ {task}: 数据不完整")
            all_valid = False
    
    # 最终统计
    print(f"\n" + "=" * 80)
    print(f"完成状态:")
    print(f"  ✓ 总任务数: {len(all_tasks)}/50")
    print(f"  ✓ 已解压: {extracted}")
    print(f"  {'✓' if all_valid else '✗'} 数据验证: {'通过' if all_valid else '失败'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
