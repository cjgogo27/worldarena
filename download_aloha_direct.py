#!/usr/bin/env python3
"""
直接从 Hugging Face 数据集下载 aloha-agilex_clean_50 文件
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List

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
HF_URL_BASE = "https://huggingface.co/api/datasets/TianxingChen/RoboTwin2.0/tree/main"


def get_download_urls() -> dict:
    """获取所有任务的正确下载 URL"""
    print("📋 正在获取 HuggingFace 数据集信息...")
    
    urls = {}
    for task in ALL_TASKS:
        # 构建 HF 的直接下载链接
        # 格式: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/dataset/{task}/aloha-agilex_clean_50.zip
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/dataset/{task}/aloha-agilex_clean_50.zip"
        urls[task] = url
    
    return urls


def download_with_curl(url: str, output_file: Path, task_name: str) -> bool:
    """使用 curl 下载文件（更稳定）"""
    try:
        print(f"  ⏳ 下载 {task_name}... ", end="", flush=True)
        
        # 使用 curl 的重试机制和代理
        cmd = [
            "curl",
            "-L",  # 跟随重定向
            "--connect-timeout", "30",
            "--max-time", "3600",
            "-C", "-",  # 继续之前的下载
            "-o", str(output_file),
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3700)
        
        if result.returncode == 0 and output_file.exists():
            size_mb = output_file.stat().st_size / 1e6
            print(f"✓ ({size_mb:.0f} MB)")
            return True
        else:
            print(f"✗")
            if result.stderr:
                print(f"    错误: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ (超时)")
        return False
    except Exception as e:
        print(f"✗ ({e})")
        return False


def download_with_wget(url: str, output_file: Path, task_name: str) -> bool:
    """使用 wget 下载文件（备选方案）"""
    try:
        print(f"  ⏳ 下载 {task_name} (wget)... ", end="", flush=True)
        
        cmd = [
            "wget",
            "-O", str(output_file),
            "--timeout=30",
            "--tries=3",
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3700)
        
        if result.returncode == 0 and output_file.exists():
            size_mb = output_file.stat().st_size / 1e6
            print(f"✓ ({size_mb:.0f} MB)")
            return True
        else:
            print(f"✗")
            return False
            
    except Exception as e:
        print(f"✗ ({e})")
        return False


def has_correct_aloha_zip(task_dir: Path) -> bool:
    """检查是否有正确的 aloha-agilex_clean_50.zip"""
    zip_file = task_dir / "aloha-agilex_clean_50.zip"
    return zip_file.exists() and zip_file.stat().st_size > 100e6  # 至少 100MB


def main():
    print("=" * 80)
    print("RoboTwin 2.0 直接下载器（Aloha 版本）")
    print("=" * 80)
    
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    urls = get_download_urls()
    
    # 统计需要下载的任务
    need_download = []
    for task in ALL_TASKS:
        task_dir = DATASET_DIR / task
        if not has_correct_aloha_zip(task_dir):
            need_download.append(task)
    
    print(f"\n📊 状态:")
    print(f"  需要下载: {len(need_download)} 个任务")
    
    if not need_download:
        print(f"  ✓ 所有任务已下载完毕！")
        return
    
    print(f"\n📥 开始下载 ({len(need_download)} 个任务):\n")
    
    downloaded = 0
    for i, task in enumerate(need_download, 1):
        print(f"[{i}/{len(need_download)}] {task}")
        
        task_dir = DATASET_DIR / task
        task_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = task_dir / "aloha-agilex_clean_50.zip"
        url = urls[task]
        
        # 尝试 curl
        if download_with_curl(url, output_file, task):
            downloaded += 1
        # 如果 curl 失败，尝试 wget
        elif not output_file.exists() or output_file.stat().st_size < 100e6:
            if not download_with_wget(url, output_file, task):
                print(f"    ⚠ {task} 下载失败，跳过")
                if output_file.exists():
                    output_file.unlink()
            else:
                downloaded += 1
    
    print(f"\n" + "=" * 80)
    print(f"✓ 下载完成: {downloaded}/{len(need_download)} 个任务")
    print("=" * 80)


if __name__ == "__main__":
    main()
