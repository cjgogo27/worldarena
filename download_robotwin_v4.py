#!/usr/bin/env python3
"""
RoboTwin 2.0 数据下载脚本 v4 - 使用 raw 链接格式
正确的 URL: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/raw/main/dataset/{task}/aloha-agilex_clean_50.zip
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple
import logging
import time

# ===================== 配置 =====================

ROBOTWIN_BASE = Path("/data/alice/cjtest/lara-wm/data/robotwin/dataset")
HF_DATASET = "TianxingChen/RoboTwin2.0"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# ===================== 任务列表 =====================

ALL_TASKS = [
    'adjust_bottle', 'beat_block_hammer', 'bend_rope', 'click_bell', 'cut_meat',
    'dump_bin_bigbin', 'grab_ball', 'grab_roller', 'handover_block', 'insert_coin',
    'insert_pen', 'insert_screw', 'insert_stick', 'knock_door', 'knock_door_with_handle',
    'move_box', 'move_clothes', 'move_notebook', 'move_sandal', 'open_dishwasher',
    'open_drawer_horizontal', 'open_drawer_vertical', 'open_laptop', 'open_trash_can',
    'open_window', 'pick_apple', 'pick_cup', 'pick_glass', 'pick_hammer', 'pick_jewelry',
    'pick_knife', 'pick_orange', 'pick_small_ball', 'pick_towel', 'place_a2b_left',
    'place_a2b_right', 'place_cup', 'place_napkin', 'place_toothbrush', 'play_game',
    'press_stapler', 'push_button', 'push_door', 'push_drawer', 'push_pull_handle',
    'put_coin_in_box', 'put_food_in_bowl', 'put_item_on_shelf', 'put_item_on_table',
    'reach_cabinet', 'reach_table', 'roll_rope', 'stack_blocks_two', 'stack_blocks_three',
    'turn_paper', 'twist_bottle'
]

# ===================== 辅助函数 =====================

def task_exists(task_name: str) -> bool:
    """检查任务是否已下载"""
    task_dir = ROBOTWIN_BASE / task_name
    
    # 检查是否有 zip 文件
    if list(task_dir.glob("*.zip")):
        return True
    
    # 检查是否已解压
    extracted_dir = task_dir / "extracted" / "aloha-agilex_clean_50"
    if (extracted_dir / "video").exists():
        video_count = len(list((extracted_dir / "video").glob("*.mp4")))
        if video_count > 0:
            return True
    
    return False

def get_pending_tasks() -> List[str]:
    """获取还需要下载的任务列表"""
    pending = []
    for task in ALL_TASKS:
        if not task_exists(task):
            pending.append(task)
    return pending

# ===================== 下载函数 =====================

def download_task(task_name: str) -> Tuple[str, bool, str]:
    """下载单个任务"""
    logger.info(f"[下载] 开始: {task_name}")
    
    task_dir = ROBOTWIN_BASE / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试多个 URL 格式
    urls = [
        # 格式1: raw 链接
        f"https://huggingface.co/datasets/{HF_DATASET}/raw/main/dataset/{task_name}/aloha-agilex_clean_50.zip",
        # 格式2: 使用 cdn-lfs (备用)
        f"https://cdn-lfs.huggingface.co/datasets/{HF_DATASET}/main/dataset/{task_name}/aloha-agilex_clean_50.zip",
    ]
    
    zip_path = task_dir / "aloha-agilex_clean_50.zip"
    
    for url_idx, zip_url in enumerate(urls, 1):
        logger.debug(f"[下载] 尝试 URL {url_idx}: {zip_url}")
        
        try:
            # 使用 curl 下载
            cmd = [
                "curl", "-L",
                "-H", "Accept: application/octet-stream",
                "--max-redirs", "5",
                "--connect-timeout", "30",
                "--max-time", "600",
                "--retry", "2",
                "--retry-delay", "3",
                "-o", str(zip_path),
                zip_url
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=700
            )
            
            if result.returncode == 0 and zip_path.exists():
                size_mb = zip_path.stat().st_size / (1024 * 1024)
                
                if size_mb > 1:  # 有效文件 > 1MB
                    logger.info(f"[下载] ✓ 成功: {task_name} ({size_mb:.1f} MB)")
                    return task_name, True, f"成功 ({size_mb:.1f} MB)"
                else:
                    logger.debug(f"[下载] 文件太小: {size_mb:.1f} MB，尝试下一个 URL")
                    zip_path.unlink(missing_ok=True)
                    continue
            else:
                logger.debug(f"[下载] URL {url_idx} 失败，尝试下一个")
                if zip_path.exists():
                    zip_path.unlink(missing_ok=True)
                continue
        
        except subprocess.TimeoutExpired:
            logger.debug(f"[下载] URL {url_idx} 超时")
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
            continue
        except Exception as e:
            logger.debug(f"[下载] URL {url_idx} 异常: {str(e)[:100]}")
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
            continue
    
    logger.warning(f"[下载] ✗ 所有 URL 都失败: {task_name}")
    return task_name, False, "所有 URL 都失败"

# ===================== 解压函数 =====================

def extract_task(task_name: str) -> Tuple[str, bool, str]:
    """解压任务数据"""
    logger.info(f"[解压] 开始: {task_name}")
    
    task_dir = ROBOTWIN_BASE / task_name
    zip_file = task_dir / "aloha-agilex_clean_50.zip"
    
    if not zip_file.exists():
        logger.warning(f"[解压] 无 zip 文件: {task_name}")
        return task_name, False, "无 zip 文件"
    
    try:
        import zipfile
        extract_dir = task_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"[解压] 解压中: {zip_file.name}")
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(extract_dir)
        
        # 删除 zip 以节省空间
        zip_file.unlink()
        
        # 验证视频
        video_dir = extract_dir / "aloha-agilex_clean_50" / "video"
        if video_dir.exists():
            video_count = len(list(video_dir.glob("*.mp4")))
            if video_count > 0:
                logger.info(f"[解压] ✓ 成功: {task_name} ({video_count} 个视频)")
                return task_name, True, f"成功 ({video_count} 个视频)"
        
        logger.warning(f"[解压] 无视频文件: {task_name}")
        return task_name, False, "无视频文件"
    
    except zipfile.BadZipFile:
        logger.warning(f"[解压] zip 损坏: {task_name}")
        zip_file.unlink(missing_ok=True)
        return task_name, False, "zip 损坏"
    except Exception as e:
        logger.error(f"[解压] 异常: {task_name} - {str(e)}")
        zip_file.unlink(missing_ok=True)
        return task_name, False, f"异常: {str(e)[:100]}"

# ===================== 主程序 =====================

def main():
    logger.info("=" * 80)
    logger.info("RoboTwin 2.0 数据下载 v4 (使用 raw 链接格式)")
    logger.info("=" * 80)
    logger.info("")
    
    # 获取待下载任务
    pending_tasks = get_pending_tasks()
    
    logger.info(f"已下载: {len(ALL_TASKS) - len(pending_tasks)}/{len(ALL_TASKS)}")
    logger.info(f"待下载: {len(pending_tasks)}/{len(ALL_TASKS)}")
    
    if len(pending_tasks) == 0:
        logger.info("所有任务已下载！")
        return
    
    logger.info("")
    logger.info(f"开始下载 {len(pending_tasks)} 个待处理任务...")
    logger.info("")
    
    # 串行下载和解压 (避免网络冲突)
    success_count = 0
    failed_tasks = []
    
    for idx, task_name in enumerate(pending_tasks, 1):
        logger.info(f"[{idx}/{len(pending_tasks)}] 处理 {task_name}")
        
        # 下载
        _, download_ok, download_msg = download_task(task_name)
        if not download_ok:
            logger.warning(f"[{idx}/{len(pending_tasks)}] ✗ 下载失败: {download_msg}")
            failed_tasks.append(task_name)
            time.sleep(1)  # 等待 1 秒后继续
            continue
        
        # 解压
        _, extract_ok, extract_msg = extract_task(task_name)
        if extract_ok:
            logger.info(f"[{idx}/{len(pending_tasks)}] ✓ 完成: {extract_msg}")
            success_count += 1
        else:
            logger.warning(f"[{idx}/{len(pending_tasks)}] ✗ 解压失败: {extract_msg}")
            failed_tasks.append(task_name)
        
        logger.info("")
    
    # 统计
    logger.info("=" * 80)
    logger.info("下载完成统计")
    logger.info("=" * 80)
    logger.info(f"成功: {success_count}/{len(pending_tasks)}")
    logger.info(f"失败: {len(failed_tasks)}/{len(pending_tasks)}")
    
    if failed_tasks:
        logger.warning("\n失败的任务:")
        for task in failed_tasks:
            logger.warning(f"  • {task}")
    
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
