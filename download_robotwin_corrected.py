#!/usr/bin/env python3
"""
RoboTwin 2.0 数据下载 - 修正版
使用正确的 HuggingFace 链接: dataset/[task_name]/...

下载流程: 下载 → 解压 → 统计
已下载的任务会自动跳过
"""

import os
import json
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse
import time
import sys
import zipfile

# ===================== 配置 =====================

ROBOTWIN_BASE = Path("/data/alice/cjtest/lara-wm/data/robotwin/dataset")
HF_DATASET_ID = "TianxingChen/RoboTwin2.0"
HF_BASE_URL = "https://huggingface.co/datasets/{}/resolve/main".format(HF_DATASET_ID)

# VideoX-Fun 指令前缀
INSTRUCTION_PREFIX = "In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. The arm maintains high stability with no deformation and enters the frame to "

# ===================== 日志配置 =====================

def setup_logging(log_file: str):
    """设置日志记录"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 清空现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # 控制台处理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = None

# ===================== 任务列表 =====================

def get_all_tasks() -> List[str]:
    """获取所有 50 个任务名称"""
    tasks = [
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
    return tasks

def check_task_exists(task_name: str) -> bool:
    """检查任务数据是否已存在且有效"""
    task_dir = ROBOTWIN_BASE / task_name
    extracted_dir = task_dir / "extracted" / "aloha-agilex_clean_50"
    video_dir = extracted_dir / "video"
    
    if video_dir.exists():
        video_count = len(list(video_dir.glob("*.mp4")))
        valid_count = sum(1 for f in video_dir.glob("*.mp4") if f.stat().st_size > 0)
        if valid_count > 10:  # 至少有 10 个有效视频
            return True
    return False

# ===================== 下载函数 =====================

def download_task_with_curl(task_name: str) -> Tuple[str, bool, str]:
    """使用 curl 下载单个任务的数据"""
    logger.info(f"[下载] 开始: {task_name}")
    
    # 检查是否已存在
    if check_task_exists(task_name):
        logger.info(f"[已存在] {task_name} - 已跳过")
        return task_name, True, "已存在"
    
    task_dir = ROBOTWIN_BASE / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否有 zip 文件已下载但未解压
    existing_zips = list(task_dir.glob("**/*.zip"))
    if existing_zips:
        logger.info(f"[跳过] {task_name} - zip 文件已存在")
        return task_name, True, "zip 已存在"
    
    try:
        # 尝试多种可能的 URL 格式
        url_patterns = [
            f"{HF_BASE_URL}/dataset/{task_name}/aloha-agilex_clean_50.zip",
            f"{HF_BASE_URL}/aloha-agilex_clean_50/{task_name}.zip",
            f"{HF_BASE_URL}/dataset/{task_name}.zip",
        ]
        
        zip_path = task_dir / f"{task_name}.zip"
        download_success = False
        last_error = ""
        
        for url in url_patterns:
            logger.debug(f"[下载] 尝试 URL: {url}")
            
            cmd = [
                "curl", "-L",
                "--max-redirs", "5",
                "--connect-timeout", "30",
                "--max-time", "600",
                "--retry", "2",
                "--retry-delay", "3",
                "-o", str(zip_path),
                url
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=650
            )
            
            if result.returncode == 0 and zip_path.exists():
                size_mb = zip_path.stat().st_size / (1024 * 1024)
                if size_mb > 0.5:  # 有效 zip 通常 > 0.5MB
                    logger.info(f"[下载] ✓ 成功 {task_name} ({size_mb:.1f} MB)")
                    download_success = True
                    return task_name, True, f"下载成功 ({size_mb:.1f} MB)"
                else:
                    logger.debug(f"[下载] 文件太小 ({size_mb:.1f} MB)，继续尝试...")
                    zip_path.unlink()
            else:
                last_error = result.stderr[:200] if result.stderr else "无响应"
        
        if not download_success:
            logger.warning(f"[下载] ✗ 所有 URL 都失败 {task_name}")
            if zip_path.exists():
                zip_path.unlink()
            return task_name, False, f"下载失败: {last_error[:100]}"
    
    except subprocess.TimeoutExpired:
        logger.warning(f"[下载] ✗ 超时 {task_name}")
        return task_name, False, "下载超时"
    except Exception as e:
        logger.error(f"[下载] ✗ 异常 {task_name}: {str(e)}")
        return task_name, False, f"异常: {str(e)[:100]}"

def extract_task(task_name: str) -> Tuple[str, bool, str]:
    """解压单个任务的数据"""
    logger.info(f"[解压] 开始: {task_name}")
    
    task_dir = ROBOTWIN_BASE / task_name
    
    # 查找 zip 文件
    zip_files = list(task_dir.glob("**/*.zip"))
    
    if not zip_files:
        logger.warning(f"[解压] 无 zip 文件: {task_name}")
        return task_name, False, "无 zip 文件"
    
    try:
        for zip_file in zip_files:
            extract_dir = task_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"[解压] 解压文件: {zip_file.name}")
            
            try:
                with zipfile.ZipFile(zip_file, 'r') as z:
                    z.extractall(extract_dir)
                
                # 删除 zip 以节省空间
                logger.debug(f"[解压] 删除 zip 文件")
                zip_file.unlink()
                
                logger.info(f"[解压] ✓ 成功 {task_name}")
                return task_name, True, "解压成功"
            except zipfile.BadZipFile:
                logger.warning(f"[解压] 损坏的 zip {task_name}")
                zip_file.unlink()
                return task_name, False, "zip 损坏"
    
    except Exception as e:
        logger.error(f"[解压] ✗ 异常 {task_name}: {str(e)}")
        return task_name, False, f"异常: {str(e)[:100]}"

def count_videos(task_name: str) -> Tuple[str, int, str]:
    """统计任务中的视频数"""
    logger.info(f"[统计] 开始: {task_name}")
    
    task_dir = ROBOTWIN_BASE / task_name
    extracted_dir = task_dir / "extracted" / "aloha-agilex_clean_50"
    video_dir = extracted_dir / "video"
    
    if not video_dir.exists():
        logger.warning(f"[统计] video 目录不存在: {task_name}")
        return task_name, 0, "video 目录不存在"
    
    try:
        valid_count = sum(1 for f in video_dir.glob("*.mp4") if f.stat().st_size > 0)
        logger.info(f"[统计] ✓ {task_name}: {valid_count} 个有效视频")
        return task_name, valid_count, f"{valid_count} 个视频"
    
    except Exception as e:
        logger.error(f"[统计] ✗ 异常 {task_name}: {str(e)}")
        return task_name, 0, f"异常: {str(e)[:100]}"

# ===================== 主管道 =====================

def process_task_pipeline(task_name: str) -> Dict:
    """处理单个任务的完整管道"""
    result = {
        'task': task_name,
        'download': False,
        'extract': False,
        'count': 0,
        'status': 'pending',
        'message': ''
    }
    
    # 下载
    _, download_ok, download_msg = download_task_with_curl(task_name)
    result['download'] = download_ok
    if not download_ok:
        result['status'] = 'download_failed'
        result['message'] = download_msg
        return result
    
    # 检查是否是已存在的任务
    if download_msg == "已存在":
        result['extract'] = True
        result['status'] = 'already_exists'
        result['message'] = "已存在"
        # 统计现有视频
        _, count, _ = count_videos(task_name)
        result['count'] = count
        return result
    
    # 解压
    _, extract_ok, extract_msg = extract_task(task_name)
    result['extract'] = extract_ok
    if not extract_ok:
        result['status'] = 'extract_failed'
        result['message'] = extract_msg
        return result
    
    # 统计
    _, count, count_msg = count_videos(task_name)
    result['count'] = count
    result['status'] = 'success' if count > 10 else 'incomplete'
    result['message'] = count_msg
    
    return result

def run_pipeline(task_list: List[str], max_workers: int = 4):
    """并行运行数据管道"""
    logger.info("=" * 90)
    logger.info("RoboTwin 2.0 数据下载管道 (修正版)")
    logger.info("=" * 90)
    logger.info(f"待处理任务数: {len(task_list)}")
    logger.info(f"并行工作数: {max_workers}")
    logger.info("")
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_task_pipeline, task): task 
            for task in task_list
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            task_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                status_icon = "✓" if result['status'] in ['success', 'already_exists'] else "✗"
                logger.info(f"[{completed:2d}/{len(task_list):2d}] {status_icon} "
                           f"{task_name:30s} | {result['count']:3d} 个视频 | "
                           f"{result.get('message', 'OK')[:40]}")
            except Exception as e:
                logger.error(f"[{completed}/{len(task_list)}] ✗ {task_name}: {str(e)}")
                results.append({
                    'task': task_name,
                    'status': 'error',
                    'message': str(e),
                    'download': False,
                    'extract': False,
                    'count': 0
                })
    
    return results, time.time() - start_time

# ===================== 报告生成 =====================

def generate_report(results: List[Dict], elapsed_time: float):
    """生成处理报告"""
    logger.info("")
    logger.info("=" * 90)
    logger.info("下载完成报告")
    logger.info("=" * 90)
    
    success = sum(1 for r in results if r['status'] == 'success')
    already = sum(1 for r in results if r['status'] == 'already_exists')
    failed = sum(1 for r in results if r['status'] in ['download_failed', 'error'])
    total_videos = sum(r['count'] for r in results)
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    logger.info(f"处理时间: {hours}h {minutes}m {seconds}s")
    logger.info(f"新下载: {success} 个任务")
    logger.info(f"已存在: {already} 个任务")
    logger.info(f"失败: {failed} 个任务")
    logger.info(f"总视频数: {total_videos}")
    logger.info("")
    
    # 显示新下载的任务
    if success > 0:
        logger.info("✓ 新下载的任务:")
        for r in sorted([r for r in results if r['status'] == 'success'], 
                       key=lambda x: x['task']):
            logger.info(f"  • {r['task']:30s} | {r['count']:3d} 个视频")
    
    if already > 0:
        logger.info(f"\n✓ 已存在的任务: {already} 个")
    
    if failed > 0:
        logger.info(f"\n✗ 下载失败的任务:")
        for r in sorted([r for r in results if r['status'] in ['download_failed', 'error']], 
                       key=lambda x: x['task']):
            logger.info(f"  • {r['task']:30s} | {r.get('message', 'N/A')[:50]}")
    
    # 保存 JSON 报告
    report_file = Path("/data/alice/cjtest/download_report.json")
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'total_tasks': len(results),
            'newly_downloaded': success,
            'already_existed': already,
            'failed': failed,
            'total_videos': total_videos,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info(f"详细报告: {report_file}")
    logger.info("=" * 90)

# ===================== 主函数 =====================

def main():
    global logger
    
    parser = argparse.ArgumentParser(
        description='RoboTwin 2.0 数据下载 (修正版) - 已下载的任务自动跳过'
    )
    parser.add_argument('--tasks', default='pending', 
                       help='任务选择: all/pending (默认: pending - 仅下载未下载的)')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行工作数 (默认: 4)')
    parser.add_argument('--log', default='/data/alice/cjtest/download.log',
                       help='日志文件路径')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log)
    
    # 获取任务列表
    all_tasks = get_all_tasks()
    
    if args.tasks == 'all':
        task_list = all_tasks
    else:  # pending
        task_list = [t for t in all_tasks if not check_task_exists(t)]
    
    logger.info(f"待处理任务: {len(task_list)} 个 (共 {len(all_tasks)} 个)")
    
    if len(task_list) == 0:
        logger.info("所有任务已完成！")
        return
    
    # 运行管道
    results, elapsed_time = run_pipeline(task_list, max_workers=args.workers)
    
    # 生成报告
    generate_report(results, elapsed_time)
    
    logger.info(f"\n日志文件: {args.log}")

if __name__ == '__main__':
    main()
