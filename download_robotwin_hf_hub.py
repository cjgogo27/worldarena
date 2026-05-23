#!/usr/bin/env python3
"""
RoboTwin 2.0 数据下载 - 使用 huggingface_hub 库
这是最可靠的方式，利用 HF 官方库来处理认证和路径问题
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse
import time
import json
import sys

try:
    from huggingface_hub import hf_hub_download, list_files_in_repo
except ImportError:
    print("正在安装 huggingface_hub...")
    import subprocess
    subprocess.run(["pip", "install", "huggingface_hub", "-q"])
    from huggingface_hub import hf_hub_download, list_files_in_repo

# ===================== 配置 =====================

ROBOTWIN_BASE = Path("/data/alice/cjtest/lara-wm/data/robotwin/dataset")
HF_DATASET_ID = "TianxingChen/RoboTwin2.0"

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

def download_task_hf_hub(task_name: str) -> Tuple[str, bool, str]:
    """使用 huggingface_hub 库下载单个任务的数据"""
    logger.info(f"[下载] 开始: {task_name}")
    
    # 检查是否已存在
    if check_task_exists(task_name):
        logger.info(f"[已存在] {task_name} - 已跳过")
        return task_name, True, "已存在"
    
    task_dir = ROBOTWIN_BASE / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 列出任务目录下的所有文件
        logger.debug(f"[下载] 列出 dataset/{task_name} 中的文件...")
        
        try:
            files = list_files_in_repo(
                HF_DATASET_ID,
                subfolder=f"dataset/{task_name}",
                repo_type="dataset"
            )
        except Exception as e:
            logger.warning(f"[下载] 无法访问 dataset/{task_name}: {str(e)[:100]}")
            return task_name, False, f"无法访问: {str(e)[:50]}"
        
        # 查找 zip 或相关文件
        zip_files = [f for f in files if f.endswith('.zip')]
        
        if not zip_files:
            logger.warning(f"[下载] 未找到 zip 文件: {task_name}")
            return task_name, False, "未找到 zip 文件"
        
        # 下载第一个 zip 文件
        zip_file_name = zip_files[0].split('/')[-1]
        logger.debug(f"[下载] 下载 {zip_file_name}")
        
        try:
            zip_path = hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=f"dataset/{task_name}/{zip_file_name}",
                repo_type="dataset",
                cache_dir=str(task_dir),
                local_dir=str(task_dir),
                local_dir_use_symlinks=False
            )
            
            # 检查文件大小
            size_mb = Path(zip_path).stat().st_size / (1024 * 1024)
            
            if size_mb > 0.5:
                logger.info(f"[下载] ✓ 成功 {task_name} ({size_mb:.1f} MB)")
                return task_name, True, f"下载成功 ({size_mb:.1f} MB)"
            else:
                logger.warning(f"[下载] ✗ 文件太小 {task_name} ({size_mb:.1f} MB)")
                return task_name, False, "文件太小"
        
        except Exception as e:
            logger.warning(f"[下载] ✗ 下载失败 {task_name}: {str(e)[:100]}")
            return task_name, False, f"下载失败: {str(e)[:50]}"
    
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
        import zipfile
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
    _, download_ok, download_msg = download_task_hf_hub(task_name)
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
    logger.info("RoboTwin 2.0 数据下载管道 (HF Hub 版)")
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
    report_file = Path("/data/alice/cjtest/hf_hub_download_report.json")
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
        description='RoboTwin 2.0 数据下载 (HF Hub 版) - 已下载的任务自动跳过'
    )
    parser.add_argument('--tasks', default='pending', 
                       help='任务选择: all/pending (默认: pending - 仅下载未下载的)')
    parser.add_argument('--workers', type=int, default=2,
                       help='并行工作数 (默认: 2，HF Hub 单账号限制)')
    parser.add_argument('--log', default='/data/alice/cjtest/hf_hub_download.log',
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
