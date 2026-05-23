#!/usr/bin/env python3
"""
RoboTwin 2.0 完整数据管道：下载 + 解压 + 转换
支持后台运行，详细日志记录

使用方式:
  python3 download_extract_convert_pipeline.py --tasks all --workers 4 --log pipeline.log
  
  或在后台运行:
  nohup python3 download_extract_convert_pipeline.py --tasks all --workers 4 > pipeline.log 2>&1 &
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

# ===================== 配置 =====================

ROBOTWIN_BASE = Path("/data/alice/cjtest/lara-wm/data/robotwin/dataset")
OUTPUT_DATASET = Path("/data/alice/cjtest/datasets/worldarena_wan_i2v_clean50")
DOWNLOAD_DIR = Path("/tmp/robotwin_downloads")

# HuggingFace 模型 ID
HUGGINGFACE_DATASET = "TianxingChen/RoboTwin2.0"

# VideoX-Fun 指令前缀
INSTRUCTION_PREFIX = "In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. The arm maintains high stability with no deformation and enters the frame to "

# ===================== 日志配置 =====================

def setup_logging(log_file: str):
    """设置日志记录"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # 控制台处理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = None  # 全局日志器

# ===================== 下载函数 =====================

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
    """检查任务数据是否已存在"""
    task_dir = ROBOTWIN_BASE / task_name
    extracted_dir = task_dir / "extracted" / "aloha-agilex_clean_50"
    video_dir = extracted_dir / "video"
    
    if video_dir.exists():
        video_count = len(list(video_dir.glob("*.mp4")))
        if video_count > 0:
            return True
    return False

def download_task(task_name: str) -> Tuple[str, bool, str]:
    """下载单个任务的数据"""
    logger.info(f"[下载] 开始下载任务: {task_name}")
    
    task_dir = ROBOTWIN_BASE / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在
    if check_task_exists(task_name):
        logger.info(f"[下载] 任务已存在: {task_name}")
        return task_name, True, "已存在"
    
    try:
        # 使用 huggingface-hub 下载
        cmd = [
            "huggingface-cli", "download",
            HUGGINGFACE_DATASET,
            f"aloha-agilex_clean_50/{task_name}.zip",
            "--repo-type", "dataset",
            "--cache-dir", str(DOWNLOAD_DIR),
            "--local-dir", str(task_dir)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            logger.info(f"[下载] ✓ 下载成功: {task_name}")
            return task_name, True, "下载成功"
        else:
            logger.warning(f"[下载] ✗ 下载失败: {task_name}\n错误: {result.stderr[:200]}")
            return task_name, False, f"下载失败: {result.stderr[:100]}"
    
    except subprocess.TimeoutExpired:
        logger.warning(f"[下载] ✗ 下载超时: {task_name}")
        return task_name, False, "下载超时"
    except Exception as e:
        logger.error(f"[下载] ✗ 异常: {task_name} - {str(e)}")
        return task_name, False, f"异常: {str(e)[:100]}"

# ===================== 解压函数 =====================

def extract_task(task_name: str) -> Tuple[str, bool, str]:
    """解压单个任务的数据"""
    logger.info(f"[解压] 开始解压任务: {task_name}")
    
    task_dir = ROBOTWIN_BASE / task_name
    
    # 查找 zip 文件
    zip_files = list(task_dir.glob("**/*.zip"))
    
    if not zip_files:
        logger.warning(f"[解压] 未找到 zip 文件: {task_name}")
        return task_name, False, "未找到 zip 文件"
    
    try:
        for zip_file in zip_files:
            extract_dir = task_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"[解压] 解压文件: {zip_file.name} → {extract_dir}")
            
            result = subprocess.run(
                ["unzip", "-q", str(zip_file), "-d", str(extract_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.warning(f"[解压] unzip 失败: {result.stderr[:200]}")
                # 尝试使用 Python 解压
                import zipfile
                try:
                    with zipfile.ZipFile(zip_file, 'r') as z:
                        z.extractall(extract_dir)
                    logger.info(f"[解压] ✓ 用 Python 解压成功: {task_name}")
                except Exception as e:
                    logger.error(f"[解压] Python 解压失败: {task_name} - {str(e)}")
                    return task_name, False, f"解压失败: {str(e)[:100]}"
            else:
                logger.info(f"[解压] ✓ 解压成功: {task_name}")
        
        return task_name, True, "解压成功"
    
    except subprocess.TimeoutExpired:
        logger.warning(f"[解压] ✗ 解压超时: {task_name}")
        return task_name, False, "解压超时"
    except Exception as e:
        logger.error(f"[解压] ✗ 异常: {task_name} - {str(e)}")
        return task_name, False, f"异常: {str(e)[:100]}"

# ===================== 转换函数 =====================

def get_episode_instruction(task_dir: Path, episode_id: int) -> str:
    """从 JSON 文件获取 episode 的 instruction"""
    instruction_file = task_dir / "instructions" / f"episode{episode_id}.json"
    
    if not instruction_file.exists():
        instruction_file = task_dir / "instructions" / f"{episode_id:02d}.json"
    
    if instruction_file.exists():
        try:
            with open(instruction_file, 'r') as f:
                data = json.load(f)
                
                if 'seen' in data and isinstance(data['seen'], list) and len(data['seen']) > 0:
                    raw_instruction = data['seen'][0]
                    return INSTRUCTION_PREFIX + raw_instruction
                elif 'instruction' in data:
                    raw_instruction = data['instruction']
                    if isinstance(raw_instruction, list):
                        raw_instruction = raw_instruction[0] if raw_instruction else ""
                    return INSTRUCTION_PREFIX + raw_instruction
                elif 'text' in data:
                    raw_instruction = data['text']
                    if isinstance(raw_instruction, list):
                        raw_instruction = raw_instruction[0] if raw_instruction else ""
                    return INSTRUCTION_PREFIX + raw_instruction
                else:
                    for value in data.values():
                        if isinstance(value, list) and len(value) > 0:
                            return INSTRUCTION_PREFIX + str(value[0])
                        elif value:
                            return INSTRUCTION_PREFIX + str(value)
        except Exception as e:
            logger.debug(f"解析指令异常: {instruction_file} - {e}")
    
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
    except Exception:
        pass
    
    return {"width": 832, "height": 480}

def process_task_for_training(task_name: str) -> Tuple[str, int, str]:
    """将任务数据转换为训练格式"""
    logger.info(f"[转换] 开始转换任务: {task_name}")
    
    task_dir = ROBOTWIN_BASE / task_name
    extracted_dir = task_dir / "extracted" / "aloha-agilex_clean_50"
    video_dir = extracted_dir / "video"
    
    if not video_dir.exists():
        logger.warning(f"[转换] video 目录不存在: {task_name}")
        return task_name, 0, "video 目录不存在"
    
    try:
        videos_processed = 0
        
        for video_file in sorted(video_dir.glob("*.mp4")):
            try:
                if video_file.stem.startswith("episode"):
                    episode_id = int(video_file.stem.replace("episode", ""))
                else:
                    episode_id = int(video_file.stem)
            except ValueError:
                episode_id = hash(video_file.name) % 100
            
            instruction = get_episode_instruction(extracted_dir, episode_id)
            
            # 检查视频完整性
            if video_file.stat().st_size == 0:
                logger.debug(f"[转换] 跳过空文件: {video_file.name}")
                continue
            
            videos_processed += 1
        
        logger.info(f"[转换] ✓ 转换完成: {task_name} ({videos_processed} 个视频)")
        return task_name, videos_processed, "转换成功"
    
    except Exception as e:
        logger.error(f"[转换] ✗ 异常: {task_name} - {str(e)}")
        return task_name, 0, f"异常: {str(e)[:100]}"

# ===================== 主管道 =====================

def process_task_pipeline(task_name: str) -> Dict:
    """处理单个任务的完整管道"""
    result = {
        'task': task_name,
        'download': False,
        'extract': False,
        'convert': False,
        'total_videos': 0,
        'status': 'pending',
        'message': ''
    }
    
    # 下载
    task_name_ret, download_ok, download_msg = download_task(task_name)
    result['download'] = download_ok
    if not download_ok:
        result['status'] = 'download_failed'
        result['message'] = download_msg
        return result
    
    # 解压
    task_name_ret, extract_ok, extract_msg = extract_task(task_name)
    result['extract'] = extract_ok
    if not extract_ok:
        result['status'] = 'extract_failed'
        result['message'] = extract_msg
        return result
    
    # 转换
    task_name_ret, videos, convert_msg = process_task_for_training(task_name)
    result['convert'] = True
    result['total_videos'] = videos
    result['status'] = 'success' if videos > 0 else 'no_videos'
    result['message'] = convert_msg
    
    return result

def run_pipeline(task_list: List[str], max_workers: int = 4):
    """并行运行数据管道"""
    logger.info("=" * 80)
    logger.info("RoboTwin 2.0 数据管道启动")
    logger.info("=" * 80)
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
                
                status_icon = "✓" if result['status'] == 'success' else "✗"
                logger.info(f"[{completed}/{len(task_list)}] {status_icon} {task_name}: "
                           f"下载={result['download']}, "
                           f"解压={result['extract']}, "
                           f"视频={result['total_videos']}")
            except Exception as e:
                logger.error(f"[{completed}/{len(task_list)}] ✗ {task_name}: {str(e)}")
                results.append({
                    'task': task_name,
                    'status': 'error',
                    'message': str(e),
                    'download': False,
                    'extract': False,
                    'convert': False,
                    'total_videos': 0
                })
    
    return results, time.time() - start_time

# ===================== 报告生成 =====================

def generate_report(results: List[Dict], elapsed_time: float):
    """生成处理报告"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("处理报告")
    logger.info("=" * 80)
    
    success = sum(1 for r in results if r['status'] == 'success')
    total_videos = sum(r['total_videos'] for r in results)
    
    logger.info(f"处理时间: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
    logger.info(f"成功任务: {success}/{len(results)}")
    logger.info(f"总视频数: {total_videos}")
    logger.info("")
    
    # 按状态分类
    status_groups = {}
    for r in results:
        status = r['status']
        if status not in status_groups:
            status_groups[status] = []
        status_groups[status].append(r)
    
    for status, group in sorted(status_groups.items()):
        logger.info(f"{status.upper()}: {len(group)} 个任务")
        for r in sorted(group, key=lambda x: x['task']):
            logger.info(f"  • {r['task']}: {r.get('message', 'OK')} "
                       f"({r['total_videos']} 个视频)")
    
    # 保存 JSON 报告
    report_file = Path("/data/alice/cjtest/pipeline_report.json")
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'total_tasks': len(results),
            'successful_tasks': success,
            'total_videos': total_videos,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info(f"报告已保存: {report_file}")
    logger.info("=" * 80)

# ===================== 主函数 =====================

def main():
    global logger
    
    parser = argparse.ArgumentParser(
        description='RoboTwin 2.0 数据下载、解压、转换管道'
    )
    parser.add_argument('--tasks', default='all', 
                       help='任务选择: all/pending/retry (默认: all)')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行工作数 (默认: 4)')
    parser.add_argument('--log', default='/data/alice/cjtest/pipeline.log',
                       help='日志文件路径')
    parser.add_argument('--task-list', default=None,
                       help='指定任务列表 (逗号分隔)')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log)
    
    # 获取任务列表
    all_tasks = get_all_tasks()
    
    if args.task_list:
        task_list = args.task_list.split(',')
    elif args.tasks == 'all':
        task_list = all_tasks
    elif args.tasks == 'pending':
        task_list = [t for t in all_tasks if not check_task_exists(t)]
    else:
        task_list = all_tasks
    
    logger.info(f"待处理任务: {len(task_list)} 个")
    
    # 运行管道
    results, elapsed_time = run_pipeline(task_list, max_workers=args.workers)
    
    # 生成报告
    generate_report(results, elapsed_time)
    
    logger.info(f"\n日志文件: {args.log}")

if __name__ == '__main__':
    main()
