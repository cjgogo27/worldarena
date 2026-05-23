#!/usr/bin/env python3
"""
并行解压所有 RoboTwin aloha-agilex_clean_50.zip 文件
"""

import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

DATASET_DIR = Path("/data/alice/cjtest/lara-wm/data/robotwin/dataset")
MAX_WORKERS = 4  # 并行解压 4 个任务


def extract_zip(task_dir: Path) -> Tuple[str, bool, str]:
    """解压单个 zip 文件"""
    task_name = task_dir.name
    zip_file = task_dir / "aloha-agilex_clean_50.zip"
    extracted_dir = task_dir / "extracted"
    
    if not zip_file.exists():
        return task_name, False, "zip 文件不存在"
    
    # 检查是否已解压
    if (extracted_dir / "aloha-agilex_clean_50").exists():
        return task_name, True, "已解压"
    
    try:
        extracted_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用 unzip 解压
        cmd = f"cd {extracted_dir} && unzip -q {zip_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            return task_name, True, f"✓ 解压成功"
        else:
            error_msg = result.stderr[:100] if result.stderr else "未知错误"
            return task_name, False, f"✗ 解压失败: {error_msg}"
    
    except subprocess.TimeoutExpired:
        return task_name, False, "✗ 解压超时"
    except Exception as e:
        return task_name, False, f"✗ 异常: {str(e)[:100]}"


def main():
    print("=" * 80)
    print("RoboTwin 数据并行解压器")
    print("=" * 80)
    
    # 获取所有任务目录
    task_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    
    print(f"\n📊 任务统计: {len(task_dirs)} 个任务")
    
    # 检查哪些需要解压
    need_extract = []
    already_extracted = 0
    missing_zip = 0
    
    for task_dir in task_dirs:
        zip_file = task_dir / "aloha-agilex_clean_50.zip"
        extracted_dir = task_dir / "extracted" / "aloha-agilex_clean_50"
        
        if extracted_dir.exists():
            already_extracted += 1
        elif zip_file.exists():
            need_extract.append(task_dir)
        else:
            missing_zip += 1
    
    print(f"  已解压: {already_extracted}")
    print(f"  需要解压: {len(need_extract)}")
    print(f"  缺少 zip: {missing_zip}")
    
    if len(need_extract) == 0:
        print("\n✓ 所有任务已解压完毕！")
        return True
    
    # 并行解压
    print(f"\n📦 开始解压 ({MAX_WORKERS} 并行线程):\n")
    
    extracted_count = 0
    failed_tasks = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_zip, task_dir): task_dir for task_dir in need_extract}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            task_name, success, message = future.result()
            
            status = "✓" if success else "✗"
            print(f"[{completed}/{len(need_extract)}] {task_name:30s} {status} {message}")
            
            if success:
                extracted_count += 1
            else:
                failed_tasks.append(task_name)
    
    # 最终统计
    print(f"\n" + "=" * 80)
    print(f"解压完成:")
    print(f"  成功: {extracted_count}/{len(need_extract)}")
    print(f"  失败: {len(failed_tasks)}")
    
    if failed_tasks:
        print(f"\n失败的任务:")
        for task in failed_tasks[:10]:
            print(f"  - {task}")
        if len(failed_tasks) > 10:
            print(f"  ... 还有 {len(failed_tasks) - 10} 个")
    
    print("=" * 80)
    
    return len(failed_tasks) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
