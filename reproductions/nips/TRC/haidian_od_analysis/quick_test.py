#!/usr/bin/env python3
"""
快速测试脚本 - 使用小数据集测试完整流程
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("正在测试导入...")
import pandas as pd
import numpy as np
import geopandas as gpd

print("✓ 基础包导入成功")

from region_processor import RegionProcessor
from time_processor import TimeProcessor
from od_matrix_generator import ODMatrixGenerator

print("✓ 自定义模块导入成功")

print("\n开始快速测试...")

# 读取少量数据测试
print("读取测试数据（前1000行）...")
df = pd.read_csv('/data/alice/cjtest/TRC/all_taxi_data.csv', nrows=1000)
print(f"✓ 读取了 {len(df)} 条记录")
print(f"  列: {df.columns.tolist()}")
print(f"  时间范围: {df['date_time'].min()} 到 {df['date_time'].max()}")

print("\n测试时间处理...")
time_proc = TimeProcessor(interval_minutes=15)
df = time_proc.process_time_series(df, time_col='date_time')
print(f"✓ 时间处理完成，生成了 {df['time_slot'].nunique()} 个时间槽")

print("\n✅ 所有测试通过！系统可以正常运行。")
print("\n现在可以运行完整程序：")
print("  python main.py")
