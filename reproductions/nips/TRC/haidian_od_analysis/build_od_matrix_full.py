#!/usr/bin/env python3
"""
完整数据OD矩阵构建脚本
从原始数据重新处理，构建完整的OD矩阵
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from datetime import datetime
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

from region_processor import RegionProcessor
from time_processor import TimeProcessor

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("完整数据OD矩阵构建程序")
print("=" * 80)

# 配置
CONFIG = {
    'shapefile_path': '/data/alice/cjtest/TRC/海淀区边界_110108_Shapefile_(poi86.com)/110108.shp',
    'region_mapping_path': 'config/region_mapping.csv',
    'trajectory_path': '/data/alice/cjtest/TRC/all_taxi_data.csv',
    'output_dir': 'output',
    'num_regions': 29,
    'interval_minutes': 15,
    'time_threshold': 30  # 停留时间阈值
}

# ==================== 步骤1: 加载区域数据 ====================
print(f"\n{'='*80}")
print("步骤 1: 加载区域边界")
print(f"{'='*80}")

region_proc = RegionProcessor(
    CONFIG['shapefile_path'],
    CONFIG['region_mapping_path']
)
region_proc.load_regions()
bounds = region_proc.regions_gdf.total_bounds

print(f"海淀区边界: 经度 {bounds[0]:.6f}-{bounds[2]:.6f}, 纬度 {bounds[1]:.6f}-{bounds[3]:.6f}")

# ==================== 步骤2: 加载并清洗轨迹数据 ====================
print(f"\n{'='*80}")
print("步骤 2: 加载轨迹数据")
print(f"{'='*80}")

print(f"读取: {CONFIG['trajectory_path']}")
df = pd.read_csv(CONFIG['trajectory_path'])
print(f"原始数据: {len(df)} 条")

# 清洗
df = df.dropna(subset=['taxi_id', 'date_time', 'longitude', 'latitude'])
df['date_time'] = pd.to_datetime(df['date_time'])

# 坐标过滤（使用实际边界 + 缓冲）
buffer = 0.01
df = df[
    (df['longitude'].between(bounds[0]-buffer, bounds[2]+buffer)) &
    (df['latitude'].between(bounds[1]-buffer, bounds[3]+buffer))
].copy()

# 去重（移除相同出租车在相同时间相同位置的重复记录）
len_before_dedup = len(df)
df = df.drop_duplicates(subset=['taxi_id', 'date_time', 'longitude', 'latitude'])
dup_removed = len_before_dedup - len(df)
print(f"清洗后: {len(df)} 条 (去重移除 {dup_removed} 条, {dup_removed/len_before_dedup*100:.2f}%)")

# ==================== 步骤3: 空间映射 ====================
print(f"\n{'='*80}")
print("步骤 3: 空间映射")
print(f"{'='*80}")

df = region_proc.batch_points_to_regions(df, lon_col='longitude', lat_col='latitude')
df = df[df['region_id'].notna()].copy()
print(f"映射成功: {len(df)} 条，覆盖 {df['region_id'].nunique()} 个区域")

# ==================== 步骤4: 时间映射 ====================
print(f"\n{'='*80}")
print("步骤 4: 时间映射")
print(f"{'='*80}")

time_proc = TimeProcessor(interval_minutes=CONFIG['interval_minutes'])
df = time_proc.process_time_series(df, time_col='date_time')
print(f"时间映射完成，时间槽范围: {df['time_slot'].min()}-{df['time_slot'].max()}")

# ==================== 步骤5: 提取trips并构建OD矩阵 ====================
print(f"\n{'='*80}")
print("步骤 5: 提取trips")
print(f"{'='*80}")

df = df.sort_values(['taxi_id', 'date_time']).reset_index(drop=True)

trips = []
vehicles = df.groupby('taxi_id')
total_vehicles = len(vehicles)
print(f"处理 {total_vehicles} 辆车...")

start_time = datetime.now()

for idx, (vid, group) in enumerate(tqdm(vehicles, desc="提取trips", mininterval=1.0)):
    if idx > 0 and idx % 100 == 0:
        elapsed = (datetime.now() - start_time).total_seconds()
        avg_time = elapsed / idx
        eta = avg_time * (total_vehicles - idx) / 60
        print(f"\r  [{idx}/{total_vehicles}] trips: {len(trips)}, ETA: {eta:.1f}min", 
              end='', flush=True)
    
    group = group.reset_index(drop=True)
    if len(group) < 2:
        continue
    
    # 计算差值
    group['time_diff'] = group['date_time'].diff().dt.total_seconds() / 60
    group['region_changed'] = group['region_id'] != group['region_id'].shift(1)
    
    current_trip = None
    for i, row in group.iterrows():
        if i == 0:
            current_trip = {
                'origin': int(row['region_id']),
                'start_time': row['date_time'],
                'start_slot': int(row['time_slot'])
            }
        elif (row['region_changed'] or row['time_diff'] > CONFIG['time_threshold']):
            if current_trip and row['region_id'] != current_trip['origin']:
                duration = (row['date_time'] - current_trip['start_time']).total_seconds() / 60
                if duration > 0 and duration < 300:  # 过滤异常时长
                    trips.append({
                        'vehicle_id': vid,
                        'origin': current_trip['origin'],
                        'dest': int(row['region_id']),
                        'start_time': current_trip['start_time'],
                        'end_time': row['date_time'],
                        'duration': duration,
                        'time_slot': current_trip['start_slot']
                    })
            
            current_trip = {
                'origin': int(row['region_id']),
                'start_time': row['date_time'],
                'start_slot': int(row['time_slot'])
            }

print(f"\n\n提取完成: {len(trips)} 条trips")

trips_df = pd.DataFrame(trips)
trips_df.to_csv(f"{CONFIG['output_dir']}/od_trips_full.csv", index=False, encoding='utf-8-sig')
print(f"✓ 保存trips: od_trips_full.csv")

# ==================== 步骤6: 构建OD矩阵 ====================
print(f"\n{'='*80}")
print("步骤 6: 构建OD矩阵")
print(f"{'='*80}")

num_slots = 96
num_regions = CONFIG['num_regions']

od_flow = np.zeros((num_slots, num_regions, num_regions), dtype=np.float32)
od_time_sum = np.zeros((num_slots, num_regions, num_regions), dtype=np.float32)
od_count = np.zeros((num_slots, num_regions, num_regions), dtype=np.int32)

print(f"维度: {num_slots} slots × {num_regions} regions × {num_regions} regions")

for _, trip in tqdm(trips_df.iterrows(), total=len(trips_df), desc="构建矩阵"):
    t = int(trip['time_slot'])
    o = int(trip['origin']) - 1
    d = int(trip['dest']) - 1
    dur = trip['duration']
    
    if 0 <= t < num_slots and 0 <= o < num_regions and 0 <= d < num_regions:
        od_flow[t, o, d] += 1
        od_time_sum[t, o, d] += dur
        od_count[t, o, d] += 1

od_time = np.zeros_like(od_time_sum)
mask = od_count > 0
od_time[mask] = od_time_sum[mask] / od_count[mask]

# 统计
total_trips = od_flow.sum()
non_zero = (od_flow > 0).sum()
sparsity = 1 - (non_zero / od_flow.size)

print(f"\nOD矩阵统计:")
print(f"  总trips: {int(total_trips)}")
print(f"  非零单元: {non_zero} / {od_flow.size}")
print(f"  稀疏度: {sparsity*100:.2f}%")

# 保存
np.save(f"{CONFIG['output_dir']}/od_flow_full.npy", od_flow)
np.save(f"{CONFIG['output_dir']}/od_time_full.npy", od_time)
print(f"✓ 保存numpy矩阵")

# 保存CSV格式
od_records = []
for t in range(num_slots):
    for o in range(num_regions):
        for d in range(num_regions):
            if od_flow[t, o, d] > 0:
                od_records.append({
                    'time_slot': t,
                    'hour': t // 4,
                    'minute': (t % 4) * 15,
                    'origin': o + 1,
                    'dest': d + 1,
                    'flow': int(od_flow[t, o, d]),
                    'avg_time': od_time[t, o, d]
                })

od_table = pd.DataFrame(od_records)
od_table.to_csv(f"{CONFIG['output_dir']}/od_flow_table_full.csv", index=False, encoding='utf-8-sig')
print(f"✓ 保存OD表: od_flow_table_full.csv ({len(od_table)} 条)")

# ==================== 步骤7: 可视化 ====================
print(f"\n{'='*80}")
print("步骤 7: 生成可视化")
print(f"{'='*80}")

os.makedirs(f"{CONFIG['output_dir']}/od_visualizations", exist_ok=True)

# 全天汇总OD流量
daily_flow = od_flow.sum(axis=0)

# 使用对数归一化突出显示区域差异
plt.figure(figsize=(16, 14))
flow_vals = daily_flow[daily_flow > 0]
vmin, vmax = np.percentile(flow_vals, [5, 95])  # 使用5-95百分位数
sns.heatmap(daily_flow, cmap='YlOrRd', 
           norm=LogNorm(vmin=max(1, vmin), vmax=vmax),
           cbar_kws={'label': 'Total Trips (log scale)'},
           xticklabels=range(1, num_regions+1),
           yticklabels=range(1, num_regions+1),
           fmt='g')
plt.title('Daily OD Flow Matrix (Complete Data)', fontsize=18, fontweight='bold')
plt.xlabel('Destination Region', fontsize=14)
plt.ylabel('Origin Region', fontsize=14)
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/od_visualizations/od_daily_flow_full.png", dpi=300)
plt.close()
print("✓ 全天OD流量热力图 (对数色阶)")

# 全天平均旅行时间
total_time_sum = np.zeros((num_regions, num_regions))
total_count = np.zeros((num_regions, num_regions))
for t in range(num_slots):
    mask = od_flow[t] > 0
    total_time_sum[mask] += od_time[t][mask] * od_flow[t][mask]
    total_count[mask] += od_flow[t][mask]

daily_avg_time = np.zeros_like(total_time_sum)
mask = total_count > 0
daily_avg_time[mask] = total_time_sum[mask] / total_count[mask]

# 使用百分位数截断突出显示时间差异
plt.figure(figsize=(16, 14))
time_vals = daily_avg_time[daily_avg_time > 0]
if len(time_vals) > 0:
    vmin_t, vmax_t = np.percentile(time_vals, [5, 95])
else:
    vmin_t, vmax_t = 0, 60
sns.heatmap(daily_avg_time, cmap='viridis', 
           vmin=vmin_t, vmax=vmax_t,
           cbar_kws={'label': 'Avg Time (min)'},
           xticklabels=range(1, num_regions+1),
           yticklabels=range(1, num_regions+1),
           fmt='.1f', annot=False)
plt.title('Daily Average Travel Time Matrix', fontsize=18, fontweight='bold')
plt.xlabel('Destination Region', fontsize=14)
plt.ylabel('Origin Region', fontsize=14)
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/od_visualizations/od_daily_time_full.png", dpi=300)
plt.close()
print("✓ 全天平均时间热力图 (5-95百分位数色阶)")

# 时序分析
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 每小时trip数
hourly = [od_flow[t].sum() for t in range(num_slots)]
hours = [t/4 for t in range(num_slots)]
axes[0,0].plot(hours, hourly, linewidth=2)
axes[0,0].set_xlabel('Hour', fontsize=12)
axes[0,0].set_ylabel('Trips', fontsize=12)
axes[0,0].set_title('Trip Volume by Hour', fontsize=14, fontweight='bold')
axes[0,0].grid(True, alpha=0.3)

# 稀疏度
sparsity_slots = [1 - (od_flow[t] > 0).sum() / (num_regions**2) for t in range(num_slots)]
axes[0,1].plot(hours, [s*100 for s in sparsity_slots], linewidth=2, color='red')
axes[0,1].set_xlabel('Hour', fontsize=12)
axes[0,1].set_ylabel('Sparsity (%)', fontsize=12)
axes[0,1].set_title('Matrix Sparsity by Hour', fontsize=14, fontweight='bold')
axes[0,1].grid(True, alpha=0.3)

# 流量分布
flow_vals = daily_flow[daily_flow > 0].flatten()
axes[1,0].hist(flow_vals, bins=50, edgecolor='black', alpha=0.7)
axes[1,0].set_xlabel('Trips (OD pair)', fontsize=12)
axes[1,0].set_ylabel('Frequency', fontsize=12)
axes[1,0].set_title('Flow Distribution', fontsize=14, fontweight='bold')
axes[1,0].set_yscale('log')

# 时间分布
time_vals = daily_avg_time[daily_avg_time > 0].flatten()
axes[1,1].hist(time_vals, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1,1].set_xlabel('Avg Travel Time (min)', fontsize=12)
axes[1,1].set_ylabel('Frequency', fontsize=12)
axes[1,1].set_title('Travel Time Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/od_visualizations/od_analysis_full.png", dpi=300)
plt.close()
print("✓ 综合分析图")

print(f"\n{'='*80}")
print("完整OD矩阵构建完成!")
print(f"{'='*80}")
print(f"\n输出文件:")
print(f"  - od_trips_full.csv: 原始trips记录")
print(f"  - od_flow_full.npy: 流量矩阵 (numpy)")
print(f"  - od_time_full.npy: 时间矩阵 (numpy)")
print(f"  - od_flow_table_full.csv: OD流量表")
print(f"  - od_visualizations/: 可视化图表")
