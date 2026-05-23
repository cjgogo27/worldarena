#!/usr/bin/env python3
"""验证OD矩阵"""
import numpy as np
import pandas as pd

print('='*80)
print('OD矩阵验证')
print('='*80)

# 加载完整数据的OD矩阵
od_flow = np.load('output/od_flow_full.npy')
od_time = np.load('output/od_time_full.npy')

print(f'\n矩阵形状: {od_flow.shape}')
print(f'  时间槽数: {od_flow.shape[0]}')
print(f'  区域数: {od_flow.shape[1]} × {od_flow.shape[2]}')

# 统计
total_trips = od_flow.sum()
total_cells = od_flow.size
non_zero = (od_flow > 0).sum()
sparsity = 1 - (non_zero / total_cells)

print(f'\n流量统计:')
print(f'  总trips: {int(total_trips):,}')
print(f'  非零单元: {non_zero:,} / {total_cells:,}')
print(f'  稀疏度: {sparsity*100:.2f}%')
print(f'  平均每时间槽: {total_trips/od_flow.shape[0]:.1f} trips')

# 全天汇总
daily_flow = od_flow.sum(axis=0)
print(f'\n全天OD矩阵:')
print(f'  非零OD对: {(daily_flow > 0).sum()}')
print(f'  最大流量: {daily_flow.max():.0f} trips')
print(f'  平均流量 (非零OD): {daily_flow[daily_flow>0].mean():.2f} trips')

# Top 10 OD对
top_od = []
for o in range(29):
    for d in range(29):
        if daily_flow[o,d] > 0:
            top_od.append((o+1, d+1, daily_flow[o,d]))
top_od.sort(key=lambda x: x[2], reverse=True)

print(f'\nTop 10 OD对:')
for rank, (o, d, flow) in enumerate(top_od[:10], 1):
    print(f'  {rank:2d}. 区域{o:2d} → 区域{d:2d}: {int(flow):>6,} trips')

# 平均旅行时间统计
total_time_sum = np.zeros((29, 29))
total_count = np.zeros((29, 29))
for t in range(od_flow.shape[0]):
    mask = od_flow[t] > 0
    total_time_sum[mask] += od_time[t][mask] * od_flow[t][mask]
    total_count[mask] += od_flow[t][mask]

daily_avg_time = np.zeros_like(total_time_sum)
mask = total_count > 0
daily_avg_time[mask] = total_time_sum[mask] / total_count[mask]

print(f'\n旅行时间统计:')
print(f'  平均旅行时间: {daily_avg_time[mask].mean():.2f} 分钟')
print(f'  中位旅行时间: {np.median(daily_avg_time[mask]):.2f} 分钟')
print(f'  最短旅行时间: {daily_avg_time[mask].min():.2f} 分钟')
print(f'  最长旅行时间: {daily_avg_time[mask].max():.2f} 分钟')

print(f'\n文件验证:')
print(f'  ✓ od_flow_full.npy: {od_flow.nbytes/1024/1024:.2f} MB')
print(f'  ✓ od_time_full.npy: {od_time.nbytes/1024/1024:.2f} MB')

# 检查CSV文件
df_table = pd.read_csv('output/od_flow_table_full.csv')
print(f'  ✓ od_flow_table_full.csv: {len(df_table):,} 条记录')

df_trips = pd.read_csv('output/od_trips_full.csv')
print(f'  ✓ od_trips_full.csv: {len(df_trips):,} 条trips')

print('\n' + '='*80)
print('验证完成! OD矩阵构建成功')
print('='*80)
