#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 od_trips_full.csv 构建带日期的时序OD矩阵
正确维度: (天数×时间槽, 29区域, 29区域) 或 (天数, 时间槽, 29区域, 29区域)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def build_temporal_od_matrix(output_dir='output'):
    """构建带日期的OD矩阵"""
    
    print("="*80)
    print("构建带日期的时序OD矩阵")
    print("="*80)
    
    # 1. 加载trips数据
    print("\n【步骤1】加载trips数据")
    trips_df = pd.read_csv(f'{output_dir}/od_trips_full.csv')
    print(f"  总trips数: {len(trips_df):,}")
    
    # 2. 解析日期
    print("\n【步骤2】解析日期和时间槽")
    trips_df['start_time'] = pd.to_datetime(trips_df['start_time'])
    trips_df['date'] = trips_df['start_time'].dt.date
    
    # 统计时间范围
    start_date = trips_df['date'].min()
    end_date = trips_df['date'].max()
    num_days = (end_date - start_date).days + 1
    
    print(f"  时间范围: {start_date} 到 {end_date}")
    print(f"  天数: {num_days}")
    
    # 3. 生成完整的日期列表
    print("\n【步骤3】生成完整时间索引")
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    print(f"  日期列表: {len(date_range)} 天")
    
    # 4. 为每天创建day_index
    trips_df['day_index'] = (trips_df['start_time'].dt.date - start_date).apply(lambda x: x.days)
    
    print(f"  day_index范围: {trips_df['day_index'].min()} - {trips_df['day_index'].max()}")
    
    # 5. 构建4D矩阵 (天数, 时间槽, 区域, 区域)
    print("\n【步骤4】构建4D OD矩阵")
    num_slots = 96
    num_regions = 29
    
    od_flow_4d = np.zeros((num_days, num_slots, num_regions, num_regions), dtype=np.float32)
    od_time_sum_4d = np.zeros((num_days, num_slots, num_regions, num_regions), dtype=np.float32)
    od_count_4d = np.zeros((num_days, num_slots, num_regions, num_regions), dtype=np.int32)
    
    print(f"  矩阵维度: ({num_days}天, {num_slots}槽, {num_regions}区域, {num_regions}区域)")
    print("  填充数据...")
    
    for _, trip in trips_df.iterrows():
        d = int(trip['day_index'])
        t = int(trip['time_slot'])
        o = int(trip['origin']) - 1
        dest = int(trip['dest']) - 1
        dur = trip['duration']
        
        if 0 <= d < num_days and 0 <= t < num_slots and 0 <= o < num_regions and 0 <= dest < num_regions:
            od_flow_4d[d, t, o, dest] += 1
            od_time_sum_4d[d, t, o, dest] += dur
            od_count_4d[d, t, o, dest] += 1
    
    # 计算平均时间
    od_time_4d = np.zeros_like(od_time_sum_4d)
    mask = od_count_4d > 0
    od_time_4d[mask] = od_time_sum_4d[mask] / od_count_4d[mask]
    
    # 6. 统计
    print("\n【步骤5】统计信息")
    total_trips = od_flow_4d.sum()
    non_zero = (od_flow_4d > 0).sum()
    print(f"  总trips: {int(total_trips):,}")
    print(f"  非零单元: {non_zero:,} / {od_flow_4d.size:,}")
    print(f"  稀疏度: {(1 - non_zero / od_flow_4d.size)*100:.2f}%")
    
    # 每天的流量
    print(f"\n  每天流量分布:")
    for d in range(num_days):
        daily_flow = od_flow_4d[d].sum()
        date = start_date + timedelta(days=d)
        print(f"    {date}: {int(daily_flow):,} trips")
    
    # 7. 保存numpy格式
    print("\n【步骤6】保存数据")
    np.save(f'{output_dir}/od_flow_temporal.npy', od_flow_4d)
    np.save(f'{output_dir}/od_time_temporal.npy', od_time_4d)
    print(f"  ✓ 保存4D矩阵: od_flow_temporal.npy, od_time_temporal.npy")
    
    # 8. 保存CSV格式（带日期）
    print("  生成CSV表格...")
    od_records = []
    for d in range(num_days):
        date = start_date + timedelta(days=d)
        for t in range(num_slots):
            for o in range(num_regions):
                for dest in range(num_regions):
                    flow = od_flow_4d[d, t, o, dest]
                    if flow > 0:
                        od_records.append({
                            'date': date,
                            'day_index': d,
                            'time_slot': t,
                            'hour': t // 4,
                            'minute': (t % 4) * 15,
                            'origin': o + 1,
                            'dest': dest + 1,
                            'flow': flow,
                            'avg_time': od_time_4d[d, t, o, dest]
                        })
    
    od_temporal_df = pd.DataFrame(od_records)
    od_temporal_df.to_csv(f'{output_dir}/od_flow_temporal.csv', index=False, encoding='utf-8-sig')
    print(f"  ✓ 保存CSV: od_flow_temporal.csv ({len(od_temporal_df):,} 条记录)")
    
    # 9. 对比验证
    print("\n【步骤7】验证数据一致性")
    print(f"  原始trips总数: {len(trips_df):,}")
    print(f"  4D矩阵总trips: {int(od_flow_4d.sum()):,}")
    print(f"  CSV记录总flow: {int(od_temporal_df['flow'].sum()):,}")
    
    # 10. 生成3D版本（时间展平）
    print("\n【步骤8】生成3D版本（时间展平）")
    od_flow_3d = od_flow_4d.reshape(num_days * num_slots, num_regions, num_regions)
    od_time_3d = od_time_4d.reshape(num_days * num_slots, num_regions, num_regions)
    
    np.save(f'{output_dir}/od_flow_temporal_3d.npy', od_flow_3d)
    np.save(f'{output_dir}/od_time_temporal_3d.npy', od_time_3d)
    print(f"  ✓ 保存3D矩阵: ({num_days * num_slots}时间点, {num_regions}, {num_regions})")
    
    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print("\n可用文件:")
    print(f"  1. od_flow_temporal.npy - 4D数组 ({num_days}, {num_slots}, {num_regions}, {num_regions})")
    print(f"  2. od_flow_temporal_3d.npy - 3D数组 ({num_days * num_slots}, {num_regions}, {num_regions})")
    print(f"  3. od_flow_temporal.csv - 带日期的表格 ({len(od_temporal_df):,} 条记录)")
    print()
    print("现在可以基于这些数据进行时间序列预测！")
    
    return od_flow_4d, od_time_4d, od_temporal_df

if __name__ == '__main__':
    build_temporal_od_matrix()
