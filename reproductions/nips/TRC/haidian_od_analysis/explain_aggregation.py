#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解释OD矩阵聚合过程和日期信息丢失原因"""

import pandas as pd
import numpy as np

print("=" * 80)
print("关键问题：为什么日期信息消失了？")
print("=" * 80)
print()

# 读取trips数据（保留日期）
trips = pd.read_csv('/data/alice/cjtest/TRC/haidian_od_analysis/output/od_trips_full.csv')
print(f"【阶段1】od_trips_full.csv: {len(trips):,} 条trips，包含完整日期")
print(f"  时间范围: {trips['start_time'].min()} 到 {trips['start_time'].max()}")
print(f"  跨越天数: 7天 (2008-02-02 到 2008-02-08)")
print()

# 查看某个OD对在不同日期的数据
sample_od = trips[(trips['origin'] == 4) & (trips['dest'] == 14) & (trips['time_slot'] == 47)]
print(f"【示例】Origin=4 → Dest=14, time_slot=47 (11:45-12:00)")
print(f"  在7天中的分布:")
sample_od['date'] = pd.to_datetime(sample_od['start_time']).dt.date
date_counts = sample_od['date'].value_counts().sort_index()
for date, count in date_counts.items():
    print(f"    {date}: {count} 次trip")
print(f"  总计: {len(sample_od)} 次trip")
print()

# 读取聚合后的OD矩阵
od_table = pd.read_csv('/data/alice/cjtest/TRC/haidian_od_analysis/output/od_flow_table_full.csv')
result = od_table[(od_table['origin'] == 4) & (od_table['dest'] == 14) & (od_table['time_slot'] == 47)]
print(f"【阶段2】od_flow_table_full.csv中的聚合结果:")
print(f"  Origin=4 → Dest=14, time_slot=47")
if len(result) > 0:
    print(f"  flow = {result['flow'].values[0]:.0f} (= 7天该时段的总和)")
    print(f"  avg_time = {result['avg_time'].values[0]:.2f}分钟 (= 所有trips的平均时长)")
print()

print("=" * 80)
print("【关键代码】build_od_matrix_full.py 的聚合逻辑 (第181-189行):")
print("=" * 80)
print("""
for _, trip in trips_df.iterrows():
    t = int(trip['time_slot'])      # ← 只使用 time_slot (0-95)
    o = int(trip['origin']) - 1     #    忽略具体的日期信息！
    d = int(trip['dest']) - 1
    
    od_flow[t, o, d] += 1           # ← 所有相同time_slot的trips累加
    od_time_sum[t, o, d] += dur     #    不管是哪一天

# 结果：OD矩阵只有 (96时间槽, 29区域, 29区域)
#       把7天的数据聚合成"典型的一天"的流量模式
""")

print("=" * 80)
print("【总结】")
print("=" * 80)
print()
print("原始数据流转过程：")
print()
print("  all_taxi_data.csv (17,652,648条GPS点)")
print("  ↓ 包含: taxi_id, date_time, longitude, latitude")
print("  ↓ 时间范围: 2008-02-02 到 2008-02-08 (7天)")
print("  ↓")
print("  od_trips_full.csv (515,949条trips)")
print("  ↓ 包含: origin, dest, start_time, end_time, time_slot")
print("  ↓ 保留: 完整的日期时间信息 ✓")
print("  ↓")
print("  ↓ 【聚合步骤】按 time_slot 累加，忽略日期")
print("  ↓")
print("  od_flow_table_full.csv (31,095个非零OD对)")
print("    包含: time_slot, hour, minute, origin, dest, flow, avg_time")
print("    丢失: 日期信息 ✗")
print()
print("=" * 80)
print("【结论】")
print("=" * 80)
print()
print("✓ od_trips_full.csv 保留了日期 → 可以做时间序列预测")
print("✗ od_flow_table_full.csv 丢失日期 → 只是典型一天的统计结果")
print()
print("如果要做预测，需要基于 od_trips_full.csv 重新构建:")
print("  → 维度应该是 (7天 × 96时间槽, 29区域, 29区域)")
print("  → 或者 (96时间槽, 29区域, 29区域, 7天)")
print()
