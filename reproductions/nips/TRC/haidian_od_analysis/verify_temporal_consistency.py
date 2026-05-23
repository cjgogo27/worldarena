#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 od_flow_temporal.csv 和 od_flow_table_full.csv 的一致性
"""

import pandas as pd
import numpy as np

print("="*80)
print("验证时序数据与聚合数据的一致性")
print("="*80)

# 1. 加载聚合数据
print("\n【步骤1】加载聚合数据 (od_flow_table_full.csv)")
df_agg = pd.read_csv('output/od_flow_table_full.csv')
print(f"  记录数: {len(df_agg):,}")
print(f"  总flow: {df_agg['flow'].sum():,}")
print(f"  列: {list(df_agg.columns)}")

# 2. 加载时序数据
print("\n【步骤2】加载时序数据 (od_flow_temporal.csv)")
df_temporal = pd.read_csv('output/od_flow_temporal.csv')
print(f"  记录数: {len(df_temporal):,}")
print(f"  总flow: {df_temporal['flow'].sum():,}")
print(f"  列: {list(df_temporal.columns)}")
print(f"  日期范围: {df_temporal['date'].min()} 到 {df_temporal['date'].max()}")
print(f"  天数: {df_temporal['date'].nunique()}")

# 3. 对时序数据按 (time_slot, origin, dest) 分组求和
print("\n【步骤3】对时序数据按 (time_slot, origin, dest) 聚合")
df_temporal_agg = df_temporal.groupby(['time_slot', 'origin', 'dest']).agg({
    'flow': 'sum',
    'avg_time': 'mean'  # 平均时间取均值
}).reset_index()

print(f"  聚合后记录数: {len(df_temporal_agg):,}")
print(f"  聚合后总flow: {df_temporal_agg['flow'].sum():,}")

# 4. 合并两个数据集进行对比
print("\n【步骤4】合并数据集对比")
merged = pd.merge(
    df_agg[['time_slot', 'origin', 'dest', 'flow', 'avg_time']],
    df_temporal_agg[['time_slot', 'origin', 'dest', 'flow', 'avg_time']],
    on=['time_slot', 'origin', 'dest'],
    how='outer',
    suffixes=('_agg', '_temporal'),
    indicator=True
)

print(f"  合并后记录数: {len(merged):,}")

# 5. 检查只在某一个数据集中的记录
only_agg = merged[merged['_merge'] == 'left_only']
only_temporal = merged[merged['_merge'] == 'right_only']

print(f"\n  只在聚合数据中: {len(only_agg)}")
print(f"  只在时序数据中: {len(only_temporal)}")
print(f"  两者都有: {len(merged[merged['_merge'] == 'both']):,}")

if len(only_agg) > 0:
    print(f"\n  警告：有 {len(only_agg)} 条记录只在聚合数据中！")
    print("  示例:")
    print(only_agg[['time_slot', 'origin', 'dest', 'flow_agg']].head(10))

if len(only_temporal) > 0:
    print(f"\n  警告：有 {len(only_temporal)} 条记录只在时序数据中！")
    print("  示例:")
    print(only_temporal[['time_slot', 'origin', 'dest', 'flow_temporal']].head(10))

# 6. 对于两者都有的记录，检查flow是否一致
print("\n【步骤5】检查flow值的一致性")
both = merged[merged['_merge'] == 'both'].copy()

both['flow_diff'] = both['flow_temporal'] - both['flow_agg']
both['flow_diff_pct'] = (both['flow_diff'] / both['flow_agg'] * 100).round(2)

# 统计差异
exact_match = (both['flow_diff'] == 0).sum()
small_diff = (both['flow_diff'].abs() < 0.01).sum()
large_diff = (both['flow_diff'].abs() >= 1).sum()

print(f"  完全匹配 (差异=0): {exact_match:,} / {len(both):,} ({100*exact_match/len(both):.2f}%)")
print(f"  近似匹配 (差异<0.01): {small_diff:,} / {len(both):,} ({100*small_diff/len(both):.2f}%)")
print(f"  较大差异 (差异>=1): {large_diff:,} / {len(both):,} ({100*large_diff/len(both):.2f}%)")

print(f"\n  Flow差异统计:")
print(f"    平均差异: {both['flow_diff'].mean():.4f}")
print(f"    标准差: {both['flow_diff'].std():.4f}")
print(f"    最大差异: {both['flow_diff'].max():.4f}")
print(f"    最小差异: {both['flow_diff'].min():.4f}")

# 7. 显示差异最大的记录
if large_diff > 0:
    print(f"\n【步骤6】差异最大的10条记录:")
    top_diff = both.sort_values('flow_diff', ascending=False, key=lambda x: x.abs()).head(10)
    print(top_diff[['time_slot', 'origin', 'dest', 'flow_agg', 'flow_temporal', 'flow_diff', 'flow_diff_pct']])

# 8. 验证总flow
print("\n【步骤7】总体验证")
total_agg = df_agg['flow'].sum()
total_temporal = df_temporal['flow'].sum()
total_temporal_agg = df_temporal_agg['flow'].sum()

print(f"  聚合数据总flow: {total_agg:,.0f}")
print(f"  时序数据总flow: {total_temporal:,.0f}")
print(f"  时序聚合后总flow: {total_temporal_agg:,.0f}")
print(f"  差异: {total_temporal - total_agg:,.0f}")

if abs(total_temporal - total_agg) < 0.01:
    print("\n✓ 验证通过！两个数据集的总flow完全一致！")
else:
    print(f"\n✗ 警告：总flow有差异 ({(total_temporal - total_agg) / total_agg * 100:.4f}%)")

# 9. 验证示例：Origin=4 → Dest=14, time_slot=47
print("\n【步骤8】示例验证：Origin=4 → Dest=14, time_slot=47")
sample_agg = df_agg[(df_agg['origin'] == 4) & (df_agg['dest'] == 14) & (df_agg['time_slot'] == 47)]
sample_temporal = df_temporal[(df_temporal['origin'] == 4) & (df_temporal['dest'] == 14) & (df_temporal['time_slot'] == 47)]

print(f"\n  聚合数据中的记录:")
if len(sample_agg) > 0:
    print(f"    flow = {sample_agg['flow'].values[0]}")
    print(f"    avg_time = {sample_agg['avg_time'].values[0]:.2f}")

print(f"\n  时序数据中的记录 ({len(sample_temporal)} 天):")
for _, row in sample_temporal.iterrows():
    print(f"    {row['date']}: flow={row['flow']:.0f}, avg_time={row['avg_time']:.2f}")

if len(sample_temporal) > 0:
    print(f"\n  时序数据求和: {sample_temporal['flow'].sum():.0f}")
    print(f"  聚合数据值: {sample_agg['flow'].values[0] if len(sample_agg) > 0 else 'N/A'}")
    if len(sample_agg) > 0:
        diff = sample_temporal['flow'].sum() - sample_agg['flow'].values[0]
        print(f"  差异: {diff:.0f}")
        if abs(diff) < 0.01:
            print("  ✓ 匹配！")
        else:
            print(f"  ✗ 不匹配！差异 {diff:.0f}")

print("\n" + "="*80)
print("验证完成")
print("="*80)
