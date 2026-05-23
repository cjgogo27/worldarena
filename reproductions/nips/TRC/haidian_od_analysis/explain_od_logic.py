#!/usr/bin/env python3
"""
OD流量提取逻辑说明和验证
"""

import pandas as pd
import sys

def explain_current_logic():
    """
    解释当前的OD流量提取逻辑
    """
    
    print("=" * 80)
    print("当前OD流量提取逻辑说明")
    print("=" * 80)
    
    print("\n原始数据格式：")
    print("  taxi_id, date_time, longitude, latitude")
    print("  每条记录 = 某辆车在某个时刻在某个地点")
    
    print("\n处理步骤：")
    print("\n1. 空间映射（步骤3）")
    print("   GPS点(经度,纬度) → 区域ID")
    print("   例如：(116.3, 39.99) → 区域10（中关村街道）")
    
    print("\n2. 时间映射（步骤4）")
    print("   时间戳 → 15分钟时间槽ID (0-95)")
    print("   例如：15:46:08 → 时间槽63 (15:45-16:00)")
    
    print("\n3. 出行提取（步骤5）- 当前逻辑")
    print("   按车辆ID分组，按时间排序")
    print("   当连续两个轨迹点属于不同区域时 → 记录为一次OD出行")
    
    print("\n示例：")
    print("  车辆1的轨迹序列：")
    print("    点1: 15:46:08, 区域5  <-- 起点")
    print("    点2: 15:56:08, 区域5  (同区域，继续)")
    print("    点3: 16:06:08, 区域8  <-- 终点（区域变化）")
    print("    点4: 16:16:08, 区域8  (同区域，作为新起点)")
    print("    点5: 16:26:08, 区域12 <-- 新终点")
    
    print("\n  提取的出行：")
    print("    出行1: 区域5 → 区域8, 开始时间15:46, 时长20分钟, 时间槽63")
    print("    出行2: 区域8 → 区域12, 开始时间16:16, 时长10分钟, 时间槽65")
    
    print("\n4. OD矩阵生成")
    print("   统计每个时间槽内，每个OD对的出行次数")
    print("   矩阵[i][j] = 时间槽T内，从区域i到区域j的出行次数")
    
    print("\n" + "=" * 80)
    print("关键逻辑点：")
    print("=" * 80)
    
    print("\n✓ 只记录跨区域的移动")
    print("  - 在同一区域内的移动不计入OD流量")
    print("  - 区域内移动算作停留")
    
    print("\n✓ 时间阈值控制（默认30分钟）")
    print("  - 如果两个点之间间隔>30分钟，认为是新的出行起点")
    print("  - 避免将长时间停车后的移动算作同一次出行")
    
    print("\n✓ 出行时长 = 从起点时间到终点时间")
    print("  - 不是直线距离/速度")
    print("  - 是实际轨迹时间差")
    
    print("\n✓ 时间槽基于出行开始时间")
    print("  - 出行归属到起点的时间槽")
    print("  - 15:46开始的出行 → 归入时间槽63 (15:45-16:00)")


def demonstrate_with_real_data():
    """
    用实际数据演示
    """
    import os
    
    sample_file = 'output/step3_mapped_trajectory_sample.csv'
    
    if not os.path.exists(sample_file):
        print("\n请先运行步骤1-3以生成样本数据")
        return
    
    print("\n\n" + "=" * 80)
    print("实际数据示例")
    print("=" * 80)
    
    df = pd.read_csv(sample_file, nrows=1000)
    
    # 选择一辆有多个点的车
    vehicle_counts = df['taxi_id'].value_counts()
    if len(vehicle_counts) == 0:
        print("样本数据中没有足够的轨迹点")
        return
    
    sample_vehicle = vehicle_counts.index[0]
    vehicle_traj = df[df['taxi_id'] == sample_vehicle].sort_values('date_time')
    
    print(f"\n车辆 {sample_vehicle} 的轨迹示例：")
    print(f"共 {len(vehicle_traj)} 个轨迹点\n")
    
    # 显示前10个点
    display_cols = ['date_time', 'longitude', 'latitude', 'region_id']
    available_cols = [col for col in display_cols if col in vehicle_traj.columns]
    
    print(vehicle_traj[available_cols].head(10).to_string(index=False))
    
    # 识别区域变化
    if 'region_id' in vehicle_traj.columns:
        vehicle_traj = vehicle_traj.reset_index(drop=True)
        region_changes = []
        
        for i in range(len(vehicle_traj) - 1):
            if vehicle_traj.iloc[i]['region_id'] != vehicle_traj.iloc[i+1]['region_id']:
                region_changes.append({
                    'from': vehicle_traj.iloc[i]['region_id'],
                    'to': vehicle_traj.iloc[i+1]['region_id'],
                    'time': vehicle_traj.iloc[i]['date_time']
                })
        
        if region_changes:
            print(f"\n识别到 {len(region_changes)} 次区域变化（潜在OD出行）：")
            for idx, change in enumerate(region_changes[:5], 1):
                print(f"  {idx}. 区域{change['from']} → 区域{change['to']}, 时间: {change['time']}")
        else:
            print("\n该车辆在样本中没有跨区域移动")


def compare_methods():
    """
    对比不同的OD提取方法
    """
    
    print("\n\n" + "=" * 80)
    print("不同OD提取方法对比")
    print("=" * 80)
    
    print("\n方法1: 相邻点法（当前使用）")
    print("  逻辑: 连续两个轨迹点属于不同区域 → 记录OD")
    print("  优点: 简单直接，反映实际移动")
    print("  缺点: 可能遗漏中间经过的区域")
    print("  示例: A→A→B→B → 提取1次OD (A→B)")
    
    print("\n方法2: 所有点对法")
    print("  逻辑: 同一出行内所有点对都记录")
    print("  优点: 捕获所有移动")
    print("  缺点: 重复计数严重")
    print("  示例: A→B→C → 提取3次OD (A→B, B→C, A→C)")
    
    print("\n方法3: 起终点法")
    print("  逻辑: 只记录出行的起点和最终终点")
    print("  优点: 反映真实出行OD")
    print("  缺点: 忽略中间转移")
    print("  示例: A→B→C→D → 提取1次OD (A→D)")
    
    print("\n推荐: 方法1（当前使用）")
    print("  适合出租车数据，反映区域间的实际流动")


def main():
    explain_current_logic()
    demonstrate_with_real_data()
    compare_methods()
    
    print("\n\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("\n当前处理完全符合您的理解：")
    print("  ✓ 车辆从一个地点（区域）开到另一个地点（区域）")
    print("  ✓ 反映区域间的流量")
    print("  ✓ 基于实际轨迹的时空特征")
    
    print("\n如需调整，可修改：")
    print("  1. 时间阈值 (config['trip_time_threshold'])")
    print("  2. 区域划分粒度 (修改shapefile或合并区域)")
    print("  3. 时间粒度 (config['interval_minutes'])")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
