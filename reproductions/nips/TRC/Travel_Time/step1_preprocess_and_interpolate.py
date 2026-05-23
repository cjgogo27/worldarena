#!/usr/bin/env python3
"""
步骤1：数据预处理和插值
========================

功能：
1. 加载原始OD流量数据
2. 构建完整的时序矩阵（所有OD对×所有时间槽）
3. 使用训练集数据进行插值（不能使用测试集）
4. 保存插值后的完整数据为CSV

输出：
- od_flow_interpolated.csv：插值后的完整OD流量和行程时间数据
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path):
    """加载原始数据"""
    print("="*80)
    print("1. 加载原始数据")
    print("="*80)
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"原始数据: {len(df):,} 条记录")
    print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"日期列表: {sorted(df['date'].unique())}")
    
    return df


def build_complete_matrix(df):
    """构建完整的4D矩阵"""
    print("\n" + "="*80)
    print("2. 构建完整的4D矩阵")
    print("="*80)
    
    all_dates = sorted(df['date'].unique())
    n_days = len(all_dates)
    n_time_slots = 96
    n_zones = df['origin'].max()
    
    print(f"维度: {n_days}天 × {n_time_slots}时间槽 × {n_zones}区域 × {n_zones}区域")
    
    # 初始化矩阵（用NaN表示缺失）
    time_matrix = np.full((n_days, n_time_slots, n_zones, n_zones), np.nan)
    flow_matrix = np.full((n_days, n_time_slots, n_zones, n_zones), np.nan)
    
    # 填充原始数据
    for _, row in df.iterrows():
        day_idx = all_dates.index(row['date'])
        t = int(row['time_slot'])
        o = int(row['origin']) - 1  # 0-based
        d = int(row['dest']) - 1
        
        time_matrix[day_idx, t, o, d] = row['avg_time']
        flow_matrix[day_idx, t, o, d] = row['flow']
    
    # 统计缺失情况
    total_cells = n_days * n_time_slots * n_zones * n_zones
    missing_cells = np.isnan(time_matrix).sum()
    
    print(f"\n数据完整性:")
    print(f"  总单元数: {total_cells:,}")
    print(f"  缺失单元: {missing_cells:,} ({missing_cells/total_cells*100:.1f}%)")
    print(f"  有效单元: {total_cells-missing_cells:,} ({(total_cells-missing_cells)/total_cells*100:.1f}%)")
    
    return time_matrix, flow_matrix, all_dates, n_zones


def interpolate_with_train_only(time_matrix, flow_matrix, all_dates, train_dates):
    """
    使用训练集数据进行插值（不使用测试集）
    
    插值策略：
    1. 对于每个OD对，只使用训练集计算历史平均值
    2. 用训练集的平均值填充所有缺失值
    3. 如果训练集中也没有该OD对的数据，用训练集全局平均值
    """
    print("\n" + "="*80)
    print("3. 插值（仅使用训练集数据）")
    print("="*80)
    
    n_days, n_time_slots, n_zones, _ = time_matrix.shape
    
    # 获取训练集的天索引
    train_day_indices = [i for i, date in enumerate(all_dates) 
                         if pd.to_datetime(date) in pd.to_datetime(train_dates)]
    
    print(f"训练集日期: {train_dates}")
    print(f"训练集天索引: {train_day_indices}")
    
    # 只使用训练集数据计算统计值
    train_time_data = time_matrix[train_day_indices, :, :, :]
    train_flow_data = flow_matrix[train_day_indices, :, :, :]
    
    # 计算训练集的全局平均值（排除NaN）
    global_mean_time = np.nanmean(train_time_data)
    print(f"\n训练集全局平均行程时间: {global_mean_time:.2f} 分钟")
    
    # 为每个OD对计算训练集的平均值
    od_mean_time = np.nanmean(train_time_data, axis=(0, 1))  # (n_zones, n_zones)
    
    # 统计每个OD对在训练集中有多少数据点
    od_data_count = np.sum(~np.isnan(train_time_data), axis=(0, 1))
    
    print(f"\nOD对统计（训练集）:")
    print(f"  有数据的OD对: {np.sum(od_data_count > 0)} / {n_zones * n_zones}")
    print(f"  平均每个OD对数据点数: {np.mean(od_data_count[od_data_count > 0]):.1f}")
    
    # 创建插值后的矩阵
    time_matrix_filled = time_matrix.copy()
    flow_matrix_filled = flow_matrix.copy()
    
    # 插值策略：用训练集的OD平均值填充
    interpolated_count = 0
    for o in range(n_zones):
        for d in range(n_zones):
            # 该OD对在训练集中的平均值
            od_mean = od_mean_time[o, d]
            
            # 如果训练集中没有这个OD对的数据，用全局平均值
            if np.isnan(od_mean):
                od_mean = global_mean_time
            
            # 填充所有天的缺失值（包括训练集、验证集、测试集）
            for day in range(n_days):
                for t in range(n_time_slots):
                    if np.isnan(time_matrix_filled[day, t, o, d]):
                        time_matrix_filled[day, t, o, d] = od_mean
                        interpolated_count += 1
                    
                    if np.isnan(flow_matrix_filled[day, t, o, d]):
                        flow_matrix_filled[day, t, o, d] = 0  # 缺失的flow填0
    
    print(f"\n插值结果:")
    print(f"  插值单元数: {interpolated_count:,}")
    print(f"  剩余缺失值: {np.isnan(time_matrix_filled).sum()}")
    
    return time_matrix_filled, flow_matrix_filled


def save_to_csv(time_matrix, flow_matrix, all_dates, n_zones, output_path):
    """保存为CSV格式"""
    print("\n" + "="*80)
    print("4. 保存为CSV")
    print("="*80)
    
    records = []
    n_days, n_time_slots, _, _ = time_matrix.shape
    
    for day_idx, date in enumerate(all_dates):
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        
        for t in range(n_time_slots):
            hour = (t * 15) // 60
            minute = (t * 15) % 60
            
            for o in range(n_zones):
                for d in range(n_zones):
                    avg_time = time_matrix[day_idx, t, o, d]
                    flow = flow_matrix[day_idx, t, o, d]
                    
                    # 只保存有数据的记录（避免文件过大）
                    # 但至少保留flow>0或avg_time>0的记录
                    if flow > 0 or avg_time > 0:
                        records.append({
                            'date': date_str,
                            'day_index': day_idx,
                            'time_slot': t,
                            'hour': hour,
                            'minute': minute,
                            'origin': o + 1,  # 1-based
                            'dest': d + 1,
                            'flow': round(flow, 6),
                            'avg_time': round(avg_time, 6),
                            'is_interpolated': flow == 0  # 标记是否为插值
                        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ 已保存: {output_path}")
    print(f"  总记录数: {len(df):,}")
    print(f"  原始数据: {df[~df['is_interpolated']].shape[0]:,}")
    print(f"  插值数据: {df[df['is_interpolated']].shape[0]:,}")
    print(f"\n文件预览:")
    print(df.head(10))
    print("\n...")
    print(df.tail(10))
    
    # 统计信息
    print(f"\n数据统计:")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  时间槽范围: {df['time_slot'].min()} ~ {df['time_slot'].max()}")
    print(f"  OD对数: {df.groupby(['origin', 'dest']).size().shape[0]}")
    print(f"  平均行程时间: {df['avg_time'].mean():.2f} 分钟")
    print(f"  平均流量: {df['flow'].mean():.2f}")
    
    return df


def main():
    """主函数"""
    print("="*80)
    print("数据预处理和插值")
    print("="*80)
    print(f"执行时间: {datetime.now()}")
    
    # 配置
    data_path = '/data/alice/cjtest/TRC/haidian_od_analysis/output/od_flow_temporal.csv'
    output_path = '/data/alice/cjtest/TRC/Travel_Time/od_flow_interpolated.csv'
    
    # 划分训练/验证/测试集
    train_dates = ['2008-02-02', '2008-02-03', '2008-02-04', '2008-02-05', '2008-02-06']
    val_dates = ['2008-02-07']
    test_dates = ['2008-02-08']
    
    print(f"\n数据集划分:")
    print(f"  训练集: {train_dates}")
    print(f"  验证集: {val_dates}")
    print(f"  测试集: {test_dates}")
    
    # 1. 加载数据
    df = load_data(data_path)
    
    # 2. 构建完整矩阵
    time_matrix, flow_matrix, all_dates, n_zones = build_complete_matrix(df)
    
    # 3. 插值（只使用训练集）
    time_matrix_filled, flow_matrix_filled = interpolate_with_train_only(
        time_matrix, flow_matrix, all_dates, train_dates
    )
    
    # 4. 保存为CSV
    df_interpolated = save_to_csv(
        time_matrix_filled, flow_matrix_filled, 
        all_dates, n_zones, output_path
    )
    
    print("\n" + "="*80)
    print("✓ 数据预处理完成！")
    print("="*80)
    print(f"\n下一步：使用插值后的数据训练模型")
    print(f"  输入文件: {output_path}")


if __name__ == '__main__':
    main()
