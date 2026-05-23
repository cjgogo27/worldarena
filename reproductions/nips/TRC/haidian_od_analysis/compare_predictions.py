#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并原始数据和LSTM预测结果进行对比
"""
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # 文件路径
    original_file = 'output/od_flow_temporal.csv'
    prediction_file = 'output/od_flow_predictions_lstm_test.csv'
    output_file = 'output/od_flow_comparison.csv'
    
    print("="*80)
    print("LSTM预测结果对比分析")
    print("="*80)
    
    # 读取原始数据（只保留测试日期2008-02-08）
    print("\n1. 读取原始数据...")
    df_original = pd.read_csv(original_file)
    df_original['date'] = pd.to_datetime(df_original['date'])
    df_test_actual = df_original[df_original['date'] == '2008-02-08'].copy()
    print(f"   原始测试数据: {len(df_test_actual)} 条记录")
    print(f"   实际总flow: {df_test_actual['flow'].sum():.0f} trips")
    
    # 读取预测数据
    print("\n2. 读取预测数据...")
    df_pred = pd.read_csv(prediction_file)
    print(f"   预测数据: {len(df_pred)} 条记录")
    print(f"   预测总flow: {df_pred['flow'].sum():.0f} trips")
    
    # 合并数据（基于time_slot, origin, dest）
    print("\n3. 合并原始数据和预测数据...")
    df_test_actual = df_test_actual.rename(columns={'flow': 'actual_flow', 'avg_time': 'actual_avg_time'})
    df_pred = df_pred.rename(columns={'flow': 'predicted_flow', 'avg_time': 'predicted_avg_time'})
    
    # 选择需要的列
    actual_cols = ['date', 'day_index', 'time_slot', 'hour', 'minute', 'origin', 'dest', 'actual_flow', 'actual_avg_time']
    pred_cols = ['time_slot', 'origin', 'dest', 'predicted_flow']
    
    df_comparison = pd.merge(
        df_test_actual[actual_cols],
        df_pred[pred_cols],
        on=['time_slot', 'origin', 'dest'],
        how='outer'
    )
    
    # 填充缺失值为0
    df_comparison['actual_flow'] = df_comparison['actual_flow'].fillna(0)
    df_comparison['predicted_flow'] = df_comparison['predicted_flow'].fillna(0)
    
    # 计算误差
    df_comparison['absolute_error'] = np.abs(df_comparison['predicted_flow'] - df_comparison['actual_flow'])
    df_comparison['squared_error'] = (df_comparison['predicted_flow'] - df_comparison['actual_flow']) ** 2
    
    # 只对非零实际值计算百分比误差
    non_zero_mask = df_comparison['actual_flow'] > 0
    df_comparison['percentage_error'] = 0.0
    df_comparison.loc[non_zero_mask, 'percentage_error'] = (
        np.abs(df_comparison.loc[non_zero_mask, 'predicted_flow'] - df_comparison.loc[non_zero_mask, 'actual_flow']) / 
        df_comparison.loc[non_zero_mask, 'actual_flow'] * 100
    )
    
    # 按time_slot, origin, dest排序
    df_comparison = df_comparison.sort_values(['time_slot', 'origin', 'dest']).reset_index(drop=True)
    
    # 保存对比结果
    df_comparison.to_csv(output_file, index=False)
    print(f"   对比结果已保存到: {output_file}")
    print(f"   总记录数: {len(df_comparison)}")
    
    # 计算评估指标
    print("\n" + "="*80)
    print("评估指标")
    print("="*80)
    
    mae = df_comparison['absolute_error'].mean()
    rmse = np.sqrt(df_comparison['squared_error'].mean())
    
    # MAPE只计算非零实际值
    mape = df_comparison.loc[non_zero_mask, 'percentage_error'].mean() if non_zero_mask.sum() > 0 else 0
    
    # R²
    ss_res = df_comparison['squared_error'].sum()
    ss_tot = ((df_comparison['actual_flow'] - df_comparison['actual_flow'].mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\n全部数据 ({len(df_comparison)} 个OD-时间槽):")
    print(f"  MAE:  {mae:.4f} trips")
    print(f"  RMSE: {rmse:.4f} trips")
    print(f"  MAPE: {mape:.2f}% (基于{non_zero_mask.sum()}个非零实际值)")
    print(f"  R²:   {r2:.4f}")
    
    # 只对非零实际值的评估
    if non_zero_mask.sum() > 0:
        df_nonzero = df_comparison[non_zero_mask]
        mae_nz = df_nonzero['absolute_error'].mean()
        rmse_nz = np.sqrt(df_nonzero['squared_error'].mean())
        mape_nz = df_nonzero['percentage_error'].mean()
        
        ss_res_nz = df_nonzero['squared_error'].sum()
        ss_tot_nz = ((df_nonzero['actual_flow'] - df_nonzero['actual_flow'].mean()) ** 2).sum()
        r2_nz = 1 - (ss_res_nz / ss_tot_nz) if ss_tot_nz > 0 else 0
        
        print(f"\n只计算非零实际值 ({non_zero_mask.sum()} 个OD-时间槽):")
        print(f"  MAE:  {mae_nz:.4f} trips")
        print(f"  RMSE: {rmse_nz:.4f} trips")
        print(f"  MAPE: {mape_nz:.2f}%")
        print(f"  R²:   {r2_nz:.4f}")
    
    # 显示一些示例
    print("\n" + "="*80)
    print("预测示例（前10个非零实际值）")
    print("="*80)
    
    sample_df = df_comparison[non_zero_mask].head(10)[
        ['time_slot', 'hour', 'minute', 'origin', 'dest', 'actual_flow', 'predicted_flow', 'absolute_error', 'percentage_error']
    ]
    
    print("\n" + sample_df.to_string(index=False))
    
    # 统计信息
    print("\n" + "="*80)
    print("数据统计")
    print("="*80)
    print(f"\n实际flow统计:")
    print(f"  总计: {df_comparison['actual_flow'].sum():.0f} trips")
    print(f"  平均: {df_comparison['actual_flow'].mean():.2f} trips")
    print(f"  最大: {df_comparison['actual_flow'].max():.0f} trips")
    print(f"  非零记录: {non_zero_mask.sum()} / {len(df_comparison)} ({non_zero_mask.sum()/len(df_comparison)*100:.1f}%)")
    
    print(f"\n预测flow统计:")
    print(f"  总计: {df_comparison['predicted_flow'].sum():.0f} trips")
    print(f"  平均: {df_comparison['predicted_flow'].mean():.2f} trips")
    print(f"  最大: {df_comparison['predicted_flow'].max():.0f} trips")
    
    # 按时间槽统计
    print("\n按时间槽统计误差:")
    time_slot_stats = df_comparison.groupby('time_slot').agg({
        'actual_flow': 'sum',
        'predicted_flow': 'sum',
        'absolute_error': 'mean'
    }).reset_index()
    time_slot_stats.columns = ['time_slot', 'actual_total', 'predicted_total', 'avg_mae']
    time_slot_stats['total_error'] = np.abs(time_slot_stats['predicted_total'] - time_slot_stats['actual_total'])
    
    # 显示误差最大的10个时间槽
    print("\n误差最大的10个时间槽:")
    worst_slots = time_slot_stats.nlargest(10, 'total_error')
    for _, row in worst_slots.iterrows():
        slot = int(row['time_slot'])
        hour = slot // 4
        minute = (slot % 4) * 15
        print(f"  时间槽 {slot:2d} ({hour:02d}:{minute:02d}): "
              f"实际={row['actual_total']:6.0f}, "
              f"预测={row['predicted_total']:6.0f}, "
              f"误差={row['total_error']:6.0f}")
    
    print("\n" + "="*80)
    print(f"完成！对比结果已保存到: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
