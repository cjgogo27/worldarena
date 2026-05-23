#!/usr/bin/env python3
"""
独立的OD矩阵构建脚本
功能：从已处理的轨迹数据构建OD流量矩阵和平均旅行时间矩阵
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ODMatrixBuilder:
    """OD矩阵构建器"""
    
    def __init__(self, num_regions=29, num_time_slots=96):
        self.num_regions = num_regions
        self.num_time_slots = num_time_slots
        
    def load_trajectory_data(self, file_path):
        """加载已处理的轨迹数据"""
        print(f"加载轨迹数据: {file_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(file_path) / 1024 / 1024
        print(f"文件大小: {file_size:.2f} MB")
        
        # 如果是sample文件，需要加载完整数据
        if 'sample' in file_path.lower():
            print("检测到sample文件，将尝试加载完整数据...")
            full_path = file_path.replace('_sample', '')
            if os.path.exists(full_path):
                file_path = full_path
                print(f"使用完整数据: {full_path}")
            else:
                print(f"警告：未找到完整数据文件，将使用sample数据")
        
        df = pd.read_csv(file_path)
        print(f"加载完成: {len(df)} 条记录")
        print(f"数据列: {df.columns.tolist()}")
        
        # 转换时间列
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'])
        
        return df
    
    def extract_trips(self, df, time_threshold_minutes=30):
        """
        从轨迹数据提取出行记录
        
        Args:
            df: 轨迹数据，需包含 taxi_id, date_time, region_id
            time_threshold_minutes: 停留时间阈值（分钟）
        
        Returns:
            trips_df: 出行记录DataFrame
        """
        print(f"\n开始提取出行记录...")
        print(f"参数: 停留时间阈值 = {time_threshold_minutes} 分钟")
        
        # 确保数据按车辆和时间排序
        df = df.sort_values(['taxi_id', 'date_time']).reset_index(drop=True)
        
        # 过滤掉region_id为空的记录
        df = df[df['region_id'].notna()].copy()
        print(f"有效轨迹点: {len(df)} 条")
        
        trips = []
        vehicle_groups = list(df.groupby('taxi_id'))
        print(f"正在处理 {len(vehicle_groups)} 辆车的轨迹...")
        
        start_time = datetime.now()
        
        # 使用更高效的算法
        for idx, (vehicle_id, group) in enumerate(tqdm(vehicle_groups, desc="提取trips", 
                                                         mininterval=0.5)):
            if idx > 0 and idx % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / idx
                remaining = avg_time * (len(vehicle_groups) - idx)
                print(f"\r  进度: {idx}/{len(vehicle_groups)} | "
                      f"已提取trips: {len(trips)} | "
                      f"预计剩余: {remaining/60:.1f}分钟", 
                      end='', flush=True)
            
            group = group.reset_index(drop=True)
            if len(group) < 2:
                continue
            
            # 计算时间差和区域变化
            group['time_diff'] = group['date_time'].diff().dt.total_seconds() / 60
            group['region_changed'] = group['region_id'] != group['region_id'].shift(1)
            
            # 识别新trip的起点：区域变化 或 时间间隔过长
            group['is_trip_start'] = (group['region_changed']) | (group['time_diff'] > time_threshold_minutes)
            
            # 记录trip
            current_trip = None
            for i, row in group.iterrows():
                if i == 0:  # 第一个点作为起点
                    current_trip = {
                        'vehicle_id': vehicle_id,
                        'origin_region': int(row['region_id']),
                        'start_time': row['date_time'],
                        'start_time_slot': row.get('time_slot', None),
                        'start_global_slot': row.get('global_time_slot', None)
                    }
                elif row['is_trip_start']:
                    # 如果区域变化，保存上一段trip
                    if current_trip and row['region_id'] != current_trip['origin_region']:
                        trip = {
                            'vehicle_id': vehicle_id,
                            'origin_region': current_trip['origin_region'],
                            'dest_region': int(row['region_id']),
                            'start_time': current_trip['start_time'],
                            'end_time': row['date_time'],
                            'duration_minutes': (row['date_time'] - current_trip['start_time']).total_seconds() / 60,
                            'start_time_slot': current_trip.get('start_time_slot'),
                            'start_global_slot': current_trip.get('start_global_slot')
                        }
                        if trip['duration_minutes'] > 0:  # 只保留有效时长的trip
                            trips.append(trip)
                    
                    # 开始新trip
                    current_trip = {
                        'vehicle_id': vehicle_id,
                        'origin_region': int(row['region_id']),
                        'start_time': row['date_time'],
                        'start_time_slot': row.get('time_slot', None),
                        'start_global_slot': row.get('global_time_slot', None)
                    }
        
        print(f"\n\n提取完成! 共获得 {len(trips)} 条出行记录")
        
        trips_df = pd.DataFrame(trips)
        if len(trips_df) > 0:
            print(f"平均出行时长: {trips_df['duration_minutes'].mean():.2f} 分钟")
            print(f"时间范围: {trips_df['start_time'].min()} 到 {trips_df['end_time'].max()}")
        
        return trips_df
    
    def build_od_matrices(self, trips_df):
        """
        构建OD矩阵
        
        Returns:
            od_flow: (num_time_slots, num_regions, num_regions) 流量矩阵
            od_time: (num_time_slots, num_regions, num_regions) 平均时间矩阵
        """
        print(f"\n构建OD矩阵...")
        print(f"维度: {self.num_time_slots} 时间槽 × {self.num_regions} 区域 × {self.num_regions} 区域")
        
        # 初始化矩阵
        od_flow = np.zeros((self.num_time_slots, self.num_regions, self.num_regions), dtype=np.float32)
        od_time_sum = np.zeros((self.num_time_slots, self.num_regions, self.num_regions), dtype=np.float32)
        od_time_count = np.zeros((self.num_time_slots, self.num_regions, self.num_regions), dtype=np.int32)
        
        # 填充矩阵
        for _, trip in tqdm(trips_df.iterrows(), total=len(trips_df), desc="构建矩阵"):
            time_slot = trip.get('start_time_slot')
            if pd.isna(time_slot):
                continue
            
            time_slot = int(time_slot)
            origin = int(trip['origin_region']) - 1  # 转为0-based索引
            dest = int(trip['dest_region']) - 1
            duration = trip['duration_minutes']
            
            if 0 <= time_slot < self.num_time_slots and 0 <= origin < self.num_regions and 0 <= dest < self.num_regions:
                od_flow[time_slot, origin, dest] += 1
                od_time_sum[time_slot, origin, dest] += duration
                od_time_count[time_slot, origin, dest] += 1
        
        # 计算平均时间
        od_time = np.zeros_like(od_time_sum)
        mask = od_time_count > 0
        od_time[mask] = od_time_sum[mask] / od_time_count[mask]
        
        # 统计
        total_trips = od_flow.sum()
        non_zero = (od_flow > 0).sum()
        total_cells = od_flow.size
        sparsity = 1 - (non_zero / total_cells)
        
        print(f"\nOD矩阵统计:")
        print(f"  总trip数: {int(total_trips)}")
        print(f"  非零单元: {non_zero} / {total_cells}")
        print(f"  稀疏度: {sparsity*100:.2f}%")
        print(f"  平均每时间槽: {total_trips/self.num_time_slots:.1f} trips")
        
        return od_flow, od_time
    
    def visualize_od_matrix(self, od_flow, od_time, output_dir='output'):
        """可视化OD矩阵"""
        print(f"\n生成可视化图表...")
        os.makedirs(f'{output_dir}/od_visualizations', exist_ok=True)
        
        # 1. 全天汇总的OD流量热力图
        daily_flow = od_flow.sum(axis=0)  # 汇总所有时间槽
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(daily_flow, cmap='YlOrRd', cbar_kws={'label': 'Total Trips'},
                   xticklabels=range(1, self.num_regions+1),
                   yticklabels=range(1, self.num_regions+1))
        plt.title('Daily OD Flow Matrix (All Time Slots Aggregated)', fontsize=16, fontweight='bold')
        plt.xlabel('Destination Region', fontsize=12)
        plt.ylabel('Origin Region', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/od_visualizations/od_daily_flow_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: od_daily_flow_heatmap.png")
        
        # 2. 全天汇总的平均旅行时间热力图
        # 重新计算全天平均时间
        total_time_sum = np.zeros((self.num_regions, self.num_regions))
        total_count = np.zeros((self.num_regions, self.num_regions))
        
        for t in range(self.num_time_slots):
            mask = od_flow[t] > 0
            total_time_sum[mask] += od_time[t][mask] * od_flow[t][mask]
            total_count[mask] += od_flow[t][mask]
        
        daily_avg_time = np.zeros_like(total_time_sum)
        mask = total_count > 0
        daily_avg_time[mask] = total_time_sum[mask] / total_count[mask]
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(daily_avg_time, cmap='viridis', cbar_kws={'label': 'Avg Travel Time (min)'},
                   xticklabels=range(1, self.num_regions+1),
                   yticklabels=range(1, self.num_regions+1))
        plt.title('Daily Average Travel Time Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Destination Region', fontsize=12)
        plt.ylabel('Origin Region', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/od_visualizations/od_daily_time_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: od_daily_time_heatmap.png")
        
        # 3. 稀疏性分析
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 3.1 每小时的trip数量
        hourly_trips = []
        for t in range(self.num_time_slots):
            hourly_trips.append(od_flow[t].sum())
        
        hours = [i/4 for i in range(self.num_time_slots)]  # 15分钟 = 0.25小时
        axes[0, 0].plot(hours, hourly_trips, linewidth=2)
        axes[0, 0].set_xlabel('Hour of Day', fontsize=12)
        axes[0, 0].set_ylabel('Number of Trips', fontsize=12)
        axes[0, 0].set_title('Trip Volume by Time Slot', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 3.2 每个时间槽的稀疏度
        sparsity_by_slot = []
        for t in range(self.num_time_slots):
            non_zero = (od_flow[t] > 0).sum()
            total_cells = self.num_regions * self.num_regions
            sparsity_by_slot.append(1 - non_zero / total_cells)
        
        axes[0, 1].plot(hours, [s*100 for s in sparsity_by_slot], linewidth=2, color='red')
        axes[0, 1].set_xlabel('Hour of Day', fontsize=12)
        axes[0, 1].set_ylabel('Sparsity (%)', fontsize=12)
        axes[0, 1].set_title('Matrix Sparsity by Time Slot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3.3 OD流量分布直方图
        flow_values = daily_flow[daily_flow > 0].flatten()
        axes[1, 0].hist(flow_values, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Number of Trips (OD Pair)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Distribution of OD Flow Values', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        
        # 3.4 旅行时间分布直方图
        time_values = daily_avg_time[daily_avg_time > 0].flatten()
        axes[1, 1].hist(time_values, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Average Travel Time (minutes)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Distribution of Travel Times', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/od_visualizations/od_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: od_analysis.png")
        
        # 4. 选择几个高峰时段的OD矩阵
        peak_hours = [8, 12, 18, 22]  # 早高峰、午间、晚高峰、夜间
        peak_slots = [h * 4 for h in peak_hours]  # 转换为时间槽
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        axes = axes.flatten()
        
        for i, (hour, slot) in enumerate(zip(peak_hours, peak_slots)):
            if slot < self.num_time_slots:
                sns.heatmap(od_flow[slot], ax=axes[i], cmap='YlOrRd',
                           xticklabels=range(1, self.num_regions+1),
                           yticklabels=range(1, self.num_regions+1),
                           cbar_kws={'label': 'Trips'})
                axes[i].set_title(f'OD Flow at {hour}:00-{hour}:15', fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Destination Region')
                axes[i].set_ylabel('Origin Region')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/od_visualizations/od_peak_hours.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: od_peak_hours.png")
        
        print(f"\n所有可视化图表已保存到: {output_dir}/od_visualizations/")
    
    def save_od_matrices(self, od_flow, od_time, trips_df, output_dir='output'):
        """保存OD矩阵到文件"""
        print(f"\n保存OD矩阵...")
        
        # 1. 保存trips记录为CSV
        trips_path = f'{output_dir}/od_trips.csv'
        trips_df.to_csv(trips_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ Trips记录: {trips_path}")
        
        # 2. 保存为numpy格式（便于Python读取）
        np.save(f'{output_dir}/od_flow_matrix.npy', od_flow)
        np.save(f'{output_dir}/od_time_matrix.npy', od_time)
        print(f"  ✓ NumPy格式: od_flow_matrix.npy, od_time_matrix.npy")
        
        # 3. 保存为CSV格式（每个时间槽一个文件 - 仅保存有流量的）
        os.makedirs(f'{output_dir}/od_matrices_csv', exist_ok=True)
        
        # 保存汇总的全天OD矩阵
        daily_flow = od_flow.sum(axis=0)
        daily_flow_df = pd.DataFrame(daily_flow, 
                                      index=range(1, self.num_regions+1),
                                      columns=range(1, self.num_regions+1))
        daily_flow_df.to_csv(f'{output_dir}/od_matrices_csv/daily_flow_matrix.csv')
        print(f"  ✓ 全天流量矩阵: od_matrices_csv/daily_flow_matrix.csv")
        
        # 保存长格式OD表（便于数据库导入和分析）
        od_records = []
        for t in range(self.num_time_slots):
            for o in range(self.num_regions):
                for d in range(self.num_regions):
                    if od_flow[t, o, d] > 0:
                        od_records.append({
                            'time_slot': t,
                            'hour': t // 4,
                            'origin_region': o + 1,
                            'dest_region': d + 1,
                            'flow': int(od_flow[t, o, d]),
                            'avg_travel_time': od_time[t, o, d]
                        })
        
        od_table = pd.DataFrame(od_records)
        od_table.to_csv(f'{output_dir}/od_flow_table.csv', index=False, encoding='utf-8-sig')
        print(f"  ✓ OD流量表: od_flow_table.csv ({len(od_table)} 条记录)")
        
        # 4. 生成统计报告
        with open(f'{output_dir}/od_matrix_report.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("OD矩阵构建报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. 数据统计\n")
            f.write(f"   总trip数: {len(trips_df)}\n")
            f.write(f"   时间范围: {trips_df['start_time'].min()} 到 {trips_df['end_time'].max()}\n")
            f.write(f"   平均出行时长: {trips_df['duration_minutes'].mean():.2f} 分钟\n")
            f.write(f"   中位出行时长: {trips_df['duration_minutes'].median():.2f} 分钟\n\n")
            
            f.write("2. OD矩阵维度\n")
            f.write(f"   时间槽数: {self.num_time_slots}\n")
            f.write(f"   区域数: {self.num_regions}\n")
            f.write(f"   矩阵形状: ({self.num_time_slots}, {self.num_regions}, {self.num_regions})\n\n")
            
            f.write("3. 稀疏性分析\n")
            total_cells = od_flow.size
            non_zero = (od_flow > 0).sum()
            sparsity = 1 - (non_zero / total_cells)
            f.write(f"   总单元数: {total_cells}\n")
            f.write(f"   非零单元: {non_zero}\n")
            f.write(f"   稀疏度: {sparsity*100:.2f}%\n\n")
            
            f.write("4. 热门OD对 (Top 20)\n")
            daily_flow = od_flow.sum(axis=0)
            top_od = []
            for o in range(self.num_regions):
                for d in range(self.num_regions):
                    if daily_flow[o, d] > 0:
                        top_od.append((o+1, d+1, daily_flow[o, d]))
            top_od.sort(key=lambda x: x[2], reverse=True)
            
            for rank, (o, d, flow) in enumerate(top_od[:20], 1):
                f.write(f"   {rank:2d}. 区域{o:2d} → 区域{d:2d}: {int(flow):4d} trips\n")
        
        print(f"  ✓ 统计报告: od_matrix_report.txt")
        print(f"\n保存完成!")


def main():
    """主函数"""
    print("=" * 80)
    print("独立OD矩阵构建程序")
    print("=" * 80)
    
    # 初始化
    builder = ODMatrixBuilder(num_regions=29, num_time_slots=96)
    
    # 1. 加载轨迹数据（尝试step4数据）
    traj_file = 'output/step4_time_mapped_sample.csv'
    if not os.path.exists(traj_file):
        print(f"错误: 找不到文件 {traj_file}")
        print("请先运行main.py的步骤1-4")
        return
    
    df = builder.load_trajectory_data(traj_file)
    
    # 检查必要的列
    required_cols = ['taxi_id', 'date_time', 'region_id', 'time_slot']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误: 缺少必要的列: {missing_cols}")
        return
    
    # 2. 提取trips
    trips_df = builder.extract_trips(df, time_threshold_minutes=30)
    
    if len(trips_df) == 0:
        print("错误: 未提取到任何trip")
        return
    
    # 3. 构建OD矩阵
    od_flow, od_time = builder.build_od_matrices(trips_df)
    
    # 4. 可视化
    builder.visualize_od_matrix(od_flow, od_time)
    
    # 5. 保存结果
    builder.save_od_matrices(od_flow, od_time, trips_df)
    
    print("\n" + "=" * 80)
    print("OD矩阵构建完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
