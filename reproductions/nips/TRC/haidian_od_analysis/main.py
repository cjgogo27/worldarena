#!/usr/bin/env python3
"""
海淀区OD流量分析主程序
功能：完整的数据处理流程，从原始轨迹数据生成15分钟粒度的OD流量矩阵和平均旅行时间

作者：数据分析团队
日期：2026-02-03
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from region_processor import RegionProcessor
from time_processor import TimeProcessor, create_time_slots_reference
from od_matrix_generator import ODMatrixGenerator
from quality_checker import DataQualityChecker


class HaidianODAnalysisPipeline:
    """海淀区OD分析完整流程"""
    
    def __init__(self, config):
        """
        初始化流程
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.region_processor = None
        self.time_processor = None
        self.od_generator = None
        self.quality_checker = None
        
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], 'visualizations'), exist_ok=True)
        
        print("=" * 80)
        print("海淀区OD流量分析系统")
        print("=" * 80)
        print(f"配置信息:")
        print(f"  区域数量: {config['num_regions']}")
        print(f"  时间粒度: {config['interval_minutes']} 分钟")
        print(f"  输出目录: {config['output_dir']}")
        print("=" * 80)
    
    def step1_load_and_prepare_regions(self):
        """步骤1: 加载和准备区域数据"""
        print("\n" + "=" * 80)
        print("步骤 1/6: 加载区域数据")
        print("=" * 80)
        
        self.region_processor = RegionProcessor(
            shapefile_path=self.config['shapefile_path'],
            region_mapping_path=self.config['region_mapping_path']
        )
        
        # 加载区域边界
        self.region_processor.load_regions()
        
        # 加载区域映射表
        self.region_mapping = self.region_processor.load_region_mapping()
        
        # 保存区域信息到CSV
        step1_csv = os.path.join(self.config['output_dir'], 'step1_regions.csv')
        self.region_mapping.to_csv(step1_csv, index=False, encoding='utf-8-sig')
        print(f"✓ 区域信息已保存: {step1_csv}")
        
        # 导出区域边界为GeoJSON
        geojson_path = os.path.join(self.config['output_dir'], 'step1_regions.geojson')
        self.region_processor.export_region_geojson(geojson_path)
        
        # 获取海淀区实际边界范围（用于轨迹数据过滤）
        bounds = self.region_processor.regions_gdf.total_bounds
        self.bounds = {
            'min_lon': bounds[0],
            'max_lon': bounds[2],
            'min_lat': bounds[1],
            'max_lat': bounds[3]
        }
        print(f"\n海淀区实际边界范围:")
        print(f"  经度: {self.bounds['min_lon']:.6f} - {self.bounds['max_lon']:.6f}")
        print(f"  纬度: {self.bounds['min_lat']:.6f} - {self.bounds['max_lat']:.6f}")
        
        # 绘制区域分布图
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 10))
        self.region_processor.regions_gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue', alpha=0.5)
        ax.set_title('Haidian District 29 Administrative Regions', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        vis_path = os.path.join(self.config['output_dir'], 'visualizations', 'step1_regions_map.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 区域分布图已保存: {vis_path}")
        
        print(f"\n✓ Step 1 completed! Loaded {len(self.region_mapping)} regions")
        return self.region_mapping
    
    def step2_load_and_filter_trajectory(self):
        """步骤2: 加载和过滤轨迹数据"""
        print("\n" + "=" * 80)
        print("步骤 2/6: 加载轨迹数据")
        print("=" * 80)
        
        # 读取轨迹数据
        print(f"正在读取: {self.config['trajectory_path']}")
        
        # 根据文件大小决定是否分块读取
        file_size = os.path.getsize(self.config['trajectory_path']) / (1024**2)  # MB
        print(f"文件大小: {file_size:.2f} MB")
        
        if file_size > 1000:  # 大于1GB
            print("文件较大，使用分块读取...")
            # 分块读取并处理
            chunk_size = 1000000
            chunks = []
            for chunk in pd.read_csv(self.config['trajectory_path'], chunksize=chunk_size):
                # 基本过滤
                chunk = chunk.dropna(subset=['taxi_id', 'date_time', 'longitude', 'latitude'])
                chunks.append(chunk)
                if len(chunks) >= 10:  # 限制内存使用
                    break
            df_traj = pd.concat(chunks, ignore_index=True)
        else:
            df_traj = pd.read_csv(self.config['trajectory_path'])
        
        print(f"原始数据: {len(df_traj)} 条记录")
        
        # 数据质量检查
        self.quality_checker = DataQualityChecker(num_regions=self.config['num_regions'])
        self.quality_checker.check_trajectory_data(df_traj)
        
        # 基本清洗
        df_traj = df_traj.dropna(subset=['taxi_id', 'date_time', 'longitude', 'latitude'])
        
        # 使用Shapefile实际边界 + 小缓冲区（0.01度约1km）
        buffer = 0.01
        min_lon = self.bounds['min_lon'] - buffer
        max_lon = self.bounds['max_lon'] + buffer
        min_lat = self.bounds['min_lat'] - buffer
        max_lat = self.bounds['max_lat'] + buffer
        
        print("\n轨迹清洗规则:")
        print("  1. 移除空值记录 (taxi_id, date_time, longitude, latitude)")
        print(f"  2. 经度范围过滤: {min_lon:.6f} - {max_lon:.6f} (海淀区实际边界+1km缓冲)")
        print(f"  3. 纬度范围过滤: {min_lat:.6f} - {max_lat:.6f} (海淀区实际边界+1km缓冲)")
        print("  4. 移除重复记录")
        print("  5. 移除异常速度点 (可选，当前未启用)")
        
        # 过滤经纬度范围（海淀区实际边界 + 缓冲区）
        before_filter = len(df_traj)
        df_traj = df_traj[
            (df_traj['longitude'].between(min_lon, max_lon)) &
            (df_traj['latitude'].between(min_lat, max_lat))
        ].copy()
        after_filter = len(df_traj)
        
        print(f"\n坐标范围过滤: 移除 {before_filter - after_filter} 条记录")
        
        # 移除重复记录
        before_dedup = len(df_traj)
        df_traj = df_traj.drop_duplicates(subset=['taxi_id', 'date_time', 'longitude', 'latitude'])
        after_dedup = len(df_traj)
        
        print(f"去重处理: 移除 {before_dedup - after_dedup} 条重复记录")
        print(f"最终清洗后数据: {len(df_traj)} 条记录")
        
        # 保存清洗后的数据样本（前10000条）
        step2_csv = os.path.join(self.config['output_dir'], 'step2_cleaned_trajectory_sample.csv')
        df_traj.head(10000).to_csv(step2_csv, index=False, encoding='utf-8-sig')
        print(f"✓ 清洗后数据样本已保存: {step2_csv}")
        
        # 绘制轨迹点分布图
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sample_size = min(50000, len(df_traj))
        sample = df_traj.sample(n=sample_size, random_state=42)
        ax.scatter(sample['longitude'], sample['latitude'], alpha=0.1, s=1, c='red')
        ax.set_title(f'Trajectory Points Distribution (Sample: {sample_size} points)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        vis_path = os.path.join(self.config['output_dir'], 'visualizations', 'step2_trajectory_distribution.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 轨迹分布图已保存: {vis_path}")
        
        self.df_trajectory = df_traj
        print(f"\n✓ Step 2 completed!")
        return df_traj
    
    def step3_spatial_mapping(self):
        """步骤3: 空间映射 - 将轨迹点映射到区域"""
        print("\n" + "=" * 80)
        print("步骤 3/6: 空间映射（轨迹点 -> 区域）")
        print("=" * 80)
        
        # 批量映射点到区域
        df_mapped = self.region_processor.batch_points_to_regions(
            self.df_trajectory,
            lon_col='longitude',
            lat_col='latitude'
        )
        
        # 移除未能映射到任何区域的点
        df_mapped = df_mapped[df_mapped['region_id'].notna()].copy()
        
        # 确保region_id为整数
        df_mapped['region_id'] = df_mapped['region_id'].astype(int)
        
        # 保存映射后的数据样本
        step3_csv = os.path.join(self.config['output_dir'], 'step3_mapped_trajectory_sample.csv')
        df_mapped.head(10000).to_csv(step3_csv, index=False, encoding='utf-8-sig')
        print(f"✓ 映射后数据样本已保存: {step3_csv}")
        
        # 统计每个区域的轨迹点数量
        region_counts = df_mapped['region_id'].value_counts().sort_index()
        region_stats = pd.DataFrame({
            'region_id': region_counts.index,
            'point_count': region_counts.values
        })
        # 合并区域名称
        region_stats = region_stats.merge(
            self.region_mapping[['region_id', 'region_name']], 
            on='region_id', 
            how='left'
        )
        
        step3_stats = os.path.join(self.config['output_dir'], 'step3_region_point_statistics.csv')
        region_stats.to_csv(step3_stats, index=False, encoding='utf-8-sig')
        print(f"✓ 区域统计已保存: {step3_stats}")
        
        # 绘制区域轨迹点分布柱状图
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Bar chart
        ax.bar(range(len(region_stats)), region_stats['point_count'])
        ax.set_xlabel('Region ID', fontsize=12)
        ax.set_ylabel('Number of Points', fontsize=12)
        ax.set_title('Point Distribution by Region', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        vis_path = os.path.join(self.config['output_dir'], 'visualizations', 'step3_region_distribution.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Region distribution chart saved: {vis_path}")
        
        print(f"\n✓ Step 3 completed! {len(df_mapped)} records, covering {df_mapped['region_id'].nunique()} regions")
        
        self.df_trajectory = df_mapped
        return df_mapped
    
    def step4_temporal_mapping(self):
        """步骤4: 时间映射 - 将时间映射到15分钟时间槽"""
        print("\n" + "=" * 80)
        print("步骤 4/6: 时间映射（时间 -> 15分钟时间槽）")
        print("=" * 80)
        
        self.time_processor = TimeProcessor(
            interval_minutes=self.config['interval_minutes']
        )
        
        # 处理时间序列
        df_with_time = self.time_processor.process_time_series(
            self.df_trajectory,
            time_col='date_time'
        )
        
        # 打印时间统计
        time_stats = self.time_processor.get_time_slot_statistics(df_with_time)
        print(f"\n时间统计:")
        for key, value in time_stats.items():
            print(f"  {key}: {value}")
        
        # 创建时间槽参考表
        time_ref_path = os.path.join(self.config['output_dir'], 'step4_time_slots_reference.csv')
        create_time_slots_reference(
            interval_minutes=self.config['interval_minutes'],
            output_path=time_ref_path
        )
        
        # 保存时间映射后的数据样本
        step4_csv = os.path.join(self.config['output_dir'], 'step4_time_mapped_sample.csv')
        df_with_time.head(10000).to_csv(step4_csv, index=False, encoding='utf-8-sig')
        print(f"✓ 时间映射数据样本已保存: {step4_csv}")
        
        # 统计时间槽分布
        time_slot_counts = df_with_time['time_slot'].value_counts().sort_index()
        time_slot_stats = pd.DataFrame({
            'time_slot': time_slot_counts.index,
            'record_count': time_slot_counts.values
        })
        
        step4_stats = os.path.join(self.config['output_dir'], 'step4_time_slot_statistics.csv')
        time_slot_stats.to_csv(step4_stats, index=False, encoding='utf-8-sig')
        print(f"✓ 时间槽统计已保存: {step4_stats}")
        
        # 绘制时间分布图
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # 24-hour distribution
        hourly = df_with_time['hour'].value_counts().sort_index()
        ax1.plot(hourly.index, hourly.values, marker='o', linewidth=2)
        ax1.fill_between(hourly.index, hourly.values, alpha=0.3)
        ax1.set_xlabel('Hour', fontsize=12)
        ax1.set_ylabel('Number of Records', fontsize=12)
        ax1.set_title('24-Hour Trajectory Point Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(24))
        
        # Time slot distribution (96 15-minute slots)
        ax2.bar(time_slot_stats['time_slot'], time_slot_stats['record_count'], width=0.8)
        ax2.set_xlabel('Time Slot (0-95)', fontsize=12)
        ax2.set_ylabel('Number of Records', fontsize=12)
        ax2.set_title('15-Minute Time Slot Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        vis_path = os.path.join(self.config['output_dir'], 'visualizations', 'step4_time_distribution.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 时间分布图已保存: {vis_path}")
        
        self.df_trajectory = df_with_time
        print(f"\n✓ Step 4 completed!")
        return df_with_time
    
    def step5_extract_trips_and_generate_od(self):
        """步骤5: 提取出行并生成OD矩阵"""
        print("\n" + "=" * 80)
        print("步骤 5/6: 提取出行记录并生成OD矩阵")
        print("=" * 80)
        
        self.od_generator = ODMatrixGenerator(
            num_regions=self.config['num_regions'],
            interval_minutes=self.config['interval_minutes']
        )
        
        # 提取出行记录
        print("\n正在提取出行记录...")
        df_trips = self.od_generator.extract_trips(
            self.df_trajectory,
            time_col='date_time',
            region_col='region_id',
            vehicle_col='taxi_id',
            time_threshold_minutes=self.config.get('trip_time_threshold', 30)
        )
        
        # 保存出行记录
        trips_path = os.path.join(self.config['output_dir'], 'step5_trips.csv')
        df_trips.to_csv(trips_path, index=False, encoding='utf-8-sig')
        print(f"✓ 出行记录已保存: {trips_path}")
        
        # 出行统计
        trip_stats = {
            'total_trips': len(df_trips),
            'avg_duration': df_trips['duration_minutes'].mean(),
            'median_duration': df_trips['duration_minutes'].median(),
            'min_duration': df_trips['duration_minutes'].min(),
            'max_duration': df_trips['duration_minutes'].max(),
            'unique_od_pairs': len(df_trips.groupby(['origin_region', 'dest_region']))
        }
        
        trip_stats_df = pd.DataFrame([trip_stats])
        step5_stats = os.path.join(self.config['output_dir'], 'step5_trip_statistics.csv')
        trip_stats_df.to_csv(step5_stats, index=False, encoding='utf-8-sig')
        print(f"✓ 出行统计已保存: {step5_stats}")
        
        # 绘制出行时长分布图
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Trip duration histogram
        ax1.hist(df_trips['duration_minutes'], bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Trip Duration (minutes)', fontsize=12)
        ax1.set_ylabel('Number of Trips', fontsize=12)
        ax1.set_title('Trip Duration Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Hourly distribution
        if 'start_time' in df_trips.columns:
            df_trips['hour'] = pd.to_datetime(df_trips['start_time']).dt.hour
            hourly_trips = df_trips['hour'].value_counts().sort_index()
            ax2.bar(hourly_trips.index, hourly_trips.values)
            ax2.set_xlabel('Hour', fontsize=12)
            ax2.set_ylabel('Number of Trips', fontsize=12)
            ax2.set_title('Trip Temporal Distribution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_xticks(range(24))
        
        # Top 10 OD pairs
        od_counts = df_trips.groupby(['origin_region', 'dest_region']).size().sort_values(ascending=False).head(10)
        od_labels = [f"{o}->{d}" for o, d in od_counts.index]
        ax3.barh(range(len(od_counts)), od_counts.values)
        ax3.set_yticks(range(len(od_counts)))
        ax3.set_yticklabels(od_labels, fontsize=10)
        ax3.set_xlabel('Number of Trips', fontsize=12)
        ax3.set_title('Top 10 OD Pairs', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Distance vs duration scatter
        if all(col in df_trips.columns for col in ['start_lon', 'start_lat', 'end_lon', 'end_lat']):
            # Simple Euclidean distance calculation
            df_trips['distance'] = np.sqrt(
                (df_trips['end_lon'] - df_trips['start_lon'])**2 + 
                (df_trips['end_lat'] - df_trips['start_lat'])**2
            ) * 111  # Approximate conversion to km
            
            sample = df_trips.sample(n=min(5000, len(df_trips)), random_state=42)
            ax4.scatter(sample['distance'], sample['duration_minutes'], alpha=0.3, s=10)
            ax4.set_xlabel('Distance (km)', fontsize=12)
            ax4.set_ylabel('Duration (minutes)', fontsize=12)
            ax4.set_title('Trip Distance vs Duration', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        vis_path = os.path.join(self.config['output_dir'], 'visualizations', 'step5_trip_analysis.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 出行分析图已保存: {vis_path}")
        
        # 质量检查
        self.quality_checker.check_trips_data(df_trips)
        
        # 生成OD流量矩阵
        print("\n正在生成OD流量矩阵...")
        od_matrices = self.od_generator.create_od_matrix(
            df_trips,
            time_slot_col='start_time_slot',
            origin_col='origin_region',
            dest_col='dest_region'
        )
        
        # 生成OD平均旅行时间矩阵
        print("\n正在生成OD平均旅行时间矩阵...")
        time_matrices = self.od_generator.create_od_travel_time_matrix(
            df_trips,
            time_slot_col='start_time_slot',
            origin_col='origin_region',
            dest_col='dest_region',
            duration_col='duration_minutes'
        )
        
        # 保存每个时间槽的OD矩阵统计
        od_summary = []
        for slot in sorted(od_matrices.keys()):
            flow_matrix = od_matrices[slot]
            time_matrix = time_matrices[slot]
            
            od_summary.append({
                'time_slot': slot,
                'total_flow': int(flow_matrix.sum()),
                'non_zero_pairs': int((flow_matrix > 0).sum()),
                'avg_flow_per_pair': float(flow_matrix[flow_matrix > 0].mean()) if (flow_matrix > 0).any() else 0,
                'avg_travel_time': float(time_matrix[time_matrix > 0].mean()) if (time_matrix > 0).any() else 0
            })
        
        od_summary_df = pd.DataFrame(od_summary)
        step5_od_summary = os.path.join(self.config['output_dir'], 'step5_od_summary_by_timeslot.csv')
        od_summary_df.to_csv(step5_od_summary, index=False, encoding='utf-8-sig')
        print(f"✓ OD矩阵汇总已保存: {step5_od_summary}")
        
        # 绘制OD流量热力图（选择几个代表性时间槽）
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        
        # Select 4 time slots: morning peak(8:00), noon(12:00), evening peak(18:00), night(22:00)
        representative_slots = [32, 48, 72, 88]  # Corresponding to 8:00, 12:00, 18:00, 22:00
        time_labels = ['08:00', '12:00', '18:00', '22:00']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, (slot, label) in enumerate(zip(representative_slots, time_labels)):
            if slot in od_matrices:
                matrix = od_matrices[slot]
                matrix_log = np.log1p(matrix)
                
                sns.heatmap(matrix_log, ax=axes[idx], cmap='YlOrRd', 
                           cbar_kws={'label': 'log(1+flow)'}, square=True)
                axes[idx].set_title(f'OD Flow Heatmap - {label}', fontsize=14, fontweight='bold')
                axes[idx].set_xlabel('Destination Region', fontsize=12)
                axes[idx].set_ylabel('Origin Region', fontsize=12)
        
        plt.tight_layout()
        vis_path = os.path.join(self.config['output_dir'], 'visualizations', 'step5_od_heatmaps.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ OD热力图已保存: {vis_path}")
        
        # Plot time slot flow trend
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        ax1.plot(od_summary_df['time_slot'], od_summary_df['total_flow'], 
                marker='o', linewidth=2, markersize=4)
        ax1.fill_between(od_summary_df['time_slot'], od_summary_df['total_flow'], alpha=0.3)
        ax1.set_xlabel('Time Slot', fontsize=12)
        ax1.set_ylabel('Total Flow', fontsize=12)
        ax1.set_title('Total Flow Trend by Time Slot', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(od_summary_df['time_slot'], od_summary_df['avg_travel_time'], 
                marker='s', linewidth=2, markersize=4, color='orange')
        ax2.fill_between(od_summary_df['time_slot'], od_summary_df['avg_travel_time'], alpha=0.3, color='orange')
        ax2.set_xlabel('Time Slot', fontsize=12)
        ax2.set_ylabel('Avg Travel Time (minutes)', fontsize=12)
        ax2.set_title('Average Travel Time by Time Slot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        vis_path = os.path.join(self.config['output_dir'], 'visualizations', 'step5_od_trends.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ OD趋势图已保存: {vis_path}")
        
        # 质量检查
        self.quality_checker.check_od_matrices(od_matrices, time_matrices)
        
        self.od_matrices = od_matrices
        self.time_matrices = time_matrices
        self.df_trips = df_trips
        
        print(f"\n✓ Step 5 completed!")
        return od_matrices, time_matrices, df_trips
    
    def step6_save_results(self):
        """步骤6: 保存结果"""
        print("\n" + "=" * 80)
        print("步骤 6/6: 保存结果")
        print("=" * 80)
        
        # 保存为HDF5格式
        h5_path = os.path.join(self.config['output_dir'], 'step6_final_od_matrices.h5')
        self.od_generator.save_od_matrices_to_h5(
            self.od_matrices,
            self.time_matrices,
            h5_path,
            region_mapping_df=self.region_mapping
        )
        
        # 保存为CSV格式（长表）
        csv_dir = os.path.join(self.config['output_dir'], 'final_results')
        df_od = self.od_generator.export_od_to_csv(
            self.od_matrices,
            self.time_matrices,
            csv_dir,
            region_mapping_df=self.region_mapping
        )
        
        # 生成总结报告
        report_path = os.path.join(self.config['output_dir'], 'step6_quality_report.json')
        self.quality_checker.generate_summary_report(report_path)
        
        # 生成最终汇总统计
        final_summary = {
            '总轨迹点数': len(self.df_trajectory),
            '总出行数': len(self.df_trips),
            '涉及区域数': self.df_trajectory['region_id'].nunique(),
            '时间槽数': len(self.od_matrices),
            '平均每时间槽流量': sum(m.sum() for m in self.od_matrices.values()) / len(self.od_matrices),
            '总OD对数': len(df_od),
            '平均出行时间_分钟': self.df_trips['duration_minutes'].mean()
        }
        
        summary_df = pd.DataFrame([final_summary])
        summary_path = os.path.join(self.config['output_dir'], 'step6_final_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"✓ 最终汇总已保存: {summary_path}")
        
        print(f"\n✓ Step 6 completed! All results saved to: {self.config['output_dir']}")
        
        # Print complete file list
        print(f"\n" + "=" * 80)
        print("Output File List (by step)")
        print("=" * 80)
        print(f"\nStep 1 - Region Data:")
        print(f"  • step1_regions.csv - Region list")
        print(f"  • step1_regions.geojson - Region boundaries")
        print(f"  • visualizations/step1_regions_map.png - Region map")
        
        print(f"\nStep 2 - Trajectory Data:")
        print(f"  • step2_cleaned_trajectory_sample.csv - Cleaned sample")
        print(f"  • visualizations/step2_trajectory_distribution.png - Trajectory distribution")
        
        print(f"\nStep 3 - Spatial Mapping:")
        print(f"  • step3_mapped_trajectory_sample.csv - Mapped sample")
        print(f"  • step3_region_point_statistics.csv - Region point statistics")
        print(f"  • visualizations/step3_region_distribution.png - Region distribution")
        
        print(f"\nStep 4 - Temporal Mapping:")
        print(f"  • step4_time_mapped_sample.csv - Time-mapped sample")
        print(f"  • step4_time_slots_reference.csv - Time slots reference")
        print(f"  • step4_time_slot_statistics.csv - Time slot statistics")
        print(f"  • visualizations/step4_time_distribution.png - Time distribution")
        
        print(f"\nStep 5 - OD Matrix Generation:")
        print(f"  • step5_trips.csv - Trip records")
        print(f"  • step5_trip_statistics.csv - Trip statistics")
        print(f"  • step5_od_summary_by_timeslot.csv - OD summary by time slot")
        print(f"  • visualizations/step5_trip_analysis.png - Trip analysis")
        print(f"  • visualizations/step5_od_heatmaps.png - OD heatmaps")
        print(f"  • visualizations/step5_od_trends.png - OD trends")
        
        print(f"\nStep 6 - Final Results:")
        print(f"  • step6_final_od_matrices.h5 - OD matrices (HDF5)")
        print(f"  • step6_final_summary.csv - Final summary")
        print(f"  • step6_quality_report.json - Quality report")
        print(f"  • final_results/od_flow_15min.csv - OD flow table")
        print("=" * 80)
    
    def run(self):
        """运行完整流程"""
        start_time = datetime.now()
        
        try:
            # 步骤1: 加载区域数据
            self.step1_load_and_prepare_regions()
            
            # 步骤2: 加载轨迹数据
            self.step2_load_and_filter_trajectory()
            
            # 步骤3: 空间映射
            self.step3_spatial_mapping()
            
            # 步骤4: 时间映射
            self.step4_temporal_mapping()
            
            # 步骤5: 提取出行和生成OD矩阵
            self.step5_extract_trips_and_generate_od()
            
            # 步骤6: 保存结果
            self.step6_save_results()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 80)
            print("✓ 处理完成！")
            print("=" * 80)
            print(f"总耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
            print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ 处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    
    # 配置参数
    config = {
        # 路径配置
        'shapefile_path': '/data/alice/cjtest/TRC/海淀区边界_110108_Shapefile_(poi86.com)/110108.shp',
        'region_mapping_path': '/data/alice/cjtest/TRC/haidian_od_analysis/config/region_mapping.csv',
        'trajectory_path': '/data/alice/cjtest/TRC/all_taxi_data.csv',
        'output_dir': '/data/alice/cjtest/TRC/haidian_od_analysis/output',
        
        # 参数配置
        'num_regions': 29,  # 22街道 + 7镇
        'interval_minutes': 15,  # 15分钟时间粒度
        'trip_time_threshold': 30,  # 出行时间阈值（分钟）
    }
    
    # 创建并运行流程
    pipeline = HaidianODAnalysisPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
