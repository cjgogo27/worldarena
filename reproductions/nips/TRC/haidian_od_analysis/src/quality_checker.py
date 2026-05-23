"""
数据质量检查模块
功能：检查数据完整性、一致性和质量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, num_regions=29):
        """初始化检查器"""
        self.num_regions = num_regions
        self.report = {}
    
    def check_trajectory_data(self, df, required_cols=None):
        """
        检查轨迹数据质量
        
        Args:
            df: 轨迹DataFrame
            required_cols: 必需的列名列表
            
        Returns:
            检查报告字典
        """
        print("=" * 60)
        print("轨迹数据质量检查")
        print("=" * 60)
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_records': len(df),
            'checks': []
        }
        
        # 1. 检查必需字段
        if required_cols is None:
            required_cols = ['taxi_id', 'date_time', 'longitude', 'latitude']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            report['checks'].append({
                'name': '必需字段检查',
                'status': 'FAIL',
                'message': f"缺少字段: {missing_cols}"
            })
            print(f"❌ 缺少必需字段: {missing_cols}")
        else:
            report['checks'].append({
                'name': '必需字段检查',
                'status': 'PASS',
                'message': '所有必需字段存在'
            })
            print(f"✓ 所有必需字段存在")
        
        # 2. 检查空值
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols) > 0:
            report['checks'].append({
                'name': '空值检查',
                'status': 'WARNING',
                'message': f"存在空值: {null_cols.to_dict()}"
            })
            print(f"\n⚠ 空值统计:")
            for col, count in null_cols.items():
                pct = count / len(df) * 100
                print(f"  {col}: {count} ({pct:.2f}%)")
        else:
            report['checks'].append({
                'name': '空值检查',
                'status': 'PASS',
                'message': '无空值'
            })
            print(f"✓ 无空值")
        
        # 3. 检查经纬度范围（北京范围）
        if 'longitude' in df.columns and 'latitude' in df.columns:
            # 北京大致范围: 经度 115.7-117.4, 纬度 39.4-41.6
            # 海淀区范围: 经度 116.0-116.5, 纬度 39.9-40.2
            lon_valid = df['longitude'].between(115.0, 118.0)
            lat_valid = df['latitude'].between(39.0, 42.0)
            invalid_coords = (~lon_valid) | (~lat_valid)
            invalid_count = invalid_coords.sum()
            
            if invalid_count > 0:
                pct = invalid_count / len(df) * 100
                report['checks'].append({
                    'name': '坐标范围检查',
                    'status': 'WARNING',
                    'message': f"异常坐标数量: {invalid_count} ({pct:.2f}%)"
                })
                print(f"\n⚠ 异常坐标: {invalid_count} 条 ({pct:.2f}%)")
                print(f"  经度范围: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]")
                print(f"  纬度范围: [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]")
            else:
                report['checks'].append({
                    'name': '坐标范围检查',
                    'status': 'PASS',
                    'message': '所有坐标在合理范围内'
                })
                print(f"✓ 所有坐标在合理范围内")
        
        # 4. 检查时间范围和顺序
        if 'date_time' in df.columns:
            try:
                time_series = pd.to_datetime(df['date_time'])
                report['checks'].append({
                    'name': '时间范围',
                    'status': 'INFO',
                    'message': f"从 {time_series.min()} 到 {time_series.max()}"
                })
                print(f"\n📅 时间范围: {time_series.min()} 到 {time_series.max()}")
                print(f"  总天数: {time_series.dt.date.nunique()}")
            except:
                report['checks'].append({
                    'name': '时间格式检查',
                    'status': 'FAIL',
                    'message': '时间格式错误'
                })
                print(f"❌ 时间格式错误")
        
        # 5. 检查车辆ID
        if 'taxi_id' in df.columns:
            num_vehicles = df['taxi_id'].nunique()
            records_per_vehicle = len(df) / num_vehicles
            report['checks'].append({
                'name': '车辆统计',
                'status': 'INFO',
                'message': f"{num_vehicles} 辆车, 平均每车 {records_per_vehicle:.1f} 条记录"
            })
            print(f"\n🚕 车辆数量: {num_vehicles}")
            print(f"  平均每车记录数: {records_per_vehicle:.1f}")
        
        # 6. 检查区域ID（如果已经映射）
        if 'region_id' in df.columns:
            valid_regions = df['region_id'].between(1, self.num_regions)
            invalid_regions = (~valid_regions) & (df['region_id'].notna())
            invalid_count = invalid_regions.sum()
            
            if invalid_count > 0:
                report['checks'].append({
                    'name': '区域ID检查',
                    'status': 'WARNING',
                    'message': f"非法区域ID: {invalid_count} 条"
                })
                print(f"\n⚠ 非法区域ID: {invalid_count} 条")
            else:
                mapped = df['region_id'].notna().sum()
                unmapped = df['region_id'].isna().sum()
                pct_mapped = mapped / len(df) * 100
                report['checks'].append({
                    'name': '区域映射率',
                    'status': 'INFO',
                    'message': f"映射率: {pct_mapped:.2f}% ({mapped}/{len(df)})"
                })
                print(f"\n🗺️  区域映射: {mapped} 条已映射 ({pct_mapped:.2f}%), {unmapped} 条未映射")
                print(f"  涉及区域数: {df['region_id'].nunique()}")
        
        self.report['trajectory_check'] = report
        print("\n" + "=" * 60)
        return report
    
    def check_trips_data(self, trips_df):
        """检查出行数据质量"""
        print("=" * 60)
        print("出行数据质量检查")
        print("=" * 60)
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_trips': len(trips_df),
            'checks': []
        }
        
        # 1. 检查出行时长
        if 'duration_minutes' in trips_df.columns:
            # 异常出行时长（过短或过长）
            too_short = trips_df['duration_minutes'] < 1
            too_long = trips_df['duration_minutes'] > 120  # 超过2小时
            
            print(f"\n⏱️  出行时长统计:")
            print(f"  平均: {trips_df['duration_minutes'].mean():.2f} 分钟")
            print(f"  中位数: {trips_df['duration_minutes'].median():.2f} 分钟")
            print(f"  范围: [{trips_df['duration_minutes'].min():.2f}, {trips_df['duration_minutes'].max():.2f}]")
            print(f"  过短(<1min): {too_short.sum()} 条")
            print(f"  过长(>120min): {too_long.sum()} 条")
            
            report['checks'].append({
                'name': '出行时长',
                'status': 'INFO',
                'message': f"平均 {trips_df['duration_minutes'].mean():.2f} 分钟"
            })
        
        # 2. 检查OD对分布
        if 'origin_region' in trips_df.columns and 'dest_region' in trips_df.columns:
            num_od_pairs = len(trips_df.groupby(['origin_region', 'dest_region']))
            max_possible = self.num_regions * (self.num_regions - 1)  # 排除自环
            coverage = num_od_pairs / max_possible * 100
            
            print(f"\n🔄 OD对统计:")
            print(f"  不同OD对数量: {num_od_pairs}")
            print(f"  理论最大数量: {max_possible} (不含自环)")
            print(f"  覆盖率: {coverage:.2f}%")
            
            # 检查自环（起点=终点）
            self_loops = trips_df['origin_region'] == trips_df['dest_region']
            print(f"  自环出行: {self_loops.sum()} 条 ({self_loops.sum()/len(trips_df)*100:.2f}%)")
            
            report['checks'].append({
                'name': 'OD对覆盖率',
                'status': 'INFO',
                'message': f"{num_od_pairs}/{max_possible} ({coverage:.2f}%)"
            })
        
        # 3. 检查时间分布
        if 'start_time' in trips_df.columns:
            trips_df['hour'] = pd.to_datetime(trips_df['start_time']).dt.hour
            hourly_dist = trips_df['hour'].value_counts().sort_index()
            
            print(f"\n📊 时段分布:")
            print(f"  最忙时段: {hourly_dist.idxmax()}:00 ({hourly_dist.max()} 次出行)")
            print(f"  最闲时段: {hourly_dist.idxmin()}:00 ({hourly_dist.min()} 次出行)")
        
        # 4. 检查空间覆盖
        if 'origin_region' in trips_df.columns:
            origin_coverage = trips_df['origin_region'].nunique()
            dest_coverage = trips_df['dest_region'].nunique()
            
            print(f"\n📍 空间覆盖:")
            print(f"  起点覆盖: {origin_coverage}/{self.num_regions} 个区域")
            print(f"  终点覆盖: {dest_coverage}/{self.num_regions} 个区域")
            
            # 找出没有覆盖的区域
            all_regions = set(range(1, self.num_regions + 1))
            origin_regions = set(trips_df['origin_region'].unique())
            dest_regions = set(trips_df['dest_region'].unique())
            
            missing_origins = all_regions - origin_regions
            missing_dests = all_regions - dest_regions
            
            if missing_origins:
                print(f"  未作为起点的区域: {sorted(missing_origins)}")
            if missing_dests:
                print(f"  未作为终点的区域: {sorted(missing_dests)}")
        
        self.report['trips_check'] = report
        print("\n" + "=" * 60)
        return report
    
    def check_od_matrices(self, od_matrices, time_matrices):
        """检查OD矩阵质量"""
        print("=" * 60)
        print("OD矩阵质量检查")
        print("=" * 60)
        
        # 1. 检查矩阵维度
        print(f"\n📏 矩阵维度检查:")
        for slot, matrix in list(od_matrices.items())[:3]:
            print(f"  时间槽 {slot}: {matrix.shape}")
            if matrix.shape != (self.num_regions, self.num_regions):
                print(f"    ❌ 维度错误！应为 ({self.num_regions}, {self.num_regions})")
        
        # 2. 统计流量
        total_flow = sum(matrix.sum() for matrix in od_matrices.values())
        avg_flow_per_slot = total_flow / len(od_matrices)
        
        print(f"\n🚦 流量统计:")
        print(f"  总流量: {int(total_flow)}")
        print(f"  时间槽数量: {len(od_matrices)}")
        print(f"  平均每时间槽流量: {avg_flow_per_slot:.2f}")
        
        # 3. 检查稀疏性
        sample_matrix = list(od_matrices.values())[0]
        non_zero = (sample_matrix > 0).sum()
        total_cells = self.num_regions * self.num_regions
        sparsity = (total_cells - non_zero) / total_cells * 100
        
        print(f"\n💧 稀疏性分析 (首个时间槽):")
        print(f"  非零元素: {non_zero}/{total_cells}")
        print(f"  稀疏率: {sparsity:.2f}%")
        
        # 4. 检查时间矩阵
        print(f"\n⏰ 平均旅行时间统计:")
        all_times = []
        for matrix in time_matrices.values():
            valid_times = matrix[matrix > 0]
            if len(valid_times) > 0:
                all_times.extend(valid_times)
        
        if all_times:
            all_times = np.array(all_times)
            print(f"  平均: {all_times.mean():.2f} 分钟")
            print(f"  中位数: {np.median(all_times):.2f} 分钟")
            print(f"  范围: [{all_times.min():.2f}, {all_times.max():.2f}]")
        
        print("\n" + "=" * 60)
    
    def visualize_od_heatmap(self, od_matrix, time_slot='total', output_path=None):
        """可视化OD流量热力图"""
        plt.figure(figsize=(12, 10))
        
        # 使用对数scale以便更好地显示
        matrix_log = np.log1p(od_matrix)  # log(1+x)
        
        sns.heatmap(matrix_log, cmap='YlOrRd', cbar_kws={'label': 'log(1+flow)'})
        plt.title(f'OD Flow Heatmap - Time Slot {time_slot}')
        plt.xlabel('Destination Region')
        plt.ylabel('Origin Region')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"热力图已保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_summary_report(self, output_path=None):
        """生成汇总报告"""
        print("\n" + "=" * 60)
        print("质量检查汇总报告")
        print("=" * 60)
        
        for check_name, check_result in self.report.items():
            print(f"\n【{check_name}】")
            if 'checks' in check_result:
                for check in check_result['checks']:
                    status_icon = {'PASS': '✓', 'WARNING': '⚠', 'FAIL': '❌', 'INFO': 'ℹ'}
                    icon = status_icon.get(check['status'], '•')
                    print(f"  {icon} {check['name']}: {check['message']}")
        
        if output_path:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.report, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n报告已保存到: {output_path}")
        
        print("=" * 60)


if __name__ == "__main__":
    print("数据质量检查模块")
    checker = DataQualityChecker(num_regions=29)
    print(f"配置: {checker.num_regions} 个区域")
