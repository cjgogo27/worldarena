"""
OD流量矩阵生成模块
功能：根据轨迹数据生成OD流量矩阵和平均旅行时间
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import h5py
from tqdm import tqdm


class ODMatrixGenerator:
    """OD流量矩阵生成器"""
    
    def __init__(self, num_regions=29, interval_minutes=15):
        """
        初始化OD矩阵生成器
        
        Args:
            num_regions: 区域数量
            interval_minutes: 时间间隔（分钟）
        """
        self.num_regions = num_regions
        self.interval_minutes = interval_minutes
        self.slots_per_day = (24 * 60) // interval_minutes
    
    def extract_trips(self, trajectory_df, time_col='date_time', 
                     region_col='region_id', vehicle_col='taxi_id',
                     time_threshold_minutes=30, distance_threshold=None):
        """
        从轨迹数据中提取出行记录（OD对）
        
        Args:
            trajectory_df: 轨迹DataFrame，需包含时间、区域、车辆ID
            time_col: 时间列名
            region_col: 区域ID列名
            vehicle_col: 车辆ID列名
            time_threshold_minutes: 停留时间阈值（分钟），超过此时间视为新的出行
            distance_threshold: 距离阈值（可选）
            
        Returns:
            trips_df: 出行记录DataFrame，包含起点、终点、时间等信息
        """
        print(f"开始从轨迹数据提取出行记录...")
        print(f"原始轨迹点数: {len(trajectory_df)}")
        
        # 确保数据按车辆ID和时间排序
        trajectory_df = trajectory_df.sort_values([vehicle_col, time_col]).reset_index(drop=True)
        
        # 过滤掉区域ID为空的记录
        trajectory_df = trajectory_df[trajectory_df[region_col].notna()].copy()
        print(f"过滤后轨迹点数: {len(trajectory_df)}")
        
        trips = []
        
        # 按车辆分组处理
        vehicle_groups = list(trajectory_df.groupby(vehicle_col))
        print(f"正在处理 {len(vehicle_groups)} 辆车的轨迹...")
        
        # 添加更详细的进度信息
        from datetime import datetime
        start_time = datetime.now()
        
        # 使用更细粒度的进度反馈
        for idx, (vehicle_id, group) in enumerate(tqdm(vehicle_groups, desc="Extracting trips", 
                                                         miniters=1, mininterval=1.0)):
            # 每处理10辆车显示一次详细进度（从100改为10）
            if idx > 0 and idx % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / idx
                remaining = avg_time * (len(vehicle_groups) - idx)
                print(f"\r  [{idx}/{len(vehicle_groups)}] vehicles | "
                      f"Trips: {len(trips)} | "
                      f"Avg: {avg_time:.2f}s/veh | "
                      f"ETA: {remaining/60:.1f}min", end='', flush=True)
            
            group = group.reset_index(drop=True)
            
            if len(group) < 2:
                continue
            
            i = 0
            while i < len(group) - 1:
                origin_idx = i
                origin = group.iloc[origin_idx]
                
                # 寻找下一个不同区域的点作为目的地
                j = i + 1
                while j < len(group):
                    dest = group.iloc[j]
                    
                    # 检查时间间隔
                    time_diff = (dest[time_col] - origin[time_col]).total_seconds() / 60
                    
                    # 如果时间间隔过长，认为是新的出行起点
                    if time_diff > time_threshold_minutes:
                        break
                    
                    # 如果区域变化，记录这次出行
                    if dest[region_col] != origin[region_col]:
                        trip = {
                            'vehicle_id': vehicle_id,
                            'origin_region': int(origin[region_col]),
                            'dest_region': int(dest[region_col]),
                            'start_time': origin[time_col],
                            'end_time': dest[time_col],
                            'duration_minutes': time_diff,
                            'start_lon': origin.get('longitude', None),
                            'start_lat': origin.get('latitude', None),
                            'end_lon': dest.get('longitude', None),
                            'end_lat': dest.get('latitude', None)
                        }
                        
                        # 添加时间槽信息
                        if 'time_slot' in origin:
                            trip['start_time_slot'] = origin['time_slot']
                        if 'global_time_slot' in origin:
                            trip['start_global_slot'] = origin['global_time_slot']
                        if 'date' in origin:
                            trip['date'] = origin['date']
                        
                        trips.append(trip)
                        
                        # 从目的地开始下一次出行
                        i = j
                        break
                    
                    j += 1
                else:
                    # 没有找到有效的目的地，移动到下一个点
                    i += 1
                    continue
                
                if j >= len(group):
                    break
        
        trips_df = pd.DataFrame(trips)
        print(f"\n成功提取 {len(trips_df)} 条出行记录")
        
        if len(trips_df) > 0:
            print(f"平均出行时间: {trips_df['duration_minutes'].mean():.2f} 分钟")
            print(f"出行时间范围: {trips_df['start_time'].min()} 到 {trips_df['end_time'].max()}")
            print(f"涉及区域数: 起点 {trips_df['origin_region'].nunique()} 个, 终点 {trips_df['dest_region'].nunique()} 个")
        
        return trips_df
    
    def create_od_matrix(self, trips_df, time_slot_col='start_time_slot',
                        origin_col='origin_region', dest_col='dest_region'):
        """
        创建OD流量矩阵
        
        Args:
            trips_df: 出行记录DataFrame
            time_slot_col: 时间槽列名
            origin_col: 起点列名
            dest_col: 终点列名
            
        Returns:
            od_matrices: dict，key为时间槽，value为OD矩阵(numpy array)
        """
        print(f"\n开始生成OD流量矩阵...")
        
        od_matrices = {}
        
        # 按时间槽分组
        if time_slot_col in trips_df.columns:
            time_slots = sorted(trips_df[time_slot_col].unique())
        else:
            print(f"警告: 未找到时间槽列 '{time_slot_col}'，将生成总体OD矩阵")
            time_slots = ['total']
        
        for slot in tqdm(time_slots, desc="生成OD矩阵"):
            # 筛选该时间槽的出行
            if slot == 'total':
                slot_trips = trips_df
            else:
                slot_trips = trips_df[trips_df[time_slot_col] == slot]
            
            # 初始化矩阵
            matrix = np.zeros((self.num_regions, self.num_regions), dtype=np.int32)
            
            # 填充矩阵
            for _, trip in slot_trips.iterrows():
                o = int(trip[origin_col]) - 1  # 转换为0-based索引
                d = int(trip[dest_col]) - 1
                
                if 0 <= o < self.num_regions and 0 <= d < self.num_regions:
                    matrix[o, d] += 1
            
            od_matrices[slot] = matrix
        
        print(f"成功生成 {len(od_matrices)} 个时间槽的OD矩阵")
        
        return od_matrices
    
    def create_od_travel_time_matrix(self, trips_df, time_slot_col='start_time_slot',
                                    origin_col='origin_region', dest_col='dest_region',
                                    duration_col='duration_minutes'):
        """
        创建OD平均旅行时间矩阵
        
        Args:
            trips_df: 出行记录DataFrame
            time_slot_col: 时间槽列名
            origin_col: 起点列名
            dest_col: 终点列名
            duration_col: 持续时间列名
            
        Returns:
            time_matrices: dict，key为时间槽，value为平均旅行时间矩阵
        """
        print(f"\n开始生成OD平均旅行时间矩阵...")
        
        time_matrices = {}
        
        # 按时间槽分组
        if time_slot_col in trips_df.columns:
            time_slots = sorted(trips_df[time_slot_col].unique())
        else:
            time_slots = ['total']
        
        for slot in tqdm(time_slots, desc="生成时间矩阵"):
            # 筛选该时间槽的出行
            if slot == 'total':
                slot_trips = trips_df
            else:
                slot_trips = trips_df[trips_df[time_slot_col] == slot]
            
            # 初始化矩阵（用于累加时间和计数）
            time_sum = np.zeros((self.num_regions, self.num_regions), dtype=np.float64)
            count = np.zeros((self.num_regions, self.num_regions), dtype=np.int32)
            
            # 累加旅行时间
            for _, trip in slot_trips.iterrows():
                o = int(trip[origin_col]) - 1
                d = int(trip[dest_col]) - 1
                
                if 0 <= o < self.num_regions and 0 <= d < self.num_regions:
                    time_sum[o, d] += trip[duration_col]
                    count[o, d] += 1
            
            # 计算平均值
            avg_time = np.zeros((self.num_regions, self.num_regions), dtype=np.float32)
            mask = count > 0
            avg_time[mask] = time_sum[mask] / count[mask]
            
            time_matrices[slot] = avg_time
        
        print(f"成功生成 {len(time_matrices)} 个时间槽的平均时间矩阵")
        
        return time_matrices
    
    def save_od_matrices_to_h5(self, od_matrices, time_matrices, output_path,
                              region_mapping_df=None):
        """
        将OD矩阵保存为HDF5格式
        
        Args:
            od_matrices: OD流量矩阵字典
            time_matrices: OD平均时间矩阵字典
            output_path: 输出文件路径
            region_mapping_df: 区域映射DataFrame（可选）
        """
        print(f"\n保存OD矩阵到 {output_path}...")
        
        with h5py.File(output_path, 'w') as f:
            # 保存流量矩阵
            flow_group = f.create_group('flow')
            for slot, matrix in od_matrices.items():
                flow_group.create_dataset(str(slot), data=matrix, compression='gzip')
            
            # 保存时间矩阵
            time_group = f.create_group('travel_time')
            for slot, matrix in time_matrices.items():
                time_group.create_dataset(str(slot), data=matrix, compression='gzip')
            
            # 保存元数据
            f.attrs['num_regions'] = self.num_regions
            f.attrs['interval_minutes'] = self.interval_minutes
            f.attrs['num_time_slots'] = len(od_matrices)
            
            # 保存区域映射
            if region_mapping_df is not None:
                region_group = f.create_group('regions')
                region_group.create_dataset('region_id', data=region_mapping_df['region_id'].values)
                region_group.create_dataset('region_code', 
                    data=region_mapping_df['region_code'].astype(str).values.astype('S'))
                region_group.create_dataset('region_name', 
                    data=region_mapping_df['region_name'].values.astype('S'))
        
        print(f"OD矩阵保存成功！")
    
    def export_od_to_csv(self, od_matrices, time_matrices, output_dir,
                        region_mapping_df=None):
        """
        将OD矩阵导出为CSV格式（长表格式）
        
        Args:
            od_matrices: OD流量矩阵字典
            time_matrices: OD平均时间矩阵字典
            output_dir: 输出目录
            region_mapping_df: 区域映射DataFrame（可选）
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n导出OD矩阵到 {output_dir}...")
        
        all_records = []
        
        for slot in od_matrices.keys():
            flow_matrix = od_matrices[slot]
            time_matrix = time_matrices[slot]
            
            for o in range(self.num_regions):
                for d in range(self.num_regions):
                    if flow_matrix[o, d] > 0:  # 只保存有流量的OD对
                        record = {
                            'time_slot': slot,
                            'origin_region': o + 1,  # 转换回1-based
                            'dest_region': d + 1,
                            'flow': int(flow_matrix[o, d]),
                            'avg_travel_time': float(time_matrix[o, d])
                        }
                        
                        # 添加区域名称
                        if region_mapping_df is not None:
                            origin_info = region_mapping_df[region_mapping_df['region_id'] == o + 1]
                            dest_info = region_mapping_df[region_mapping_df['region_id'] == d + 1]
                            
                            if len(origin_info) > 0:
                                record['origin_name'] = origin_info.iloc[0]['region_name']
                            if len(dest_info) > 0:
                                record['dest_name'] = dest_info.iloc[0]['region_name']
                        
                        all_records.append(record)
        
        df_od = pd.DataFrame(all_records)
        output_path = f"{output_dir}/od_flow_15min.csv"
        df_od.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"OD流量数据已导出到: {output_path}")
        print(f"总共 {len(df_od)} 条OD记录")
        
        return df_od


if __name__ == "__main__":
    # 测试代码
    print("OD矩阵生成器模块")
    generator = ODMatrixGenerator(num_regions=29, interval_minutes=15)
    print(f"区域数量: {generator.num_regions}")
    print(f"时间间隔: {generator.interval_minutes} 分钟")
    print(f"每日时间槽数: {generator.slots_per_day}")
