"""
时间处理模块
功能：将原始时间数据映射到15分钟时间粒度
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TimeProcessor:
    """时间处理器"""
    
    def __init__(self, interval_minutes=15):
        """
        初始化时间处理器
        
        Args:
            interval_minutes: 时间间隔（分钟），默认15分钟
        """
        self.interval_minutes = interval_minutes
    
    def round_to_interval(self, dt):
        """
        将时间向下取整到最近的时间间隔
        
        Args:
            dt: datetime对象或时间字符串
            
        Returns:
            取整后的datetime对象
        """
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        # 向下取整到最近的interval_minutes
        minutes = (dt.minute // self.interval_minutes) * self.interval_minutes
        return dt.replace(minute=minutes, second=0, microsecond=0)
    
    def create_time_slot_id(self, dt):
        """
        创建时间槽ID（每天从0开始）
        
        Args:
            dt: datetime对象
            
        Returns:
            时间槽ID (0-95，15分钟间隔时每天有96个时间槽)
        """
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        total_minutes = dt.hour * 60 + dt.minute
        slot_id = total_minutes // self.interval_minutes
        return slot_id
    
    def create_global_time_slot(self, dt, base_date=None):
        """
        创建全局时间槽ID（从基准日期开始计数）
        
        Args:
            dt: datetime对象
            base_date: 基准日期，默认为数据中最早的日期
            
        Returns:
            全局时间槽ID
        """
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        if base_date is None:
            base_date = dt.date()
        
        if isinstance(base_date, str):
            base_date = pd.to_datetime(base_date).date()
        elif isinstance(base_date, datetime):
            base_date = base_date.date()
        
        # 计算距离基准日期的天数
        days_diff = (dt.date() - base_date).days
        
        # 计算当天的时间槽
        day_slot = self.create_time_slot_id(dt)
        
        # 全局时间槽 = 天数 * 每天时间槽数 + 当天时间槽
        slots_per_day = (24 * 60) // self.interval_minutes
        global_slot = days_diff * slots_per_day + day_slot
        
        return global_slot
    
    def process_time_series(self, df, time_col='date_time'):
        """
        处理时间序列数据，添加时间相关字段
        
        Args:
            df: 包含时间列的DataFrame
            time_col: 时间列名
            
        Returns:
            添加了时间处理字段的DataFrame
        """
        print(f"开始处理时间序列数据，共 {len(df)} 条记录...")
        
        # 转换为datetime类型
        df[time_col] = pd.to_datetime(df[time_col])
        
        # 向下取整到时间间隔
        df['time_rounded'] = df[time_col].apply(self.round_to_interval)
        
        # 提取日期
        df['date'] = df['time_rounded'].dt.date
        
        # 提取小时
        df['hour'] = df['time_rounded'].dt.hour
        
        # 创建每日时间槽ID (0-95)
        df['time_slot'] = df['time_rounded'].apply(self.create_time_slot_id)
        
        # 创建全局时间槽ID
        base_date = df['date'].min()
        df['global_time_slot'] = df['time_rounded'].apply(
            lambda x: self.create_global_time_slot(x, base_date)
        )
        
        # 创建时间槽标签 (HH:MM-HH:MM格式)
        df['time_slot_label'] = df['time_rounded'].apply(
            lambda x: f"{x.strftime('%H:%M')}-{(x + timedelta(minutes=self.interval_minutes)).strftime('%H:%M')}"
        )
        
        print(f"时间处理完成！")
        print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
        print(f"时间槽数量: {df['time_slot'].nunique()} (每日), {df['global_time_slot'].nunique()} (全局)")
        
        return df
    
    def get_time_slot_statistics(self, df):
        """获取时间槽统计信息"""
        if 'time_slot' not in df.columns:
            raise ValueError("请先运行 process_time_series")
        
        stats = {
            'total_records': len(df),
            'date_range': (df['date'].min(), df['date'].max()),
            'total_days': df['date'].nunique(),
            'slots_per_day': (24 * 60) // self.interval_minutes,
            'total_slots': df['global_time_slot'].nunique(),
            'records_per_slot_avg': len(df) / df['global_time_slot'].nunique()
        }
        
        return stats


def create_time_slots_reference(interval_minutes=15, output_path=None):
    """
    创建时间槽参考表
    
    Args:
        interval_minutes: 时间间隔
        output_path: 输出文件路径（可选）
        
    Returns:
        时间槽参考DataFrame
    """
    slots_per_day = (24 * 60) // interval_minutes
    
    time_slots = []
    for slot_id in range(slots_per_day):
        start_minutes = slot_id * interval_minutes
        end_minutes = start_minutes + interval_minutes
        
        start_hour = start_minutes // 60
        start_min = start_minutes % 60
        end_hour = end_minutes // 60
        end_min = end_minutes % 60
        
        time_slots.append({
            'time_slot_id': slot_id,
            'start_time': f"{start_hour:02d}:{start_min:02d}",
            'end_time': f"{end_hour:02d}:{end_min:02d}",
            'label': f"{start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}",
            'hour': start_hour,
            'is_peak_morning': 7 <= start_hour < 9,
            'is_peak_evening': 17 <= start_hour < 19,
            'is_daytime': 6 <= start_hour < 22
        })
    
    df_slots = pd.DataFrame(time_slots)
    
    if output_path:
        df_slots.to_csv(output_path, index=False, encoding='utf-8')
        print(f"时间槽参考表已保存到: {output_path}")
    
    return df_slots


if __name__ == "__main__":
    # 测试代码
    processor = TimeProcessor(interval_minutes=15)
    
    # 测试时间处理
    test_time = "2008-02-02 15:46:08"
    rounded = processor.round_to_interval(test_time)
    slot_id = processor.create_time_slot_id(test_time)
    
    print(f"原始时间: {test_time}")
    print(f"取整后: {rounded}")
    print(f"时间槽ID: {slot_id}")
    
    # 创建时间槽参考表
    slots_ref = create_time_slots_reference(
        interval_minutes=15,
        output_path="/data/alice/cjtest/TRC/haidian_od_analysis/config/time_slots_reference.csv"
    )
    print(f"\n时间槽参考表（前10行）:\n{slots_ref.head(10)}")
