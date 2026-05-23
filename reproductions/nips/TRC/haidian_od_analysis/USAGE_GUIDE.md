# 海淀区OD流量分析 - 使用指南

## 快速开始（3步完成）

### 方法1: 使用自动化脚本（推荐）

```bash
cd /data/alice/cjtest/TRC/haidian_od_analysis
./setup_and_run.sh
```

脚本会自动完成:
1. ✓ 创建Python虚拟环境
2. ✓ 安装所有依赖包
3. ✓ 运行分析程序

### 方法2: 手动执行

```bash
# 1. 进入项目目录
cd /data/alice/cjtest/TRC/haidian_od_analysis

# 2. 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行程序
python main.py
```

## 输入数据要求

### 必需文件

1. **区域边界文件** (Shapefile)
   - 位置: `/data/alice/cjtest/TRC/海淀区边界_110108_Shapefile_(poi86.com)/110108.shp`
   - 格式: Shapefile (.shp + .shx + .dbf)
   - 坐标系: WGS84 (EPSG:4326)

2. **轨迹数据** (CSV)
   - 位置: `/data/alice/cjtest/TRC/all_taxi_data.csv`
   - 必需字段:
     - `taxi_id`: 车辆ID
     - `date_time`: 时间戳 (格式: YYYY-MM-DD HH:MM:SS)
     - `longitude`: 经度
     - `latitude`: 纬度

3. **区域映射表** (自动生成)
   - 位置: `config/region_mapping.csv`
   - 已包含海淀区29个区域的ID和名称

## 输出结果

所有结果保存在 `output/` 目录:

```
output/
├── od_matrices.h5              # OD矩阵（HDF5格式）
├── trips.csv                   # 出行记录
├── haidian_regions.geojson     # 区域边界（GeoJSON）
├── quality_report.json         # 质量检查报告
├── time_slots_reference.csv    # 时间槽参考表
├── csv/
│   └── od_flow_15min.csv      # OD流量（CSV长表）
└── visualizations/
    └── od_heatmap_*.png        # OD流量热力图
```

## 核心功能模块

### 1. 区域处理 (region_processor.py)

```python
from src.region_processor import RegionProcessor

# 初始化
processor = RegionProcessor(
    shapefile_path='path/to/shapefile.shp',
    region_mapping_path='path/to/mapping.csv'
)

# 加载区域
processor.load_regions()

# 点映射
region_id = processor.point_to_region(lon=116.3, lat=39.99)
```

### 2. 时间处理 (time_processor.py)

```python
from src.time_processor import TimeProcessor

# 初始化（15分钟粒度）
processor = TimeProcessor(interval_minutes=15)

# 处理时间序列
df_with_time = processor.process_time_series(df, time_col='date_time')

# 时间映射
time_slot = processor.create_time_slot_id(datetime_obj)  # 返回0-95
```

### 3. OD矩阵生成 (od_matrix_generator.py)

```python
from src.od_matrix_generator import ODMatrixGenerator

# 初始化
generator = ODMatrixGenerator(num_regions=29, interval_minutes=15)

# 提取出行
trips_df = generator.extract_trips(trajectory_df)

# 生成OD矩阵
od_matrices = generator.create_od_matrix(trips_df)
time_matrices = generator.create_od_travel_time_matrix(trips_df)
```

### 4. 质量检查 (quality_checker.py)

```python
from src.quality_checker import DataQualityChecker

# 初始化
checker = DataQualityChecker(num_regions=29)

# 检查轨迹数据
checker.check_trajectory_data(df)

# 检查出行数据
checker.check_trips_data(trips_df)

# 检查OD矩阵
checker.check_od_matrices(od_matrices, time_matrices)
```

## 读取结果示例

### 读取HDF5格式的OD矩阵

```python
import h5py
import numpy as np

# 打开文件
with h5py.File('output/od_matrices.h5', 'r') as f:
    # 查看所有时间槽
    time_slots = list(f['flow'].keys())
    print(f"时间槽: {time_slots}")
    
    # 读取特定时间槽的流量矩阵
    flow = f['flow']['0'][:]  # 时间槽0 (00:00-00:15)
    travel_time = f['travel_time']['0'][:]
    
    # 读取元数据
    num_regions = f.attrs['num_regions']
    interval = f.attrs['interval_minutes']
    
    print(f"区域数: {num_regions}")
    print(f"时间粒度: {interval}分钟")
    print(f"流量矩阵形状: {flow.shape}")
    print(f"总流量: {flow.sum()}")
```

### 读取CSV格式的OD数据

```python
import pandas as pd

# 读取OD流量表
df_od = pd.read_csv('output/csv/od_flow_15min.csv')

# 查看早高峰（7:00-9:00）的数据
morning_peak = df_od[(df_od['time_slot'] >= 28) & (df_od['time_slot'] < 36)]

# 找出最热门的OD对
top_od = df_od.nlargest(10, 'flow')
print("Top 10 OD对:")
print(top_od[['origin_name', 'dest_name', 'flow', 'avg_travel_time']])

# 统计某个区域的出行
region_name = '中关村街道'
from_region = df_od[df_od['origin_name'] == region_name]
print(f"\n从{region_name}出发的出行数: {from_region['flow'].sum()}")
```

### 分析出行记录

```python
import pandas as pd

# 读取出行记录
trips = pd.read_csv('output/trips.csv')

# 转换时间
trips['start_time'] = pd.to_datetime(trips['start_time'])
trips['hour'] = trips['start_time'].dt.hour

# 按小时统计出行量
hourly_trips = trips.groupby('hour').size()

# 分析平均出行时间
avg_duration = trips.groupby(['origin_region', 'dest_region'])['duration_minutes'].mean()

# 找出最长的出行
longest_trips = trips.nlargest(10, 'duration_minutes')
```

## 参数调优

### 修改时间粒度

在 `main.py` 中修改:

```python
config = {
    'interval_minutes': 30,  # 改为30分钟粒度
    # ...
}
```

影响:
- 15分钟: 96个时间槽/天 (精细)
- 30分钟: 48个时间槽/天 (平衡)
- 60分钟: 24个时间槽/天 (粗粒度)

### 调整出行提取参数

```python
config = {
    'trip_time_threshold': 30,  # 停留超过30分钟视为新出行
    # ...
}
```

建议:
- 出租车: 20-30分钟
- 私家车: 30-60分钟
- 公交车: 10-20分钟

## 性能优化

### 处理大规模数据

系统自动处理:
- 文件 < 1GB: 一次性读入内存
- 文件 > 1GB: 分块读取（每块100万条）

手动限制数据量:

```python
# 在main.py的step2_load_and_filter_trajectory中
# 只读取前N条记录
df_traj = pd.read_csv(trajectory_path, nrows=1000000)

# 或只读取某几天的数据
df_traj = df_traj[df_traj['date'] == '2008-02-02']
```

### 并行处理（高级）

对于多日数据，可以按日期分组并行处理:

```python
from multiprocessing import Pool

def process_date(date):
    # 处理单日数据
    pass

dates = df['date'].unique()
with Pool(4) as p:  # 4个进程
    results = p.map(process_date, dates)
```

## 常见问题排查

### 问题1: 区域映射率低

**现象**: 大量轨迹点未能映射到区域

**原因**:
- GPS数据不在海淀区范围内
- Shapefile坐标系不匹配
- 区域边界数据不准确

**解决**:
```python
# 检查数据范围
print(df['longitude'].min(), df['longitude'].max())
print(df['latitude'].min(), df['latitude'].max())

# 海淀区大致范围: 116.0-116.5, 39.9-40.2
```

### 问题2: 内存不足

**现象**: 程序运行时内存溢出

**解决**:
1. 限制读取数据量（见上文）
2. 使用分块处理
3. 增加swap空间

### 问题3: Shapefile读取错误

**现象**: 无法读取.shp文件

**解决**:
```bash
# 检查文件完整性
ls -lh /data/alice/cjtest/TRC/海淀区边界_110108_Shapefile_\(poi86.com\)/

# 应包含: .shp, .shx, .dbf文件
```

### 问题4: 依赖安装失败

**解决**:
```bash
# 单独安装可能失败的包
pip install geopandas --no-cache-dir

# 或使用conda
conda install geopandas
```

## 进阶使用

### 自定义区域划分

编辑 `config/region_mapping.csv`:

```csv
region_id,region_code,region_name,region_type
1,110108001,万寿路街道,街道
...
30,110108200,新增区域,其他
```

### 扩展时间维度分析

```python
# 添加星期几分析
trips['weekday'] = pd.to_datetime(trips['start_time']).dt.dayofweek

# 工作日 vs 周末
weekday_od = trips[trips['weekday'] < 5].groupby(['origin_region', 'dest_region']).size()
weekend_od = trips[trips['weekday'] >= 5].groupby(['origin_region', 'dest_region']).size()
```

### 集成其他数据源

```python
# 加载POI数据
poi_df = pd.read_csv('poi_data.csv')

# 统计每个区域的POI数量
region_poi = poi_df.groupby('region_id').size()

# 结合OD分析
od_with_poi = pd.merge(df_od, region_poi, 
                       left_on='origin_region', 
                       right_index=True)
```

## 技术支持

### 日志查看

所有处理步骤都有详细的控制台输出，建议保存日志:

```bash
python main.py 2>&1 | tee analysis.log
```

### 调试模式

在代码中添加断点:

```python
import pdb; pdb.set_trace()
```

### 性能分析

```bash
# 使用time命令
time python main.py

# 使用Python profiler
python -m cProfile -o profile.stats main.py
```

## 更新日志

**v1.0.0** (2026-02-03)
- ✓ 初始版本发布
- ✓ 支持29个区域的OD分析
- ✓ 15分钟时间粒度
- ✓ 完整的质量检查系统
- ✓ 多格式输出支持

---

如有其他问题，请查看 README.md 或联系维护团队。
