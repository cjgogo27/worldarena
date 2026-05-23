# 海淀区OD流量分析系统

基于海淀区29个行政区划（22街道+7镇）的时空数据处理系统，生成15分钟粒度的OD流量矩阵和平均旅行时间。

## 项目概述

### 功能特性

- ✅ **区域映射**: 基于Shapefile的29个行政区划精确映射
- ✅ **时间粒度**: 15分钟时间间隔的精细化分析
- ✅ **OD矩阵**: 完整的起点-终点流量矩阵
- ✅ **旅行时间**: 各OD对的平均旅行时间计算
- ✅ **质量检查**: 全流程数据质量监控和报告
- ✅ **多格式输出**: 支持HDF5和CSV格式

### 区域划分

海淀区共29个行政区域:

**22个街道:**
1. 万寿路街道 (110108001)
2. 永定路街道 (110108002)
3. 羊坊店街道 (110108003)
4. 甘家口街道 (110108004)
5. 八里庄街道 (110108005)
6. 紫竹院街道 (110108006)
7. 北下关街道 (110108007)
8. 北太平庄街道 (110108008)
9. 学院路街道 (110108010)
10. 中关村街道 (110108011)
11. 海淀街道 (110108012)
12. 青龙桥街道 (110108013)
13. 清华园街道 (110108014)
14. 燕园街道 (110108015)
15. 香山街道 (110108016)
16. 清河街道 (110108017)
17. 花园路街道 (110108018)
18. 西三旗街道 (110108019)
19. 马连洼街道 (110108020)
20. 田村路街道 (110108021)
21. 上地街道 (110108022)
22. 曙光街道 (110108023)

**7个镇:**
23. 温泉镇 (110108100)
24. 四季青镇 (110108101)
25. 西北旺镇 (110108102)
26. 苏家坨镇 (110108103)
27. 上庄镇 (110108104)
28. 东升镇 (110108105)
29. 海淀镇 (110108106)

## 项目结构

```
haidian_od_analysis/
├── config/                      # 配置文件
│   ├── region_mapping.csv      # 区域ID映射表
│   └── time_slots_reference.csv # 时间槽参考表（自动生成）
├── src/                        # 源代码
│   ├── region_processor.py    # 区域处理模块
│   ├── time_processor.py      # 时间处理模块
│   ├── od_matrix_generator.py # OD矩阵生成模块
│   └── quality_checker.py     # 质量检查模块
├── output/                     # 输出目录（自动创建）
│   ├── od_matrices.h5         # OD矩阵（HDF5格式）
│   ├── trips.csv              # 出行记录
│   ├── haidian_regions.geojson # 区域边界
│   ├── quality_report.json    # 质量报告
│   ├── csv/                   # CSV格式输出
│   │   └── od_flow_15min.csv # OD流量（长表格式）
│   └── visualizations/        # 可视化结果
│       └── od_heatmap_*.png  # OD热力图
├── main.py                    # 主程序
├── requirements.txt           # 依赖包
└── README.md                  # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保以下数据文件就绪:

- **Shapefile**: `海淀区边界_110108_Shapefile_(poi86.com)/110108.shp`
- **轨迹数据**: `all_taxi_data.csv` (格式: taxi_id, date_time, longitude, latitude)

### 3. 运行分析

```bash
# 直接运行主程序
python main.py
```

## 处理流程

系统分为6个主要步骤:

### 步骤1: 加载区域数据
- 读取Shapefile文件
- 加载区域映射表
- 导出GeoJSON格式边界

### 步骤2: 加载轨迹数据
- 读取原始GPS轨迹数据
- 基本数据清洗
- 质量检查

### 步骤3: 空间映射
- 将GPS点批量映射到29个区域
- 空间连接处理
- 统计映射成功率

### 步骤4: 时间映射
- 将时间戳映射到15分钟时间槽
- 生成时间槽ID（每天96个时间槽）
- 创建时间参考表

### 步骤5: 提取出行和生成OD矩阵
- 从轨迹提取出行记录（OD对）
- 生成29x29的OD流量矩阵（每个时间槽）
- 计算OD平均旅行时间矩阵
- 数据质量检查

### 步骤6: 保存结果
- 保存HDF5格式矩阵
- 导出CSV格式数据
- 生成可视化图表
- 输出质量报告

## 输出说明

### 1. OD流量矩阵 (od_matrices.h5)

HDF5格式,包含:
- `/flow/{time_slot}`: 各时间槽的流量矩阵 (29x29)
- `/travel_time/{time_slot}`: 各时间槽的平均时间矩阵 (29x29)
- `/regions/*`: 区域映射信息

读取示例:
```python
import h5py
import numpy as np

with h5py.File('output/od_matrices.h5', 'r') as f:
    # 读取时间槽0的流量矩阵
    flow_matrix = f['flow']['0'][:]
    # 读取时间槽0的时间矩阵
    time_matrix = f['travel_time']['0'][:]
    # 读取元数据
    num_regions = f.attrs['num_regions']
```

### 2. OD流量表 (csv/od_flow_15min.csv)

长表格式,列包括:
- `time_slot`: 时间槽ID (0-95)
- `origin_region`: 起点区域ID (1-29)
- `dest_region`: 终点区域ID (1-29)
- `origin_name`: 起点区域名称
- `dest_name`: 终点区域名称
- `flow`: 流量数量
- `avg_travel_time`: 平均旅行时间（分钟）

### 3. 出行记录 (trips.csv)

每条出行记录包括:
- `vehicle_id`: 车辆ID
- `origin_region`: 起点区域
- `dest_region`: 终点区域
- `start_time`: 出发时间
- `end_time`: 到达时间
- `duration_minutes`: 旅行时间（分钟）
- `start_time_slot`: 出发时间槽
- `date`: 日期

### 4. 质量报告 (quality_report.json)

包含各阶段的质量检查结果:
- 轨迹数据检查
- 出行数据检查
- OD矩阵检查

## 配置说明

在 `main.py` 中可修改以下配置:

```python
config = {
    # 路径配置
    'shapefile_path': '...',          # Shapefile路径
    'region_mapping_path': '...',     # 区域映射表路径
    'trajectory_path': '...',         # 轨迹数据路径
    'output_dir': '...',              # 输出目录
    
    # 参数配置
    'num_regions': 29,                # 区域数量
    'interval_minutes': 15,           # 时间粒度（分钟）
    'trip_time_threshold': 30,        # 出行时间阈值（分钟）
}
```

## 时间槽说明

- **时间粒度**: 15分钟
- **每日时间槽数**: 96个 (24小时 × 60分钟 / 15分钟)
- **时间槽ID**: 0-95
  - 0: 00:00-00:15
  - 1: 00:15-00:30
  - ...
  - 95: 23:45-24:00

时间槽参考表自动生成在 `output/time_slots_reference.csv`

## 数据质量检查

系统自动执行以下质量检查:

### 轨迹数据
- ✓ 必需字段完整性
- ✓ 空值检测
- ✓ 坐标范围验证
- ✓ 时间格式检查
- ✓ 车辆统计
- ✓ 区域映射率

### 出行数据
- ✓ 出行时长分布
- ✓ OD对覆盖率
- ✓ 时间分布
- ✓ 空间覆盖

### OD矩阵
- ✓ 矩阵维度验证
- ✓ 流量统计
- ✓ 稀疏性分析
- ✓ 平均时间统计

## 常见问题

### Q: 如何处理大规模数据？
A: 系统自动检测文件大小，对于大文件（>1GB）会使用分块读取。

### Q: 某些区域没有数据怎么办？
A: 质量检查会识别出未覆盖的区域，OD矩阵中对应位置为0。

### Q: 如何修改时间粒度？
A: 修改 `config['interval_minutes']` 参数，如设置为30表示30分钟粒度。

### Q: 如何可视化结果？
A: 系统自动生成示例热力图，也可使用提供的可视化函数自定义图表。

## 技术栈

- **Python 3.8+**
- **GeoPandas**: 地理空间数据处理
- **Pandas**: 数据处理
- **NumPy**: 数值计算
- **Shapely**: 几何操作
- **H5py**: HDF5文件读写
- **Matplotlib/Seaborn**: 可视化

## 维护和更新

### 更新区域映射
如需修改区域划分，编辑 `config/region_mapping.csv`

### 扩展功能
核心模块均可独立使用和扩展:
- `region_processor.py`: 区域处理
- `time_processor.py`: 时间处理
- `od_matrix_generator.py`: OD矩阵生成
- `quality_checker.py`: 质量检查

## 许可证

本项目仅供研究和学习使用。

## 联系方式

如有问题或建议，请联系项目维护团队。

---

**最后更新**: 2026-02-03
