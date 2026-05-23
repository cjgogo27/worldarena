# AutoGeo 快速使用指南

## 环境要求

```bash
pip install -r requirements.txt  # 或直接运行，依赖很少
```

## 快速开始

### 1. 基础运行

```python
from autogeo import MasterAgent, initialize_tools

# 初始化工具
initialize_tools()

# 创建主智能体
master = MasterAgent(master_id="geo_001")

# 输入图像列表（可以是多张）
images = ["img_paris_1", "img_paris_2", "img_paris_3"]
master.initialize(images)

# 运行定位
results = master.run()

# 查看结果
print(f"联合预测: {results['joint_prediction']}")
print(f"平均置信度: {results['average_confidence']:.2f}")
```

### 2. 查看详细结果

```python
from autogeo.utils import format_results

print(format_results(results))
```

### 3. 计算评估指标

```python
from autogeo.utils import MetricsCalculator

# 准备预测数据
predictions = [
    {"location": "Paris", "confidence": 0.9, "error_km": 5},
    {"location": "Paris", "confidence": 0.8, "error_km": 10},
]

# 计算指标
recall_1km = MetricsCalculator.recall_at_k(predictions, 1)
recall_25km = MetricsCalculator.recall_at_k(predictions, 25)
median_error = MetricsCalculator.median_error(predictions)
geo_score = MetricsCalculator.geo_score(predictions)
consistency = MetricsCalculator.consistency_score(predictions)

print(f"Recall@1km: {recall_1km:.2%}")
print(f"Recall@25km: {recall_25km:.2%}")
print(f"中位误差: {median_error:.1f}km")
print(f"GeoScore: {geo_score:.1f}")
print(f"一致性: {consistency:.2f}")
```

### 4. 运行完整示例

```bash
cd /data/alice/cjtest
python autogeo/examples.py
```

## 输出结果说明

运行后会输出：

```
GEOLOCALIZATION RESULTS
==================================================
Total Rounds: 6
Average Confidence: 0.50

Individual Predictions:
  - Image img_paris_1: Paris (confidence: 0.50)
  - Image img_paris_2: Paris (confidence: 0.50)

Joint Prediction: Paris
Joint Score: 0.50
==================================================
```

## 核心概念

| 概念 | 说明 |
|------|------|
| SubAgent | 每个图像对应一个子智能体，自主调用工具 |
| MasterAgent | 主智能体，协调所有子智能体 |
| BeliefState | 信念状态，存储当前位置假设和置信度 |
| Staged Communication | 阶段性通信，子智能体定期同步信息 |
