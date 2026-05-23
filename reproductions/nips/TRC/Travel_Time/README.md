# 行程时间预测模型

## 📊 项目概述

本项目实现了基于机器学习的**行程时间预测模型**，使用历史OD（Origin-Destination）流量数据预测未来的出行时间。

---

## 📁 数据说明

### 数据来源
- **输入数据**: `/data/alice/cjtest/TRC/haidian_od_analysis/output/od_flow_temporal.csv`
- **数据量**: 104,296条记录（2008-02-02至2008-02-08共7天的OD流量数据）

### 数据集划分
- **训练集**: 2008-02-02 ~ 2008-02-06 (80,901条)
- **验证集**: 2008-02-07 (14,217条)
- **测试集**: 2008-02-08 (9,178条)

### 原始数据字段
- `date`: 日期
- `hour`, `minute`: 时间
- `time_slot`: 时间槽
- `origin`: 起点区域ID
- `dest`: 终点区域ID
- `flow`: 该时段该OD对的出行流量
- **`avg_time`**: 平均出行时间（分钟）**← 这是我们要预测的目标**

---

## 🎯 预测任务

### 输入特征（共21个）

#### 1. 时间特征 (8个)
- `hour`: 小时 (0-23)
- `minute`: 分钟 (0-59)
- `time_slot`: 时间槽编号
- `day_of_week`: 星期几 (0=周一, 6=周日)
- `is_weekend`: 是否周末 (0/1)
- `is_morning_peak`: 是否早高峰7-9点 (0/1)
- `is_evening_peak`: 是否晚高峰17-19点 (0/1)
- `is_peak_hour`: 是否高峰时段 (0/1)

#### 2. 空间特征 (2个)
- `origin`: 起点区域ID
- `dest`: 终点区域ID

#### 3. 流量特征 (2个)
- `flow`: 当前时段的出行流量
- `log_flow`: 流量的对数变换 (处理极端值)

#### 4. 历史统计特征 (9个)
- `od_hist_mean`: OD对的历史平均出行时间
- `od_hist_std`: OD对的历史标准差
- `od_hist_median`: OD对的历史中位数
- `hour_hist_mean`: 该小时的历史平均出行时间
- `hour_hist_std`: 该小时的历史标准差
- `origin_hist_mean`: 该起点的历史平均出行时间
- `origin_hist_std`: 该起点的历史标准差
- `dest_hist_mean`: 该终点的历史平均出行时间
- `dest_hist_std`: 该终点的历史标准差

### 输出（预测目标）
- **`predicted_time_ensemble`**: 预测的平均出行时间（分钟）

---

## 🤖 预测方法

### 模型架构
采用**集成学习（Ensemble Learning）**方法，结合两个梯度提升树模型：

1. **XGBoost (eXtreme Gradient Boosting)**
   - 优化的梯度提升决策树算法
   - 参数设置：
     - `max_depth`: 8
     - `learning_rate`: 0.05
     - `n_estimators`: 500
     - 正则化项防止过拟合

2. **LightGBM (Light Gradient Boosting Machine)**
   - 微软开发的高效梯度提升框架
   - 参数设置：
     - `num_leaves`: 64
     - `learning_rate`: 0.05
     - `n_estimators`: 500

3. **集成策略**
   - 最终预测 = (XGBoost预测 + LightGBM预测) / 2
   - 通过平均多个模型降低预测方差，提高稳定性

### 预测流程

```
Step 1: 数据加载与预处理
    ↓
Step 2: 特征工程（创建时间、空间、流量特征）
    ↓
Step 3: 基于训练集计算历史统计特征
    ↓
Step 4: 训练XGBoost模型（在验证集上调优）
    ↓
Step 5: 训练LightGBM模型（在验证集上调优）
    ↓
Step 6: 集成预测（两模型平均）
    ↓
Step 7: 在测试集上评估性能
```

### 核心思想

**通过学习历史OD流量模式，捕捉以下规律：**

1. **时间规律**: 不同时段（早晚高峰vs平峰）的出行时间差异
2. **空间规律**: 不同OD对之间的距离和拥堵特征
3. **流量关系**: 流量与拥堵程度的相关性
4. **历史模式**: 相似时空条件下的历史出行时间

---

## 📈 模型性能

### 测试集评估结果

| 模型 | MAE (分钟) | RMSE (分钟) | R² | MAPE (%) |
|------|-----------|------------|-----|----------|
| XGBoost | 18.49 | 34.78 | 0.222 | 277.81 |
| LightGBM | 19.27 | 33.48 | 0.279 | 309.24 |
| **Ensemble** | **18.72** | **33.67** | **0.271** | **292.53** |

### 性能指标解读

- **MAE (平均绝对误差)**: 平均预测误差约18.72分钟
- **RMSE (均方根误差)**: 考虑极端值的误差约33.67分钟
- **R² (决定系数)**: 模型解释了27.1%的出行时间变异
- **MAPE (平均绝对百分比误差)**: 约292.53%（由于很多短途出行时间小，导致MAPE较大）

### Top 10 重要特征

| 特征 | 重要性 | 说明 |
|------|--------|------|
| od_hist_mean | 36.75% | **OD对历史平均时间**（最重要） |
| od_hist_median | 10.40% | OD对历史中位数 |
| hour_hist_mean | 3.23% | 小时历史平均时间 |
| hour_hist_std | 3.17% | 小时历史标准差 |
| dest_hist_std | 3.14% | 终点历史标准差 |
| dest_hist_mean | 3.09% | 终点历史平均时间 |
| is_morning_peak | 3.09% | 是否早高峰 |
| is_evening_peak | 3.05% | 是否晚高峰 |
| is_peak_hour | 2.97% | 是否高峰时段 |
| origin_hist_std | 2.95% | 起点历史标准差 |

**关键发现**：
- **历史模式是最强预测因子**：OD对的历史平均时间贡献了近37%的重要性
- **时段特征很重要**：是否高峰时段对预测有显著影响
- **空间特征**：起终点的历史出行模式也很关键

---

## 📂 输出文件

所有结果保存在：`/data/alice/cjtest/TRC/Travel_Time/`

| 文件名 | 说明 | 大小 |
|--------|------|------|
| `test_predictions.csv` | 测试集的详细预测结果 | 2.8MB |
| `model_evaluation.csv` | 三个模型的性能对比 | 278B |
| `model_summary.txt` | 模型总结报告 | 2.3KB |
| `feature_importance_xgboost.csv` | XGBoost特征重要性 | 515B |
| `feature_importance_lightgbm.csv` | LightGBM特征重要性 | 337B |
| `travel_time_prediction_results.png` | 可视化结果图表 | 1MB |

### 预测结果文件格式

`test_predictions.csv` 包含以下关键列：

- 原始数据的所有字段
- `predicted_time_xgb`: XGBoost的预测时间
- `predicted_time_lgb`: LightGBM的预测时间
- **`predicted_time_ensemble`**: 集成模型的预测时间 ← **最终预测结果**
- `absolute_error`: 绝对误差 = |真实值 - 预测值|
- `relative_error`: 相对误差 = 绝对误差 / 真实值

---

## 🚀 使用方法

### 环境要求
```bash
conda activate trc
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

### 运行预测
```bash
cd /data/alice/cjtest/TRC/Travel_Time
python travel_time_prediction.py
```

### 使用预测结果
```python
import pandas as pd

# 加载预测结果
predictions = pd.read_csv('test_predictions.csv')

# 查看某个OD对的预测
od_15_to_23 = predictions[(predictions['origin'] == 15) & (predictions['dest'] == 23)]
print(od_15_to_23[['hour', 'minute', 'avg_time', 'predicted_time_ensemble']])

# 统计预测误差
print(f"平均误差: {predictions['absolute_error'].mean():.2f} 分钟")
print(f"中位数误差: {predictions['absolute_error'].median():.2f} 分钟")
```

---

## 📊 可视化结果

`travel_time_prediction_results.png` 包含6个子图：

1. **真实值vs预测值散点图**: 查看预测的准确性
2. **误差分布直方图**: 查看误差的分布情况
3. **模型比较柱状图**: 对比三个模型的MAE
4. **按小时的误差分析**: 发现哪些时段预测困难
5. **特征重要性**: Top 10重要特征可视化
6. **相对误差分布**: 查看预测的相对准确性

---

## 💡 模型优势与局限

### ✅ 优势
1. **融合多维特征**: 时间、空间、流量、历史统计信息
2. **集成学习**: 结合XGBoost和LightGBM的优势
3. **可解释性**: 通过特征重要性了解影响因素
4. **实用性强**: 可用于交通规划、路径推荐

### ⚠️ 局限
1. **依赖历史数据**: 对新的OD对预测效果可能不佳
2. **未考虑外部因素**: 天气、事故、节假日等
3. **时间粒度固定**: 当前按时间槽预测，无法细化到秒级
4. **MAPE较高**: 短途出行导致相对误差被放大

### 🔧 改进方向
- 添加天气、事故等外部数据
- 引入深度学习模型（LSTM、GRU）捕捉时序依赖
- 使用图神经网络（GNN）建模路网拓扑
- 加入实时交通流量数据

---

## 📞 联系方式

如有问题或建议，请联系项目负责人。

**生成时间**: 2026-02-11
**版本**: v1.0
