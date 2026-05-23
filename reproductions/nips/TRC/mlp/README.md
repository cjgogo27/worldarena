# MLP 行程时间预测

## 运行方式

```bash
cd /data/alice/cjtest/TRC/mlp
conda activate trc
python train_mlp.py 2>&1 | tee train_mlp.log
```

---

## 输入 / 输出

| 类型 | 路径 | 说明 |
|------|------|------|
| **输入** | `TRC/Travel_Time/od_flow_interpolated.csv` | 插值后的完整OD数据（step1生成） |
| **输出** | `TRC/mlp/mlp_evaluation.csv` | 评估指标（MAE / RMSE / R² / 准确率） |
| **输出** | `TRC/mlp/mlp_predictions.csv` | 测试集逐条预测结果（含真实值对比） |
| **输出** | `TRC/mlp/mlp_model.pt` | PyTorch 模型权重 |
| **输出** | `TRC/mlp/train_mlp.log` | 完整训练日志 |
| **输出** | `TRC/mlp/figures/01_true_vs_predicted.png` | 真实值 vs 预测值散点图 |
| **输出** | `TRC/mlp/figures/02_error_distribution.png` | 绝对误差直方图 |
| **输出** | `TRC/mlp/figures/03_flow_vs_travel_time.png` | 按流量分组的行程时间对比 |
| **输出** | `TRC/mlp/figures/04_error_by_hour.png` | 各小时平均误差折线图 |
| **输出** | `TRC/mlp/figures/05_training_curve.png` | 训练/验证 Loss 曲线 |
| **输出** | `TRC/mlp/figures/06_error_cdf.png` | 累积误差分布（CDF） |
| **输出** | `TRC/mlp/figures/07_flow_scatter.png` | Flow vs 行程时间散点（颜色=误差） |
| **输出** | `TRC/mlp/figures/08_error_by_period.png` | 高峰/平峰误差柱状图 |

---

## 数据说明

- **原始数据**：104,296 条（18.5%），真实观测值
- **插值数据**：460,856 条（81.5%），用**训练集**各OD对历史均值填充，未知OD对用全局均值 29.46 min
- **数据集划分**

| 集合 | 日期 | 样本数 |
|------|------|--------|
| 训练集 | 2008-02-02 ~ 02-06 | 403,680 |
| 验证集 | 2008-02-07 | 80,736 |
| 测试集 | 2008-02-08 | 80,736 |

---

## 模型结构

```
输入层 (25 维)
  ↓
Linear(25→256) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Linear(256→128) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Linear(128→64) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Linear(64→1)   ← 输出行程时间（分钟）
```

总参数量：**48,769**

---

## 关键超参数

| 参数 | 值 | 说明 |
|------|----|------|
| `hidden_dims` | (256, 128, 64) | 各隐藏层神经元数 |
| `dropout` | 0.2 | 防过拟合 |
| `epochs` | 100 | 最大训练轮数 |
| `batch_size` | 2048 | 每批样本数 |
| `lr` | 1e-3 | 初始学习率（Adam） |
| `patience` | 10 | 早停耐心轮数 |
| `loss` | HuberLoss(δ=10) | 对大误差鲁棒 |
| `scheduler` | ReduceLROnPlateau | val_loss无改善×5轮 → lr×0.5 |

---

## 25 个输入特征

| 类别 | 特征 |
|------|------|
| **时间** | `hour_norm`, `minute_norm`, `time_slot_norm`, `day_of_week`, `is_weekend` |
| **高峰** | `is_morning_peak`(7-9点), `is_evening_peak`(17-19点), `is_peak` |
| **Flow** | `flow_raw`, `flow_log`, `flow_sqrt`, `flow_squared` |
| **Flow×高峰** | `flow_density_morning`, `flow_density_evening` |
| **历史统计** | `hour_mean`, `hour_std`（训练集各小时均值/标准差） |
| **Flow-bin统计** | `flow_bin_mean`, `flow_bin_std`（训练集各流量区间均值/标准差） |
| **时序lag** | `lag_1`, `lag_2`, `lag_4`, `lag_8`, `lag_16`（前1/2/4/8/16个时间槽） |
| **Rolling** | `rolling_4h_mean`, `rolling_4h_std`（过去4小时=16槽滑动统计） |

> **注**：不含 origin/dest ID，防止模型记忆OD对而非学习真实规律。

---

## 插值方法

### 原理

原始数据中 **81.5% 的时间槽没有观测值**（出租车从未在该时刻走过该OD对）。
插值在训练开始前完成（`step1_preprocess_and_interpolate.py`），**只用训练集（2月2-6日）的数据**，测试集的信息完全隔离。

### 两级填充策略

```
缺失值
  ├─ 该OD对在训练集中有历史数据？
  │     → 用该OD对的训练集平均行程时间填充
  │
  └─ 该OD对在训练集中从未出现？
        → 用训练集全局平均值（29.46 分钟）填充

缺失的 flow → 一律填 0（没有车辆经过）
```

### 举例

假设 OD 对 **区域3 → 区域7**，训练集（2月2-6日）共出现过 5 次观测：

| 日期 | 时间槽 | 行程时间 |
|------|--------|---------|
| 02-02 | 32 (08:00) | 18 min |
| 02-03 | 32 (08:00) | 22 min |
| 02-04 | 68 (17:00) | 25 min |
| 02-05 | 32 (08:00) | 20 min |
| 02-06 | 45 (11:15) | 15 min |

训练集均值 = (18+22+25+20+15) / 5 = **20 min**

插值结果（7天 × 96时刻 = 672 个格子）：

```
有观测值的 5 个格子 → 保留原始值（is_interpolated = False）
剩余 667 个空格子  → 全部填入 20 min（is_interpolated = True）
```

> **注意**：所有时刻统一填同一个均值，没有区分早高峰/晚高峰。
> 这是当前方案的主要局限，后续可改为按时段（峰/平/谷）分别计算均值。

---

## 评估指标说明

| 指标 | 含义 |
|------|------|
| MAE | 平均绝对误差（分钟） |
| RMSE | 均方根误差，对大误差敏感 |
| R² | 拟合优度，越接近1越好 |
| MAPE | 平均绝对百分比误差（%） |
| Acc@5min | 预测误差 ≤5 分钟的样本占比（%) |
| Acc@10min | 预测误差 ≤10 分钟的样本占比（%） |
| Acc@20min | 预测误差 ≤20 分钟的样本占比（%） |
