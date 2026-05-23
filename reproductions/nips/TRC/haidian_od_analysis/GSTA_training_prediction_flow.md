# GSTA模型训练和预测流程详解

## 📊 数据划分策略

### ✅ **有明确的训练集/验证集/测试集划分**

```python
# 从预先划分好的CSV文件加载
X_train = pd.read_csv('../Chengdu Data/X_train.csv')
Y_train = pd.read_csv('../Chengdu Data/Y_train.csv')
X_val = pd.read_csv('../Chengdu Data/X_val.csv')
Y_val = pd.read_csv('../Chengdu Data/Y_val.csv')
X_test = pd.read_csv('../Chengdu Data/X_test.csv')
Y_test = pd.read_csv('../Chengdu Data/Y_test.csv')
```

**数据已经预先划分好**：
- ✓ **训练集 (X_train, Y_train)**：用于训练模型参数
- ✓ **验证集 (X_val, Y_val)**：用于早停和调参
- ✓ **测试集 (X_test, Y_test)**：最终评估性能

**划分方式**：
- 通常按时间顺序划分（例如：前70%训练，15%验证，15%测试）
- 或随机抽样（但出租车数据更常用时间序列划分）

---

## 🔄 训练流程

### 第1步：数据预处理

```python
# 1. 类型转换（分类特征需要embedding）
X_train[['pickup_cluster', 'dropoff_cluster', 'pickup_geohash', 'dropoff_geohash']] = \
    X_train[['pickup_cluster', 'dropoff_cluster', 'pickup_geohash', 'dropoff_geohash']].astype('object')

# 2. 数据格式转换（适配模型输入）
X_train_list, X_val_list, X_test_list = preproc(X_train, X_val, X_test)
```

**preproc函数的作用**：
- 为每个分类变量创建独立的输入列表
- 对分类变量进行编码映射（0到vocab_size-1）
- 将数值特征单独组织成一个输入

---

### 第2步：模型编译

```python
GSTA = Model(inputs=input_models, outputs=output)
GSTA.compile(
    optimizer=Adam(lr=0.001),       # Adam优化器，学习率0.001
    loss='mean_absolute_error',     # 损失函数：MAE
    metrics=['mae', 'mape'],        # 评估指标：MAE和MAPE
    run_eagerly=True
)
```

---

### 第3步：设置回调函数

```python
callbacks_list = [
    # 1. 早停回调
    EarlyStopping(
        monitor='val_loss',      # 监控验证集损失
        min_delta=0.0001,        # 最小改善阈值
        patience=20,             # 20轮不改善就停止
        verbose=0, 
        mode='auto'
    ),
    
    # 2. 保存最佳模型
    ModelCheckpoint(
        '../Models/GSTA_Chengdu_Best_weights.hdf5',
        monitor='val_loss',      # 监控验证集损失
        save_best_only=True,     # 只保存最好的模型
        save_weights_only=True
    ),
    
    # 3. 学习率衰减
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,              # 学习率衰减为原来的0.1倍
        patience=20,             # 20轮不改善就降低学习率
        min_lr=0.0001            # 最小学习率
    ),
    
    # 4. 检查点保存
    ModelCheckpoint(
        filepath="../Models/GSTA_Chengdu_CKPT.ckpt",
        save_weights_only=True,
        verbose=1
    )
]
```

**回调函数的作用**：
- **EarlyStopping**：防止过拟合，验证损失不再下降时自动停止
- **ModelCheckpoint**：自动保存验证集上表现最好的模型
- **ReduceLROnPlateau**：动态调整学习率，训练平台期降低学习率

---

### 第4步：训练模型

```python
history = GSTA.fit(
    X_train_list,                         # 训练数据（多个输入的列表）
    Y_train,                              # 训练标签（行程时间）
    validation_data=(X_val_list, Y_val),  # 验证数据
    epochs=50,                            # 最多训练50轮
    batch_size=256,                       # 批次大小256
    callbacks=callbacks_list,             # 回调函数
    shuffle=False                         # 不打乱顺序（保持时间序列）
)
```

**训练过程**：
```
Epoch 1/50
  训练集：计算损失 → 反向传播 → 更新参数
  验证集：计算损失 → 保存最佳模型（如果val_loss降低）
  
Epoch 2/50
  训练集：...
  验证集：...
  检查早停条件
  
...

Epoch N
  验证集损失连续20轮不降低 → 触发早停 → 结束训练
```

---

## 🎯 预测流程

### 第1步：加载最佳模型

```python
# 训练结束后，模型已经是验证集上表现最好的版本
# (因为ModelCheckpoint会自动加载最佳权重)
```

### 第2步：在测试集上预测

```python
GSTA_predictions = GSTA.predict(X_test_list)
```

**预测过程**：
```
输入: X_test_list (测试集的所有特征)
  ↓
GSTA模型（已训练好）
  ↓ 前向传播
  1. Embedding层（处理分类特征）
  2. 空间-时间注意力
  3. 多头注意力
  4. 前馈网络
  ↓
输出: 预测的travel_time (单位：秒)
```

### 第3步：计算评估指标

```python
y_test = np.array(Y_test)           # 真实值
y_pred = np.array(GSTA_predictions) # 预测值

# 计算MAPE, MAE, RMSE, MdAE, R²
MAPE = mean_absolute_percentage_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)
```

---

## 📈 完整流程图

```
┌─────────────────────────────────────────────────────────┐
│ 数据准备（预先完成）                                      │
├─────────────────────────────────────────────────────────┤
│ 原始数据 → 特征工程 → 划分数据集                          │
│                      ↓                                  │
│          ┌──────────────────────┐                       │
│          │ 训练集 (70%)         │                       │
│          │ 验证集 (15%)         │                       │
│          │ 测试集 (15%)         │                       │
│          └──────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 模型构建                                                 │
├─────────────────────────────────────────────────────────┤
│ 1. 加载数据 (X_train, Y_train, X_val, Y_val)           │
│ 2. 数据预处理 (embedding编码 + 数值归一化)              │
│ 3. 构建GSTA模型架构                                      │
│ 4. 编译模型 (优化器 + 损失函数)                          │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 训练阶段                                                 │
├─────────────────────────────────────────────────────────┤
│ For epoch in 1..50:                                     │
│   ┌─────────────────────────────────────────┐           │
│   │ 训练步骤:                                │           │
│   │   - 前向传播 (X_train → predictions)    │           │
│   │   - 计算损失 (MAE)                      │           │
│   │   - 反向传播 (更新权重)                  │           │
│   └─────────────────────────────────────────┘           │
│   ┌─────────────────────────────────────────┐           │
│   │ 验证步骤:                                │           │
│   │   - 前向传播 (X_val → predictions)      │           │
│   │   - 计算val_loss                        │           │
│   │   - 如果val_loss降低 → 保存模型          │           │
│   │   - 检查早停条件                         │           │
│   └─────────────────────────────────────────┘           │
│                                                         │
│ → 早停触发或达到50轮 → 训练结束                          │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 测试阶段                                                 │
├─────────────────────────────────────────────────────────┤
│ 1. 加载最佳模型权重                                      │
│ 2. 在测试集上预测: GSTA.predict(X_test)                 │
│ 3. 计算评估指标: MAPE, MAE, RMSE, R²                    │
│ 4. 可视化结果                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🔍 关键设计细节

### 1. **为什么shuffle=False？**

```python
history = GSTA.fit(..., shuffle=False)
```

**原因**：
- 出租车数据是时间序列数据
- 保持时间顺序可能有助于模型学习时间相关的模式
- 但也可以设置shuffle=True（不影响最终结果）

---

### 2. **为什么用验证集？**

```python
validation_data=(X_val_list, Y_val)
```

**作用**：
- **监控过拟合**：训练损失下降但验证损失上升 = 过拟合
- **早停依据**：基于验证集损失决定何时停止
- **超参数调优**：可以用验证集选择最佳超参数
- **保存最佳模型**：验证集上表现最好的模型

---

### 3. **多输入模型架构**

```python
input_models = [
    input_pickup_cluster,     # 输入1
    input_dropoff_cluster,    # 输入2
    input_pickup_geohash,     # 输入3
    input_dropoff_geohash,    # 输入4
    input_numeric,            # 输入5（所有数值特征）
]

GSTA = Model(inputs=input_models, outputs=output)
```

**为什么多输入？**
- 分类特征需要独立的embedding层
- 每个embedding层有自己的词汇表大小
- 最后在模型内部拼接

---

## 🎓 与LSTM的对比

| 维度 | **LSTM (OD流量预测)** | **GSTA (行程时间预测)** |
|------|---------------------|----------------------|
| **数据类型** | 时间序列（连续的时间槽） | 独立样本（每条trip） |
| **输入形状** | (batch, seq_len, features) | (batch, features) |
| **训练样本** | 滑动窗口生成 | 每条trip是一个样本 |
| **数据划分** | 按日期划分（5天train，1天val，1天test） | 按样本划分（70%/15%/15%） |
| **shuffle** | 必须False（保持时间顺序） | 可以True或False |
| **预测方式** | 自回归（用预测值预测下一步） | 单步预测（一次预测一个值） |
| **验证集作用** | 早停 + 超参数调优 | 早停 + 超参数调优 |

---

## 💡 总结

### GSTA的预测方式：

1. **有明确的数据划分**：训练集 + 验证集 + 测试集
2. **训练时使用验证集**：监控过拟合 + 早停 + 保存最佳模型
3. **测试时一次性预测**：加载最佳模型 → 预测全部测试集 → 计算指标
4. **每个样本独立**：不是时间序列预测，每条trip独立预测
5. **监督学习**：输入特征 → 预测travel_time

### 关键优势：

- ✓ 验证集防止过拟合
- ✓ 自动保存最佳模型
- ✓ 动态学习率调整
- ✓ 早停节省训练时间

### 与您的LSTM的区别：

- GSTA：预测单次行程时间（回归）
- LSTM：预测时间序列流量（序列预测）
- GSTA：样本独立
- LSTM：样本有时间依赖性

---

## 📝 如果您要实现类似的流程

```python
# 1. 准备数据（预先划分）
X_train, Y_train = load_training_data()
X_val, Y_val = load_validation_data()
X_test, Y_test = load_test_data()

# 2. 构建模型
model = build_gsta_model()
model.compile(optimizer=Adam(lr=0.001), loss='mae')

# 3. 设置回调
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20),
    ModelCheckpoint('best_model.h5', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
]

# 4. 训练
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=256,
    callbacks=callbacks
)

# 5. 测试
predictions = model.predict(X_test)
evaluate_metrics(Y_test, predictions)
```

这就是GSTA的完整训练和预测流程！
