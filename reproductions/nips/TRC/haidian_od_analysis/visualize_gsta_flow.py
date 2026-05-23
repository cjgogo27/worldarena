#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSTA训练和预测流程可视化
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(18, 14))

# ============================================================================
# 创建3个子图
# ============================================================================
ax1 = plt.subplot(3, 1, 1)  # 数据划分
ax2 = plt.subplot(3, 1, 2)  # 训练流程
ax3 = plt.subplot(3, 1, 3)  # 预测流程

# ============================================================================
# 图1: 数据划分策略
# ============================================================================
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 8)
ax1.axis('off')
ax1.set_title('GSTA数据划分策略（预先完成）', fontsize=16, fontweight='bold', pad=20)

# 原始数据
y_start = 6.5
box_data = FancyBboxPatch((2, y_start), 16, 1,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#E3F2FD',
                          linewidth=2)
ax1.add_patch(box_data)
ax1.text(10, y_start+0.5, '原始出租车行程数据 (百万级样本)', 
        ha='center', va='center', fontsize=12, fontweight='bold')
ax1.text(10, y_start+0.1, '每条记录 = 一次行程（起点、终点、时间、duration等）',
        ha='center', va='center', fontsize=9, color='#666')

# 向下箭头
arrow = FancyArrowPatch((10, y_start), (10, y_start-0.8),
                       arrowstyle='->', mutation_scale=30,
                       linewidth=3, color='black')
ax1.add_patch(arrow)

# 三个数据集
y_base = 4
datasets = [
    {'name': '训练集', 'ratio': '70%', 'color': '#4CAF50', 'x': 2, 'width': 5.6, 
     'desc': 'X_train.csv\nY_train.csv', 'purpose': '训练模型参数'},
    {'name': '验证集', 'ratio': '15%', 'color': '#FF9800', 'x': 8, 'width': 2.4,
     'desc': 'X_val.csv\nY_val.csv', 'purpose': '早停+调参'},
    {'name': '测试集', 'ratio': '15%', 'color': '#2196F3', 'x': 10.8, 'width': 2.4,
     'desc': 'X_test.csv\nY_test.csv', 'purpose': '最终评估'},
]

for ds in datasets:
    box = FancyBboxPatch((ds['x'], y_base), ds['width'], 1.5,
                        boxstyle="round,pad=0.05",
                        edgecolor='black', facecolor=ds['color'],
                        linewidth=2)
    ax1.add_patch(box)
    ax1.text(ds['x'] + ds['width']/2, y_base+1.1, 
            f"{ds['name']} ({ds['ratio']})",
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax1.text(ds['x'] + ds['width']/2, y_base+0.5,
            ds['desc'],
            ha='center', va='center', fontsize=8, color='white')

# 用途说明
for i, ds in enumerate(datasets):
    ax1.text(ds['x'] + ds['width']/2, y_base-0.6,
            f"用途: {ds['purpose']}",
            ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))

# 说明
ax1.text(10, 1.5, '✓ 数据已预先划分好（不需要在代码中划分）', 
        ha='center', fontsize=11, fontweight='bold', color='green')
ax1.text(10, 0.8, '✓ 通常按时间顺序或随机划分', 
        ha='center', fontsize=10, color='#666')

# ============================================================================
# 图2: 训练流程
# ============================================================================
ax2.set_xlim(0, 20)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('GSTA训练流程（使用训练集+验证集）', fontsize=16, fontweight='bold', pad=20)

# Epoch循环框
y_epoch = 8.5
box_epoch = Rectangle((1, 1.5), 18, 7,
                      edgecolor='purple', facecolor='none',
                      linewidth=2, linestyle='--')
ax2.add_patch(box_epoch)
ax2.text(1.5, y_epoch+0.3, 'For Epoch = 1, 2, ..., 50 (或早停)', 
        fontsize=12, fontweight='bold', color='purple')

# 训练步骤
y_train = 6.5
box_train = FancyBboxPatch((2, y_train), 7, 1.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='#4CAF50', facecolor='#E8F5E9',
                          linewidth=2)
ax2.add_patch(box_train)
ax2.text(5.5, y_train+1.4, '训练步骤 (X_train → Y_train)', 
        ha='center', fontsize=11, fontweight='bold', color='#4CAF50')
steps_train = [
    '1. 前向传播: predictions = GSTA(X_train)',
    '2. 计算损失: loss = MAE(predictions, Y_train)',
    '3. 反向传播: 计算梯度',
    '4. 更新参数: Adam优化器'
]
for i, step in enumerate(steps_train):
    ax2.text(2.3, y_train+0.9-i*0.35, step, fontsize=8, color='#333')

# 验证步骤
box_val = FancyBboxPatch((11, y_train), 7, 1.8,
                        boxstyle="round,pad=0.1",
                        edgecolor='#FF9800', facecolor='#FFF3E0',
                        linewidth=2)
ax2.add_patch(box_val)
ax2.text(14.5, y_train+1.4, '验证步骤 (X_val → Y_val)', 
        ha='center', fontsize=11, fontweight='bold', color='#FF9800')
steps_val = [
    '1. 前向传播: predictions = GSTA(X_val)',
    '2. 计算val_loss',
    '3. 如果val_loss降低 → 保存最佳模型',
    '4. 检查早停: 20轮不改善 → 停止'
]
for i, step in enumerate(steps_val):
    ax2.text(11.3, y_train+0.9-i*0.35, step, fontsize=8, color='#333')

# 箭头连接
arrow1 = FancyArrowPatch((9, y_train+0.9), (11, y_train+0.9),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='black')
ax2.add_patch(arrow1)

# 回调函数
y_callback = 4
ax2.text(2, y_callback+0.8, '回调函数 (Callbacks):', fontsize=11, fontweight='bold')
callbacks = [
    '✓ EarlyStopping: 20轮val_loss不降→停止',
    '✓ ModelCheckpoint: 保存最佳模型',
    '✓ ReduceLROnPlateau: 动态降低学习率',
]
for i, cb in enumerate(callbacks):
    ax2.text(2.5, y_callback+0.3-i*0.4, cb, fontsize=9, color='#555')

# 训练配置
y_config = 3
ax2.text(11, y_config+0.8, '训练配置:', fontsize=11, fontweight='bold')
configs = [
    'Epochs: 50 (最多)',
    'Batch Size: 256',
    'Optimizer: Adam (lr=0.001)',
    'Loss: MAE (Mean Absolute Error)',
]
for i, cfg in enumerate(configs):
    ax2.text(11.5, y_config+0.3-i*0.4, f'• {cfg}', fontsize=9, color='#555')

# 结果
box_result = FancyBboxPatch((6, 0.3), 8, 0.8,
                           boxstyle="round,pad=0.1",
                           edgecolor='green', facecolor='#C8E6C9',
                           linewidth=2)
ax2.add_patch(box_result)
ax2.text(10, 0.7, '训练完成 → 保存最佳模型权重 (best_weights.hdf5)', 
        ha='center', fontsize=10, fontweight='bold', color='green')

# ============================================================================
# 图3: 预测流程
# ============================================================================
ax3.set_xlim(0, 20)
ax3.set_ylim(0, 8)
ax3.axis('off')
ax3.set_title('GSTA预测流程（在测试集上）', fontsize=16, fontweight='bold', pad=20)

# 流程图
y_pred = 6

# 步骤1: 加载模型
box1 = FancyBboxPatch((1, y_pred), 4, 1.2,
                     boxstyle="round,pad=0.1",
                     edgecolor='#2196F3', facecolor='#E3F2FD',
                     linewidth=2)
ax3.add_patch(box1)
ax3.text(3, y_pred+0.8, '步骤1', fontsize=10, fontweight='bold', color='#2196F3')
ax3.text(3, y_pred+0.4, '加载最佳模型', fontsize=9)
ax3.text(3, y_pred+0.05, 'best_weights.hdf5', fontsize=8, color='#666')

# 步骤2: 准备测试数据
box2 = FancyBboxPatch((6, y_pred), 4, 1.2,
                     boxstyle="round,pad=0.1",
                     edgecolor='#2196F3', facecolor='#E3F2FD',
                     linewidth=2)
ax3.add_patch(box2)
ax3.text(8, y_pred+0.8, '步骤2', fontsize=10, fontweight='bold', color='#2196F3')
ax3.text(8, y_pred+0.4, '准备测试数据', fontsize=9)
ax3.text(8, y_pred+0.05, 'X_test_list', fontsize=8, color='#666')

# 步骤3: 预测
box3 = FancyBboxPatch((11, y_pred), 4, 1.2,
                     boxstyle="round,pad=0.1",
                     edgecolor='#2196F3', facecolor='#E3F2FD',
                     linewidth=2)
ax3.add_patch(box3)
ax3.text(13, y_pred+0.8, '步骤3', fontsize=10, fontweight='bold', color='#2196F3')
ax3.text(13, y_pred+0.4, '一次性预测', fontsize=9)
ax3.text(13, y_pred+0.05, 'GSTA.predict()', fontsize=8, color='#666')

# 步骤4: 评估
box4 = FancyBboxPatch((16, y_pred), 3, 1.2,
                     boxstyle="round,pad=0.1",
                     edgecolor='#2196F3', facecolor='#E3F2FD',
                     linewidth=2)
ax3.add_patch(box4)
ax3.text(17.5, y_pred+0.8, '步骤4', fontsize=10, fontweight='bold', color='#2196F3')
ax3.text(17.5, y_pred+0.4, '计算指标', fontsize=9)
ax3.text(17.5, y_pred+0.05, 'MAE/RMSE/R²', fontsize=8, color='#666')

# 箭头连接
for i in range(3):
    arrow = FancyArrowPatch((5+i*5, y_pred+0.6), (6+i*5, y_pred+0.6),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax3.add_patch(arrow)

# 详细说明
y_detail = 4
ax3.text(2, y_detail+0.8, '预测过程详解:', fontsize=11, fontweight='bold')

details = [
    '输入: X_test_list (测试集所有特征)',
    '  ↓ Embedding层 (处理分类特征)',
    '  ↓ 空间-时间注意力',
    '  ↓ 多头注意力 (4个头)',
    '  ↓ 前馈网络 (3层Dense)',
    '输出: predictions (预测的travel_time)',
]

for i, detail in enumerate(details):
    color = '#2196F3' if '↓' in detail else '#333'
    weight = 'bold' if '输入' in detail or '输出' in detail else 'normal'
    ax3.text(2.3, y_detail+0.3-i*0.35, detail, fontsize=9, color=color, fontweight=weight)

# 评估指标
y_metrics = 3.5
ax3.text(12, y_metrics+0.8, '评估指标:', fontsize=11, fontweight='bold')
metrics = [
    'MAPE: 平均绝对百分比误差',
    'MAE: 平均绝对误差',
    'RMSE: 均方根误差',
    'MdAE: 中位数绝对误差',
    'R²: 决定系数',
]
for i, metric in enumerate(metrics):
    ax3.text(12.3, y_metrics+0.3-i*0.35, f'• {metric}', fontsize=9, color='#555')

# 关键特点
box_key = FancyBboxPatch((4, 0.3), 12, 1.2,
                        boxstyle="round,pad=0.1",
                        edgecolor='#FF5722', facecolor='#FFEBEE',
                        linewidth=2)
ax3.add_patch(box_key)
ax3.text(10, 1.2, '关键特点', fontsize=11, fontweight='bold', color='#FF5722')
ax3.text(10, 0.8, '✓ 每个样本独立预测（不是时间序列） | ✓ 一次性处理全部测试集 | ✓ 使用验证集上的最佳模型',
        ha='center', fontsize=9, color='#333')

plt.tight_layout()
plt.savefig('output/GSTA_training_prediction_flow.png', dpi=300, bbox_inches='tight')
print("✓ GSTA训练和预测流程图已保存: output/GSTA_training_prediction_flow.png")
plt.close()

print("\n" + "="*80)
print("GSTA关键流程总结")
print("="*80)
print("\n【数据划分】")
print("  ✓ 训练集 (70%): 训练模型参数")
print("  ✓ 验证集 (15%): 早停 + 保存最佳模型 + 超参数调优")
print("  ✓ 测试集 (15%): 最终评估性能")

print("\n【训练流程】")
print("  1. 加载预先划分好的数据 (X_train, Y_train, X_val, Y_val)")
print("  2. 每个epoch:")
print("     - 训练步骤: 前向传播 → 计算损失 → 反向传播 → 更新参数")
print("     - 验证步骤: 计算val_loss → 保存最佳模型 → 检查早停")
print("  3. 训练结束: 保存验证集上表现最好的模型")

print("\n【预测流程】")
print("  1. 加载最佳模型权重")
print("  2. 在测试集上一次性预测: GSTA.predict(X_test)")
print("  3. 计算评估指标: MAPE, MAE, RMSE, R²")

print("\n【与LSTM的区别】")
print("  GSTA  : 每个样本独立 | 回归预测 | 预测单次行程时间")
print("  LSTM  : 样本有依赖   | 序列预测 | 预测OD流量序列")

print("\n" + "="*80)
