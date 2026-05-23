#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化LSTM预测机制
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# ============================================================================
# 图1: 数据连接方式
# ============================================================================
ax = axes[0]
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('LSTM数据准备：多天数据连接成连续序列', 
             fontsize=16, fontweight='bold', pad=20)

# 绘制日期块
days = [
    {'name': '2月2日', 'color': '#E8F4F8', 'x': 1, 'slots': '0-95'},
    {'name': '2月3日', 'color': '#D4E9F2', 'x': 3.5, 'slots': '96-191'},
    {'name': '...', 'color': 'white', 'x': 6, 'slots': ''},
    {'name': '2月7日', 'color': '#FFE6CC', 'x': 8.5, 'slots': '480-575'},
    {'name': '2月8日', 'color': '#FFD9B3', 'x': 11, 'slots': '576-671'},
]

y_base = 5
for day in days:
    if day['name'] != '...':
        box = FancyBboxPatch((day['x'], y_base), 2, 1.5, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=day['color'],
                            linewidth=2)
        ax.add_patch(box)
        ax.text(day['x'] + 1, y_base + 1.1, day['name'], 
               ha='center', va='center', fontsize=11, fontweight='bold')
        if day['slots']:
            ax.text(day['x'] + 1, y_base + 0.5, f'槽{day["slots"]}', 
                   ha='center', va='center', fontsize=9)
    else:
        ax.text(day['x'] + 1, y_base + 0.75, day['name'],
               ha='center', va='center', fontsize=16, fontweight='bold')

# 箭头连接
arrow_y = y_base + 0.75
for i in range(len(days) - 1):
    if days[i]['name'] != '...' and days[i+1]['name'] != '...':
        start_x = days[i]['x'] + 2.1
        end_x = days[i+1]['x'] - 0.1
        arrow = FancyArrowPatch((start_x, arrow_y), (end_x, arrow_y),
                              arrowstyle='->', mutation_scale=20, 
                              linewidth=2, color='#333')
        ax.add_patch(arrow)

# 说明文字
ax.text(8, 3.5, '连接形成连续时间序列', 
       ha='center', fontsize=13, style='italic')
ax.text(8, 2.8, '☑ 优点：能捕捉跨天的流量变化', 
       ha='center', fontsize=11, color='green')
ax.text(8, 2.2, '☑ 优点：学习短期连续趋势（3小时窗口）', 
       ha='center', fontsize=11, color='green')
ax.text(8, 1.6, '☐ 缺点：忽略日周期性（每天同时段规律）', 
       ha='center', fontsize=11, color='orange')

# ============================================================================
# 图2: 滑动窗口预测机制
# ============================================================================
ax = axes[1]
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('滑动窗口预测：前12槽→预测下1槽', 
             fontsize=16, fontweight='bold', pad=20)

# 示例1: 预测2月8日凌晨00:00
y1 = 5.5
# 输入框 (2月7日21:00-23:45)
box1 = FancyBboxPatch((1, y1), 5, 1.2,
                     boxstyle="round,pad=0.05",
                     edgecolor='blue', facecolor='#E3F2FD',
                     linewidth=2)
ax.add_patch(box1)
ax.text(3.5, y1+0.6, '输入: 2月7日 21:00-23:45', 
       ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(3.5, y1+0.2, '(12个连续时间槽)', 
       ha='center', va='center', fontsize=9, color='#666')

# LSTM
box2 = FancyBboxPatch((7, y1+0.1), 1.5, 1,
                     boxstyle="round,pad=0.05",
                     edgecolor='purple', facecolor='#F3E5F5',
                     linewidth=2)
ax.add_patch(box2)
ax.text(7.75, y1+0.6, 'LSTM', 
       ha='center', va='center', fontsize=11, fontweight='bold')

# 输出框
box3 = FancyBboxPatch((10, y1), 4, 1.2,
                     boxstyle="round,pad=0.05",
                     edgecolor='red', facecolor='#FFEBEE',
                     linewidth=2)
ax.add_patch(box3)
ax.text(12, y1+0.6, '预测: 2月8日 00:00', 
       ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(12, y1+0.2, '(1个时间槽)', 
       ha='center', va='center', fontsize=9, color='#666')

# 箭头
arrow1 = FancyArrowPatch((6, y1+0.6), (7, y1+0.6),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='black')
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch((8.5, y1+0.6), (10, y1+0.6),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='black')
ax.add_patch(arrow2)

# 示例2: 预测2月8日00:15
y2 = 3.5
box4 = FancyBboxPatch((1, y2), 5, 1.2,
                     boxstyle="round,pad=0.05",
                     edgecolor='blue', facecolor='#E3F2FD',
                     linewidth=1.5, linestyle='--')
ax.add_patch(box4)
ax.text(3.5, y2+0.6, '输入: 2月7日 21:15 - 2月8日 00:00', 
       ha='center', va='center', fontsize=10)
ax.text(3.5, y2+0.2, '(跨天输入)', 
       ha='center', va='center', fontsize=9, color='#666', style='italic')

box5 = FancyBboxPatch((10, y2), 4, 1.2,
                     boxstyle="round,pad=0.05",
                     edgecolor='red', facecolor='#FFEBEE',
                     linewidth=1.5, linestyle='--')
ax.add_patch(box5)
ax.text(12, y2+0.6, '预测: 2月8日 00:15', 
       ha='center', va='center', fontsize=10)

ax.text(8, y2+0.6, '→', ha='center', fontsize=20)

# 关键说明
ax.text(8, 1.5, '🔥 关键: 预测新一天的凌晨需要前一天晚上的数据！', 
       ha='center', fontsize=12, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ============================================================================
# 图3: 当前方法 vs 周期性方法
# ============================================================================
ax = axes[2]
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('两种预测方法对比', 
             fontsize=16, fontweight='bold', pad=20)

# 当前方法
y_current = 5
ax.text(1, y_current+1.2, '✅ 当前方法：连续时间序列', 
       fontsize=12, fontweight='bold', color='green')

# 输入可视化
slots_x = np.linspace(2, 6, 12)
for i, x in enumerate(slots_x):
    color = '#FFE6CC' if i < 8 else '#FFD9B3'  # 前8个是2月7日，后4个是2月8日
    box = FancyBboxPatch((x-0.15, y_current), 0.25, 0.5,
                        facecolor=color, edgecolor='black', linewidth=0.5)
    ax.add_patch(box)

ax.text(4, y_current-0.5, '2月7日21:00-23:45 + 2月8日00:00-00:45', 
       ha='center', fontsize=9)

# 箭头和预测
ax.text(6.5, y_current+0.25, '→', fontsize=20)
box_pred = FancyBboxPatch((7, y_current), 0.5, 0.5,
                         facecolor='#FFEBEE', edgecolor='red', linewidth=2)
ax.add_patch(box_pred)
ax.text(7.25, y_current-0.5, '预测2月8日01:00', ha='center', fontsize=9)

ax.text(10, y_current+0.5, '学习：短期趋势', fontsize=10, style='italic')
ax.text(10, y_current, '例如: 流量逐渐减少', fontsize=9, color='#666')

# 周期性方法
y_period = 2
ax.text(1, y_period+1.2, '🔄 周期性方法：同时段历史', 
       fontsize=12, fontweight='bold', color='blue')

# 输入可视化
days_x = [2, 3, 4, 5, 6]
for i, x in enumerate(days_x):
    box = FancyBboxPatch((x-0.15, y_period), 0.25, 0.5,
                        facecolor='#E8F4F8', edgecolor='black', linewidth=0.5)
    ax.add_patch(box)

ax.text(4, y_period-0.5, '2月2-6日每天的00:00', ha='center', fontsize=9)

# 箭头和预测
ax.text(6.5, y_period+0.25, '→', fontsize=20)
box_pred2 = FancyBboxPatch((7, y_period), 0.5, 0.5,
                          facecolor='#FFEBEE', edgecolor='red', linewidth=2)
ax.add_patch(box_pred2)
ax.text(7.25, y_period-0.5, '预测2月8日00:00', ha='center', fontsize=9)

ax.text(10, y_period+0.5, '学习：日周期性', fontsize=10, style='italic')
ax.text(10, y_period, '例如: 凌晨都很少', fontsize=9, color='#666')

# 图例
legend_elements = [
    mpatches.Patch(facecolor='#FFE6CC', edgecolor='black', label='2月7日数据'),
    mpatches.Patch(facecolor='#FFD9B3', edgecolor='black', label='2月8日数据'),
    mpatches.Patch(facecolor='#E8F4F8', edgecolor='black', label='历史同时段'),
    mpatches.Patch(facecolor='#FFEBEE', edgecolor='red', label='预测目标', linewidth=2),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('output/lstm_mechanism_visualization.png', dpi=300, bbox_inches='tight')
print("✓ 可视化图表已保存: output/lstm_mechanism_visualization.png")
plt.close()

print("\n图表包含三个部分:")
print("  1. 数据连接方式 - 展示如何将多天数据连接成连续序列")
print("  2. 滑动窗口机制 - 展示如何用前12槽预测下1槽")
print("  3. 方法对比 - 对比连续时间序列vs周期性方法")
