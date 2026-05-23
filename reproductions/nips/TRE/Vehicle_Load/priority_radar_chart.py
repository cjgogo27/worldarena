import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.path import Path
import matplotlib.cm as cm

# 设置高质量图表参数
plt.rcParams['font.family'] = ['Arial', 'SimHei', 'sans-serif']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10
plt.rcParams['savefig.dpi'] = 300

# 准备数据
priority_ratios = ['Passenger high priority', 'Equal']
distance = [610, 420]
served_requests = [2, 2]
request_cost = [1737, 2094]
delay_penalty = [0, 375]
number_used_vehicles = [2, 1]

# 对数据进行归一化处理，便于在雷达图上展示
data = np.array([distance, served_requests, request_cost, delay_penalty, number_used_vehicles]).T

# 对于成本类指标，较低的值表示更好的性能，我们需要反转这些指标的数据
# 对于距离，我们也反转，因为较短的距离可能意味着更高效的路径
# 创建需要反转的索引
invert_indices = [0, 2, 3]  # distance, request_cost, delay_penalty

# 对每个需要反转的指标，计算其最大值，然后用最大值减去当前值
data_normalized = data.copy().astype(float)
for i in invert_indices:
    max_val = np.max(data[:, i])
    data_normalized[:, i] = max_val - data[:, i] + np.min(data[:, i])

# 对所有指标进行0-1归一化
for i in range(data_normalized.shape[1]):
    min_val = np.min(data_normalized[:, i])
    max_val = np.max(data_normalized[:, i])
    if max_val - min_val > 0:  # 避免除以零
        data_normalized[:, i] = (data_normalized[:, i] - min_val) / (max_val - min_val)

# 雷达图的标签
categories = ['Distance (m)', 'Served Requests', 'Request Cost (¥)', 'Delay Penalty', 'Vehicles Used']
n = len(categories)

# 计算每个类别的角度
angles = [i / float(n) * 2 * np.pi for i in range(n)]
angles += angles[:1]  # 闭合雷达图

# 为每个数据点也添加闭合
values1 = data_normalized[0].tolist() + data_normalized[0].tolist()[:1]
values2 = data_normalized[1].tolist() + data_normalized[1].tolist()[:1]

# 创建雷达图
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# 设置雷达图的角度和标签
ax.set_theta_offset(np.pi / 2)  # 从顶部开始
ax.set_theta_direction(-1)  # 顺时针方向
plt.xticks(angles[:-1], categories, fontsize=14, fontweight='bold')

# 设置y轴的范围和刻度
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.set_rlabel_position(0)  # 将y轴标签放在顶部

# 添加网格线，使图表更清晰
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)

# 定义颜色
colors = ['#1f77b4', '#ff7f0e']

# 绘制两个优先级的雷达图区域，提高透明度以便更好地显示重合部分
ax.fill(angles, values1, color=colors[0], alpha=0.35, label=priority_ratios[0])
ax.fill(angles, values2, color=colors[1], alpha=0.35, label=priority_ratios[1])

# 绘制两个优先级的雷达图轮廓线
ax.plot(angles, values1, color=colors[0], linewidth=2.5, marker='o', markersize=8, label='')
ax.plot(angles, values2, color=colors[1], linewidth=2.5, marker='s', markersize=8, label='')

# 为每个点添加数值标签
for i in range(n):
    # 乘客高优先级的标签
    ax.text(angles[i], values1[i] + 0.05, f'{data[0, i]}', 
            ha='center', va='center', fontsize=11, fontweight='bold', color=colors[0])
    # 平等优先级的标签
    ax.text(angles[i], values2[i] - 0.08, f'{data[1, i]}', 
            ha='center', va='center', fontsize=11, fontweight='bold', color=colors[1])

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14, 
           frameon=True, framealpha=0.9, edgecolor='gray')

# 添加标题
plt.title('Performance Comparison by Priority Ratio', fontsize=22, fontweight='bold', pad=20)

# 添加说明框，解释数据的实际含义
explanation_text = "Note:\n"
explanation_text += "• Lower values are better for: Distance, Request Cost, Delay Penalty, Vehicles Used\n"
explanation_text += "• Higher values are better for: Served Requests\n"
explanation_text += "• Data is normalized for visualization purposes"

fig.text(0.02, 0.02, explanation_text, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=1'))

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('priority_comparison_radar_chart1.png', dpi=600, bbox_inches='tight')
plt.savefig('priority_comparison_radar_chart1.svg', format='svg', bbox_inches='tight')
plt.savefig('priority_comparison_radar_chart1.pdf', format='pdf', bbox_inches='tight')

# 打印分析摘要
print("=== Priority Comparison Radar Chart Analysis ===")
print(f"1. Passenger high priority settings use 2 vehicles compared to 1 in equal priority")
print(f"2. Equal priority achieves shorter distance ({distance[1]} vs {distance[0]})")
print(f"3. Request cost is lower with passenger high priority ({request_cost[0]} vs {request_cost[1]})")
print(f"4. Delay penalty is zero with passenger high priority but {delay_penalty[1]} with equal priority")
print(f"5. Both settings serve the same number of requests ({served_requests[0]})")
print("\nRadar chart saved in PNG, SVG, and PDF formats.")

# 显示图表
plt.show()