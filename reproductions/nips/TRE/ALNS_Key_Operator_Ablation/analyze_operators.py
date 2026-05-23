import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 数据
data = {
    'operator': ['ALL', 'Greedy Insertion', 'Historical Removal', 'Worst Removal', 
                 'Route Removal', 'Related Removal', 'Random Removal', 'Random Insertion'],
    'r': [20] * 8,
    'served_requests': [20, 6, 20, 6, 6, 6, 20, 6],
    'best_iteration_number': [162, 0, 149, 0, 0, 0, 95, 0],
    'best_cost': [12592.933, 7052.055, 12917.879, 7052.055, 7052.055, 7052.055, 12904.174, 7052.055],
    'best_request_cost': [9972.734, 6070.22, 10319.421, 6070.22, 6070.22, 6070.22, 10321.637, 6070.22],
    'best_unload_cost': [2554.5, 952.5, 2526.5, 952.5, 952.5, 952.5, 2514.5, 952.5],
    'best_emission_cost': [65.698, 29.334, 71.959, 29.334, 29.334, 29.334, 68.037, 29.334],
    'best_storage_cost': [80.213, 0, 80.213, 0, 0, 0, 0, 0],
    'initial_time': [5.266, 11.303, 11.518, 11.641, 11.183, 11.270, 11.083, 11.096],
    'add_initial_best_time': [367.939, 11.303, 510.921, 11.641, 11.183, 11.270, 331.359, 11.096],
    'number_used_vehicles': [14, 5, 14, 5, 5, 5, 15, 5],
    'barge_seved_r_portion': [70, 66.667, 75, 66.667, 66.667, 66.667, 70, 66.667],
    'train_seved_r_portion': [15, 33.333, 10, 33.333, 33.333, 33.333, 10, 33.333],
    'truck_seved_r_portion': [15, 0, 15, 0, 0, 0, 20, 0]
}

df = pd.DataFrame(data)

# 定义颜色 - 为8个算子定义颜色
all_color = '#2ecc71'  # 绿色 - ALL (最好的结果)
good_colors = ['#3498db', '#9b59b6']  # 蓝色和紫色 - 有效的算子
poor_colors = ['#e74c3c', '#e67e22', '#95a5a6', '#34495e', '#7f8c8d', '#c0392b']  # 红色系 - 无效的算子

# 8个算子的颜色：ALL, Greedy Insertion, Historical Removal, Worst Removal, Route Removal, Related Removal, Random Removal, Random Insertion
colors = [all_color, poor_colors[0], good_colors[0], poor_colors[1], poor_colors[2], poor_colors[3], good_colors[1], poor_colors[4]]

output_dir = r"E:\TRE\ALNS_Key_Operator_Ablation\Figures"

# 图1: 服务请求数对比 - 关键指标
print("生成图1: 服务请求数对比")
fig = plt.figure(figsize=(12, 7))
bars = plt.bar(range(len(df)), df['served_requests'], color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Operator', fontsize=14, fontweight='bold')
plt.ylabel('Number of Served Requests', fontsize=14, fontweight='bold')
plt.title('Impact of Operators on Service Quality: Served Requests\n(Key Metric for Solution Quality)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(df)), df['operator'], rotation=45, ha='right', fontsize=11)
plt.ylim(0, 25)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, df['served_requests'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 添加参考线
plt.axhline(y=20, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Maximum Requests (20)')
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_dir}/operator_analysis_1_served_requests.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/operator_analysis_1_served_requests.pdf', bbox_inches='tight')
print(f"  保存到: {output_dir}/operator_analysis_1_served_requests.png")
plt.close()

# 图2: 迭代次数对比 - 展示算法改进能力
print("生成图2: 最佳解迭代次数对比")
fig = plt.figure(figsize=(12, 7))
bars = plt.bar(range(len(df)), df['best_iteration_number'], color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Operator', fontsize=14, fontweight='bold')
plt.ylabel('Iteration Number at Best Solution', fontsize=14, fontweight='bold')
plt.title('Operators Contribution to Solution Improvement: Best Iteration Number\n(Higher values indicate continuous improvement capability)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(df)), df['operator'], rotation=45, ha='right', fontsize=11)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, df['best_iteration_number'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
             f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/operator_analysis_2_iteration_number.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/operator_analysis_2_iteration_number.pdf', bbox_inches='tight')
print(f"  保存到: {output_dir}/operator_analysis_2_iteration_number.png")
plt.close()

# 图3: 总成本对比
print("生成图3: 总成本对比")
fig = plt.figure(figsize=(12, 7))
bars = plt.bar(range(len(df)), df['best_cost'], color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Operator', fontsize=14, fontweight='bold')
plt.ylabel('Total Cost', fontsize=14, fontweight='bold')
plt.title('Solution Quality: Total Cost by Operator\n(Higher cost indicates more requests served with complex routing)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(df)), df['operator'], rotation=45, ha='right', fontsize=11)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, df['best_cost'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
             f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/operator_analysis_3_total_cost.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/operator_analysis_3_total_cost.pdf', bbox_inches='tight')
print(f"  保存到: {output_dir}/operator_analysis_3_total_cost.png")
plt.close()

# 图4: 成本结构堆叠图
print("生成图4: 成本结构对比")
fig = plt.figure(figsize=(12, 7))

cost_components = ['best_request_cost', 'best_unload_cost', 'best_emission_cost', 'best_storage_cost']
cost_labels = ['Request Cost', 'Unload Cost', 'Emission Cost', 'Storage Cost']
cost_colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']

bottoms = np.zeros(len(df))
for cost_col, label, color in zip(cost_components, cost_labels, cost_colors):
    plt.bar(range(len(df)), df[cost_col], bottom=bottoms, label=label, 
            color=color, edgecolor='black', linewidth=0.5)
    bottoms += df[cost_col].values

plt.xlabel('Operator', fontsize=14, fontweight='bold')
plt.ylabel('Cost Components', fontsize=14, fontweight='bold')
plt.title('Cost Structure Analysis by Operator\n(Breakdown showing complexity of solutions)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(df)), df['operator'], rotation=45, ha='right', fontsize=11)
plt.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig(f'{output_dir}/operator_analysis_4_cost_structure.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/operator_analysis_4_cost_structure.pdf', bbox_inches='tight')
print(f"  保存到: {output_dir}/operator_analysis_4_cost_structure.png")
plt.close()

# 图5: 计算时间对比
print("生成图5: 计算时间对比")
fig = plt.figure(figsize=(12, 7))
bars = plt.bar(range(len(df)), df['add_initial_best_time'], color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Operator', fontsize=14, fontweight='bold')
plt.ylabel('Computation Time (seconds)', fontsize=14, fontweight='bold')
plt.title('Computational Efficiency vs Solution Quality\n(Time investment for improved solutions)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(df)), df['operator'], rotation=45, ha='right', fontsize=11)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, df['add_initial_best_time'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'{val:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/operator_analysis_5_computation_time.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/operator_analysis_5_computation_time.pdf', bbox_inches='tight')
print(f"  保存到: {output_dir}/operator_analysis_5_computation_time.png")
plt.close()

# 图6: 使用车辆数对比
print("生成图6: 使用车辆数对比")
fig = plt.figure(figsize=(12, 7))
bars = plt.bar(range(len(df)), df['number_used_vehicles'], color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Operator', fontsize=14, fontweight='bold')
plt.ylabel('Number of Vehicles Used', fontsize=14, fontweight='bold')
plt.title('Resource Utilization: Number of Vehicles Required\n(Indicator of solution complexity and efficiency)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(df)), df['operator'], rotation=45, ha='right', fontsize=11)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, df['number_used_vehicles'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
             f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/operator_analysis_6_vehicles_used.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/operator_analysis_6_vehicles_used.pdf', bbox_inches='tight')
print(f"  保存到: {output_dir}/operator_analysis_6_vehicles_used.png")
plt.close()

# 图7: 运输模式分布（多模式柱状图）
print("生成图7: 运输模式分布对比")
fig = plt.figure(figsize=(14, 7))

x = np.arange(len(df))
width = 0.25

bars1 = plt.bar(x - width, df['barge_seved_r_portion'], width, label='Barge', 
                color='#3498db', edgecolor='black', linewidth=1)
bars2 = plt.bar(x, df['train_seved_r_portion'], width, label='Train', 
                color='#2ecc71', edgecolor='black', linewidth=1)
bars3 = plt.bar(x + width, df['truck_seved_r_portion'], width, label='Truck', 
                color='#e74c3c', edgecolor='black', linewidth=1)

plt.xlabel('Operator', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Requests Served (%)', fontsize=14, fontweight='bold')
plt.title('Transportation Mode Distribution by Operator\n(Diversity in solution approach - intermodal vs single mode)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, df['operator'], rotation=45, ha='right', fontsize=11)
plt.legend(fontsize=12, loc='upper right')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/operator_analysis_7_transport_mode.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/operator_analysis_7_transport_mode.pdf', bbox_inches='tight')
print(f"  保存到: {output_dir}/operator_analysis_7_transport_mode.png")
plt.close()

# 图8: 效率对比（服务请求数 vs 计算时间）散点图
print("生成图8: 效率分析散点图")
fig = plt.figure(figsize=(12, 8))

for i, row in df.iterrows():
    plt.scatter(row['add_initial_best_time'], row['served_requests'], 
               s=500, color=colors[i], edgecolor='black', linewidth=2, alpha=0.7)
    plt.annotate(row['operator'], 
                xy=(row['add_initial_best_time'], row['served_requests']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

plt.xlabel('Computation Time (seconds)', fontsize=14, fontweight='bold')
plt.ylabel('Number of Served Requests', fontsize=14, fontweight='bold')
plt.title('Efficiency Analysis: Solution Quality vs Computational Cost\n(Top-right quadrant shows best operators)', 
          fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')

# 添加参考线
plt.axhline(y=20, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Max Requests')
plt.axhline(y=6, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Poor Performance')
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig(f'{output_dir}/operator_analysis_8_efficiency_scatter.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/operator_analysis_8_efficiency_scatter.pdf', bbox_inches='tight')
print(f"  保存到: {output_dir}/operator_analysis_8_efficiency_scatter.png")
plt.close()

# 图9: 综合性能雷达图
print("生成图9: 综合性能雷达图 - ALL vs Key Operators")

# 选择关键算子进行对比
key_operators = ['ALL', 'Historical Removal', 'Random Removal']
key_df = df[df['operator'].isin(key_operators)].copy()

# 标准化指标（0-100）
metrics = {
    'Served Requests': 'served_requests',
    'Iteration Progress': 'best_iteration_number',
    'Solution Quality\n(inverted cost)': 'best_cost',
    'Vehicle Efficiency\n(inverted)': 'number_used_vehicles',
    'Time Efficiency\n(inverted)': 'add_initial_best_time'
}

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

for idx, row in key_df.iterrows():
    values = []
    values.append(row['served_requests'] / 20 * 100)  # 服务请求比例
    values.append(row['best_iteration_number'] / 162 * 100)  # 迭代进度
    values.append((1 - row['best_cost'] / 13000) * 100)  # 成本倒置
    values.append((1 - row['number_used_vehicles'] / 15) * 100)  # 车辆数倒置
    values.append((1 - row['add_initial_best_time'] / 600) * 100)  # 时间倒置
    values += values[:1]
    
    color = colors[df[df['operator'] == row['operator']].index[0]]
    ax.plot(angles, values, 'o-', linewidth=2, label=row['operator'], color=color, markersize=8)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics.keys(), fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=True, shadow=True)
plt.title('Comprehensive Performance Comparison\n(Normalized scores: larger area = better performance)', 
          fontsize=16, fontweight='bold', pad=30, y=1.08)

plt.tight_layout()
plt.savefig(f'{output_dir}/operator_analysis_9_radar_chart.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/operator_analysis_9_radar_chart.pdf', bbox_inches='tight')
print(f"  保存到: {output_dir}/operator_analysis_9_radar_chart.png")
plt.close()

# 生成分析总结
print("\n" + "="*80)
print("算子消融实验分析总结 - 回答审稿人问题")
print("="*80)
print("\n关键发现 (Key Findings):")
print("\n1. 算子的必要性 (Necessity of Operators):")
print("   - ALL (所有算子组合): 服务20个请求，达到最优解")
print("   - 单一移除算子 (Greedy, Worst, Route, Related, Random Insertion): 仅服务6个请求 (70%性能下降)")
print("   - Historical Removal: 服务20个请求，性能接近ALL")
print("   - Random Removal: 服务20个请求，性能接近ALL")
print("\n2. 算子对解质量的显著影响 (Significant Impact on Solution Quality):")
print(f"   - 最佳组合 (ALL): {df[df['operator']=='ALL']['served_requests'].values[0]} 请求服务")
print(f"   - 最差单一算子: {df[df['operator']=='Greedy Insertion']['served_requests'].values[0]} 请求服务")
print(f"   - 性能差异: {(20-6)/6*100:.1f}% 改进")
print("\n3. 迭代改进能力 (Iterative Improvement Capability):")
print("   - ALL: 在第162次迭代达到最优 (持续改进)")
print("   - Historical Removal: 在第149次迭代达到最优")
print("   - 无效算子: 0次迭代 (无改进能力)")
print("\n4. 计算效率权衡 (Computational Trade-off):")
print("   - 高质量解需要更多计算时间 (367-511秒)")
print("   - 低质量解虽然快速 (11秒) 但无实用价值")
print("\n结论 (Conclusion):")
print("实验清楚证明了多样化算子的必要性:")
print("- Historical Removal 和 Random Removal 是关键算子")
print("- 单一算子严重限制了搜索空间")
print("- 算子组合显著提升解的质量 (服务率提升233%)")
print("="*80)

print("\n所有图表已生成完成！")
print(f"保存位置: {output_dir}")
