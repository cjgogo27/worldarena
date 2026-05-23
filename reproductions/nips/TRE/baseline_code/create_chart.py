import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 定义数据 - R数量固定为20
operators = ['None', 'Greedy Insertion', 'Historical Removal', 'Worst Removal', 
             'Route Removal', 'Related Removal', 'Random Removal', 'Random Insertion']

# 定义每个算子的数据（R=20时的服务比例）
operator_data = {
    'evtol': [70, 66.66666667, 75, 66.66666667, 66.66666667, 66.66666667, 70, 66.66666667],
    'gv': [15, 33.33333333, 10, 33.33333333, 33.33333333, 33.33333333, 10, 33.33333333],
    'drone': [15, 0, 15, 0, 0, 0, 20, 0]
}

# 设置颜色
colors = {
    'evtol': 'green',  # 绿色
    'gv': 'blue',      # 蓝色
    'drone': 'orange'  # 橙色
}

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))  # 创建单个图表

# 设置颜色
colors = {
    'evtol': 'green',  # 绿色
    'gv': 'blue',      # 蓝色
    'drone': 'orange'  # 橙色
}

# 绘制堆叠柱状图
x = np.arange(len(operators))  # 横轴位置
width = 0.6  # 柱子宽度

# 底部基线，初始为0
bottom = np.zeros(len(operators))

# 依次绘制每个类别的堆叠部分
for category, values in operator_data.items():
    bars = ax.bar(x, values, width, label=category, color=colors[category], bottom=bottom)
    bottom += values
    
    # 在每个柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
                f'{height:.1f}%', ha='center', va='center', color='white', fontweight='bold')

# 设置图表标题和标签
ax.set_title('不同算子在R=20时的服务比例（堆叠柱状图）', fontsize=16)
ax.set_xlabel('算子', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(operators, rotation=45, ha='right')
ax.set_ylim(0, 100)  # 设置纵轴范围
ax.legend()

# 添加网格线
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局，确保标签不被截断
plt.tight_layout()

# 保存图表
plt.savefig('e:\\TRE\\baseline_code\\stacked_bar_chart_R20.png')
plt.show()