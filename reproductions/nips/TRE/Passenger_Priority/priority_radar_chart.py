import matplotlib.pyplot as plt
import numpy as np

# 设置字体 - 使用Times New Roman或serif字体使其更接近原图
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

# 数据 - 按照原图顺序：Distance在顶部，然后顺时针
categories = ['Distance', 'Served requests', 'Request cost', 'Delay penalty', 'Number of vehicles']
N = len(categories)

# 原始数据
high_priority_raw = [610, 2, 1737, 0, 2]
equal_priority_raw = [420, 2, 2094, 375, 1]

# 归一化为百分比（相对于每个维度的最大值），然后缩小到70%
max_values = [max(high_priority_raw[i], equal_priority_raw[i]) for i in range(N)]
# 特殊处理：如果最大值为0，设为1避免除0
max_values = [m if m > 0 else 1 for m in max_values]

# 转换为百分比并缩放到70%
scale_factor = 0.7
high_priority = [(high_priority_raw[i] / max_values[i]) * 100 * scale_factor for i in range(N)]
equal_priority = [(equal_priority_raw[i] / max_values[i]) * 100 * scale_factor for i in range(N)]

# 计算角度
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
high_priority_plot = high_priority + [high_priority[0]]
equal_priority_plot = equal_priority + [equal_priority[0]]
angles += angles[:1]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# 设置起始角度为顶部，顺时针方向
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 绘制高优先级数据（蓝色，圆形标记）- 使用原图颜色
ax.plot(angles, high_priority_plot, 'o-', linewidth=2.5, color='#5B9FBF', 
        label='Passenger High Priority', markersize=9)
ax.fill(angles, high_priority_plot, alpha=0.3, color='#7DB9D3')

# 绘制相等优先级数据（黄色，方形标记）- 使用原图颜色
ax.plot(angles, equal_priority_plot, 's-', linewidth=2.5, color='#D4C341', 
        label='Equal Priority', markersize=9)
ax.fill(angles, equal_priority_plot, alpha=0.3, color='#E8D96F')

# 设置类别标签 - 使用serif字体，黑色
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=18, fontweight='bold', family='serif')

# 设置Y轴范围和网格 - 显示百分比
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=14, color='gray')
ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8, color='gray')

# 添加原始数值标签 - 使用原始值而非百分比，字体更大
# Distance: 610(蓝) 420(黄)
ax.text(angles[0], high_priority[0] + 8.5, '610', ha='center', va='bottom', 
        fontsize=19, color='#3A7FA0', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.47', facecolor='white', edgecolor='#5B9FBF', linewidth=1.85))
ax.text(angles[0], equal_priority[0] - 5.5, '420', ha='center', va='top', 
        fontsize=19, color='#A89B1E', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.47', facecolor='#FFFACD', edgecolor='#D4C341', linewidth=1.85))

# Served requests: 2(蓝和黄都是2，在同一位置)
ax.text(angles[1] + 0.01, high_priority[1] + 3.9, '2', ha='left', va='center', 
        fontsize=19, color='#3A7FA0', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.47', facecolor='white', edgecolor='#5B9FBF', linewidth=1.85))
ax.text(angles[1] + 0.2, equal_priority[1] + 11.6, '2', ha='left', va='center', 
        fontsize=19, color='#A89B1E', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.47', facecolor='#FFFACD', edgecolor='#D4C341', linewidth=1.85))

# Request cost: 1737(蓝) 2094(黄)
ax.text(angles[2], high_priority[2] - 8.5, '1737', ha='center', va='top', 
        fontsize=19, color='#3A7FA0', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.47', facecolor='white', edgecolor='#5B9FBF', linewidth=1.85))
ax.text(angles[2] + 0.09, equal_priority[2] + 8.5, '2094', ha='center', va='bottom', 
        fontsize=19, color='#A89B1E', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.47', facecolor='#FFFACD', edgecolor='#D4C341', linewidth=1.85))

# Delay penalty: 0(蓝-中心) 375(黄)
ax.text(angles[3], 5, '0', ha='center', va='center', 
        fontsize=19, color='#3A7FA0', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.47', facecolor='white', edgecolor='#5B9FBF', linewidth=1.85))
ax.text(angles[3] - 0.155, equal_priority[3] - 8.5, '375', ha='right', va='top', 
        fontsize=19, color='#A89B1E', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.47', facecolor='#FFFACD', edgecolor='#D4C341', linewidth=1.85))

# Number of vehicles: 2(蓝) 1(黄)
ax.text(angles[4] - 0.125, high_priority[4] + 3.5, '2', ha='right', va='center', 
        fontsize=19, color='#3A7FA0', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.47', facecolor='white', edgecolor='#5B9FBF', linewidth=1.85))
ax.text(angles[4] - 0.125, equal_priority[4] - 3.5, '1', ha='right', va='center', 
        fontsize=19, color='#A89B1E', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.47', facecolor='#FFFACD', edgecolor='#D4C341', linewidth=1.85))

# 添加图例 - 右上角
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.08), fontsize=14, 
                   frameon=True, edgecolor='black', fancybox=False, framealpha=1)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_linewidth(1.2)

# 调整布局
plt.tight_layout()

# 保存图片为PNG
plt.savefig('/data/mayue/cjy/priority_radar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
print("雷达图已保存为: priority_radar_chart.png")

# 保存图片为PDF
plt.savefig('/data/mayue/cjy/priority_radar_chart.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("雷达图已保存为: priority_radar_chart.pdf")

print(f"\n归一化信息：")
print(f"Distance: max={max_values[0]} -> High:{high_priority[0]:.1f}%, Equal:{equal_priority[0]:.1f}%")
print(f"Served requests: max={max_values[1]} -> High:{high_priority[1]:.1f}%, Equal:{equal_priority[1]:.1f}%")
print(f"Request cost: max={max_values[2]} -> High:{high_priority[2]:.1f}%, Equal:{equal_priority[2]:.1f}%")
print(f"Delay penalty: max={max_values[3]} -> High:{high_priority[3]:.1f}%, Equal:{equal_priority[3]:.1f}%")
print(f"Number of vehicles: max={max_values[4]} -> High:{high_priority[4]:.1f}%, Equal:{equal_priority[4]:.1f}%")

# 显示图片
plt.show()
