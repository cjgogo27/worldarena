import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 设置中文字体和科学绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 准备数据
K = [5, 10, 20, 30]  # 提供的车辆数目
R = [10, 20, 50, 100]  # 订单数目
number_used_vehicles = [4, 5, 11, 24]  # 使用的车辆数目
load_factor = [97.80, 94.75, 93.57, 92.81]  # 负载因子(%)
request_per_vehicle = [2.5, 4, 4.54, 4.16]  # 每辆车的订单数量

# 计算车辆使用率
vehicle_utilization = [used / provided * 100 for used, provided in zip(number_used_vehicles, K)]

# 创建DataFrame便于处理
data = pd.DataFrame({
    '提供的车辆数目(K)': K,
    '订单数目(R)': R,
    '使用的车辆数目': number_used_vehicles,
    '负载因子(%)': load_factor,
    '每辆车的订单数量': request_per_vehicle,
    '车辆使用率(%)': vehicle_utilization
})

# 创建一个高级的组合图表
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig)

# 设置整体标题
fig.suptitle('车辆资源分配与订单处理效率分析', fontsize=20, fontweight='bold', y=0.98)

# 第一个子图：负载因子和车辆使用率随订单数变化
ax1 = fig.add_subplot(gs[0, 0])

# 创建双Y轴
twin1 = ax1.twinx()

# 绘制负载因子曲线（左Y轴）
l1, = ax1.plot(data['订单数目(R)'], data['负载因子(%)'], 'o-', 
               color='#1f77b4', linewidth=3, markersize=8, label='负载因子(%)')
ax1.set_xlabel('订单数目(R)', fontsize=14)
ax1.set_ylabel('负载因子(%)', fontsize=14, color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=12)
ax1.grid(True, alpha=0.3)

# 绘制车辆使用率曲线（右Y轴）
l2, = twin1.plot(data['订单数目(R)'], data['车辆使用率(%)'], 's--', 
                color='#ff7f0e', linewidth=3, markersize=8, label='车辆使用率(%)')
twin1.set_ylabel('车辆使用率(%)', fontsize=14, color='#ff7f0e')
twin1.tick_params(axis='y', labelcolor='#ff7f0e', labelsize=12)

# 合并图例
lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=12)

ax1.set_title('订单数目对负载因子和车辆使用率的影响', fontsize=16, pad=20)

# 第二个子图：每辆车的订单数量
ax2 = fig.add_subplot(gs[0, 1])

# 创建柱状图
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax2.bar(data['订单数目(R)'], data['每辆车的订单数量'], color=colors, alpha=0.8, width=5)

# 在柱状图上添加数值标签
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.2f}', ha='center', va='bottom', fontsize=12)

ax2.set_xlabel('订单数目(R)', fontsize=14)
ax2.set_ylabel('每辆车的订单数量', fontsize=14)
ax2.set_title('每辆车处理的订单数量分析', fontsize=16, pad=20)
ax2.grid(True, axis='y', alpha=0.3)

# 第三个子图：3D散点图展示三个变量的关系
from mpl_toolkits.mplot3d import Axes3D
ax3 = fig.add_subplot(gs[1, :], projection='3d')

# 创建3D散点图
scatter = ax3.scatter(data['提供的车辆数目(K)'], data['订单数目(R)'], data['负载因子(%)'],
                    c=data['负载因子(%)'], cmap='viridis', s=100, alpha=0.8, edgecolors='k')

# 添加颜色条
cbar = fig.colorbar(scatter, ax=ax3, pad=0.1)
cbar.set_label('负载因子(%)', fontsize=12)

# 设置坐标轴标签
ax3.set_xlabel('提供的车辆数目(K)', fontsize=14, labelpad=10)
ax3.set_ylabel('订单数目(R)', fontsize=14, labelpad=10)
ax3.set_zlabel('负载因子(%)', fontsize=14, labelpad=10)

ax3.set_title('车辆数目、订单数目与负载因子的三维关系', fontsize=16, pad=20)

# 添加网格线
ax3.grid(True, alpha=0.3)

# 优化布局
plt.tight_layout(rect=[0, 0, 1, 0.97])

# 添加注释说明
fig.text(0.5, 0.01, '实验数据显示：随着订单数目增加，负载因子略有下降但保持在90%以上，体现了系统的稳定性。', 
         ha='center', fontsize=12, style='italic')

# 保存图表为高分辨率图片
plt.savefig('vehicle_resource_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('vehicle_resource_analysis.svg', format='svg', bbox_inches='tight')

# 显示图表
plt.show()

print("图表已成功生成并保存为vehicle_resource_analysis.png和vehicle_resource_analysis.svg")
print("\n数据分析摘要：")
print(f"1. 负载因子范围：{min(load_factor):.2f}% - {max(load_factor):.2f}%")
print(f"2. 平均每辆车订单数：{np.mean(request_per_vehicle):.2f}")
print(f"3. 最高车辆使用率：{max(vehicle_utilization):.2f}%")
print(f"4. 当订单数为{data.loc[data['每辆车的订单数量'].idxmax(), '订单数目(R)']}时，")
print(f"   每辆车的订单数量达到最大值：{data['每辆车的订单数量'].max():.2f}")