#!/usr/bin/env python3
"""
按家庭可视化所有成员的轨迹
每个家庭的所有成员画在同一张图里
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from collections import defaultdict

# 文件路径
TRAJECTORY_FILE = "/data/alice/cjtest/FinalTraj/Trajectory_Generation_multi_agent/output_trajectories/all_trajectories_20251214_180155_California.json"
OUTPUT_DIR = "/data/alice/cjtest/FinalTraj/evaluation/case_study/household_trajectories"

# 活动类型到颜色的映射
ACTIVITY_COLORS = {
    'home': '#90EE90',       # 浅绿色
    'work': '#FFB6C1',       # 浅粉色
    'education': '#87CEEB',  # 天蓝色
    'shopping': '#FFD700',   # 金色
    'service': '#DDA0DD',    # 梅红色
    'medical': '#FF6347',    # 番茄红
    'dine_out': '#FFA500',   # 橙色
    'socialize': '#9370DB',  # 中紫色
    'exercise': '#32CD32',   # 酸橙绿
    'dropoff_pickup': '#FF69B4'  # 热粉色
}


def time_to_minutes(time_str):
    """将时间字符串转换为分钟数"""
    if time_str == "24:00":
        return 1440
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])


def load_trajectories(file_path):
    """加载轨迹数据并按家庭分组"""
    with open(file_path, 'r', encoding='utf-8') as f:
        trajectories = json.load(f)
    
    # 按家庭分组
    households = defaultdict(list)
    for traj in trajectories:
        user_id = traj['user_id']
        # 提取家庭ID (household_id_member)
        household_id = user_id.rsplit('_', 1)[0]
        households[household_id].append(traj)
    
    # 按成员编号排序
    for household_id in households:
        households[household_id].sort(key=lambda x: x['user_id'])
    
    return households


def plot_household(household_id, members, output_dir):
    """为一个家庭绘制所有成员的轨迹"""
    num_members = len(members)
    
    # 创建图表
    fig, axes = plt.subplots(num_members, 1, figsize=(18, 3 * num_members))
    
    # 如果只有一个成员，axes不是列表
    if num_members == 1:
        axes = [axes]
    
    # 设置主标题
    fig.suptitle(f'Household {household_id} - {num_members} Member(s)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 为每个成员绘制轨迹
    for idx, (ax, member) in enumerate(zip(axes, members)):
        user_id = member['user_id']
        schedule = member['schedule']
        
        # 设置坐标轴
        ax.set_xlim(0, 1440)  # 0-24小时（分钟）
        ax.set_ylim(0, 1)
        ax.set_ylabel(f'{user_id}', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        
        # 只在最后一个子图显示x轴标签
        if idx == num_members - 1:
            ax.set_xlabel('Time', fontsize=14, fontweight='bold')
        else:
            ax.set_xticks([])
        
        # 绘制活动条
        for seg in schedule:
            start = time_to_minutes(seg['start_time'])
            end = time_to_minutes(seg['end_time'])
            activity = seg['activity']
            color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
            
            # 绘制活动条
            ax.barh(0.5, end - start, left=start, height=0.6, 
                   color=color, edgecolor='black', linewidth=1.5)
            
            # 添加活动标签
            mid = (start + end) / 2
            width = end - start
            if width > 60:  # 只在较宽的段上显示文字
                ax.text(mid, 0.5, activity, ha='center', va='center', 
                       fontsize=11, fontweight='bold')
        
        # 设置x轴刻度（每2小时）
        ax.set_xticks([i * 120 for i in range(13)])
        if idx == num_members - 1:
            ax.set_xticklabels([f'{i*2:02d}:00' for i in range(13)], fontsize=12)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # 添加图例
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=activity)
                      for activity, color in ACTIVITY_COLORS.items()]
    
    # 将图例放在图表外侧
    fig.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=11,
              title='Activities', title_fontsize=12)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 0.95, 0.99])
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'household_{household_id}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def create_summary_statistics(households, output_dir):
    """创建汇总统计信息"""
    stats = {
        'total_households': len(households),
        'total_members': sum(len(members) for members in households.values()),
        'household_sizes': {}
    }
    
    # 统计家庭规模分布
    for household_id, members in households.items():
        size = len(members)
        stats['household_sizes'][size] = stats['household_sizes'].get(size, 0) + 1
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, 'summary_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 创建统计摘要文本
    summary_text = f"""
========================================
Household Trajectory Visualization Summary
========================================

Total Households: {stats['total_households']}
Total Members: {stats['total_members']}

Household Size Distribution:
"""
    for size in sorted(stats['household_sizes'].keys()):
        count = stats['household_sizes'][size]
        summary_text += f"  {size} members: {count} households\n"
    
    # 保存摘要文本
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(summary_text)
    
    return stats


def main():
    """主函数"""
    print("="*70)
    print("Household Trajectory Visualization")
    print("="*70)
    
    # 加载数据
    print(f"\nLoading trajectories from:")
    print(f"  {TRAJECTORY_FILE}")
    households = load_trajectories(TRAJECTORY_FILE)
    
    print(f"\n✓ Loaded {len(households)} households")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # 生成每个家庭的可视化
    print(f"\nGenerating visualizations...")
    generated_files = []
    
    for household_id, members in sorted(households.items()):
        print(f"  Processing Household {household_id} ({len(members)} members)...", end=' ')
        try:
            output_file = plot_household(household_id, members, OUTPUT_DIR)
            generated_files.append(output_file)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n✓ Generated {len(generated_files)} household visualizations")
    
    # 创建汇总统计
    print(f"\nGenerating summary statistics...")
    stats = create_summary_statistics(households, OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print("Visualization Complete!")
    print(f"{'='*70}")
    print(f"\nAll visualizations saved to:")
    print(f"  {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    print(f"  - {len(generated_files)} household PNG images")
    print(f"  - summary_statistics.json")
    print(f"  - summary.txt")
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
