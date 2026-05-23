import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set professional style
plt.style.use('default')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rcParams['axes.linewidth'] = 1.5
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3

# Data for Mode 1 (模式1)
mode1_K = np.array([15, 30, 70, 118])
mode1_R = np.array([10, 20, 50, 100])
mode1_vehicles_used = np.array([4, 5, 11, 24])
mode1_load_factor = np.array([0.978, 0.948, 0.936, 0.928])
mode1_requests_per_vehicle = np.array([2.5, 4, 4.54, 4.16])

# Data for Mode 2 (模式2)
mode2_K = np.array([15, 30, 70, 118])
mode2_R = np.array([10, 20, 50, 100])
mode2_vehicles_used = np.array([10, 19, 46, 89])
mode2_load_factor = np.array([0.969, 0.934, 0.927, 0.899])
mode2_requests_per_vehicle = np.array([1, 0.95, 0.92, 0.89])

# Color scheme
color_mode1_used = '#2E86AB'      # Blue for Mode 1 used
color_mode1_provided = '#A8DADC'  # Light blue for Mode 1 provided
color_mode2_used = '#E63946'      # Red for Mode 2 used
color_mode2_provided = '#F4A6A3'  # Light red for Mode 2 provided

# ============= FIGURE 1: Vehicle Allocation =============
fig1, ax1 = plt.subplots(figsize=(14, 8))
fig1.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.12)

x_pos = np.arange(len(mode1_R))
width = 0.2

# Mode 1 bars
bars1_used = ax1.bar(x_pos - 1.5*width, mode1_vehicles_used, width, 
                     label='Mode 1: Vehicles Used', color=color_mode1_used, 
                     alpha=0.85, edgecolor='black', linewidth=1.5)
bars1_provided = ax1.bar(x_pos - 0.5*width, mode1_K, width,
                         label='Mode 1: Vehicles Provided', color=color_mode1_provided,
                         alpha=0.85, edgecolor='black', linewidth=1.5)

# Mode 2 bars
bars2_used = ax1.bar(x_pos + 0.5*width, mode2_vehicles_used, width,
                     label='Mode 2: Vehicles Used', color=color_mode2_used,
                     alpha=0.85, edgecolor='black', linewidth=1.5)
bars2_provided = ax1.bar(x_pos + 1.5*width, mode2_K, width,
                         label='Mode 2: Vehicles Provided', color=color_mode2_provided,
                         alpha=0.85, edgecolor='black', linewidth=1.5)

# Add value labels for all bars
for bars in [bars1_used, bars1_provided, bars2_used, bars2_provided]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                 f'{int(height)}', ha='center', va='bottom', 
                 fontsize=10, fontweight='bold')

# Add utilization rate annotations
for i, x in enumerate(x_pos):
    # Mode 1 utilization
    util1 = (mode1_vehicles_used[i] / mode1_K[i]) * 100
    ax1.text(x - width, -7, f'{util1:.1f}%', ha='center', va='top', 
            fontsize=9, color=color_mode1_used, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                     alpha=0.9, edgecolor=color_mode1_used, linewidth=1))
    
    # Mode 2 utilization
    util2 = (mode2_vehicles_used[i] / mode2_K[i]) * 100
    ax1.text(x + width, -7, f'{util2:.1f}%', ha='center', va='top', 
            fontsize=9, color=color_mode2_used, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                     alpha=0.9, edgecolor=color_mode2_used, linewidth=1))

ax1.set_xlabel('Number of Orders (R)', fontsize=17, fontweight='bold')
ax1.set_ylabel('Number of Vehicles', fontsize=17, fontweight='bold')
ax1.set_title('Vehicle Allocation: Mode 1 vs Mode 2', 
             fontsize=20, fontweight='bold', pad=20)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(mode1_R, fontsize=14)
ax1.tick_params(axis='both', labelsize=13)
ax1.set_ylim(0, max(max(mode2_K), max(mode1_K)) * 1.15)
ax1.legend(loc='upper left', fontsize=12, framealpha=0.95, 
          edgecolor='black', ncol=2, columnspacing=1)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)

# Add text annotation
ax1.text(0.98, 0.02, 'Utilization Rate (%)', 
        transform=ax1.transAxes, fontsize=11, 
        horizontalalignment='right', verticalalignment='bottom',
        style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save figure 1
plt.savefig('/data/mayue/cjy/Other_method/Vehicle_Load/combined_figure1_vehicle_allocation.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/data/mayue/cjy/Other_method/Vehicle_Load/combined_figure1_vehicle_allocation.pdf', 
           dpi=300, bbox_inches='tight', facecolor='white')
print("Figure 1 saved: Vehicle Allocation (Mode 1 vs Mode 2)")


# ============= FIGURE 2: Performance Metrics =============
fig2, ax2 = plt.subplots(figsize=(14, 8))
fig2.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.12)

x_pos = np.arange(len(mode1_R))

# Plot Load Factor
line1 = ax2.plot(x_pos, mode1_load_factor, 'D-', label='Mode 1: Load Factor', 
                color=color_mode1_used, linewidth=3.5, markersize=13, 
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
line2 = ax2.plot(x_pos, mode2_load_factor, 's-', label='Mode 2: Load Factor',
                color=color_mode2_used, linewidth=3.5, markersize=13,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)

# Add value labels for Load Factor
for i, (x, y) in enumerate(zip(x_pos, mode1_load_factor)):
    ax2.text(x - 0.08, y + 0.005, f'{y:.3f}', ha='right', va='bottom', 
            fontsize=10, color=color_mode1_used, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     alpha=0.85, edgecolor=color_mode1_used, linewidth=1.2))

for i, (x, y) in enumerate(zip(x_pos, mode2_load_factor)):
    ax2.text(x + 0.08, y - 0.005, f'{y:.3f}', ha='left', va='top', 
            fontsize=10, color=color_mode2_used, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     alpha=0.85, edgecolor=color_mode2_used, linewidth=1.2))

ax2.set_xlabel('Number of Orders (R)', fontsize=17, fontweight='bold')
ax2.set_ylabel('Load Factor', fontsize=17, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(mode1_R, fontsize=14)
ax2.tick_params(axis='both', labelsize=13)
ax2.set_ylim(0.85, 1.0)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)

# Add second y-axis for Requests per Vehicle
ax3 = ax2.twinx()
line3 = ax3.plot(x_pos, mode1_requests_per_vehicle, 'o-', 
                label='Mode 1: Requests per Vehicle',
                color='#F18F01', linewidth=3.5, markersize=13,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
line4 = ax3.plot(x_pos, mode2_requests_per_vehicle, '^-',
                label='Mode 2: Requests per Vehicle',
                color='#C77DFF', linewidth=3.5, markersize=13,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)

# Add value labels for Requests per Vehicle
for i, (x, y) in enumerate(zip(x_pos, mode1_requests_per_vehicle)):
    ax3.text(x - 0.08, y + 0.12, f'{y:.2f}', ha='right', va='bottom', 
            fontsize=10, color='#F18F01', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     alpha=0.85, edgecolor='#F18F01', linewidth=1.2))

for i, (x, y) in enumerate(zip(x_pos, mode2_requests_per_vehicle)):
    ax3.text(x + 0.08, y - 0.05, f'{y:.2f}', ha='left', va='top', 
            fontsize=10, color='#C77DFF', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     alpha=0.85, edgecolor='#C77DFF', linewidth=1.2))

ax3.set_ylabel('Requests per Vehicle', fontsize=17, fontweight='bold')
ax3.tick_params(axis='y', labelsize=13)
ax3.set_ylim(0.5, 5.0)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_linewidth(2)

# Combine legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper right', fontsize=12, framealpha=0.95, 
          edgecolor='black', ncol=2, columnspacing=1)

# Title
fig2.suptitle('Performance Metrics: Load Factor and Service Efficiency', 
             fontsize=20, fontweight='bold', y=0.97)

# Save figure 2
plt.savefig('/data/mayue/cjy/Other_method/Vehicle_Load/combined_figure2_performance_metrics.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/data/mayue/cjy/Other_method/Vehicle_Load/combined_figure2_performance_metrics.pdf', 
           dpi=300, bbox_inches='tight', facecolor='white')
print("Figure 2 saved: Performance Metrics (Mode 1 vs Mode 2)")

print("\n" + "="*70)
print("All combined figures saved successfully!")
print("="*70)
print("\nFigure 1: Vehicle Allocation (Mode 1 & Mode 2 in one plot)")
print("  - combined_figure1_vehicle_allocation.png (300 DPI)")
print("  - combined_figure1_vehicle_allocation.pdf (vector format)")
print("\nFigure 2: Performance Metrics (Mode 1 & Mode 2 in one plot)")
print("  - combined_figure2_performance_metrics.png (300 DPI)")
print("  - combined_figure2_performance_metrics.pdf (vector format)")

plt.show()
