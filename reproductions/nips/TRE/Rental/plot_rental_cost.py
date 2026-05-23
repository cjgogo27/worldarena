import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Rectangle

# Read Excel file
file_path = '/data/mayue/cjy/Other_method/Rental/结果汇总.xlsx'
df = pd.read_excel(file_path)

print("Data preview:")
print(df)

# Set publication-quality plotting parameters
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2

# Extract data
total_vehicles = df['Total number']
rental_cost = df['Rental cost']
evtol = df['Number of eVTOL']
gv = df['Number of GV']
drone = df['Number of Drone']

# Create single comprehensive figure
fig, ax1 = plt.subplots(figsize=(14, 8))

# X-axis positions
x_pos = np.arange(len(total_vehicles))
bar_width = 0.5

# ===== Primary Y-axis: Stacked bar chart for vehicle composition =====
p1 = ax1.bar(x_pos, evtol, bar_width, label='eVTOL', 
             color='#1f77b4', edgecolor='white', linewidth=2, alpha=0.85)
p2 = ax1.bar(x_pos, gv, bar_width, bottom=evtol, label='GV', 
             color='#ff7f0e', edgecolor='white', linewidth=2, alpha=0.85)
p3 = ax1.bar(x_pos, drone, bar_width, bottom=evtol+gv, label='Drone', 
             color="#2ca02c", edgecolor='white', linewidth=2, alpha=0.85)

ax1.set_xlabel('Total Number of Vehicles', fontsize=16, fontweight='bold', labelpad=12)
ax1.set_ylabel('Number of Vehicles by Type', fontsize=16, fontweight='bold', labelpad=12, color='#2c3e50')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(total_vehicles, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#2c3e50', labelsize=11)
ax1.set_ylim(0, max(total_vehicles) * 1.15)

# ===== Secondary Y-axis: Line chart for rental cost =====
ax2 = ax1.twinx()

# Plot rental cost line
line = ax2.plot(x_pos, rental_cost, 
                marker='D', 
                linewidth=3.5, 
                markersize=10,
                color='#e74c3c',
                markerfacecolor='#c0392b',
                markeredgewidth=2.5,
                markeredgecolor='#e74c3c',
                label='Rental Cost',
                zorder=10,
                linestyle='-',
                alpha=0.9)

# Add value labels for rental cost
for i, (x, y) in enumerate(zip(x_pos, rental_cost)):
    ax2.annotate(f'{y:,.0f}', 
                xy=(x, y), 
                xytext=(0, 16),
                textcoords='offset points',
                ha='center',
                fontsize=14,
                fontweight='bold',
                color='#c0392b',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef5e7', 
                         edgecolor='#e74c3c', linewidth=1.5, alpha=0.9))

ax2.set_ylabel('Rental Cost', fontsize=16, fontweight='bold', labelpad=12, color='#e74c3c')
ax2.tick_params(axis='y', labelcolor='#e74c3c', labelsize=11)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Set y-axis limits for better visualization
cost_range = rental_cost.max() - rental_cost.min()
ax2.set_ylim(rental_cost.min() - cost_range * 0.2, rental_cost.max() + cost_range * 0.35)

# # ===== Title and Grid =====
# plt.title('Impact of Vehicle Fleet Composition on Rental Cost', 
#           fontsize=16, fontweight='bold', pad=20, color='#2c3e50')

ax1.grid(True, axis='y', alpha=0.2, linestyle='--', linewidth=1, zorder=0)
ax1.set_axisbelow(True)

# ===== Combined Legend =====
# Get handles and labels from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine legends
ax1.legend(handles1 + handles2, labels1 + labels2, 
          loc='upper left', frameon=True, framealpha=0.95, 
          fontsize=11, edgecolor='#34495e', fancybox=True,
          shadow=True, ncol=1)

props = dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', edgecolor='#34495e', 
             linewidth=2, alpha=0.95)
# ax1.text(0.98, 0.55, transform=ax1.transAxes, fontsize=10,
#         verticalalignment='top', horizontalalignment='right', bbox=props,
#         family='monospace', fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save figure
output_png = '/data/mayue/cjy/Other_method/Rental/rental_cost_combined.png'
output_pdf = '/data/mayue/cjy/Other_method/Rental/rental_cost_combined.pdf'

plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')

print(f"\nCombined figure saved:")
print(f"  PNG: {output_png}")
print(f"  PDF: {output_pdf}")

print("\n" + "="*70)
print("COMPREHENSIVE ANALYSIS - ALL DATA IN ONE FIGURE")
print("="*70)
print(f"✓ Vehicle fleet composition shown as stacked bars")
print(f"✓ Rental cost trend shown as line with dual Y-axis")
print(f"✓ All vehicle types included: eVTOL, Ground Vehicle, Drone")
print("="*70)

plt.show()
