import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Set scientific plotting style with high-quality fonts
plt.rcParams['font.family'] = ['Arial', 'sans-serif']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (15, 10)

# Prepare the data
K = [5, 10, 20, 30]  # Number of vehicles provided
R = [10, 20, 50, 100]  # Number of requests
number_used_vehicles = [4, 5, 11, 24]  # Number of vehicles used
load_factor = [97.80, 94.75, 93.57, 92.81]  # Load factor (%)
request_per_vehicle = [2.5, 4, 4.54, 4.16]  # Requests per vehicle

# Calculate vehicle utilization
vehicle_utilization = [used / provided * 100 for used, provided in zip(number_used_vehicles, K)]

# Create a single comprehensive plot
fig, ax1 = plt.subplots(figsize=(16, 10))

# Define a professional color palette with improved contrast
colors = ['#1a5fb4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot 1: Number of Vehicles Used (Bar chart, left Y-axis)
x = np.arange(len(R))
width = 0.4

bars = ax1.bar(x, number_used_vehicles, width, color=colors[0], 
               alpha=0.8, edgecolor='white', linewidth=1.5, 
               label='Vehicles Used')

# Add data labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom', 
            fontweight='bold', fontsize=12, color='black')

# Set left Y-axis properties for Vehicles Used
ax1.set_ylabel('Number of Vehicles Used', fontsize=14, fontweight='semibold', color='black')
ax1.tick_params(axis='y', labelsize=12, width=1.5, grid_color='lightgray', grid_linewidth=0.5)
ax1.tick_params(axis='x', labelsize=12, width=1.5)
ax1.set_ylim(0, max(number_used_vehicles) * 1.2)

# Create second Y-axis for Load Factor and Requests per Vehicle
ax2 = ax1.twinx()

# Plot 2: Load Factor (Line with markers, second Y-axis)
l1, = ax2.plot(x, load_factor, 'o-', color=colors[1], 
               markersize=12, markeredgewidth=1.5, markeredgecolor='white',
               label='Load Factor (%)', zorder=3)

# Plot 3: Requests per Vehicle (Line with different markers, second Y-axis)
l2, = ax2.plot(x, request_per_vehicle, 's--', color=colors[2],
               markersize=12, markeredgewidth=1.5, markeredgecolor='white',
               label='Requests per Vehicle', zorder=3)

# Plot 4: Vehicle Utilization (Dashed line, second Y-axis)
l3, = ax2.plot(x, vehicle_utilization, 'D-.', color=colors[3],
               markersize=12, markeredgewidth=1.5, markeredgecolor='white',
               label='Vehicle Utilization (%)', zorder=3)

# Add data labels for all line plots with improved positioning
for i, (lf, rp, vu) in enumerate(zip(load_factor, request_per_vehicle, vehicle_utilization)):
    ax2.text(i, lf + 0.8, f'{lf}%', ha='center', va='bottom', 
             fontweight='bold', fontsize=11, color=colors[1], 
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
    ax2.text(i, rp + 0.2, f'{rp:.2f}', ha='center', va='bottom', 
             fontweight='bold', fontsize=11, color=colors[2],
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
    ax2.text(i, vu - 2.5, f'{vu:.1f}%', ha='center', va='top', 
             fontweight='bold', fontsize=11, color=colors[3],
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

# Set second Y-axis properties
ax2.set_ylabel('Percentage (%) / Requests per Vehicle', 
               fontsize=14, fontweight='semibold', color='black')
ax2.tick_params(axis='y', labelsize=12, width=1.5)
ax2.set_ylim(0, max(max(load_factor), max(vehicle_utilization)) * 1.15)

# Set X-axis properties
ax1.set_xlabel('Number of Requests (R)', fontsize=14, fontweight='semibold')
ax1.set_xticks(x)
ax1.set_xticklabels(R)

# Add grid lines with improved transparency and style
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, zorder=0)

# Create a sophisticated legend with custom handles
legend_elements = [
    Patch(facecolor=colors[0], alpha=0.8, edgecolor='white', linewidth=1.5, 
          label='Vehicles Used'),
    Line2D([0], [0], color=colors[1], marker='o', linestyle='-', 
           markersize=12, markeredgewidth=1.5, markeredgecolor='white',
           label='Load Factor (%)'),
    Line2D([0], [0], color=colors[2], marker='s', linestyle='--', 
           markersize=12, markeredgewidth=1.5, markeredgecolor='white',
           label='Requests per Vehicle'),
    Line2D([0], [0], color=colors[3], marker='D', linestyle='-.', 
           markersize=12, markeredgewidth=1.5, markeredgecolor='white',
           label='Vehicle Utilization (%)')
]

# Position the legend at the top center with improved styling
ax1.legend(handles=legend_elements, loc='upper center', 
           bbox_to_anchor=(0.5, 1.12), ncol=2, 
           frameon=True, framealpha=0.95, edgecolor='gray', 
           fontsize=13, shadow=False, borderpad=1)

# Add a professional title with improved styling
plt.title('Vehicle Resource Allocation and Request Processing Efficiency', 
          fontsize=24, fontweight='bold', pad=40, loc='center', color='black')

# Add a detailed annotation box with key insights
insights_text = ""
insights_text += "Key Performance Metrics:\n"
insights_text += f"• Load Factor Range: {min(load_factor):.2f}% - {max(load_factor):.2f}%\n"
insights_text += f"• Average Requests per Vehicle: {np.mean(request_per_vehicle):.2f}\n"
insights_text += f"• Maximum Vehicle Utilization: {max(vehicle_utilization):.1f}%\n"
insights_text += f"• Peak Requests per Vehicle: {max(request_per_vehicle):.2f} (at R={R[np.argmax(request_per_vehicle)]})"

# Add the annotation box to the figure with improved styling
fig.text(0.02, 0.02, insights_text, fontsize=14, fontweight='semibold',
         bbox=dict(facecolor='white', alpha=0.98, edgecolor='gray', 
                   boxstyle='round,pad=1.5', linewidth=1.5),
         verticalalignment='bottom', horizontalalignment='left',
         zorder=5)

# Add a data source note
plt.figtext(0.98, 0.02, 'Data Source: Vehicle Resource Allocation Experiments', 
            ha='right', fontsize=12, style='italic', color='gray')

# Adjust layout to prevent overlap with improved margins
plt.tight_layout(rect=[0.03, 0.08, 0.98, 0.9])

# Save the figure in multiple high-quality formats
plt.savefig('single_comprehensive_vehicle_analysis.png', dpi=600, bbox_inches='tight')
plt.savefig('single_comprehensive_vehicle_analysis.svg', format='svg', bbox_inches='tight')
plt.savefig('single_comprehensive_vehicle_analysis.pdf', format='pdf', bbox_inches='tight')

# Print analysis summary
print("=== Single Comprehensive Vehicle Resource Analysis ===")
print(f"1. Load Factor Performance: {min(load_factor):.2f}% - {max(load_factor):.2f}%")
print(f"2. Average Requests per Vehicle: {np.mean(request_per_vehicle):.2f}")
print(f"3. Vehicle Utilization Range: {min(vehicle_utilization):.1f}% - {max(vehicle_utilization):.1f}%")
print(f"4. Optimal Request Handling: {R[np.argmax(request_per_vehicle)]} requests (achieving {max(request_per_vehicle):.2f} requests per vehicle)")
print("\nVisualization saved as PNG, SVG, and PDF formats for publication use.")

# Display the plot
plt.show()