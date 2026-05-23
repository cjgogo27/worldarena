import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Set scientific plotting style with high-quality fonts
plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8
plt.rcParams['savefig.dpi'] = 300

# Prepare the data
K = [5, 10, 20, 30]  # Number of vehicles provided
R = [10, 20, 50, 100]  # Number of requests
number_used_vehicles = [4, 5, 11, 24]  # Number of vehicles used
load_factor = [97.80, 94.75, 93.57, 92.81]  # Load factor (%)
request_per_vehicle = [2.5, 4, 4.54, 4.16]  # Requests per vehicle

# Calculate vehicle utilization
vehicle_utilization = [used / provided * 100 for used, provided in zip(number_used_vehicles, K)]

# Create a high-quality advanced visualization
fig = plt.figure(figsize=(18, 14), constrained_layout=True)
fig.suptitle('Vehicle Resource Allocation and Request Processing Efficiency Analysis', 
             fontsize=22, fontweight='bold')
fig.subplots_adjust(top=0.94)  # Adjust top margin to create space for title

# Create a GridSpec for the layout
gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1.5])

# Define a professional color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# First subplot: Load Factor and Vehicle Utilization
ax1 = fig.add_subplot(gs[0])

# Create twin axes for dual Y-axis
twin1 = ax1.twinx()

# Plot Load Factor (left Y-axis)
l1, = ax1.plot(R, load_factor, 'o-', color=colors[0], 
               markersize=10, markeredgewidth=1.5, markeredgecolor='white',
               label='Load Factor (%)')
ax1.set_xlabel('Number of Requests (R)', fontsize=14, fontweight='semibold')
ax1.set_ylabel('Load Factor (%)', fontsize=14, fontweight='semibold', color=colors[0])
ax1.tick_params(axis='y', labelcolor=colors[0], labelsize=12, width=1.5)
ax1.tick_params(axis='x', labelsize=12, width=1.5)
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot Vehicle Utilization (right Y-axis)
l2, = twin1.plot(R, vehicle_utilization, 's--', color=colors[1],
                markersize=10, markeredgewidth=1.5, markeredgecolor='white',
                label='Vehicle Utilization (%)')
twin1.set_ylabel('Vehicle Utilization (%)', fontsize=14, fontweight='semibold', color=colors[1])
twin1.tick_params(axis='y', labelcolor=colors[1], labelsize=12, width=1.5)

# Add a sophisticated legend with custom handles
legend_elements = [
    Line2D([0], [0], color=colors[0], marker='o', linestyle='-', markersize=10, 
           markeredgewidth=1.5, markeredgecolor='white', label='Load Factor (%)'),
    Line2D([0], [0], color=colors[1], marker='s', linestyle='--', markersize=10, 
           markeredgewidth=1.5, markeredgecolor='white', label='Vehicle Utilization (%)')
]
ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2), 
           ncol=2, frameon=True, framealpha=0.9, edgecolor='gray', fontsize=12)

ax1.set_title('Impact of Request Volume on System Performance', fontsize=18, pad=15, fontweight='semibold')

# Second subplot: Requests per Vehicle and Number of Vehicles Used
ax2 = fig.add_subplot(gs[1])

# Create a grouped bar chart
x = np.arange(len(R))
width = 0.35

# Plot Number of Vehicles Used (left bars)
bars1 = ax2.bar(x - width/2, number_used_vehicles, width, color=colors[2], 
                alpha=0.8, edgecolor='white', linewidth=1.5, label='Vehicles Used')

# Create a secondary Y-axis for Requests per Vehicle
ax2_twin = ax2.twinx()

# Plot Requests per Vehicle (right axis)
line2, = ax2_twin.plot(x, request_per_vehicle, 'D-', color=colors[3], 
                      markersize=10, markeredgewidth=1.5, markeredgecolor='white',
                      label='Requests per Vehicle')

# Add data labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add data labels near line points
for i, value in enumerate(request_per_vehicle):
    ax2_twin.text(i, value + 0.15, f'{value:.2f}', ha='center', va='bottom', 
                 fontweight='bold', fontsize=12, color=colors[3])

# Customize axes
ax2.set_xlabel('Number of Requests (R)', fontsize=14, fontweight='semibold')
ax2.set_ylabel('Number of Vehicles Used', fontsize=14, fontweight='semibold', color=colors[2])
ax2_twin.set_ylabel('Requests per Vehicle', fontsize=14, fontweight='semibold', color=colors[3])

# Set X-axis ticks and labels
ax2.set_xticks(x)
ax2.set_xticklabels(R)

# Customize ticks
ax2.tick_params(axis='y', labelcolor=colors[2], labelsize=12, width=1.5)
ax2_twin.tick_params(axis='y', labelcolor=colors[3], labelsize=12, width=1.5)
ax2.tick_params(axis='x', labelsize=12, width=1.5)

# Add grid
ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

# Add legend with custom elements
legend_elements = [
    Patch(facecolor=colors[2], alpha=0.8, edgecolor='white', linewidth=1.5, label='Vehicles Used'),
    Line2D([0], [0], color=colors[3], marker='D', linestyle='-', markersize=10, 
           markeredgewidth=1.5, markeredgecolor='white', label='Requests per Vehicle')
]
ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2), 
           ncol=2, frameon=True, framealpha=0.9, edgecolor='gray', fontsize=12)

ax2.set_title('Resource Utilization and Service Capacity Analysis', fontsize=18, pad=15, fontweight='semibold')

# Third subplot: Comprehensive 3D visualization
ax3 = fig.add_subplot(gs[2], projection='3d')

# Create a 3D scatter plot with enhanced visual effects
scatter = ax3.scatter(K, R, load_factor,
                    c=load_factor, cmap='viridis', 
                    s=200, alpha=0.9, 
                    edgecolors='black', linewidths=1.5,
                    depthshade=True)

# Add a color bar with a professional label
cbar = fig.colorbar(scatter, ax=ax3, pad=0.15, shrink=0.7)
cbar.set_label('Load Factor (%)', fontsize=14, fontweight='semibold')
cbar.ax.tick_params(labelsize=12)

# Customize the 3D axes with scientific formatting
ax3.set_xlabel('Vehicles Provided (K)', fontsize=14, fontweight='semibold', labelpad=15)
ax3.set_ylabel('Requests (R)', fontsize=14, fontweight='semibold', labelpad=15)
ax3.set_zlabel('Load Factor (%)', fontsize=14, fontweight='semibold', labelpad=15)

# Enhance the 3D plot appearance
ax3.xaxis._axinfo['grid'].update(color='gray', linestyle='--', alpha=0.3)
ax3.yaxis._axinfo['grid'].update(color='gray', linestyle='--', alpha=0.3)
ax3.zaxis._axinfo['grid'].update(color='gray', linestyle='--', alpha=0.3)

# Add annotations for key data points
for i, (k, r, lf) in enumerate(zip(K, R, load_factor)):
    ax3.text(k, r, lf + 0.5, f'{lf}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax3.set_title('Three-Dimensional Relationship: Vehicles, Requests, and Load Factor', 
             fontsize=18, pad=20, fontweight='semibold')

# Add a professional footer/legend box with key insights
insights_text = ""
insights_text += "Key Insights:\n"
insights_text += f"• Load Factor Range: {min(load_factor):.2f}% - {max(load_factor):.2f}%\n"
insights_text += f"• Average Requests per Vehicle: {np.mean(request_per_vehicle):.2f}\n"
insights_text += f"• Maximum Vehicle Utilization: {max(vehicle_utilization):.2f}%\n"
insights_text += f"• Peak Requests per Vehicle: {max(request_per_vehicle):.2f} (at R={R[np.argmax(request_per_vehicle)]})"

fig.text(0.02, 0.02, insights_text, fontsize=13, fontweight='semibold',
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=1'))

# Save the figure in multiple formats for publication quality
plt.savefig('comprehensive_vehicle_analysis.png', dpi=600, bbox_inches='tight')
plt.savefig('comprehensive_vehicle_analysis.svg', format='svg', bbox_inches='tight')
plt.savefig('comprehensive_vehicle_analysis.pdf', format='pdf', bbox_inches='tight')

# Print analysis summary
print("=== Comprehensive Vehicle Resource Analysis Summary ===")
print(f"1. Load Factor Performance: {min(load_factor):.2f}% - {max(load_factor):.2f}%")
print(f"2. Average Requests Handled per Vehicle: {np.mean(request_per_vehicle):.2f}")
print(f"3. Highest Vehicle Utilization Rate: {max(vehicle_utilization):.2f}%")
print(f"4. Optimal Request Volume: {R[np.argmax(request_per_vehicle)]} requests (maximizes requests per vehicle at {max(request_per_vehicle):.2f})")
print("\nVisualizations saved as PNG, SVG, and PDF formats for publication use.")

# Display the plot
plt.show()