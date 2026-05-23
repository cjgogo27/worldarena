import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Set style for a more professional look
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read data
df = pd.read_excel('E:\\TRE\\Passenger_Priority\\Figures\\overall.xlsx')

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Define categories with units - ALL data columns used
categories = ['Distance\n(km)', 'Served\nRequests', 'Request Cost\n($)', 'Delay Penalty\n(min)', 'Number of\nVehicles']

# Extract data for each priority ratio - using text labels instead of ratios
data_high_priority = [
    float(df.loc[0, 'distance']),
    float(df.loc[0, 'served_requests']),
    float(df.loc[0, 'request_cost']),
    float(df.loc[0, 'delay_penalty']),
    float(df.loc[0, 'number_used_vehicles'])
]

data_equal_priority = [
    float(df.loc[1, 'distance']),
    float(df.loc[1, 'served_requests']),
    float(df.loc[1, 'request_cost']),
    float(df.loc[1, 'delay_penalty']),
    float(df.loc[1, 'number_used_vehicles'])
]

print(f"High Priority - Distance: {data_high_priority[0]}, Served: {data_high_priority[1]}, Cost: {data_high_priority[2]}, Delay: {data_high_priority[3]}, Vehicles: {data_high_priority[4]}")
print(f"Equal Priority - Distance: {data_equal_priority[0]}, Served: {data_equal_priority[1]}, Cost: {data_equal_priority[2]}, Delay: {data_equal_priority[3]}, Vehicles: {data_equal_priority[4]}")

# Normalize data to 0-1 scale for better visualization
def normalize_data(data_list, data1, data2):
    max_vals = [max(data1[i], data2[i]) for i in range(len(data1))]
    min_vals = [min(data1[i], data2[i]) for i in range(len(data1))]
    normalized = []
    for i, val in enumerate(data_list):
        if max_vals[i] == min_vals[i]:
            normalized.append(0.5)
        else:
            # For cost and delay, lower is better, so invert
            if i in [2, 3]:  # Request Cost and Delay Penalty
                normalized.append(1 - (val - min_vals[i]) / (max_vals[i] - min_vals[i]))
            else:
                normalized.append((val - min_vals[i]) / (max_vals[i] - min_vals[i]))
    return normalized

data_high_priority_norm = normalize_data(data_high_priority, data_high_priority, data_equal_priority)
data_equal_priority_norm = normalize_data(data_equal_priority, data_high_priority, data_equal_priority)

# Number of variables
num_vars = len(categories)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Complete the circle
data_high_priority_norm += data_high_priority_norm[:1]
data_equal_priority_norm += data_equal_priority_norm[:1]
angles += angles[:1]

# Create figure with high DPI for quality
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'), dpi=300)

# Plot data
ax.plot(angles, data_high_priority_norm, 'o-', linewidth=2.5, label='High Priority Passenger', color='#FF6B6B', markersize=8)
ax.fill(angles, data_high_priority_norm, alpha=0.25, color='#FF6B6B')

ax.plot(angles, data_equal_priority_norm, 'o-', linewidth=2.5, label='Equal Priority (Passenger & Parcel)', color='#4ECDC4', markersize=8)
ax.fill(angles, data_equal_priority_norm, alpha=0.25, color='#4ECDC4')

# Fix axis to go in the right order and start at 12 o'clock
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Set category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, weight='bold', linespacing=1.5)

# Set y-axis labels with better formatting
ax.set_ylim(0, 1.15)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10, color='gray')

# Add grid styling
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7, color='gray')
ax.spines['polar'].set_color('lightgray')
ax.spines['polar'].set_linewidth(2)

# Add data labels on the chart
for i, (angle, val_high, val_equal) in enumerate(zip(angles[:-1], data_high_priority_norm, data_equal_priority_norm)):
    # Label for High Priority (red)
    x_offset = 1.08 if val_high > 0.85 else 1.05
    ax.text(angle, val_high * x_offset, f'{data_high_priority[i]:.0f}', 
            ha='center', va='center', size=9, weight='bold',
            color='#FF6B6B', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#FF6B6B', alpha=0.8))
    
    # Label for Equal Priority (cyan)
    x_offset = 1.08 if val_equal > 0.85 else 1.05
    ax.text(angle, val_equal * x_offset, f'{data_equal_priority[i]:.0f}', 
            ha='center', va='center', size=9, weight='bold',
            color='#4ECDC4', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#4ECDC4', alpha=0.8))

# Add title
plt.title('Performance Comparison: Priority Ratio Analysis\n(Passenger to Parcel)', 
          size=16, weight='bold', pad=30, y=1.08)

# Add legend with better positioning
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
                   fontsize=12, frameon=True, shadow=True, fancybox=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Add note about normalization
note_text = "Note: Cost and Delay values are inverted (lower is better)"
plt.figtext(0.5, 0.02, note_text, ha='center', fontsize=9, style='italic', color='gray')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save figure as PNG
output_path_png = 'E:\\TRE\\Passenger_Priority\\Figures\\radar_chart_advanced.png'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Radar chart (PNG) saved to: {output_path_png}")

# Save figure as PDF
output_path_pdf = 'E:\\TRE\\Passenger_Priority\\Figures\\radar_chart_advanced.pdf'
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Radar chart (PDF) saved to: {output_path_pdf}")

# Show plot
plt.show()
