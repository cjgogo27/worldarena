import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Set style for scientific publication
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# Read data
df = pd.read_excel('E:\\TRE\\Vehicle_Load\\结果汇总.xlsx')

# Sort by Total number for better visualization
df_sorted = df.sort_values('Total number')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Extract data
x = df_sorted['Total number'].values
y = df_sorted['Rental cost'].values

# Plot with markers and lines
ax.plot(x, y, marker='o', markersize=8, linewidth=2.5, 
        color='#2E86AB', markerfacecolor='#A23B72', 
        markeredgecolor='#2E86AB', markeredgewidth=1.5,
        label='Rental Cost', alpha=0.8)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Labels and title
ax.set_xlabel('Total Number of Vehicles', fontsize=14, fontweight='bold')
ax.set_ylabel('Rental Cost ($)', fontsize=14, fontweight='bold')
ax.set_title('Impact of Vehicle Fleet Size on Rental Cost', 
             fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks
ax.set_xticks(x)

# Format y-axis
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Add legend
ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('E:\\TRE\\Vehicle_Load\\Figures\\vehicle_rental_cost_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.savefig('E:\\TRE\\Vehicle_Load\\Figures\\vehicle_rental_cost_analysis.pdf', 
            bbox_inches='tight')

print("✓ Figure saved successfully!")
print(f"  - PNG: E:\\TRE\\Vehicle_Load\\Figures\\vehicle_rental_cost_analysis.png")
print(f"  - PDF: E:\\TRE\\Vehicle_Load\\Figures\\vehicle_rental_cost_analysis.pdf")

# Show statistics
print(f"\nData Summary:")
print(f"  Total vehicles range: {x.min()} - {x.max()}")
print(f"  Rental cost range: ${y.min():,.0f} - ${y.max():,.0f}")
print(f"  Average cost increase per vehicle: ${(y.max()-y.min())/(x.max()-x.min()):,.0f}")

plt.show()
