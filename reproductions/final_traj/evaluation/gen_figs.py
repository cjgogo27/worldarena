import sys
import os

# Add evaluation dir to path
sys.path.insert(0, '/data/alice/cjtest/FinalTraj/evaluation')

from run_baselines_and_figures import (
    generate_activity_start_time_figure,
    generate_cross_city_comparison_figure,
    generate_subgroup_analysis_figure,
    generate_gender_bias_figure
)

print("Generating Activity Start Time Figure...")
generate_activity_start_time_figure()

print("Generating Cross City Comparison Figure...")
generate_cross_city_comparison_figure()

print("Generating Subgroup Analysis Figure...")
generate_subgroup_analysis_figure()

print("Generating Gender Bias Figure...")
generate_gender_bias_figure()

print("Done!")
