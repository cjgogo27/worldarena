#!/usr/bin/env python3
"""
Compare generated and original trajectories for each household side by side
Generated trajectories on top, original trajectories on bottom
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from collections import defaultdict

# File paths
GENERATED_TRAJECTORY_FILE = "/data/alice/cjtest/FinalTraj/Trajectory_Generation_multi_agent/output_trajectories/all_trajectories_20251214_180155_California.json"
ORIGINAL_TRAJECTORY_FILE = "/data/alice/cjtest/FinalTraj/California/processed_data/all_user_schedules.json"
HOUSEHOLD_FILE = "/data/alice/cjtest/FinalTraj/California/processed_data/california_household_static.json"
OUTPUT_DIR = "/data/alice/cjtest/FinalTraj/evaluation/case_study/household_comparison"

# Activity type to color mapping
ACTIVITY_COLORS = {
    'home': '#90EE90',       # Light green
    'work': '#FFB6C1',       # Light pink
    'education': '#87CEEB',  # Sky blue
    'shopping': '#FFD700',   # Gold
    'service': '#DDA0DD',    # Plum
    'medical': '#FF6347',    # Tomato
    'dine_out': '#FFA500',   # Orange
    'socialize': '#9370DB',  # Medium purple
    'exercise': '#32CD32',   # Lime green
    'dropoff_pickup': '#FF69B4'  # Hot pink
}


def time_to_minutes(time_str):
    """Convert time string to minutes"""
    if time_str == "24:00":
        return 1440
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])


def load_trajectories(file_path):
    """Load trajectory data and group by household"""
    with open(file_path, 'r', encoding='utf-8') as f:
        trajectories = json.load(f)
    
    # Group by household
    households = defaultdict(list)
    for traj in trajectories:
        user_id = traj['user_id']
        # Extract household ID (household_id_member)
        household_id = user_id.rsplit('_', 1)[0]
        households[household_id].append(traj)
    
    return households


def load_household_info(file_path):
    """Load household static information"""
    with open(file_path, 'r', encoding='utf-8') as f:
        households = json.load(f)
    
    household_dict = {}
    for h in households:
        household_dict[h['household_id']] = h
    
    return household_dict


def plot_trajectory_on_axis(ax, members, title, show_xlabel=False):
    """
    Plot trajectories for all members on a given axis
    
    Args:
        ax: Matplotlib axis object
        members: List of member trajectories
        title: Title for this subplot
        show_xlabel: Whether to show x-axis label
    """
    # Sort by user_id for consistency
    members = sorted(members, key=lambda x: x['user_id'])
    n_members = len(members)
    
    if n_members == 0:
        return
    
    # Plot trajectory for each member
    for member_idx, member in enumerate(members):
        user_id = member['user_id']
        schedule = member['schedule']
        
        # Calculate y-coordinate position (top to bottom)
        y_pos = n_members - member_idx - 1
        
        # Draw each activity segment
        for segment in schedule:
            activity = segment['activity']
            start_time = time_to_minutes(segment['start_time'])
            end_time = time_to_minutes(segment['end_time'])
            
            # Get color
            color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
            
            # Draw rectangle
            rect = mpatches.Rectangle(
                (start_time, y_pos - 0.4),
                end_time - start_time,
                0.8,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(rect)
            
            # Add activity label for longer time periods
            duration = end_time - start_time
            if duration > 60:  # Only label activities longer than 1 hour
                mid_point = (start_time + end_time) / 2
                ax.text(
                    mid_point, y_pos,
                    activity.replace('_', '\n'),
                    ha='center', va='center',
                    fontsize=7,
                    weight='bold'
                )
    
    # Set y-axis
    ax.set_ylim(-0.5, n_members - 0.5)
    ax.set_yticks(range(n_members))
    ax.set_yticklabels([m['user_id'].split('_')[-1] for m in members[::-1]])
    ax.set_ylabel('Member', fontsize=11, fontweight='bold')
    
    # Set x-axis (time axis)
    ax.set_xlim(0, 1440)
    hours = [0, 6, 12, 18, 24]
    ax.set_xticks([h * 60 for h in hours])
    if show_xlabel:
        ax.set_xticklabels([f'{h:02d}:00' for h in hours])
        ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    else:
        ax.set_xticklabels([])
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Set title
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)


def plot_household_comparison(household_id, generated_members, original_members, household_info, output_dir):
    """
    Create comparison visualization for one household
    
    Args:
        household_id: Household ID
        generated_members: Generated trajectories for household members
        original_members: Original trajectories for household members
        household_info: Dictionary of household static information
        output_dir: Output directory
    """
    n_members = max(len(generated_members), len(original_members))
    if n_members == 0:
        return
    
    # Create figure with 2 rows (generated on top, original on bottom)
    fig, (ax_gen, ax_orig) = plt.subplots(2, 1, figsize=(18, 3 * n_members), 
                                           gridspec_kw={'hspace': 0.3})
    
    # Get household info for main title
    h_info = household_info.get(household_id, {})
    household_size = h_info.get('household_size', n_members)
    num_vehicles = h_info.get('num_vehicles', 'Unknown')
    
    # Main title
    main_title = f'Household {household_id} Comparison\n'
    main_title += f'Size: {household_size} | Vehicles: {num_vehicles} | Members: {n_members}'
    fig.suptitle(main_title, fontsize=15, fontweight='bold', y=0.995)
    
    # Plot generated trajectories (top)
    plot_trajectory_on_axis(
        ax_gen, 
        generated_members, 
        'Generated Trajectories (LLM Multi-Agent)',
        show_xlabel=False
    )
    
    # Plot original trajectories (bottom)
    plot_trajectory_on_axis(
        ax_orig, 
        original_members, 
        'Original Trajectories (Ground Truth)',
        show_xlabel=True
    )
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=color, edgecolor='black', label=activity)
        for activity, color in sorted(ACTIVITY_COLORS.items())
    ]
    
    # Add legend to the right of the figure
    fig.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        fontsize=10,
        title='Activities',
        title_fontsize=11
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.92, 0.98])
    
    # Save figure
    output_path = os.path.join(output_dir, f'household_{household_id}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: household_{household_id}_comparison.png ({n_members} members)')


def main():
    """Main function"""
    print("="*70)
    print("Household Trajectory Comparison Visualization")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load data
    print(f"\nLoading data...")
    print(f"  Generated trajectories: {GENERATED_TRAJECTORY_FILE}")
    print(f"  Original trajectories: {ORIGINAL_TRAJECTORY_FILE}")
    print(f"  Household info: {HOUSEHOLD_FILE}")
    
    generated_households = load_trajectories(GENERATED_TRAJECTORY_FILE)
    original_households = load_trajectories(ORIGINAL_TRAJECTORY_FILE)
    household_info = load_household_info(HOUSEHOLD_FILE)
    
    print(f"\n✓ Loaded {len(generated_households)} generated households")
    print(f"✓ Loaded {len(original_households)} original households")
    
    # Find common households
    common_household_ids = set(generated_households.keys()) & set(original_households.keys())
    print(f"✓ Found {len(common_household_ids)} households to compare")
    
    if len(common_household_ids) == 0:
        print("\n❌ No common households found!")
        return
    
    # Generate comparison visualizations
    print(f"\nGenerating comparison visualizations...")
    print("-"*70)
    
    for idx, household_id in enumerate(sorted(common_household_ids), 1):
        generated_members = generated_households[household_id]
        original_members = original_households[household_id]
        
        plot_household_comparison(
            household_id, 
            generated_members, 
            original_members,
            household_info,
            OUTPUT_DIR
        )
        
        if idx % 5 == 0:
            print(f"  Progress: {idx}/{len(common_household_ids)}")
    
    print("-"*70)
    print(f"\n✅ Complete! Generated {len(common_household_ids)} comparison images")
    print(f"📁 Saved to: {OUTPUT_DIR}")
    print("="*70)
    
    # Statistics
    print(f"\n📊 Statistics:")
    gen_member_counts = [len(members) for household_id, members in generated_households.items() 
                         if household_id in common_household_ids]
    orig_member_counts = [len(members) for household_id, members in original_households.items() 
                          if household_id in common_household_ids]
    
    print(f"  Generated - Avg members per household: {np.mean(gen_member_counts):.2f}")
    print(f"  Original  - Avg members per household: {np.mean(orig_member_counts):.2f}")
    
    # Check for member count mismatches
    mismatches = 0
    for household_id in common_household_ids:
        gen_count = len(generated_households[household_id])
        orig_count = len(original_households[household_id])
        if gen_count != orig_count:
            mismatches += 1
            print(f"\n  ⚠️  Mismatch in household {household_id}: "
                  f"Generated={gen_count}, Original={orig_count}")
    
    if mismatches == 0:
        print(f"\n  ✓ All households have matching member counts")
    else:
        print(f"\n  ⚠️  {mismatches} households have mismatched member counts")


if __name__ == '__main__':
    main()
