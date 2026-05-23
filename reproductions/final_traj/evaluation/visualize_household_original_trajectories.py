#!/usr/bin/env python3
"""
Visualize original (ground truth) trajectories for all members in each household
All members of a household are plotted in the same figure
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
OUTPUT_DIR = "/data/alice/cjtest/FinalTraj/evaluation/case_study/household_trajectories_original"

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


def load_generated_household_ids(file_path):
    """Load household IDs from generated trajectory file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        trajectories = json.load(f)
    
    household_ids = set()
    for traj in trajectories:
        user_id = traj['user_id']
        # Extract household ID (household_id_member)
        household_id = user_id.rsplit('_', 1)[0]
        household_ids.add(household_id)
    
    return household_ids


def load_trajectories(file_path, target_household_ids=None):
    """Load trajectory data and group by household"""
    with open(file_path, 'r', encoding='utf-8') as f:
        trajectories = json.load(f)
    
    # Group by household
    households = defaultdict(list)
    for traj in trajectories:
        user_id = traj['user_id']
        # Extract household ID (household_id_member)
        household_id = user_id.rsplit('_', 1)[0]
        
        # Only include households in target list if specified
        if target_household_ids is None or household_id in target_household_ids:
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


def plot_household_trajectories(household_id, members, household_info, output_dir):
    """
    Visualize trajectories for all members of a household
    
    Args:
        household_id: Household ID
        members: List of trajectories for all members in the household
        household_info: Dictionary of household static information
        output_dir: Output directory
    """
    # Sort by user_id for consistency
    members = sorted(members, key=lambda x: x['user_id'])
    
    n_members = len(members)
    if n_members == 0:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 2 * n_members))
    
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
                    fontsize=8,
                    weight='bold'
                )
    
    # Set y-axis
    ax.set_ylim(-0.5, n_members - 0.5)
    ax.set_yticks(range(n_members))
    ax.set_yticklabels([m['user_id'].split('_')[-1] for m in members[::-1]])
    ax.set_ylabel('Household Members', fontsize=12, fontweight='bold')
    
    # Set x-axis (time axis)
    ax.set_xlim(0, 1440)
    hours = [0, 6, 12, 18, 24]
    ax.set_xticks([h * 60 for h in hours])
    ax.set_xticklabels([f'{h:02d}:00' for h in hours])
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add title (including household information)
    h_info = household_info.get(household_id, {})
    household_size = h_info.get('household_size', len(members))
    num_vehicles = h_info.get('num_vehicles', 'Unknown')
    
    title = f'Household {household_id} - Original Trajectory\n'
    title += f'Household Size: {household_size} | Vehicles: {num_vehicles} | Members: {n_members}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=color, edgecolor='black', label=activity)
        for activity, color in sorted(ACTIVITY_COLORS.items())
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        ncol=1,
        fontsize=10
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'household_{household_id}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Saved: household_{household_id}.png ({n_members} members)')


def main():
    """Main function"""
    print("="*70)
    print("Original Household Trajectory Visualization")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load generated household IDs
    print(f"\nLoading generated household IDs...")
    print(f"  Generated file: {GENERATED_TRAJECTORY_FILE}")
    target_household_ids = load_generated_household_ids(GENERATED_TRAJECTORY_FILE)
    print(f"✓ Found {len(target_household_ids)} households in generated data")
    
    # Load data
    print(f"\nLoading original trajectory data...")
    print(f"  Original trajectory file: {ORIGINAL_TRAJECTORY_FILE}")
    print(f"  Household info: {HOUSEHOLD_FILE}")
    
    households = load_trajectories(ORIGINAL_TRAJECTORY_FILE, target_household_ids)
    household_info = load_household_info(HOUSEHOLD_FILE)
    
    print(f"\n✓ Loaded {len(households)} households' data")
    
    # Statistics
    total_members = sum(len(members) for members in households.values())
    print(f"✓ Total {total_members} members")
    
    # Generate visualization
    print(f"\nGenerating visualizations...")
    print("-"*70)
    
    for idx, (household_id, members) in enumerate(sorted(households.items()), 1):
        plot_household_trajectories(household_id, members, household_info, OUTPUT_DIR)
        
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{len(households)}")
    
    print("-"*70)
    print(f"\n✅ Complete! Generated {len(households)} images")
    print(f"📁 Saved to: {OUTPUT_DIR}")
    print("="*70)
    
    # Statistics
    print(f"\n📊 Statistics:")
    member_counts = [len(members) for members in households.values()]
    print(f"  Average members per household: {np.mean(member_counts):.2f}")
    print(f"  Min members: {min(member_counts)}")
    print(f"  Max members: {max(member_counts)}")
    
    # Distribution by member count
    from collections import Counter
    member_distribution = Counter(member_counts)
    print(f"\n  Member count distribution:")
    for size in sorted(member_distribution.keys()):
        count = member_distribution[size]
        print(f"    {size}-member households: {count}")


if __name__ == '__main__':
    main()
