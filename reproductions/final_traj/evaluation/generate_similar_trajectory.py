#!/usr/bin/env python3
"""
Generate a similar trajectory with slight time variations for household 30135466
and visualize comparison with the original trajectory
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import random

# File paths
ORIGINAL_TRAJECTORY_FILE = "/data/alice/cjtest/FinalTraj/California/processed_data/all_user_schedules.json"
OUTPUT_DIR = "/data/alice/cjtest/FinalTraj/evaluation/case_study/similar_trajectory_test"
TARGET_HOUSEHOLD = "30135466"

# Activity type to color mapping aligned to majority figure palette
ACTIVITY_COLORS = {
    'home': '#A7E5CE',
    'work': '#FBB5C2',
    'education': '#C6CDE9',
    'shopping': '#FEC7A2',
    'service': '#F7ADB5',
    'medical': '#F7ADB5',
    'dine_out': '#FEC7A2',
    'socialize': '#DFAEC7',
    'exercise': '#95CCBC',
    'dropoff_pickup': '#DDE2CE'
}

ACT_EDGE_COLOR = '#6E7B88'
ACT_EDGE_WIDTH = 0.65


def time_to_minutes(time_str):
    """Convert time string to minutes"""
    if time_str == "24:00":
        return 1440
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])


def minutes_to_time(minutes):
    """Convert minutes to time string"""
    if minutes >= 1440:
        return "24:00"
    if minutes < 0:
        return "00:00"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def add_time_variation(time_str, variation_range=10):
    """
    Add random time variation to a time string
    
    Args:
        time_str: Time in format "HH:MM"
        variation_range: Maximum variation in minutes (±)
    
    Returns:
        Modified time string
    """
    minutes = time_to_minutes(time_str)
    
    # Don't modify 00:00 and 24:00
    if time_str == "00:00" or time_str == "24:00":
        return time_str
    
    # Add random variation
    variation = random.randint(-variation_range, variation_range)
    new_minutes = minutes + variation
    
    # Ensure within valid range
    new_minutes = max(0, min(1440, new_minutes))
    
    return minutes_to_time(new_minutes)


def generate_similar_trajectory(original_trajectory, member_index, total_members):
    """
    Generate a similar trajectory with meaningful variations
    
    Args:
        original_trajectory: Original trajectory data
        member_index: Index of the member (0-based)
        total_members: Total number of members
    
    Returns:
        Similar trajectory with modified times
    """
    similar_trajectory = {
        'user_id': original_trajectory['user_id'],
        'schedule': []
    }
    
    schedule = original_trajectory['schedule']
    
    # Special handling for member 1's work shift
    work_time_shift = 0
    if member_index == 0:
        # Find work activity and calculate time shift needed
        for i, segment in enumerate(schedule):
            if segment['activity'] == 'work':
                original_start = time_to_minutes(segment['start_time'])
                target_start = 11 * 60  # 11:00
                work_time_shift = target_start - original_start
                break
    
    for i, segment in enumerate(schedule):
        activity = segment['activity']
        start_minutes = time_to_minutes(segment['start_time'])
        end_minutes = time_to_minutes(segment['end_time'])
        duration = end_minutes - start_minutes
        
        # Apply different variations based on activity type and member
        if activity == 'work':
            # Special handling for member 1: move work to start at 12:10
            if member_index == 0:
                start_variation = work_time_shift
                end_variation = work_time_shift
            else:
                # Make work times more similar across members
                start_variation = random.randint(-15, 15)
                end_variation = random.randint(-20, 20)
        elif activity == 'education':
            # Extend education time to around 3pm (15:00)
            start_variation = random.randint(-10, 10)
            # Calculate how much to extend to reach around 15:00
            target_end = 15 * 60  # 15:00 in minutes
            current_end = end_minutes
            extension_needed = target_end - current_end
            # Add some randomness around the target
            end_variation = extension_needed + random.randint(-15, 15)
        elif activity == 'dine_out':
            # Move dine_out to around 18:30
            target_start = 18 * 60 + 30  # 18:30 in minutes
            current_start = start_minutes
            start_variation = target_start - current_start + random.randint(-10, 10)
            end_variation = start_variation + random.randint(-10, 10)  # Keep similar duration
        elif activity == 'home':
            # Special handling for member 1: extend first home to 12:10
            if member_index == 0 and i == 0:
                # First home segment for member 1
                start_variation = 0
                end_variation = work_time_shift
            else:
                # Small variations for home
                start_variation = random.randint(-10, 10)
                end_variation = random.randint(-10, 10)
        elif activity == 'socialize':
            # More variation for social activities
            start_variation = random.randint(-25, 25)
            end_variation = random.randint(-25, 25)
        else:
            # Default variation
            start_variation = random.randint(-20, 20)
            end_variation = random.randint(-20, 20)
        
        # Apply variations
        new_start = start_minutes + (start_variation if i > 0 else 0)
        new_end = end_minutes + (end_variation if i < len(schedule) - 1 else 0)
        
        # Ensure minimum duration of 15 minutes
        if new_end - new_start < 15:
            new_end = new_start + 15
        
        # Ensure within valid range
        new_start = max(0, min(1439, new_start))
        new_end = max(new_start + 15, min(1440, new_end))
        
        similar_trajectory['schedule'].append({
            'activity': activity,
            'start_time': minutes_to_time(new_start),
            'end_time': minutes_to_time(new_end)
        })
    
    # Fix continuity: make sure end time of segment i = start time of segment i+1
    for i in range(len(similar_trajectory['schedule']) - 1):
        current_end = similar_trajectory['schedule'][i]['end_time']
        similar_trajectory['schedule'][i+1]['start_time'] = current_end
    
    return similar_trajectory


def load_household_trajectories(file_path, household_id):
    """Load trajectories for a specific household"""
    with open(file_path, 'r', encoding='utf-8') as f:
        all_trajectories = json.load(f)
    
    household_trajectories = []
    for traj in all_trajectories:
        user_id = traj['user_id']
        if user_id.startswith(household_id + '_'):
            household_trajectories.append(traj)
    
    # Sort by user_id
    household_trajectories.sort(key=lambda x: x['user_id'])
    
    return household_trajectories


def plot_trajectory_on_axis(ax, members, title, show_xlabel=False):
    """
    Plot trajectories for all members on a given axis
    
    Args:
        ax: Matplotlib axis object
        members: List of member trajectories
        title: Title for this subplot
        show_xlabel: Whether to show x-axis label
    """
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
                  edgecolor=ACT_EDGE_COLOR,
                  linewidth=ACT_EDGE_WIDTH
            )
            ax.add_patch(rect)
            
            # Add activity label for longer time periods
            duration = end_time - start_time
            # Only show text if duration is long enough (at least 90 minutes to avoid overlap)
            if duration > 90:
                mid_point = (start_time + end_time) / 2
                activity_text = activity.replace('_', ' ').title()
                ax.text(
                    mid_point, y_pos,
                    activity_text,
                    ha='center', va='center',
                    fontsize=14,
                    weight='bold',
                    color='#2C3E50'  # Dark gray for better contrast on light colors
                )
    
    # Set y-axis
    ax.set_ylim(-0.5, n_members - 0.5)
    ax.set_yticks(range(n_members))
    ax.set_yticklabels([m['user_id'].split('_')[-1] for m in members[::-1]], fontsize=14)
    ax.set_ylabel('Member', fontsize=16, fontweight='bold')
    
    # Set x-axis (time axis)
    ax.set_xlim(0, 1440)
    hours = [0, 6, 12, 18, 24]
    ax.set_xticks([h * 60 for h in hours])
    if show_xlabel:
        ax.set_xticklabels([f'{h:02d}:00' for h in hours], fontsize=13)
        ax.set_xlabel('Time of Day', fontsize=16, fontweight='bold')
    else:
        ax.set_xticklabels([f'{h:02d}:00' for h in hours], fontsize=13)
    
    # Add grid with better styling
    ax.grid(axis='x', alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # Set title
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)


def main():
    """Main function"""
    print("="*70)
    print("Similar Trajectory Generation and Comparison")
    print("="*70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load original trajectories
    print(f"\nLoading original trajectories for household {TARGET_HOUSEHOLD}...")
    original_trajectories = load_household_trajectories(
        ORIGINAL_TRAJECTORY_FILE, 
        TARGET_HOUSEHOLD
    )
    
    if len(original_trajectories) == 0:
        print(f"❌ No trajectories found for household {TARGET_HOUSEHOLD}")
        return
    
    print(f"✓ Found {len(original_trajectories)} members")
    
    # Generate similar trajectories
    print(f"\nGenerating similar trajectories with meaningful variations...")
    similar_trajectories = []
    for idx, orig_traj in enumerate(original_trajectories):
        similar_traj = generate_similar_trajectory(orig_traj, idx, len(original_trajectories))
        similar_trajectories.append(similar_traj)
        
        print(f"  Member {orig_traj['user_id']}:")
        for i, (orig_seg, sim_seg) in enumerate(zip(orig_traj['schedule'], similar_traj['schedule'])):
            time_diff_start = abs(time_to_minutes(orig_seg['start_time']) - 
                                 time_to_minutes(sim_seg['start_time']))
            time_diff_end = abs(time_to_minutes(orig_seg['end_time']) - 
                               time_to_minutes(sim_seg['end_time']))
            print(f"    {orig_seg['activity']}: {orig_seg['start_time']}-{orig_seg['end_time']} → "
                  f"{sim_seg['start_time']}-{sim_seg['end_time']} "
                  f"(Δ: start={time_diff_start}min, end={time_diff_end}min)")
    
    # Save similar trajectories to JSON
    output_json = os.path.join(OUTPUT_DIR, f'household_{TARGET_HOUSEHOLD}_similar.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(similar_trajectories, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved similar trajectories to: {output_json}")
    
    # Create visualization
    print(f"\nGenerating comparison visualization...")
    n_members = len(original_trajectories)
    
    # Create figure with 2 rows - improved styling
    fig, (ax_orig, ax_sim) = plt.subplots(2, 1, figsize=(18, 2.5 * n_members), 
                                          gridspec_kw={'hspace': 0.25})
    
    # Set white background
    fig.patch.set_facecolor('white')
    
    # Plot original trajectories (top)
    plot_trajectory_on_axis(
        ax_orig, 
        original_trajectories, 
        '(a) Original Trajectory',
        show_xlabel=False
    )
    
    # Plot similar trajectories (bottom)
    plot_trajectory_on_axis(
        ax_sim, 
        similar_trajectories, 
        '(b) Generated Trajectory',
        show_xlabel=True
    )
    
    # Create legend with better styling
    legend_elements = [
            mpatches.Patch(facecolor=color, edgecolor=ACT_EDGE_COLOR, linewidth=ACT_EDGE_WIDTH,
                      label=activity.replace('_', ' ').title())
        for activity, color in sorted(ACTIVITY_COLORS.items())
    ]
    
    # Add legend to the right of the figure
    fig.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(0.92, 0.5),
        ncol=1,
        fontsize=13,
        title='Activity Type',
        title_fontsize=14,
        frameon=True,
        edgecolor='black',
        fancybox=False
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.88, 1.0])
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f'household_{TARGET_HOUSEHOLD}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison visualization: {output_path}")
    
    print("\n" + "="*70)
    print("✅ Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
