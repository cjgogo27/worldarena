#!/usr/bin/env python3
"""RoboTwin to Diffusion Policy conversion script.

Converts RoboTwin HDF5 episodes to Diffusion Policy Zarr format.
Standalone - does not require torch for basic conversion.
"""

import argparse
import os
import sys
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import zarr
except ImportError:
    print("Error: zarr not installed. Install with: pip install zarr numcodecs")
    sys.exit(1)

ROBOTTWIN_KEY_MAPPING = {
    "action": ["action", "joint_action/right_arm", "joint_action/left_arm", "actions/right_arm", "actions/left_arm", "right_arm"],
    "gripper": ["joint_action/right_gripper", "joint_action/left_gripper", "actions/right_gripper", "right_gripper"],
    "state": ["observations/qpos", "joint_states", "observations/joint_states", "endpose/right_endpose"],
    "image": ["right_camera/rgb", "head_camera/rgb", "observation/right_camera/rgb"],
}


def find_key_recursive(h5_file, possible_keys):
    """Find first available key from list in HDF5 file."""
    for key in possible_keys:
        if key in h5_file:
            return key
        parts = key.split("/")
        obj = h5_file
        for part in parts:
            if part in obj:
                obj = obj[part]
            else:
                break
        else:
            if isinstance(obj, h5py.Dataset):
                return key
    return None


def load_episode_hdf5(hdf5_path, load_images=True):
    """Load single HDF5 episode."""
    episode = {"observations": {}, "actions": {}}
    
    with h5py.File(hdf5_path, "r") as f:
        for name, key_set in ROBOTTWIN_KEY_MAPPING.items():
            if name == "image":
                continue
            found_key = find_key_recursive(f, key_set)
            if found_key and name in ["action", "gripper"]:
                data = f[found_key][:]
                if name == "action":
                    episode["actions"]["action"] = data
                elif name == "gripper":
                    episode["actions"]["gripper"] = data
        
        found_key = find_key_recursive(f, ROBOTTWIN_KEY_MAPPING["state"])
        if found_key:
            episode["observations"]["state"] = f[found_key][:]
    
    return episode


def convert_hdf5_to_zarr(input_dir, output_path, max_episodes=None, chunk_length=100):
    """Convert RoboTwin HDF5 episodes to Diffusion Policy Zarr format."""
    input_dir = Path(input_dir)
    hdf5_files = sorted(input_dir.glob("episode*.hdf5"))
    
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")
    
    if max_episodes:
        hdf5_files = hdf5_files[:max_episodes]
    
    print(f"Converting {len(hdf5_files)} episodes...")
    
    os.makedirs(output_path, exist_ok=True)
    zarr_root = zarr.open_group(output_path, mode="w")
    
    actions_list = []
    states_list = []
    episode_lengths = []
    
    for hdf5_path in tqdm(hdf5_files, desc="Loading episodes"):
        ep = load_episode_hdf5(hdf5_path, load_images=False)
        
        action = ep["actions"].get("action")
        state = ep["observations"].get("state")
        
        if action is not None:
            actions_list.append(action.astype(np.float32))
            episode_lengths.append(len(action))
            
            if state is not None and len(state) == len(action):
                states_list.append(state.astype(np.float32))
            else:
                states_list.append(action.astype(np.float32))
    
    if not actions_list:
        raise ValueError("No action data found")
    
    total_steps = sum(episode_lengths)
    action_dim = actions_list[0].shape[-1] if actions_list[0].ndim > 1 else 1
    state_dim = states_list[0].shape[-1] if states_list[0].ndim > 1 else 1
    
    print(f"Total steps: {total_steps}, Action dim: {action_dim}, State dim: {state_dim}")
    
    all_actions = np.concatenate(actions_list, axis=0)
    all_states = np.concatenate(states_list, axis=0)
    
    zarr_root.create_dataset(
        "data/action",
        data=all_actions,
        chunks=(chunk_length, action_dim),
        compressor=None,
    )
    
    zarr_root.create_dataset(
        "data/state",
        data=all_states,
        chunks=(chunk_length, state_dim),
        compressor=None,
    )
    
    zarr_root.create_dataset(
        "meta/episode_length",
        data=np.array(episode_lengths, dtype=np.int32),
        compressor=None,
    )
    
    n_episodes = len(episode_lengths)
    zarr_root.attrs["n_episodes"] = n_episodes
    zarr_root.attrs["total_steps"] = total_steps
    
    print(f"\nDone! Output: {output_path}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Action dim: {action_dim}")
    print(f"  State dim: {state_dim}")
    
    return zarr_root


def main():
    parser = argparse.ArgumentParser(
        description="Convert RoboTwin HDF5 to Diffusion Policy Zarr"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input directory with HDF5 episodes"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output Zarr path"
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Max episodes to convert"
    )
    parser.add_argument(
        "--chunk-length", type=int, default=100,
        help="Chunk length for Zarr"
    )
    
    args = parser.parse_args()
    
    convert_hdf5_to_zarr(
        args.input,
        args.output,
        max_episodes=args.max_episodes,
        chunk_length=args.chunk_length,
    )


if __name__ == "__main__":
    main()
