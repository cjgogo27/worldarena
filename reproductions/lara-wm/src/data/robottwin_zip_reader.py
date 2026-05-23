"""RoboTwin Zip HDF5 reader for LaRA-WM.

Reads episode data from zipped HDF5 files with actual RoboTwin structure:
- observation/{head,left,right}_camera/rgb: JPEG bytes per timestep
- joint_action/vector: (T, 16) actions
- endpose/left_endpose, right_endpose: (T, 7) end effector poses
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from PIL import Image


@dataclass
class RoboTwinSample:
    """Single timestep sample."""
    # Images: (H, W, 3) uint8
    head_image: np.ndarray | None = None
    left_image: np.ndarray | None = None
    right_image: np.ndarray | None = None
    # State: (7,) for left arm + gripper or (16,) combined
    state: np.ndarray | None = None
    # Action: (16,) combined action
    action: np.ndarray | None = None
    # End effector: (7,) per arm
    left_endpose: np.ndarray | None = None
    right_endpose: np.ndarray | None = None
    left_gripper: np.ndarray | None = None
    right_gripper: np.ndarray | None = None


@dataclass
class RoboTwinEpisode:
    """Single episode from RoboTwin."""
    episode_id: str
    task_name: str
    # List of samples per timestep
    samples: list[RoboTwinSample]
    
    @property
    def episode_length(self) -> int:
        return len(self.samples)
    
    @property
    def images(self) -> np.ndarray:
        """Get all head camera images as (T, H, W, 3) uint8."""
        return np.array([s.head_image for s in self.samples if s.head_image is not None])
    
    @property
    def actions(self) -> np.ndarray:
        """Get all actions as (T, 16) float64."""
        return np.array([s.action for s in self.samples if s.action is not None])


def decode_jpeg_bytes(jpeg_data: bytes) -> np.ndarray:
    """Decode JPEG bytes to RGB image array."""
    img = Image.open(io.BytesIO(jpeg_data))
    return np.array(img)


def read_episode_from_hdf5(h5_data, include_images: bool = True) -> RoboTwinEpisode:
    """Read a single episode from an HDF5 file object."""
    samples = []
    
    # Get action data
    joint_action = h5_data['joint_action']
    actions = joint_action['vector'][:]  # (T, 16)
    left_arm = joint_action['left_arm'][:]  # (T, 7)
    left_gripper = joint_action['left_gripper'][:]  # (T,)
    right_arm = joint_action['right_arm'][:]  # (T, 7)
    right_gripper = joint_action['right_gripper'][:]  # (T,)
    
    # Get end effector poses
    endpose = h5_data['endpose']
    left_endpose = endpose['left_endpose'][:]  # (T, 7)
    right_endpose = endpose['right_endpose'][:]  # (T, 7)
    
    # Get image data
    obs = h5_data['observation']
    head_rgb = obs['head_camera']['rgb'][:] if include_images else None
    left_rgb = obs['left_camera']['rgb'][:] if include_images else None
    right_rgb = obs['right_camera']['rgb'][:] if include_images else None
    
    T = len(actions)
    
    for t in range(T):
        sample = RoboTwinSample(
            action=actions[t].astype(np.float32),
            state=left_arm[t].astype(np.float32),  # Use left arm as state
            left_endpose=left_endpose[t].astype(np.float32),
            right_endpose=right_endpose[t].astype(np.float32),
            left_gripper=np.array([left_gripper[t]], dtype=np.float32),
            right_gripper=np.array([right_gripper[t]], dtype=np.float32),
        )
        
        if include_images:
            try:
                sample.head_image = decode_jpeg_bytes(bytes(head_rgb[t]))
                sample.left_image = decode_jpeg_bytes(bytes(left_rgb[t]))
                sample.right_image = decode_jpeg_bytes(bytes(right_rgb[t]))
            except Exception:
                pass  # Skip image decode errors
        
        samples.append(sample)
    
    return RoboTwinEpisode(
        episode_id="unknown",
        task_name="unknown",
        samples=samples
    )


def read_episodes_from_zip(zip_path: str | Path, max_episodes: int | None = None, include_images: bool = True) -> list[RoboTwinEpisode]:
    """Read all episodes from a RoboTwin zipped HDF5 file."""
    episodes = []
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find all hdf5 files
        hdf5_files = sorted([f for f in zf.namelist() if 'data/episode' in f and f.endswith('.hdf5')])
        
        if max_episodes:
            hdf5_files = hdf5_files[:max_episodes]
        
        for i, h5_path in enumerate(hdf5_files):
            with zf.open(h5_path) as f:
                h5_data = h5py.File(io.BytesIO(f.read()), 'r')
                episode = read_episode_from_hdf5(h5_data, include_images=include_images)
                episode.episode_id = f"episode_{i}"
                episodes.append(episode)
    
    return episodes


class RoboTwinDataLoader:
    """DataLoader for RoboTwin zipped HDF5 files."""
    
    def __init__(
        self,
        data_dir: str | Path,
        tasks: list[str] | None = None,
        batch_size: int = 8,
        include_images: bool = True,
        max_episodes_per_task: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.include_images = include_images
        
        # Find available task zip files
        self.episodes: list[RoboTwinEpisode] = []
        
        available_tasks = tasks or [
            'grab_roller',
            'place_a2b_left',
            'stack_blocks_two',
            'handover_block',
            'open_laptop',
            'adjust_bottle',
            'beat_block_hammer',
            'click_bell',
            'dump_bin_bigbin',
            'press_stapler',
        ]
        
        for task in available_tasks:
            # Check in robottwin_hf/dataset/
            hf_path = self.data_dir / 'dataset' / task / 'franka_clean_50.zip'
            if not hf_path.exists():
                hf_path = self.data_dir / 'dataset' / f'{task}.zip'
            
            if not hf_path.exists():
                # Check in data/robotwin/dataset/
                hf_path = self.data_dir.parent / 'robotwin' / 'dataset' / f'{task}_franka_clean_50.zip'
            
            if hf_path.exists():
                size = hf_path.stat().st_size
                if size > 1000000:  # At least 1MB
                    print(f"Loading {task} from {hf_path}...")
                    try:
                        episodes = read_episodes_from_zip(
                            hf_path, 
                            max_episodes=max_episodes_per_task,
                            include_images=include_images
                        )
                        for ep in episodes:
                            ep.task_name = task
                        self.episodes.extend(episodes)
                    except Exception as e:
                        print(f"  Error loading {task}: {e}")
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> RoboTwinEpisode:
        return self.episodes[idx]
    
    def get_batch(self, indices: list[int]) -> list[RoboTwinEpisode]:
        return [self.episodes[i] for i in indices]
    
    def iter_batches(self) -> list[list[RoboTwinEpisode]]:
        """Yield batches of episodes."""
        for i in range(0, len(self.episodes), self.batch_size):
            yield self.episodes[i:i + self.batch_size]
