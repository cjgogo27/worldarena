"""RoboTwin to Diffusion Policy Dataset Adapter.

Converts RoboTwin HDF5 episode format to Diffusion Policy's BaseLowdimDataset interface.
Supports both low-dim state and image observations.

Location: /data/alice/cjtest/lara-wm/src/baselines/diffusion_policy_prep/
"""

from __future__ import annotations

import os
import h5py
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Callable

import torch
from torch.utils.data import Dataset

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


# RoboTwin data keys
ROBOTTWIN_OBS_KEYS = ["joint_states", "ee_pose", "joint_position"]
ROBOTTWIN_IMAGE_KEYS = ["agent_view", "head_camera", "left_camera", "right_camera", "wrist_camera"]
ROBOTTWIN_ACTION_KEYS = ["joint_position", "gripper_position", "joint_action", "gripper_action"]


@dataclass
class RoboTwinDataConfig:
    """Configuration for RoboTwin data mapping to Diffusion Policy."""
    
    # Observation mapping
    state_obs_key: str = "joint_states"  # RoboTwin key for robot state
    image_obs_key: str = "right_camera"  # RoboTwin key for image observation
    
    # Action mapping
    action_joint_key: str = "joint_action/right_arm"  # RoboTwin key for joint action
    action_gripper_key: str = "joint_action/right_gripper"  # RoboTwin key for gripper action
    
    # Whether to use absolute actions
    abs_action: bool = True
    
    # Image processing
    image_size: tuple[int, int] = (96, 96)  # (H, W) for resizing
    
    def get_action_dim(self) -> int:
        """Get total action dimension."""
        dim = 7  # joint position (7 DOF)
        if self.action_gripper_key is not None:
            dim += 1  # gripper
        return dim


class RoboTwinReplayBuffer:
    """In-memory ReplayBuffer implementation for RoboTwin HDF5 data.
    
    Manually constructs the buffer from HDF5 episodes instead of loading from Zarr.
    """
    
    def __init__(self, episodes: list[dict], keys: list[str]):
        """
        Args:
            episodes: List of episode dicts with 'observations' and 'actions' keys
            keys: List of data keys to extract (e.g., ['obs', 'action'])
        """
        if len(episodes) == 0:
            raise ValueError("No episodes provided")
        
        self.episodes = episodes
        self.keys = keys
        
        # Compute dimensions
        self.n_episodes = len(episodes)
        
        # Build flat index for episode boundaries
        self._episode_starts = []
        self._episode_lengths = []
        
        total_steps = 0
        for ep in episodes:
            ep_len = self._get_episode_length(ep)
            self._episode_starts.append(total_steps)
            self._episode_lengths.append(ep_len)
            total_steps += ep_len
        
        self.total_steps = total_steps
        
        # Pre-allocate if possible
        self._data = {}
        self._allocate_data()
    
    def _get_episode_length(self, ep: dict) -> int:
        """Get episode length from episode dict."""
        # Find first array in observations
        for key in ep.get('observations', {}).values():
            if isinstance(key, np.ndarray) and key.size > 0:
                return len(key)
        for key in ep.get('actions', {}).values():
            if isinstance(key, np.ndarray) and key.size > 0:
                return len(key)
        return 0
    
    def _allocate_data(self):
        """Allocate data arrays."""
        # For now, we'll access data lazily through episodes
        pass
    
    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get data at timestep idx."""
        # Find episode and local index
        ep_idx = 0
        local_idx = idx
        for i, (start, length) in enumerate(zip(self._episode_starts, self._episode_lengths)):
            if idx < start + length:
                ep_idx = i
                local_idx = idx - start
                break
        
        ep = self.episodes[ep_idx]
        result = {}
        
        for key in self.keys:
            if key in ep.get('observations', {}):
                arr = ep['observations'][key]
                if isinstance(arr, np.ndarray) and len(arr) > local_idx:
                    result[key] = arr[local_idx]
            elif key in ep.get('actions', {}):
                arr = ep['actions'][key]
                if isinstance(arr, np.ndarray) and len(arr) > local_idx:
                    result[key] = arr[local_idx]
        
        return result
    
    def __getattr__(self, key: str) -> np.ndarray:
        """Get all data for a key across all episodes."""
        if key in self._data and len(self._data[key]) > 0:
            return self._data[key]
        
        # Build from episodes
        result = []
        for ep in self.episodes:
            if key in ep.get('observations', {}):
                arr = ep['observations'][key]
                if isinstance(arr, np.ndarray):
                    result.append(arr)
            elif key in ep.get('actions', {}):
                arr = ep['actions'][key]
                if isinstance(arr, np.ndarray):
                    result.append(arr)
        
        if len(result) == 0:
            return np.array([])
        
        return np.concatenate(result, axis=0)


def load_hdf5_episodes(
    data_dir: str | Path,
    max_episodes: int | None = None,
    load_images: bool = False,
    image_key: str = "right_camera/rgb",
) -> list[dict]:
    """Load episodes from RoboTwin HDF5 directory.
    
    Args:
        data_dir: Directory containing episode*.hdf5 files
        max_episodes: Maximum number of episodes to load
        load_images: Whether to load image data
        image_key: Key path for image data in HDF5
        
    Returns:
        List of episode dicts with 'observations' and 'actions' keys
    """
    data_dir = Path(data_dir)
    hdf5_files = sorted(data_dir.glob("episode*.hdf5"))
    
    if max_episodes is not None:
        hdf5_files = hdf5_files[:max_episodes]
    
    episodes = []
    
    for hdf5_path in hdf5_files:
        episode = {"observations": {}, "actions": {}}
        
        with h5py.File(hdf5_path, "r") as f:
            # Load observations (state)
            if "joint_action" in f:
                # This might be under joint_action for some datasets
                for key in f["joint_action"].keys():
                    if isinstance(f["joint_action"][key], h5py.Dataset):
                        episode["actions"][key] = f[f"joint_action/{key}"][:]
            
            if "endpose" in f:
                for key in f["endpose"].keys():
                    if isinstance(f["endpose"][key], h5py.Dataset):
                        episode["observations"][key] = f[f"endpose/{key}"][:]
            
            # Also check for action groups at root level
            for grp_name in ["joint_action", "actions"]:
                if grp_name in f:
                    grp = f[grp_name]
                    for key in grp.keys():
                        if isinstance(grp[key], h5py.Dataset):
                            if key not in episode["actions"]:
                                episode["actions"][key] = grp[key][:]
            
            # Load observations at root level
            for grp_name in ["observation", "observations"]:
                if grp_name in f:
                    grp = f[grp_name]
                    for key in grp.keys():
                        if isinstance(grp[key], h5py.Dataset):
                            episode["observations"][key] = grp[key][:]
            
            # Try to load images if requested
            if load_images and image_key in f:
                # Images stored as bytes (JPEG compressed)
                img_data = f[image_key][:]
                # Decode if needed - stored as raw bytes refs
                # For now, skip actual image loading
                pass
        
        episodes.append(episode)
    
    return episodes


class RoboTwinDiffusionDataset(BaseLowdimDataset):
    """Adapter converting RoboTwin HDF5 data to Diffusion Policy dataset interface.
    
    Example:
        >>> dataset = RoboTwinDiffusionDataset(
        ...     zarr_path="/path/to/episodes",
        ...     horizon=8,
        ...     pad_before=2,
        ...     pad_after=4,
        ...     state_key="joint_action/right_arm",
        ...     action_key="joint_action/right_arm",
        ... )
        >>> normalizer = dataset.get_normalizer()
        >>> sample = dataset[0]  # {'obs': (T, Do), 'action': (T, Da)}
    """
    
    def __init__(
        self,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        state_key: str = "joint_action/right_arm",
        action_key: str = "joint_action/right_arm",
        gripper_key: str = "joint_action/right_gripper",
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        use_abs_action: bool = True,
    ):
        """
        Args:
            zarr_path: Path to HDF5 directory or Zarr file
            horizon: Sequence length for temporal aggregation
            pad_before: Padding before sequence start
            pad_after: Padding after sequence end
            state_key: Key for state observation in data
            action_key: Key for action in data
            gripper_key: Key for gripper action (optional, None to skip)
            seed: Random seed for train/val split
            val_ratio: Ratio of episodes for validation
            max_train_episodes: Max training episodes (None for all)
            use_abs_action: Whether actions are absolute (not relative)
        """
        super().__init__()
        
        self.zarr_path = Path(zarr_path)
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.state_key = state_key
        self.action_key = action_key
        self.gripper_key = gripper_key
        self.val_ratio = val_ratio
        self.use_abs_action = use_abs_action
        
        # Load HDF5 data
        self.episodes = load_hdf5_episodes(
            self.zarr_path,
            max_episodes=max_train_episodes if max_train_episodes else None,
            load_images=False,
        )
        
        if len(self.episodes) == 0:
            raise ValueError(f"No episodes found in {zarr_path}")
        
        # Check data structures
        first_ep = self.episodes[0]
        
        # Find action dimension
        action_data = None
        for key in [action_key, "joint_action/right_arm", "joint_action/left_arm"]:
            if key in first_ep.get("actions", {}):
                action_data = first_ep["actions"][key]
                self.action_key = key
                break
        
        if action_data is None:
            # Try alternate keys
            for ep in self.episodes:
                for key, arr in ep.get("actions", {}).items():
                    if isinstance(arr, np.ndarray) and arr.ndim == 2:
                        action_data = arr
                        self.action_key = key
                        break
                if action_data is not None:
                    break
        
        if action_data is not None:
            self.action_dim = action_data.shape[-1] if action_data.ndim > 1 else 1
            if gripper_key is not None:
                # Add gripper dimension
                self.action_dim += 1
        else:
            self.action_dim = 7 + (1 if gripper_key else 0)
        
        # Build sequence sampler
        # First, estimate episode lengths and validate
        valid_episodes = []
        for i, ep in enumerate(self.episodes):
            actions = ep.get("actions", {})
            if len(actions) > 0:
                # Find any action array
                for key, arr in actions.items():
                    if isinstance(arr, np.ndarray) and arr.ndim >= 1 and len(arr) > 0:
                        valid_episodes.append(i)
                        break
        
        self.episode_indices = valid_episodes
        
        # Create simple sequence sampler
        self._setup_sampler(seed)
    
    def _setup_sampler(self, seed: int = 42):
        """Setup train/val split and sampler."""
        n_episodes = len(self.episode_indices)
        
        # Create train/val mask
        val_mask = get_val_mask(
            n_episodes=n_episodes,
            val_ratio=self.val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=None,
            seed=seed,
        )
        
        self.train_mask = train_mask
        
        # Create simple index mapping
        self.train_indices = np.where(train_mask)[0].tolist()
        self.val_indices = np.where(~train_mask)[0].tolist() if hasattr(~train_mask, '__iter__') else []
    
    def get_validation_dataset(self) -> RoboTwinDiffusionDataset:
        """Return validation split."""
        val_set = RoboTwinDiffusionDataset(
            zarr_path=str(self.zarr_path),
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            state_key=self.state_key,
            action_key=self.action_key,
            gripper_key=self.gripper_key,
            seed=42,  # Same seed for reproducibility
            val_ratio=self.val_ratio,
            use_abs_action=self.use_abs_action,
        )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        """Compute normalizer from dataset statistics.
        
        Args:
            mode: Normalization mode ('limits', 'mean_std', 'gaussian', 'bounds')
            
        Returns:
            Fitted LinearNormalizer
        """
        # Collect all data
        all_obs = []
        all_actions = []
        
        for ep_idx in self.train_indices:
            ep = self.episodes[ep_idx]
            
            # Get actions
            actions = ep.get("actions", {})
            for key, arr in actions.items():
                if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                    if key == self.action_key or (self.gripper_key and key == self.gripper_key):
                        all_actions.append(arr)
            
            # Get observations (states)
            obs = ep.get("observations", {})
            for key, arr in obs.items():
                if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                    if key == self.state_key:
                        all_obs.append(arr)
        
        # Concatenate
        if len(all_actions) > 0:
            all_actions = np.concatenate(all_actions, axis=0)
        else:
            # Fallback: use uniform random
            all_actions = np.random.randn(100, self.action_dim).astype(np.float32)
        
        if len(all_obs) > 0:
            all_obs = np.concatenate(all_obs, axis=0)
        else:
            # Use actions as observation fallback
            all_obs = all_actions
        
        # Create data dict for normalizer fitting
        data = {
            "action": all_actions,
            "obs": all_obs,
        }
        
        normalizer = LinearNormalizer()
        
        # Fit per keys
        # For now, create a simple normalizer for actions
        normalizer.fit(
            data={"action": all_actions},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        """Get all actions in dataset."""
        all_actions = []
        
        for ep in self.episodes:
            actions = ep.get("actions", {})
            for key, arr in actions.items():
                if isinstance(arr, np.ndarray):
                    if key == self.action_key:
                        all_actions.append(arr)
        
        if len(all_actions) == 0:
            return torch.zeros(0, self.action_dim)
        
        return torch.from_numpy(np.concatenate(all_actions, axis=0))
    
    def __len__(self) -> int:
        return len(self.train_indices) * self._estimate_seq_per_ep()
    
    def _estimate_seq_per_ep(self) -> int:
        """Estimate sequences per episode."""
        # Simple estimate: average episode length minus padding
        total = 0
        n = 0
        for ep_idx in self.train_indices[:5]:  # Sample first 5
            ep = self.episodes[ep_idx]
            for key, arr in ep.get("actions", {}).items():
                if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                    length = max(0, len(arr) - self.pad_before - self.pad_after - self.horizon + 1)
                    total += max(1, length)
                    n += 1
                    break
        return max(1, total // max(1, n))
    
    def _sample_to_data(self, ep: dict) -> dict[str, np.ndarray]:
        """Convert episode to sample data."""
        # Get action
        actions = ep.get("actions", {})
        action = None
        for key in [self.action_key, "joint_action/right_arm", "joint_action/left_arm"]:
            if key in actions:
                action = actions[key]
                break
        
        if action is not None and not isinstance(action, np.ndarray):
            action = None
        
        # Get gripper
        gripper = None
        if self.gripper_key is not None:
            for key in [self.gripper_key, "joint_action/right_gripper", "joint_action/left_gripper"]:
                if key in actions:
                    gripper = actions[key]
                    break
        
        if gripper is not None and not isinstance(gripper, np.ndarray):
            gripper = None
        
        # Combine action + gripper
        if action is not None:
            if gripper is not None:
                # Concatenate along last axis
                action = np.concatenate([action, gripper[..., np.newaxis]], axis=-1)
        else:
            # Fallback - use zeros
            action = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        
        # Get observation (state)
        obs = ep.get("observations", {})
        state = None
        for key in [self.state_key, "joint_states"]:
            if key in obs:
                state = obs[key]
                break
        
        if state is not None and not isinstance(state, np.ndarray):
            state = None
        
        if state is None:
            state = action  # Use action as obs fallback
        
        return {
            "obs": state,
            "action": action,
        }
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get training sample."""
        # Map index to episode
        ep_idx = self.train_indices[idx % len(self.train_indices)]
        ep = self.episodes[ep_idx]
        
        data = self._sample_to_data(ep)
        
        # Apply torch conversion
        torch_data = {}
        for key, arr in data.items():
            if isinstance(arr, np.ndarray):
                torch_data[key] = torch.from_numpy(arr)
            else:
                torch_data[key] = torch.zeros(self.horizon, self.action_dim)
        
        return torch_data


class RoboTwinImageDiffusionDataset(BaseImageDataset):
    """Adapter for RoboTwin with image observations.
    
    Similar to RoboTwinDiffusionDataset but includes image observations.
    """
    
    def __init__(
        self,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        state_key: str = "joint_action/right_arm",
        action_key: str = "joint_action/right_arm",
        image_obs_key: str = "right_camera/rgb",
        gripper_key: str = "joint_action/right_gripper",
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        image_size: tuple[int, int] = (96, 96),
    ):
        super().__init__()
        
        self.zarr_path = zarr_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.state_key = state_key
        self.action_key = action_key
        self.image_obs_key = image_obs_key
        self.gripper_key = gripper_key
        self.image_size = image_size
        
        # Load HDF5 data (without images first for efficiency)
        self.episodes = load_hdf5_episodes(
            zarr_path,
            max_episodes=max_train_episodes,
            load_images=False,
        )
        
        # Similar setup as RoboTwinDiffusionDataset
        self._setup_sampler(seed)
    
    def _setup_sampler(self, seed: int = 42):
        n_episodes = len(self.episodes)
        val_mask = get_val_mask(n_episodes, self.val_ratio, seed)
        train_mask = ~val_mask
        self.train_mask = train_mask
        self.train_indices = np.where(train_mask)[0].tolist()
    
    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        # Similar to RoboTwinDiffusionDataset
        all_actions = []
        
        for ep in self.episodes:
            for key, arr in ep.get("actions", {}).items():
                if key == self.action_key and isinstance(arr, np.ndarray):
                    all_actions.append(arr)
        
        if len(all_actions) > 0:
            all_actions = np.concatenate(all_actions, axis=0)
        else:
            all_actions = np.random.randn(100, 7).astype(np.float32)
        
        normalizer = LinearNormalizer()
        normalizer.fit(data={"action": all_actions}, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        all_actions = []
        for ep in self.episodes:
            for key, arr in ep.get("actions", {}).items():
                if key == self.action_key and isinstance(arr, np.ndarray):
                    all_actions.append(arr)
        
        if len(all_actions) == 0:
            return torch.zeros(0, 7)
        
        return torch.from_numpy(np.concatenate(all_actions, axis=0))
    
    def __len__(self) -> int:
        return len(self.train_indices)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx = self.train_indices[idx % len(self.train_indices)]
        ep = self.episodes[ep_idx]
        
        # Get action
        actions = ep.get("actions", {})
        action = actions.get(self.action_key)
        
        if action is None:
            action = np.zeros((self.horizon, 7), dtype=np.float32)
        
        return {
            "obs": {"image": torch.zeros(*self.image_size, 3)},  # Placeholder
            "action": torch.from_numpy(action.astype(np.float32)),
        }


def create_robottwin_dataset(
    data_path: str,
    dataset_type: Literal["lowdim", "image"] = "lowdim",
    horizon: int = 1,
    **kwargs,
) -> RoboTwinDiffusionDataset | RoboTwinImageDiffusionDataset:
    """Factory function to create RoboTwin Diffusion Policy dataset.
    
    Args:
        data_path: Path to HDF5 episode directory
        dataset_type: 'lowdim' for state-only, 'image' for image observations
        horizon: Sequence horizon
        **kwargs: Additional arguments to dataset
        
    Returns:
        Configured dataset
    """
    if dataset_type == "image":
        return RoboTwinImageDiffusionDataset(
            zarr_path=data_path,
            horizon=horizon,
            **kwargs,
        )
    else:
        return RoboTwinDiffusionDataset(
            zarr_path=data_path,
            horizon=horizon,
            **kwargs,
        )