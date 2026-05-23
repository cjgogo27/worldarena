"""Raw RoboTwin HDF5 reader for LaRA-WM.

Reads episode data from HDF5 format with support for:
- Image observations (agent_view, wrist_camera)
- State observations (joint_states, end_effector_pose)
- Actions (joint_position, gripper_position)
- Rewards
"""

from __future__ import annotations

import h5py
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal

from numpy.typing import NDArray


ObservationKey = Literal["agent_view", "wrist_camera", "joint_states", "ee_pose"]
ActionKey = Literal["joint_position", "gripper_position"]


@dataclass(frozen=True, slots=True)
class RoboTwinEpisode:
    """Single episode from RoboTwin dataset.

    Attributes:
        episode_id: Unique episode identifier.
        observations: Dict of observation arrays keyed by sensor type.
        actions: Dict of action arrays.
        rewards: Array of reward values per timestep.
        episode_length: Number of timesteps in episode.
    """

    episode_id: str
    observations: dict[str, NDArray[np.float32]]
    actions: dict[str, NDArray[np.float32]]
    rewards: NDArray[np.float32]
    episode_length: int

    @property
    def image_obs(self) -> dict[str, NDArray[np.uint8]]:
        """Get image observations only."""
        return {
            k: v for k, v in self.observations.items()
            if "image" in k or "camera" in k or "view" in k
        }

    @property
    def state_obs(self) -> dict[str, NDArray[np.float32]]:
        """Get state observations only."""
        return {
            k: v for k, v in self.observations.items()
            if "state" in k or "pose" in k
        }

    def get(self, key: str, default: NDArray[np.float32] | None = None) -> NDArray[np.float32] | None:
        """Get observation or action by key."""
        if key in self.observations:
            return self.observations[key]
        if key in self.actions:
            return self.actions[key]
        return default


@dataclass
class RoboTwinDataset:
    """RoboTwin HDF5 dataset reader.

    Supports batched loading and iteration over episodes.

    Expected HDF5 structure:
        /episodes/
            episode_{idx}/
                observations/
                    agent_view_{t}.png or agent_view (dataset)
                    wrist_camera_{t}.png or wrist_camera (dataset)
                    joint_states (dataset): (T, DoF)
                    ee_pose (dataset): (T, 7)
                actions/
                    joint_position (dataset): (T, DoF)
                    gripper_position (dataset): (T, 1)
                rewards (dataset): (T,) or rewards (scalar per step)
        /metadata/
            episode_names
            episode_lengths

    Args:
        data_path: Path to HDF5 file or directory containing HDF5 files.
        batch_size: Number of episodes to load per batch.
        load_images: Whether to load image data (memory intensive).
        transform_obs: Optional callable to transform observations.
        transform_action: Optional callable to transform actions.
    """

    data_path: Path
    batch_size: int = 1
    load_images: bool = True
    transform_obs: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None
    transform_action: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None

    _h5_file: h5py.File | None = field(default=None, init=False, repr=False)
    _episode_keys: list[str] = field(default_factory=list, init=False, repr=False)
    _cache: dict[int, RoboTwinEpisode] = field(default_factory=dict, init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        if self.data_path.is_file():
            self._open(self.data_path)
        elif self.data_path.is_dir():
            h5_files = list(self.data_path.glob("*.h5")) + list(self.data_path.glob("*.hdf5"))
            if h5_files:
                self._open(h5_files[0])
            else:
                raise FileNotFoundError(f"No HDF5 files found in {self.data_path}")
        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def _open(self, path: Path) -> None:
        """Open HDF5 file and index episodes."""
        self._h5_file = h5py.File(path, "r")
        self._episode_keys = self._index_episodes()

    def _index_episodes(self) -> list[str]:
        keys: list[str] = []
        if self._h5_file is None:
            return keys
        if "episodes" in self._h5_file:
            for key in self._h5_file["episodes"].keys():
                if key.startswith("episode_"):
                    keys.append(key)
        keys.sort(key=lambda x: int(x.split("_")[1]))
        return keys

    def _read_observations(self, ep_group: h5py.Group) -> dict[str, NDArray[np.float32]]:
        """Read observation data from episode group."""
        obs: dict[str, NDArray[np.float32]] = {}

        if "observations" in ep_group:
            obs_group = ep_group["observations"]
            for key in obs_group.keys():
                dataset = obs_group[key]
                if isinstance(dataset, h5py.Dataset):
                    if self.load_images or not self._is_image_dataset(key, dataset):
                        if dataset.dtype == np.uint8:
                            # Image data - keep as uint8, don't convert
                            obs[key] = dataset[:]
                        else:
                            data = dataset[:].astype(np.float32)
                            obs[key] = self._maybe_transform(data, self.transform_obs)

        return obs

    def _read_actions(self, ep_group: h5py.Group) -> dict[str, NDArray[np.float32]]:
        """Read action data from episode group."""
        actions: dict[str, NDArray[np.float32]] = {}

        if "actions" in ep_group:
            act_group = ep_group["actions"]
            for key in act_group.keys():
                dataset = act_group[key]
                if isinstance(dataset, h5py.Dataset):
                    data = dataset[:].astype(np.float32)
                    actions[key] = self._maybe_transform(data, self.transform_action)

        return actions

    def _read_rewards(self, ep_group: h5py.Group) -> NDArray[np.float32]:
        """Read reward data from episode group."""
        if "rewards" in ep_group:
            dataset = ep_group["rewards"]
            if isinstance(dataset, h5py.Dataset):
                return dataset[:].astype(np.float32)

        # Check if rewards stored as scalar per step (legacy format)
        if "reward" in ep_group:
            dataset = ep_group["reward"]
            if isinstance(dataset, h5py.Dataset):
                return dataset[:].astype(np.float32)

        return np.array([], dtype=np.float32)

    def _is_image_dataset(self, key: str, dataset: h5py.Dataset) -> bool:
        """Check if dataset contains image data."""
        if dataset.dtype == np.uint8:
            # Check shape - images typically have 3+ dims
            if len(dataset.shape) >= 3:
                return True
        return False

    def _maybe_transform(
        self,
        data: NDArray[np.float32],
        transform: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None,
    ) -> NDArray[np.float32]:
        """Apply transform if provided."""
        if transform is not None:
            return transform(data)
        return data

    def __len__(self) -> int:
        return len(self._episode_keys)

    def __iter__(self) -> Iterator[RoboTwinEpisode]:
        """Iterate over all episodes."""
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> RoboTwinEpisode:
        ep: RoboTwinEpisode | list[RoboTwinEpisode] = self._getitem_impl(index)
        if isinstance(ep, list):
            return ep[0]
        return ep

    def _getitem_impl(self, index: int | slice) -> RoboTwinEpisode | list[RoboTwinEpisode]:
        """Get episode(s) by index."""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        if index < 0:
            index = len(self) + index

        if index in self._cache:
            return self._cache[index]

        if self._h5_file is None:
            raise RuntimeError("HDF5 file not open")

        key = self._episode_keys[index]
        ep_group = self._h5_file["episodes"][key]

        observations = self._read_observations(ep_group)
        actions = self._read_actions(ep_group)
        rewards = self._read_rewards(ep_group)

        episode_length = len(rewards) if len(rewards) > 0 else 0
        obs_joint_states = ep_group["observations/joint_states"]
        episode_length = max(episode_length, len(obs_joint_states))

        episode = RoboTwinEpisode(
            episode_id=key,
            observations=observations,
            actions=actions,
            rewards=rewards,
            episode_length=episode_length,
        )

        self._cache[index] = episode
        return episode

    def get_batch(self, indices: list[int]) -> list[RoboTwinEpisode]:
        """Load batch of episodes by indices."""
        return [self[i] for i in indices]

    def load_episode(self, episode_id: str) -> RoboTwinEpisode | None:
        """Load specific episode by ID."""
        for idx, key in enumerate(self._episode_keys):
            if key == episode_id or key == f"episode_{episode_id}":
                return self[idx]
        return None

    def close(self) -> None:
        """Close HDF5 file."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __enter__(self: RoboTwinDataset) -> RoboTwinDataset:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def create_reader(
    data_path: str | Path,
    batch_size: int = 1,
    load_images: bool = True,
) -> RoboTwinDataset:
    """Factory function to create a RoboTwin reader.

    Args:
        data_path: Path to HDF5 data file or directory.
        batch_size: Default batch size for loading.
        load_images: Whether to load image data.

    Returns:
        Configured RoboTwinDataset reader.
    """
    return RoboTwinDataset(
        data_path=Path(data_path),
        batch_size=batch_size,
        load_images=load_images,
    )