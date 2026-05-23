#!/usr/bin/env python3
"""Convert RoboTwin episodes to ACT-compatible HDF5 format.

This script exports the current RoboTwin offline episodes into ACT-format
HDF5 episode files that can be used with the original ACT training pipeline.

ACT HDF5 Format:
    /observations/qpos: (T, state_dim) - joint positions
    /observations/qvel: (T, state_dim) - velocities (zeros if unavailable)
    /observations/images/<cam>: (T, H, W, 3) - RGB images
    /action: (T, action_dim) - actions
    root.attrs['sim'] = True

Dimension Mapping:
    RoboTwin provides 16D combined actions (left 7D + right 7D + 2 gripper)
    ACT expects 5D (for low-cost robot) or configurable (typically 5-7D)
    
    For fair comparison with lara-wm baselines:
    - state_dim = 7 (left arm joint positions)
    - action_dim = 7 (left arm joint actions)
    
    This maps to the 7DoF arm configuration matching RoboTwin's left arm.

Split Alignment:
    Uses the same task-stratified split logic from train_and_eval.py:
    - train_ratio: 0.7 (default)
    - val_ratio: 0.15 (default)  
    - test_ratio: 0.15 (default)
    
    ACT's internal loader uses 80/20 split, but we preserve the exact
    RoboTwin split for fairness.

Usage:
    python scripts/convert_robottwin_to_act.py --task grab_roller --output-dir data/act_format
    
    # Then train with ACT:
    cd third_party/act && python train.py --task grab_roller
    
    # Note: Set TASK_CONFIG in config/config.py to match:
    #   'state_dim': 7
    #   'action_dim': 7
    #   'camera_names': ['front']
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.robottwin_zip_reader import RoboTwinDataLoader, RoboTwinEpisode, RoboTwinSample


STATE_DIM = 7
ACTION_DIM = 7
CAMERA_NAMES = ["head_cam"]


@dataclass
class ConversionConfig:
    data_dir: str = "data/robottwin_hf"
    output_dir: str = "data/act_format"
    tasks: list[str] | None = None
    max_episodes_per_task: int = 10
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    camera_names: list[str] | None = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 2026
    image_size: tuple[int, int] = (640, 480)
    include_images: bool = True


def _split_counts(num_episodes: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if num_episodes <= 0:
        return 0, 0, 0
    if num_episodes == 1:
        return 1, 0, 0
    if num_episodes == 2:
        return 1, 0, 1
    if num_episodes == 3:
        return 1, 1, 1

    raw_train = int(round(num_episodes * train_ratio))
    raw_val = int(round(num_episodes * val_ratio))
    raw_test = num_episodes - raw_train - raw_val

    train_count = max(raw_train, 1)
    val_count = max(raw_val, 1)
    test_count = max(raw_test, 1)

    while train_count + val_count + test_count > num_episodes:
        if train_count >= max(val_count, test_count) and train_count > 1:
            train_count -= 1
        elif val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    while train_count + val_count + test_count < num_episodes:
        train_count += 1

    return train_count, val_count, test_count


def split_episodes(
    episodes: list[RoboTwinEpisode],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[RoboTwinEpisode]]:
    grouped: dict[str, list[RoboTwinEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.task_name].append(episode)

    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for task_name in sorted(grouped):
        task_episodes = list(grouped[task_name])
        order = rng.permutation(len(task_episodes))
        shuffled = [task_episodes[index] for index in order]
        train_count, val_count, test_count = _split_counts(
            len(shuffled), train_ratio, val_ratio, test_ratio
        )

        split["train"].extend(shuffled[:train_count])
        split["val"].extend(shuffled[train_count:train_count + val_count])
        split["test"].extend(shuffled[train_count + val_count:train_count + val_count + test_count])

    return split


def sample_is_valid(sample: RoboTwinSample) -> bool:
    return sample.head_image is not None and sample.action is not None


def episode_has_valid_transition(episode: RoboTwinEpisode) -> bool:
    return any(
        sample_is_valid(episode.samples[index]) and sample_is_valid(episode.samples[index + 1])
        for index in range(len(episode.samples) - 1)
    )


def load_episodes(config: ConversionConfig) -> list[RoboTwinEpisode]:
    if config.tasks is None:
        config.tasks = [
            "grab_roller", "place_a2b_left", "stack_blocks_two", "open_laptop",
            "adjust_bottle", "beat_block_hammer", "click_bell", "dump_bin_bigbin",
            "press_stapler",
        ]
    if config.camera_names is None:
        config.camera_names = list(CAMERA_NAMES)

    loader = RoboTwinDataLoader(
        data_dir=config.data_dir,
        tasks=config.tasks,
        batch_size=32,
        include_images=config.include_images,
        max_episodes_per_task=config.max_episodes_per_task,
    )
    episodes: list[RoboTwinEpisode] = []
    for episode in loader.episodes:
        if episode_has_valid_transition(episode):
            episodes.append(episode)
    return episodes


def episode_to_hdf5(
    episode: RoboTwinEpisode,
    output_path: Path,
    config: ConversionConfig,
) -> None:
    """Convert a single RoboTwin episode to ACT HDF5 format.

    ACT expects:
        /observations/qpos: (T, state_dim)
        /observations/qvel: (T, state_dim) 
        /observations/images/<cam>: (T, H, W, 3)
        /action: (T, action_dim)
        root.attrs['sim'] = True
    """
    samples = [s for s in episode.samples if sample_is_valid(s)]
    T = len(samples)

    if T == 0:
        raise ValueError(f"Episode {episode.episode_id} has no valid samples")

    H, W = config.image_size

    with h5py.File(output_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = True

        obs = root.create_group("observations")
        image_group = obs.create_group("images")

        for camera_name in config.camera_names:
            image_group.create_dataset(
                camera_name,
                (T, H, W, 3),
                dtype="uint8",
                chunks=(1, H, W, 3),
            )
        obs.create_dataset("qpos", (T, config.state_dim))
        obs.create_dataset("qvel", (T, config.state_dim))
        root.create_dataset("action", (T, config.action_dim))

        for t, sample in enumerate(samples):
            camera_map = {
                "front": sample.head_image,
                "head_cam": sample.head_image,
                "left_cam": sample.left_image,
                "right_cam": sample.right_image,
            }
            for camera_name in config.camera_names:
                img = camera_map.get(camera_name)
                if img is None:
                    continue
                if img.shape[:2] != (H, W):
                    img_pil = Image.fromarray(img)
                    img_pil = img_pil.resize((W, H), Image.BILINEAR)
                    img = np.array(img_pil)
                root[f"/observations/images/{camera_name}"][t] = img

            action = sample.action
            if action is not None:
                root["/action"][t] = action[:config.action_dim]

            state = sample.state
            if config.state_dim > 7 and sample.action is not None:
                root["/observations/qpos"][t] = sample.action[:config.state_dim]
            elif state is not None:
                root["/observations/qpos"][t] = state[:config.state_dim]

            root["/observations/qvel"][t] = np.zeros(config.state_dim)


def get_dataset_stats(episode_paths: list[Path], config: ConversionConfig) -> dict:
    """Compute normalization stats for ACT format dataset."""
    all_qpos = []
    all_action = []

    for ep_path in episode_paths:
        with h5py.File(ep_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            action = root["/action"][()]
            all_qpos.append(qpos)
            all_action.append(action)

    # Pad arrays to same length for stacking
    max_len_qpos = max(arr.shape[0] for arr in all_qpos)
    max_len_action = max(arr.shape[0] for arr in all_action)

    padded_qpos = []
    padded_action = []
    for qpos, action in zip(all_qpos, all_action):
        if qpos.shape[0] < max_len_qpos:
            pad_width = [(0, max_len_qpos - qpos.shape[0]), (0, 0)]
            qpos = np.pad(qpos, pad_width, mode='constant', constant_values=0)
        padded_qpos.append(qpos)

        if action.shape[0] < max_len_action:
            pad_width = [(0, max_len_action - action.shape[0]), (0, 0)]
            action = np.pad(action, pad_width, mode='constant', constant_values=0)
        padded_action.append(action)

    all_qpos = np.stack(padded_qpos)
    all_action = np.stack(padded_action)

    action_mean = all_action.mean(axis=(0, 1))
    action_std = all_action.std(axis=(0, 1)).clip(min=1e-2)

    qpos_mean = all_qpos.mean(axis=(0, 1))
    qpos_std = all_qpos.std(axis=(0, 1)).clip(min=1e-2)

    return {
        "action_mean": action_mean,
        "action_std": action_std,
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "num_episodes": len(episode_paths),
    }


def convert_robottwin_to_act(config: ConversionConfig) -> dict:
    """Convert RoboTwin episodes to ACT format.

    Creates directory structure:
        output_dir/
            train/
                episode_0.hdf5
                episode_1.hdf5
                ...
            val/
                episode_0.hdf5
                ...
            test/
                episode_0.hdf5
                ...

    Also saves split metadata and stats.
    """
    print(f"Loading RoboTwin episodes from {config.data_dir}...")
    episodes = load_episodes(config)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        raise RuntimeError("No valid episodes found")

    print(f"Splitting episodes (train={config.train_ratio}, val={config.val_ratio}, test={config.test_ratio})...")
    split = split_episodes(
        episodes,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.split_seed,
    )

    for split_name, eps in split.items():
        print(f"  {split_name}: {len(eps)} episodes")

    output_base = Path(config.output_dir)
    split_metadata = {}

    for split_name, ep_list in split.items():
        split_dir = output_base / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_episode_paths = []
        for ep_idx, episode in enumerate(ep_list):
            hdf5_path = split_dir / f"episode_{ep_idx}.hdf5"
            print(f"  Converting {episode.task_name}/{episode.episode_id} -> {hdf5_path}")
            episode_to_hdf5(episode, hdf5_path, config)
            split_episode_paths.append(hdf5_path)

        split_metadata[split_name] = {
            "num_episodes": len(ep_list),
            "episode_ids": [f"{e.task_name}/{e.episode_id}" for e in ep_list],
        }

        stats = get_dataset_stats(split_episode_paths, config)
        stats_path = split_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump({k: v.tolist() if hasattr(v, "tolist") else v for k, v in stats.items()}, f, indent=2)
        print(f"  Saved stats to {stats_path}")

    metadata = {
        "split_metadata": split_metadata,
        "config": {
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "camera_names": config.camera_names,
            "image_size": config.image_size,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "split_seed": config.split_seed,
        },
        "source": "RoboTwin offline episodes",
        "note": (
            f"Dimension mapping: RoboTwin 16D -> ACT {config.action_dim}D (left arm 7DoF). "
            f"Use left_arm[:{config.action_dim}] from RoboTwin samples as qpos and action."
        ),
    }

    metadata_path = output_base / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print(f"\nConversion complete!")
    print(f"Output directory: {output_base}")
    print(f"  train/: {len(split['train'])} episodes")
    print(f"  val/: {len(split['val'])} episodes")
    print(f"  test/: {len(split['test'])} episodes")

    return metadata


def parse_args() -> ConversionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/robottwin_hf")
    parser.add_argument("--output-dir", default="data/act_format")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=10)
    parser.add_argument("--state-dim", type=int, default=STATE_DIM)
    parser.add_argument("--action-dim", type=int, default=ACTION_DIM)
    parser.add_argument("--camera-names", nargs="+", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=2026)
    args = parser.parse_args()

    return ConversionConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tasks=args.tasks,
        max_episodes_per_task=args.max_episodes_per_task,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        camera_names=args.camera_names,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )


def main() -> None:
    config = parse_args()
    convert_robottwin_to_act(config)


if __name__ == "__main__":
    main()
