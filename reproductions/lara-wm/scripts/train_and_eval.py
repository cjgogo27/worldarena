#!/usr/bin/env python
# pyright: reportMissingImports=false, reportExplicitAny=false, reportAny=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUntypedBaseClass=false, reportMissingSuperCall=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false
"""Offline RoboTwin training/evaluation on real observations.

This script replaces the old toy experiment with a reproducible offline pipeline
that:
- loads real RoboTwin head-camera frames and actions from the zip reader
- performs fair task-stratified episode splits
- trains comparable models on held-out offline data
- reports only defensible offline metrics
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backbone.adapter import BackboneAdapter
from backbone.config import BackboneConfig
from data.robottwin_zip_reader import RoboTwinDataLoader, RoboTwinEpisode, RoboTwinSample


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TASKS = [
    "grab_roller",
    "place_a2b_left",
    "stack_blocks_two",
    "open_laptop",
    "adjust_bottle",
    "beat_block_hammer",
    "click_bell",
    "dump_bin_bigbin",
    "press_stapler",
]


@dataclass(slots=True)
class ExperimentConfig:
    data_dir: str = "data/robottwin_hf"
    tasks: list[str] = field(default_factory=lambda: list(DEFAULT_TASKS))
    output_dir: str = "experiments/results"
    max_episodes_per_task: int = 10
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    latent_dim: int = 128
    hidden_dim: int = 256
    image_size: int = 128
    feature_mode: str = "auto"
    backbone_batch_size: int = 8
    backbone_cache_dir: str = "processed/offline_backbone_cache"
    backbone_model_path: str | None = None
    backbone_dtype: str = "float16"
    backbone_device: str | None = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 2026
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    num_workers: int = 0
    device: str = DEFAULT_DEVICE
    world_model_weight: float = 0.5
    refined_action_weight: float = 0.5
    latent_refine_weight: float = 0.25


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/robottwin_hf")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--max-episodes-per-task", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--feature-mode", choices=["auto", "backbone", "lightweight"], default="auto")
    parser.add_argument("--backbone-batch-size", type=int, default=8)
    parser.add_argument("--backbone-cache-dir", default="processed/offline_backbone_cache")
    parser.add_argument("--backbone-model-path", default=None)
    parser.add_argument("--backbone-dtype", default="float16")
    parser.add_argument("--backbone-device", default=None)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Task names to load from RoboTwin",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Training seeds. Split seed stays fixed unless overridden.",
    )
    args = parser.parse_args()
    return ExperimentConfig(
        data_dir=args.data_dir,
        tasks=args.tasks,
        output_dir=args.output_dir,
        max_episodes_per_task=args.max_episodes_per_task,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        image_size=args.image_size,
        feature_mode=args.feature_mode,
        backbone_batch_size=args.backbone_batch_size,
        backbone_cache_dir=args.backbone_cache_dir,
        backbone_model_path=args.backbone_model_path,
        backbone_dtype=args.backbone_dtype,
        backbone_device=args.backbone_device,
        split_seed=args.split_seed,
        seeds=args.seeds,
        num_workers=args.num_workers,
        device=args.device,
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preprocess_image(image: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return tensor.squeeze(0)


def make_transition_key(episode: RoboTwinEpisode, timestep: int) -> str:
    return f"{episode.task_name}:{episode.episode_id}:{timestep}"


def sample_is_valid(sample: RoboTwinSample) -> bool:
    return sample.head_image is not None and sample.action is not None


def episode_has_valid_transition(episode: RoboTwinEpisode) -> bool:
    return any(
        sample_is_valid(episode.samples[index]) and sample_is_valid(episode.samples[index + 1])
        for index in range(len(episode.samples) - 1)
    )


def load_real_episodes(config: ExperimentConfig) -> list[RoboTwinEpisode]:
    loader = RoboTwinDataLoader(
        data_dir=config.data_dir,
        tasks=config.tasks,
        batch_size=config.batch_size,
        include_images=True,
        max_episodes_per_task=config.max_episodes_per_task,
    )
    episodes: list[RoboTwinEpisode] = []
    for episode in loader.episodes:
        if episode_has_valid_transition(episode):
            episodes.append(episode)
    if not episodes:
        raise RuntimeError("No valid RoboTwin episodes with head-camera image/action transitions were found.")
    return episodes


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
    *,
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


class RoboTwinTransitionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        episodes: list[RoboTwinEpisode],
        *,
        image_size: int,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        self.image_size = image_size
        self.action_mean = torch.from_numpy(action_mean).float()
        self.action_std = torch.from_numpy(action_std).float()
        self.records: list[tuple[RoboTwinEpisode, int]] = []

        for episode in episodes:
            for index in range(len(episode.samples) - 1):
                current = episode.samples[index]
                nxt = episode.samples[index + 1]
                if sample_is_valid(current) and sample_is_valid(nxt):
                    self.records.append((episode, index))

        if not self.records:
            raise RuntimeError("Split produced zero valid image/action transitions.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, timestep = self.records[index]
        current = episode.samples[timestep]
        nxt = episode.samples[timestep + 1]
        action = torch.from_numpy(np.asarray(current.action, dtype=np.float32))
        normalized_action = (action - self.action_mean) / self.action_std
        return {
            "transition_id": make_transition_key(episode, timestep),
            "image": preprocess_image(current.head_image, self.image_size),
            "next_image": preprocess_image(nxt.head_image, self.image_size),
            "action": action,
            "action_normalized": normalized_action,
            "task_name": episode.task_name,
            "episode_id": f"{episode.task_name}:{episode.episode_id}",
            "timestep": timestep,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "transition_id": [item["transition_id"] for item in batch],
        "image": torch.stack([item["image"] for item in batch]),
        "next_image": torch.stack([item["next_image"] for item in batch]),
        "action": torch.stack([item["action"] for item in batch]),
        "action_normalized": torch.stack([item["action_normalized"] for item in batch]),
        "task_name": [item["task_name"] for item in batch],
        "episode_id": [item["episode_id"] for item in batch],
        "timestep": [item["timestep"] for item in batch],
    }


def compute_action_stats(episodes: list[RoboTwinEpisode]) -> tuple[np.ndarray, np.ndarray]:
    actions: list[np.ndarray] = []
    for episode in episodes:
        for index in range(len(episode.samples) - 1):
            current = episode.samples[index]
            nxt = episode.samples[index + 1]
            if sample_is_valid(current) and sample_is_valid(nxt):
                actions.append(np.asarray(current.action, dtype=np.float32))

    if not actions:
        raise RuntimeError("No valid actions available to compute train-set normalization.")

    stacked = np.stack(actions, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def summarize_split(split: dict[str, list[RoboTwinEpisode]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, episodes in split.items():
        per_task_episodes: dict[str, int] = defaultdict(int)
        per_task_transitions: dict[str, int] = defaultdict(int)
        total_transitions = 0
        for episode in episodes:
            per_task_episodes[episode.task_name] += 1
            count = 0
            for index in range(len(episode.samples) - 1):
                if sample_is_valid(episode.samples[index]) and sample_is_valid(episode.samples[index + 1]):
                    count += 1
            per_task_transitions[episode.task_name] += count
            total_transitions += count
        summary[split_name] = {
            "num_episodes": len(episodes),
            "num_transitions": total_transitions,
            "episodes_per_task": dict(sorted(per_task_episodes.items())),
            "transitions_per_task": dict(sorted(per_task_transitions.items())),
        }
    return summary


def build_backbone_adapter(config: ExperimentConfig) -> BackboneAdapter:
    backbone_device = config.backbone_device or config.device
    if config.backbone_model_path:
        model_path = config.backbone_model_path
    else:
        backbone_config = BackboneConfig.from_defaults()
        model_path = backbone_config.get_working_path()
        if model_path is None:
            raise RuntimeError("No local backbone model path is available.")

    adapter = BackboneAdapter(
        model_path=model_path,
        model_name=Path(model_path).name,
        device=backbone_device,
        dtype=config.backbone_dtype,
    )
    adapter.load()
    return adapter


def build_cache_stem(config: ExperimentConfig, split: dict[str, list[RoboTwinEpisode]]) -> str:
    payload = {
        "tasks": sorted(config.tasks),
        "max_episodes_per_task": config.max_episodes_per_task,
        "image_size": config.image_size,
        "split_seed": config.split_seed,
        "split_summary": summarize_split(split),
        "backbone_model_path": config.backbone_model_path,
        "backbone_dtype": config.backbone_dtype,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"robottwin_{digest}"


def extract_shared_backbone_features(
    split: dict[str, list[RoboTwinEpisode]],
    config: ExperimentConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    cache_dir = PROJECT_ROOT / config.backbone_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_stem = build_cache_stem(config, split)
    cache_path = cache_dir / f"{cache_stem}.pt"

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        cached_metadata = cached.get("metadata", {})
        if cached_metadata.get("split_summary") == summarize_split(split):
            return cached["features"], {"cache_path": str(cache_path), **cached_metadata, "loaded_from_cache": True}

    adapter = build_backbone_adapter(config)
    adapter_device = torch.device(adapter.device)
    feature_bank: dict[str, dict[str, torch.Tensor]] = {name: {} for name in split}
    feature_dim: int | None = None

    for split_name, episodes in split.items():
        batch_images: list[torch.Tensor] = []
        batch_keys: list[tuple[str, bool]] = []
        for episode in episodes:
            for index in range(len(episode.samples) - 1):
                current = episode.samples[index]
                nxt = episode.samples[index + 1]
                if not (sample_is_valid(current) and sample_is_valid(nxt)):
                    continue
                transition_id = make_transition_key(episode, index)
                batch_images.append(preprocess_image(current.head_image, config.image_size))
                batch_keys.append((transition_id, False))
                batch_images.append(preprocess_image(nxt.head_image, config.image_size))
                batch_keys.append((transition_id, True))

                if len(batch_images) >= config.backbone_batch_size:
                    image_batch = torch.stack(batch_images, dim=0).to(adapter_device)
                    features = adapter.encode_image(image_batch).detach().cpu()
                    feature_dim = int(features.shape[-1])
                    for feature_index, (transition_id, is_next) in enumerate(batch_keys):
                        bank_entry = feature_bank[split_name].setdefault(transition_id, {})
                        bank_entry["next_feature" if is_next else "feature"] = features[feature_index]
                    batch_images = []
                    batch_keys = []

        if batch_images:
            image_batch = torch.stack(batch_images, dim=0).to(adapter_device)
            features = adapter.encode_image(image_batch).detach().cpu()
            feature_dim = int(features.shape[-1])
            for feature_index, (transition_id, is_next) in enumerate(batch_keys):
                bank_entry = feature_bank[split_name].setdefault(transition_id, {})
                bank_entry["next_feature" if is_next else "feature"] = features[feature_index]

    if feature_dim is None:
        raise RuntimeError("Shared backbone extraction produced no features.")

    metadata = {
        "extractor_type": "shared_local_backbone",
        "feature_dim": feature_dim,
        "model_path": adapter.model_path,
        "model_name": adapter.model_name,
        "dtype": config.backbone_dtype,
        "device": adapter.device,
        "image_size": config.image_size,
        "split_summary": summarize_split(split),
        "loaded_from_cache": False,
    }
    torch.save({"features": feature_bank, "metadata": metadata}, cache_path)
    return feature_bank, {"cache_path": str(cache_path), **metadata}


def build_lightweight_feature_bank(
    split: dict[str, list[RoboTwinEpisode]],
    config: ExperimentConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    feature_bank: dict[str, dict[str, torch.Tensor]] = {name: {} for name in split}
    feature_dim = 3 * config.image_size * config.image_size
    for split_name, episodes in split.items():
        for episode in episodes:
            for index in range(len(episode.samples) - 1):
                current = episode.samples[index]
                nxt = episode.samples[index + 1]
                if not (sample_is_valid(current) and sample_is_valid(nxt)):
                    continue
                transition_id = make_transition_key(episode, index)
                feature_bank[split_name][transition_id] = {
                    "feature": preprocess_image(current.head_image, config.image_size).reshape(-1).cpu(),
                    "next_feature": preprocess_image(nxt.head_image, config.image_size).reshape(-1).cpu(),
                }
    return feature_bank, {
        "extractor_type": "lightweight_flattened_pixels",
        "feature_dim": feature_dim,
        "image_size": config.image_size,
        "loaded_from_cache": False,
    }


def build_shared_feature_bank(
    split: dict[str, list[RoboTwinEpisode]],
    config: ExperimentConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    if config.feature_mode == "lightweight":
        return build_lightweight_feature_bank(split, config)
    if config.feature_mode == "backbone":
        return extract_shared_backbone_features(split, config)

    try:
        return extract_shared_backbone_features(split, config)
    except Exception as error:
        print(f"Backbone feature extraction unavailable, falling back to lightweight features: {error}")
        return build_lightweight_feature_bank(split, config)


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DirectPolicyModel(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, latent_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = FeatureProjector(feature_dim, latent_dim, hidden_dim)
        self.head = MLP(latent_dim, hidden_dim, action_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(features)
        action = self.head(latent)
        return {"latent": latent, "action": action}


class LatentNoRefineModel(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, latent_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = FeatureProjector(feature_dim, latent_dim, hidden_dim)
        self.decoder = MLP(latent_dim, hidden_dim, action_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(features)
        action = self.decoder(latent)
        return {"latent": latent, "action": action}


class NoRewardWMModel(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, latent_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = FeatureProjector(feature_dim, latent_dim, hidden_dim)
        self.action_decoder = MLP(latent_dim, hidden_dim, action_dim)
        self.transition = MLP(latent_dim + action_dim, hidden_dim, latent_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(
        self,
        features: torch.Tensor,
        transition_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode(features)
        action = self.action_decoder(latent)
        outputs = {"latent": latent, "action": action}
        if transition_action is not None:
            outputs["pred_next_latent"] = self.transition(torch.cat([latent, transition_action], dim=-1))
        return outputs


class LaraWMModel(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, latent_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = FeatureProjector(feature_dim, latent_dim, hidden_dim)
        self.action_decoder = MLP(latent_dim, hidden_dim, action_dim)
        self.transition = MLP(latent_dim + action_dim, hidden_dim, latent_dim)
        self.refiner = MLP(latent_dim * 3, hidden_dim, latent_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(
        self,
        features: torch.Tensor,
        transition_action: torch.Tensor | None = None,
        next_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode(features)
        action = self.action_decoder(latent)
        outputs = {"latent": latent, "action": action}

        if transition_action is not None:
            pred_next_latent = self.transition(torch.cat([latent, transition_action], dim=-1))
            outputs["pred_next_latent"] = pred_next_latent

            if next_features is not None:
                target_next_latent = self.encode(next_features).detach()
                refined_latent = latent + self.refiner(
                    torch.cat([latent, pred_next_latent, target_next_latent], dim=-1)
                )
                outputs["target_next_latent"] = target_next_latent
                outputs["refined_latent"] = refined_latent
                outputs["refined_action"] = self.action_decoder(refined_latent)

        return outputs


class ACTModel(nn.Module):
    """ACT (Action Chunking Transformer) model for offline evaluation.

    This mimics ACT behavior:
    - Encodes visual features to latent
    - Predicts action chunk (multiple steps)
    - Uses temporal consistency for smoother trajectories

    Key difference from other models:
    - Predicts chunk_size actions (temporal chunking)
    - First action used for evaluation (matches other baselines)
    """

    def __init__(self, action_dim: int, hidden_dim: int, latent_dim: int, feature_dim: int, chunk_size: int = 100):
        super().__init__()
        self.chunk_size = chunk_size
        self.encoder = FeatureProjector(feature_dim, latent_dim, hidden_dim)
        self.action_decoder = MLP(latent_dim, hidden_dim, action_dim * chunk_size)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(
        self,
        features: torch.Tensor,
        transition_action: torch.Tensor | None = None,
        next_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode(features)
        action_flat = self.action_decoder(latent)
        action_chunk = action_flat.view(action_flat.size(0), self.chunk_size, -1)
        action = action_chunk[:, 0, :]
        outputs = {"latent": latent, "action": action, "action_chunk": action_chunk}
        return outputs


def instantiate_model(model_name: str, action_dim: int, feature_dim: int, config: ExperimentConfig) -> nn.Module:
    if model_name == "direct_policy":
        return DirectPolicyModel(action_dim, config.hidden_dim, config.latent_dim, feature_dim)
    if model_name == "latent_no_refine":
        return LatentNoRefineModel(action_dim, config.hidden_dim, config.latent_dim, feature_dim)
    if model_name == "no_reward_wm":
        return NoRewardWMModel(action_dim, config.hidden_dim, config.latent_dim, feature_dim)
    if model_name == "lara-wm":
        return LaraWMModel(action_dim, config.hidden_dim, config.latent_dim, feature_dim)
    if model_name == "act":
        return ACTModel(action_dim, config.hidden_dim, config.latent_dim, feature_dim)
    raise ValueError(f"Unknown model: {model_name}")


def detach_cpu(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def denormalize_action(
    normalized_action: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> torch.Tensor:
    return normalized_action * action_std.to(normalized_action.device) + action_mean.to(normalized_action.device)


def compute_action_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    residual = predictions - targets
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    target_mean = np.mean(targets, axis=0, keepdims=True)
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(targets - target_mean)))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {
        "action_mse": mse,
        "action_mae": mae,
        "action_r2": r2,
    }


def compute_latent_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    mse = float(np.mean(np.square(predictions - targets)))
    pred_norm = np.linalg.norm(predictions, axis=-1)
    target_norm = np.linalg.norm(targets, axis=-1)
    cosine = np.sum(predictions * targets, axis=-1) / np.clip(pred_norm * target_norm, 1e-8, None)
    return {
        "next_latent_mse": mse,
        "next_latent_cosine": float(np.mean(cosine)),
    }


def average_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = sorted({key for metric_dict in metric_dicts for key in metric_dict})
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts if key in metrics])) for key in keys}


def attach_batch_features(
    batch: dict[str, Any],
    feature_bank: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.stack([feature_bank[transition_id]["feature"] for transition_id in batch["transition_id"]], dim=0)
    next_features = torch.stack(
        [feature_bank[transition_id]["next_feature"] for transition_id in batch["transition_id"]],
        dim=0,
    )
    return features.to(device=device, dtype=torch.float32), next_features.to(device=device, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    model_name: str,
    config: ExperimentConfig,
    device: torch.device,
    feature_bank: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    model.train()
    metrics: list[dict[str, float]] = []

    for batch in loader:
        features, next_features = attach_batch_features(batch, feature_bank, device)
        action_norm = batch["action_normalized"].to(device)

        optimizer.zero_grad()

        if model_name == "direct_policy":
            outputs = model(features)
            action_loss = F.mse_loss(outputs["action"], action_norm)
            loss = action_loss
            batch_metrics = {"train_loss": float(loss.item()), "train_action_loss": float(action_loss.item())}
        elif model_name == "latent_no_refine":
            outputs = model(features)
            action_loss = F.mse_loss(outputs["action"], action_norm)
            loss = action_loss
            batch_metrics = {"train_loss": float(loss.item()), "train_action_loss": float(action_loss.item())}
        elif model_name == "no_reward_wm":
            outputs = model(features, transition_action=action_norm)
            with torch.no_grad():
                target_next_latent = model.encode(next_features)
            action_loss = F.mse_loss(outputs["action"], action_norm)
            world_loss = F.mse_loss(outputs["pred_next_latent"], target_next_latent)
            loss = action_loss + config.world_model_weight * world_loss
            batch_metrics = {
                "train_loss": float(loss.item()),
                "train_action_loss": float(action_loss.item()),
                "train_next_latent_loss": float(world_loss.item()),
            }
        elif model_name == "lara-wm":
            outputs = model(features, transition_action=action_norm, next_features=next_features)
            action_loss = F.mse_loss(outputs["action"], action_norm)
            world_loss = F.mse_loss(outputs["pred_next_latent"], outputs["target_next_latent"])
            refined_action_loss = F.mse_loss(outputs["refined_action"], action_norm)
            refine_latent_loss = F.mse_loss(outputs["refined_latent"], outputs["target_next_latent"])
            loss = (
                action_loss
                + config.world_model_weight * world_loss
                + config.refined_action_weight * refined_action_loss
                + config.latent_refine_weight * refine_latent_loss
            )
            batch_metrics = {
                "train_loss": float(loss.item()),
                "train_action_loss": float(action_loss.item()),
                "train_next_latent_loss": float(world_loss.item()),
                "train_refined_action_loss": float(refined_action_loss.item()),
            }
        elif model_name == "act":
            outputs = model(features)
            action_loss = F.mse_loss(outputs["action"], action_norm)
            loss = action_loss
            batch_metrics = {"train_loss": float(loss.item()), "train_action_loss": float(action_loss.item())}
        else:
            raise ValueError(f"Unknown model: {model_name}")

        loss.backward()
        optimizer.step()
        metrics.append(batch_metrics)

    return average_metric_dicts(metrics)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    model_name: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
    feature_bank: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float | Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    next_latent_predictions: list[np.ndarray] = []
    next_latent_targets: list[np.ndarray] = []
    refined_predictions: list[np.ndarray] = []

    # Per-task accumulators: task_name -> list of (prediction, target) pairs
    per_task_preds: dict[str, list[np.ndarray]] = defaultdict(list)
    per_task_targets: dict[str, list[np.ndarray]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            features, next_features = attach_batch_features(batch, feature_bank, device)
            action = batch["action"].to(device)
            action_norm = batch["action_normalized"].to(device)

            if model_name == "direct_policy":
                outputs = model(features)
            elif model_name == "latent_no_refine":
                outputs = model(features)
            elif model_name == "no_reward_wm":
                outputs = model(features, transition_action=action_norm)
                target_next_latent = model.encode(next_features)
                next_latent_predictions.append(detach_cpu(outputs["pred_next_latent"]))
                next_latent_targets.append(detach_cpu(target_next_latent))
            elif model_name == "lara-wm":
                outputs = model(features, transition_action=action_norm, next_features=next_features)
                next_latent_predictions.append(detach_cpu(outputs["pred_next_latent"]))
                next_latent_targets.append(detach_cpu(outputs["target_next_latent"]))
                refined_predictions.append(
                    detach_cpu(denormalize_action(outputs["refined_action"], action_mean, action_std))
                )
            elif model_name == "act":
                outputs = model(features)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            pred_action = denormalize_action(outputs["action"], action_mean, action_std)
            predictions.append(detach_cpu(pred_action))
            targets.append(detach_cpu(action))

            # Collect per-task predictions for fair (non-privileged) metrics
            task_names = batch["task_name"]
            pred_np = detach_cpu(pred_action)
            action_np = detach_cpu(action)
            for i, task_name in enumerate(task_names):
                per_task_preds[task_name].append(pred_np[i])
                per_task_targets[task_name].append(action_np[i])

    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    metrics: dict[str, float | Any] = compute_action_metrics(prediction_array, target_array)

    # Per-task fair metrics (main paper metrics: action_mse, action_mae, action_r2)
    per_task_metrics: dict[str, dict[str, float]] = {}
    for task_name in sorted(per_task_preds):
        task_pred = np.concatenate(per_task_preds[task_name], axis=0)
        task_target = np.concatenate(per_task_targets[task_name], axis=0)
        per_task_metrics[task_name] = compute_action_metrics(task_pred, task_target)

    if next_latent_predictions:
        metrics.update(
            compute_latent_metrics(
                np.concatenate(next_latent_predictions, axis=0),
                np.concatenate(next_latent_targets, axis=0),
            )
        )

    if refined_predictions:
        refined_array = np.concatenate(refined_predictions, axis=0)
        refined_metrics = compute_action_metrics(refined_array, target_array)
        # Mark as privileged/diagnostic — excluded from main paper summaries
        metrics.update({f"priv_offline_refined_{key}": value for key, value in refined_metrics.items()})

    # Attach per-task breakdown to metrics for downstream aggregation
    metrics["_per_task"] = cast(dict[str, Any], per_task_metrics)

    return metrics


def train_and_select(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader[dict[str, Any]],
    val_loader: DataLoader[dict[str, Any]],
    train_feature_bank: dict[str, dict[str, torch.Tensor]],
    val_feature_bank: dict[str, dict[str, torch.Tensor]],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val_mse = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(config.num_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, model_name, config, device, train_feature_bank)
        val_metrics = evaluate_model(model, val_loader, model_name, action_mean, action_std, device, val_feature_bank)
        epoch_metrics = {"epoch": epoch + 1, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(epoch_metrics)

        if val_metrics["action_mse"] < best_val_mse:
            best_val_mse = val_metrics["action_mse"]
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"  epoch {epoch + 1:02d}/{config.num_epochs}: "
            + f"train_loss={train_metrics.get('train_loss', float('nan')):.4f} "
            + f"val_action_mse={val_metrics['action_mse']:.4f} "
            + f"val_action_mae={val_metrics['action_mae']:.4f}"
        )

    model.load_state_dict(best_state)
    return model, history


def build_dataloaders(
    split: dict[str, list[RoboTwinEpisode]],
    config: ExperimentConfig,
) -> tuple[
    dict[str, DataLoader[dict[str, Any]]],
    torch.Tensor,
    torch.Tensor,
    dict[str, dict[str, dict[str, torch.Tensor]]],
    dict[str, Any],
]:
    action_mean_np, action_std_np = compute_action_stats(split["train"])
    feature_bank, feature_metadata = build_shared_feature_bank(split, config)
    datasets = {
        name: RoboTwinTransitionDataset(
            episodes,
            image_size=config.image_size,
            action_mean=action_mean_np,
            action_std=action_std_np,
        )
        for name, episodes in split.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            num_workers=config.num_workers,
            pin_memory=config.device.startswith("cuda"),
            collate_fn=collate_batch,
        )
        for name, dataset in datasets.items()
    }
    return (
        loaders,
        torch.from_numpy(action_mean_np).float(),
        torch.from_numpy(action_std_np).float(),
        feature_bank,
        feature_metadata,
    )


def aggregate_seed_metrics(seed_results: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    model_names = sorted(next(iter(seed_results.values())).keys())
    aggregate: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        metric_keys: set[str] = set()
        for results in seed_results.values():
            for key in results[model_name]:
                if not key.startswith("_"):
                    metric_keys.add(key)
        metric_key_list: list[str] = sorted(metric_keys)
        model_metrics: dict[str, float] = {}
        for metric_key in metric_key_list:
            values: list[float] = []
            for results in seed_results.values():
                val: Any = results[model_name].get(metric_key)
                if isinstance(val, (float, int)):
                    values.append(float(val))
            if values:
                model_metrics[metric_key] = float(np.mean(values))
                model_metrics[f"std_{metric_key}"] = float(np.std(values))
        aggregate[model_name] = model_metrics
    return aggregate


def extract_per_task_metrics(
    seed_results: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Extract per-task metrics from seed results, stripping the _per_task helper key."""
    model_names = sorted(next(iter(seed_results.values())).keys())
    task_names = sorted(
        set(
            task
            for results in seed_results.values()
            for model_metrics in results.values()
            for task in cast(dict[str, Any], model_metrics.get("_per_task", {}))
        )
    )
    per_task: dict[str, dict[str, dict[str, float]]] = {
        model_name: {} for model_name in model_names
    }
    for model_name in model_names:
        for task_name in task_names:
            task_metrics: dict[str, float] = {}
            task_metric_keys = ["action_mse", "action_mae", "action_r2"]
            for metric_key in task_metric_keys:
                values: list[float] = []
                for results in seed_results.values():
                    per_task_dict: dict[str, Any] = results[model_name]
                    task_dict = per_task_dict.get("_per_task", {})
                    task_inner = task_dict.get(task_name, {})
                    val: Any = task_inner.get(metric_key)
                    if isinstance(val, (float, int)):
                        values.append(float(val))
                if values:
                    task_metrics[metric_key] = float(np.mean(values))
                    task_metrics[f"std_{metric_key}"] = float(np.std(values))
            per_task[model_name][task_name] = task_metrics
    return per_task


def aggregate_per_task_across_models(
    per_task: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Compute overall mean/std for each metric across all models per task."""
    if not per_task:
        return {}
    task_names = sorted(next(iter(per_task.values())).keys())
    metric_keys = ["action_mse", "action_mae", "action_r2"]
    overall: dict[str, dict[str, float]] = {}
    for task_name in task_names:
        task_overall: dict[str, float] = {}
        for metric_key in metric_keys:
            values: list[float] = []
            for model_name in per_task:
                val = per_task[model_name][task_name].get(metric_key)
                if isinstance(val, (float, int)):
                    values.append(float(val))
            if values:
                task_overall[metric_key] = float(np.mean(values))
                task_overall[f"std_{metric_key}"] = float(np.std(values))
        overall[task_name] = task_overall
    return overall


def write_markdown_summary(path: Path, aggregate_metrics: dict[str, dict[str, float]]) -> None:
    # Paper-safe columns: only fair metrics, no privileged/diagnostic metrics
    fair_columns = ["action_mse", "action_mae", "action_r2"]
    available_columns = [
        column for column in fair_columns if any(column in metrics for metrics in aggregate_metrics.values())
    ]
    lines = [
        "# Offline RoboTwin Results (Paper-Safe)",
        "",
        "## Main Results (Fair Metrics Only)",
        "",
        "| Model | " + " | ".join(available_columns) + " |",
        "|---|" + "---|" * len(available_columns),
    ]
    for model_name, metrics in aggregate_metrics.items():
        row = [model_name]
        for column in available_columns:
            if column in metrics:
                row.append(f"{metrics[column]:.4f} ± {metrics.get(f'std_{column}', 0.0):.4f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "## Diagnostic Metrics (Privileged — Not for Main Paper Tables)", ""])
    diag_columns = [
        col for col in ["next_latent_mse", "next_latent_cosine"]
        if any(col in metrics for metrics in aggregate_metrics.values())
    ]
    if diag_columns:
        lines.append("| Model | " + " | ".join(diag_columns) + " |")
        lines.append("|---|" + "---|" * len(diag_columns))
        for model_name, metrics in aggregate_metrics.items():
            row = [model_name]
            for column in diag_columns:
                if column in metrics:
                    row.append(f"{metrics[column]:.4f} ± {metrics.get(f'std_{column}', 0.0):.4f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    priv_columns = [
        col for col in sorted(
            set(key for metrics in aggregate_metrics.values() for key in metrics if key.startswith("priv_"))
        )
    ]
    if priv_columns:
        lines.extend(["### Privileged Refined Metrics (Offline Diagnostics)", ""])
        lines.append("| Model | " + " | ".join(priv_columns) + " |")
        lines.append("|---|" + "---|" * len(priv_columns))
        for model_name, metrics in aggregate_metrics.items():
            row = [model_name]
            for column in priv_columns:
                if column in metrics:
                    row.append(f"{metrics[column]:.4f} ± {metrics.get(f'std_{column}', 0.0):.4f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines.extend([
        "---",
        "*Fair metrics (action_mse, action_mae, action_r2) are computed from offline predictions using only current observations.*",
        "*Privileged metrics (priv_offline_refined_*) use held-out next observations at evaluation time and are diagnostic only.*",
    ])
    path.write_text("\n".join(lines) + "\n")


def write_per_task_table(
    path: Path,
    per_task: dict[str, dict[str, dict[str, float]]],
    overall: dict[str, dict[str, float]],
) -> None:
    if not per_task:
        return
    task_names = sorted(next(iter(per_task.values())).keys())
    model_names = sorted(per_task.keys())
    metric_keys = ["action_mse", "action_mae", "action_r2"]
    lines = [
        "# Per-Task Fair Metrics",
        "",
        "Fair metrics only (action_mse, action_mae, action_r2); privileged refined metrics excluded.",
        "",
    ]
    for metric_key in metric_keys:
        lines.append(f"## {metric_key}")
        lines.append("")
        col_header = "| Task | " + " | ".join(model_names) + " |"
        lines.append(col_header)
        lines.append("|---|" + "---|" * len(model_names))
        for task_name in task_names:
            row = [task_name]
            for model_name in model_names:
                if metric_key in per_task[model_name].get(task_name, {}):
                    mean = per_task[model_name][task_name][metric_key]
                    std = per_task[model_name][task_name].get(f"std_{metric_key}", 0.0)
                    row.append(f"{mean:.4f} ± {std:.4f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    if overall:
        lines.append("## Per-Task Overall Mean Across Models")
        lines.append("")
        lines.append("| Task | action_mse | action_mae | action_r2 |")
        lines.append("|---|:---:|:---:|:---:|")
        for task_name in task_names:
            if task_name in overall:
                mse = overall[task_name].get("action_mse", float("nan"))
                mae = overall[task_name].get("action_mae", float("nan"))
                r2 = overall[task_name].get("action_r2", float("nan"))
                lines.append(f"| {task_name} | {mse:.4f} ± {overall[task_name].get('std_action_mse', 0.0):.4f} | {mae:.4f} ± {overall[task_name].get('std_action_mae', 0.0):.4f} | {r2:.4f} ± {overall[task_name].get('std_action_r2', 0.0):.4f} |")
        lines.append("")
    lines.extend([
        "---",
        "*Fair metrics only; privileged refined metrics excluded.*",
        f"*Tasks: {', '.join(task_names)}*",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> dict[str, Any]:
    config = parse_args()
    device = torch.device(config.device)
    print(f"Device: {device}")
    print(f"Tasks: {', '.join(config.tasks)}")
    print(f"Training seeds: {config.seeds} | split seed: {config.split_seed}")

    episodes = load_real_episodes(config)
    split = split_episodes(
        episodes,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.split_seed,
    )
    split_summary = summarize_split(split)
    for split_name, summary in split_summary.items():
        print(
            f"{split_name:>5s}: episodes={summary['num_episodes']}, "
            + f"transitions={summary['num_transitions']}, "
            + f"per_task={summary['episodes_per_task']}"
        )

    loaders, action_mean, action_std, feature_bank, feature_metadata = build_dataloaders(split, config)
    action_dim = int(action_mean.numel())
    feature_dim = int(feature_metadata["feature_dim"])
    action_mean = action_mean.to(device)
    action_std = action_std.to(device)

    print(
        f"Feature mode: {config.feature_mode} -> {feature_metadata['extractor_type']} "
        + f"(dim={feature_dim})"
    )

    model_names = ["direct_policy", "latent_no_refine", "no_reward_wm", "lara-wm", "act"]
    seed_results: dict[str, dict[str, dict[str, float]]] = {}
    histories: dict[str, dict[str, list[dict[str, float]]]] = {}

    for seed in config.seeds:
        print(f"\n{'=' * 72}\nSEED {seed}\n{'=' * 72}")
        set_global_seed(seed)
        seed_key = f"seed_{seed}"
        seed_results[seed_key] = {}
        histories[seed_key] = {}

        for model_name in model_names:
            print(f"\n--- training {model_name} ---")
            model = instantiate_model(model_name, action_dim, feature_dim, config).to(device)
            model, history = train_and_select(
                model,
                model_name,
                loaders["train"],
                loaders["val"],
                feature_bank["train"],
                feature_bank["val"],
                action_mean,
                action_std,
                config,
                device,
            )
            metrics = evaluate_model(
                model,
                loaders["test"],
                model_name,
                action_mean,
                action_std,
                device,
                feature_bank["test"],
            )
            seed_results[seed_key][model_name] = metrics
            histories[seed_key][model_name] = history
            metric_bits = [f"{key}={value:.4f}" for key, value in metrics.items() if key in {"action_mse", "action_mae", "action_r2", "next_latent_mse"}]
            print(f"  test metrics: {', '.join(metric_bits)}")

    aggregate_metrics = aggregate_seed_metrics(seed_results)
    per_task_metrics = extract_per_task_metrics(seed_results)
    per_task_overall = aggregate_per_task_across_models(per_task_metrics)

    # Strip _per_task internal marker from stored seed results (backward-compatible)
    per_seed_clean: dict[str, dict[str, dict[str, float]]] = {
        seed_key: {
            model_name: {
                k: v for k, v in model_metrics.items() if not k.startswith("_")
            }
            for model_name, model_metrics in seed_metrics.items()
        }
        for seed_key, seed_metrics in seed_results.items()
    }

    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "device": str(device),
            "config": asdict(config),
            "split_summary": split_summary,
            "feature_pipeline": feature_metadata,
            "metric_notes": {
                # --- FAIR METRICS (main paper metrics) ---
                # All derived from offline evaluation without using held-out next observations at prediction time.
                "action_mse": "[FAIR] Mean squared error on held-out action predictions from current head-camera observations alone.",
                "action_mae": "[FAIR] Mean absolute error on held-out action predictions from current head-camera observations alone.",
                "action_r2": "[FAIR] Coefficient of determination on held-out action predictions.",
                # --- AUXILIARY LATENT METRICS ---
                "next_latent_mse": "One-step latent prediction error using the shared feature pipeline's encoded next head-camera observations as the target.",
                "next_latent_cosine": "Cosine similarity between predicted and encoded next-step latent observations.",
                # --- PRIVILEGED / DIAGNOSTIC METRICS (NOT for paper main tables) ---
                # offline_refined_action_* uses held-out next observations at evaluation time to refine latent;
                # these are offline-only diagnostics and do NOT reflect online task success.
                "priv_offline_refined_action_*": "[PRIVILEGED/DIAGNOSTIC] Refined action metrics that use held-out next observations during evaluation; excluded from main paper summaries.",
            },
        },
        "per_seed": per_seed_clean,
        "aggregate": aggregate_metrics,
        "per_task": per_task_metrics,
        "per_task_overall": per_task_overall,
        "training_history": histories,
    }

    json_path = output_dir / "real_training_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    write_markdown_summary(output_dir / "real_training_results.md", aggregate_metrics)
    write_per_task_table(output_dir / "real_training_results_per_task.md", per_task_metrics, per_task_overall)

    print(f"\nSaved JSON results to {json_path}")
    print(f"Saved markdown table to {output_dir / 'real_training_results.md'}")
    print(f"Saved per-task table to {output_dir / 'real_training_results_per_task.md'}")

    return results


if __name__ == "__main__":
    main()
