#!/usr/bin/env python3
from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitOverride=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false, reportMissingImports=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import train_and_eval as offline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DP_ROOT = PROJECT_ROOT / "third_party" / "diffusion_policy"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(DP_ROOT))

from src.baselines.robottwin_diffusion_policy import (  # noqa: E402
    RoboTwinDiffusionPolicy,
    RoboTwinDiffusionPolicyConfig,
    export_robottwin_diffusion_policy_checkpoint,
)


DEFAULT_TASKS = ["grab_roller", "adjust_bottle"]


@dataclass(slots=True)
class TrainConfig:
    data_dir: str = "data/robottwin_hf"
    tasks: list[str] | None = None
    output_dir: str = "experiments/diffusion_policy_robottwin"
    checkpoint_path: str = "experiments/rollout_ckpts/robottwin_diffusion_policy.pt"
    max_episodes_per_task: int = 10
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    image_size: int = 128
    split_seed: int = 2026
    train_seed: int = 42
    num_workers: int = 0
    device: str = offline.DEFAULT_DEVICE
    backbone_model_path: str | None = None
    backbone_dtype: str = "float16"
    backbone_device: str | None = None
    backbone_batch_size: int = 8
    cache_dir: str = "processed/offline_backbone_cache"
    num_inference_steps: int = 100


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a RoboTwin-compatible 16D multi-view diffusion policy.")
    parser.add_argument("--data-dir", default="data/robottwin_hf")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--output-dir", default="experiments/diffusion_policy_robottwin")
    parser.add_argument("--checkpoint-path", default="experiments/rollout_ckpts/robottwin_diffusion_policy.pt")
    parser.add_argument("--max-episodes-per-task", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=offline.DEFAULT_DEVICE)
    parser.add_argument("--backbone-model-path", default=None)
    parser.add_argument("--backbone-dtype", default="float16")
    parser.add_argument("--backbone-device", default=None)
    parser.add_argument("--backbone-batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", default="processed/offline_backbone_cache")
    parser.add_argument("--num-inference-steps", type=int, default=100)
    return TrainConfig(**vars(parser.parse_args()))


def sample_has_full_obs(sample: Any) -> bool:
    action = getattr(sample, "action", None)
    return (
        getattr(sample, "head_image", None) is not None
        and getattr(sample, "left_image", None) is not None
        and getattr(sample, "right_image", None) is not None
        and action is not None
        and np.asarray(action).shape[-1] >= 16
    )


def extract_state(sample: Any) -> np.ndarray:
    state = getattr(sample, "state", None)
    if state is not None and np.asarray(state).shape[-1] >= 16:
        return np.asarray(state, dtype=np.float32)[:16]
    return np.asarray(sample.action, dtype=np.float32)[:16]


def make_sample_key(episode: Any, timestep: int) -> str:
    return f"{episode.task_name}:{episode.episode_id}:{timestep}"


def build_feature_cache_stem(config: TrainConfig, split: dict[str, list[Any]]) -> str:
    payload = {
        "tasks": sorted(config.tasks or []),
        "max_episodes_per_task": config.max_episodes_per_task,
        "image_size": config.image_size,
        "split_seed": config.split_seed,
        "summary": offline.summarize_split(split),
        "backbone_model_path": config.backbone_model_path,
        "backbone_dtype": config.backbone_dtype,
        "variant": "robottwin_multiview_diffusion_policy",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"robottwin_multiview_dp_{digest}"


def build_multiview_feature_bank(
    split: dict[str, list[Any]],
    config: TrainConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], int, dict[str, Any]]:
    cache_dir = PROJECT_ROOT / config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{build_feature_cache_stem(config, split)}.pt"
    summary = offline.summarize_split(split)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        metadata = dict(cached.get("metadata", {}))
        if metadata.get("summary") == summary:
            return cached["features"], int(metadata["image_feature_dim"]), {**metadata, "loaded_from_cache": True}

    backbone_config = offline.ExperimentConfig(
        data_dir=config.data_dir,
        tasks=list(config.tasks or []),
        max_episodes_per_task=config.max_episodes_per_task,
        batch_size=config.batch_size,
        image_size=config.image_size,
        backbone_model_path=config.backbone_model_path,
        backbone_dtype=config.backbone_dtype,
        backbone_device=config.backbone_device,
        backbone_batch_size=config.backbone_batch_size,
        device=config.device,
        split_seed=config.split_seed,
        seeds=[config.train_seed],
    )
    adapter = offline.build_backbone_adapter(backbone_config)
    device = torch.device(adapter.device)
    feature_bank: dict[str, dict[str, torch.Tensor]] = {split_name: {} for split_name in split}
    image_feature_dim = 0
    for split_name, episodes in split.items():
        image_batch: list[torch.Tensor] = []
        batch_keys: list[tuple[str, int]] = []
        for episode in episodes:
            for timestep, sample in enumerate(episode.samples):
                if not sample_has_full_obs(sample):
                    continue
                image_batch.extend(
                    [
                        offline.preprocess_image(sample.head_image, config.image_size),
                        offline.preprocess_image(sample.left_image, config.image_size),
                        offline.preprocess_image(sample.right_image, config.image_size),
                    ]
                )
                batch_keys.append((make_sample_key(episode, timestep), 3))
                if len(batch_keys) >= config.backbone_batch_size:
                    stacked = torch.stack(image_batch, dim=0).to(device)
                    with torch.no_grad():
                        features = adapter.encode_image(stacked).detach().cpu().float()
                    view_dim = int(features.shape[-1])
                    image_feature_dim = view_dim * 3
                    cursor = 0
                    for record_key, num_views in batch_keys:
                        sample_features = features[cursor:cursor + num_views].reshape(-1)
                        feature_bank[split_name][record_key] = sample_features
                        cursor += num_views
                    image_batch = []
                    batch_keys = []
        if image_batch:
            stacked = torch.stack(image_batch, dim=0).to(device)
            with torch.no_grad():
                features = adapter.encode_image(stacked).detach().cpu().float()
            view_dim = int(features.shape[-1])
            image_feature_dim = view_dim * 3
            cursor = 0
            for record_key, num_views in batch_keys:
                sample_features = features[cursor:cursor + num_views].reshape(-1)
                feature_bank[split_name][record_key] = sample_features
                cursor += num_views
    if image_feature_dim <= 0:
        raise RuntimeError("Failed to extract multiview image features for diffusion-policy training.")
    metadata = {
        "summary": summary,
        "image_feature_dim": image_feature_dim,
        "image_size": config.image_size,
        "tasks": list(config.tasks or []),
        "extractor_type": "shared_local_backbone_multiview",
        "loaded_from_cache": False,
    }
    torch.save({"features": feature_bank, "metadata": metadata}, cache_path)
    return feature_bank, image_feature_dim, {**metadata, "cache_path": str(cache_path)}


class SequenceDiffusionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        episodes: list[Any],
        *,
        feature_bank: dict[str, torch.Tensor],
        image_feature_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        self.episodes = episodes
        self.feature_bank = feature_bank
        self.image_feature_dim = image_feature_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.obs_mean = torch.from_numpy(obs_mean).float()
        self.obs_std = torch.from_numpy(obs_std).float()
        self.action_mean = torch.from_numpy(action_mean).float()
        self.action_std = torch.from_numpy(action_std).float()
        self.records: list[tuple[Any, int]] = []
        min_future = n_action_steps - 1
        for episode in episodes:
            valid = [sample_has_full_obs(sample) for sample in episode.samples]
            for timestep in range(len(episode.samples)):
                start = max(0, timestep - (n_obs_steps - 1))
                end = min(len(episode.samples) - 1, start + horizon - 1)
                if timestep + min_future > len(episode.samples) - 1:
                    continue
                if all(valid[index] for index in range(start, end + 1)):
                    self.records.append((episode, timestep))
        if not self.records:
            raise RuntimeError("No valid multi-camera 16D diffusion-policy sequences were found.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, timestep = self.records[index]
        sequence_start = timestep - (self.n_obs_steps - 1)
        obs_seq = torch.zeros(self.horizon, 16 + self.image_feature_dim, dtype=torch.float32)
        action_seq = torch.zeros(self.horizon, 16, dtype=torch.float32)
        max_index = len(episode.samples) - 1
        for offset in range(self.horizon):
            sample_index = min(max(sequence_start + offset, 0), max_index)
            sample = episode.samples[sample_index]
            record_key = make_sample_key(episode, sample_index)
            state = torch.from_numpy(extract_state(sample))
            image_feature = self.feature_bank[record_key]
            obs_seq[offset] = torch.cat([state, image_feature], dim=0)
            action_seq[offset] = torch.from_numpy(np.asarray(sample.action, dtype=np.float32)[:16])
        obs_normalized = (obs_seq - self.obs_mean.view(1, -1)) / self.obs_std.view(1, -1)
        action_normalized = (action_seq - self.action_mean.view(1, -1)) / self.action_std.view(1, -1)
        return {
            "obs": obs_seq,
            "obs_normalized": obs_normalized,
            "action": action_seq,
            "action_normalized": action_normalized,
            "task_name": episode.task_name,
        }


def compute_norm_stats(
    episodes: list[Any],
    *,
    feature_bank: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    for episode in episodes:
        for timestep, sample in enumerate(episode.samples):
            if not sample_has_full_obs(sample):
                continue
            feature = feature_bank[make_sample_key(episode, timestep)].numpy()
            state = extract_state(sample)
            obs_rows.append(np.concatenate([state, feature.astype(np.float32)], axis=0))
            action_rows.append(np.asarray(sample.action, dtype=np.float32)[:16])
    if not obs_rows or not action_rows:
        raise RuntimeError("Unable to compute diffusion-policy normalization stats from training data.")
    obs_array = np.stack(obs_rows).astype(np.float32)
    action_array = np.stack(action_rows).astype(np.float32)
    obs_mean = obs_array.mean(axis=0)
    obs_std = np.clip(obs_array.std(axis=0), 1e-3, None)
    action_mean = action_array.mean(axis=0)
    action_std = np.clip(action_array.std(axis=0), 1e-3, None)
    return obs_mean, obs_std, action_mean, action_std


def compute_action_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    residual = predictions - targets
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    target_mean = np.mean(targets, axis=0, keepdims=True)
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(targets - target_mean)))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {"action_mse": mse, "action_mae": mae, "action_r2": r2}


def train_epoch(
    model: RoboTwinDiffusionPolicy,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    rows: list[dict[str, float]] = []
    for batch in loader:
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)
        optimizer.zero_grad()
        loss = model.compute_loss(obs, action)
        loss.backward()
        optimizer.step()
        rows.append({"train_loss": float(loss.item())})
    return offline.average_metric_dicts(rows)


def evaluate(
    model: RoboTwinDiffusionPolicy,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    per_task_preds: dict[str, list[np.ndarray]] = defaultdict(list)
    per_task_targets: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            losses.append(float(model.compute_loss(obs, action).item()))
            pred = model.predict_action(obs[:, : model.config.n_obs_steps, :])
            start = model.config.n_obs_steps - 1
            end = start + model.config.n_action_steps
            target = action[:, start:end, :]
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            preds.append(pred_np)
            targets.append(target_np)
            for idx, task_name in enumerate(batch["task_name"]):
                per_task_preds[task_name].append(pred_np[idx])
                per_task_targets[task_name].append(target_np[idx])
    metrics: dict[str, Any] = compute_action_metrics(np.concatenate(preds), np.concatenate(targets))
    metrics["diffusion_loss"] = float(np.mean(losses))
    metrics["_per_task"] = {
        task_name: compute_action_metrics(np.stack(per_task_preds[task_name]), np.stack(per_task_targets[task_name]))
        for task_name in sorted(per_task_preds)
    }
    return metrics


def build_datasets(
    split: dict[str, list[Any]],
    *,
    feature_bank_by_split: dict[str, dict[str, torch.Tensor]],
    image_feature_dim: int,
    config: TrainConfig,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> tuple[dict[str, SequenceDiffusionDataset], dict[str, str]]:
    datasets: dict[str, SequenceDiffusionDataset] = {}
    overrides: dict[str, str] = {}
    train_dataset = SequenceDiffusionDataset(
        split["train"],
        feature_bank=feature_bank_by_split["train"],
        image_feature_dim=image_feature_dim,
        horizon=config.horizon,
        n_obs_steps=config.n_obs_steps,
        n_action_steps=config.n_action_steps,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    datasets["train"] = train_dataset
    for split_name in ("val", "test"):
        if not split[split_name]:
            datasets[split_name] = train_dataset
            overrides[split_name] = "train"
            continue
        try:
            datasets[split_name] = SequenceDiffusionDataset(
                split[split_name],
                feature_bank=feature_bank_by_split[split_name],
                image_feature_dim=image_feature_dim,
                horizon=config.horizon,
                n_obs_steps=config.n_obs_steps,
                n_action_steps=config.n_action_steps,
                obs_mean=obs_mean,
                obs_std=obs_std,
                action_mean=action_mean,
                action_std=action_std,
            )
        except RuntimeError:
            datasets[split_name] = train_dataset
            overrides[split_name] = "train"
    return datasets, overrides


def main() -> None:
    config = parse_args()
    offline.set_global_seed(config.train_seed)
    device = torch.device(config.device)
    offline_config = offline.ExperimentConfig(
        data_dir=config.data_dir,
        tasks=list(config.tasks or []),
        max_episodes_per_task=config.max_episodes_per_task,
        batch_size=config.batch_size,
        image_size=config.image_size,
        split_seed=config.split_seed,
        seeds=[config.train_seed],
        device=config.device,
    )
    episodes = offline.load_real_episodes(offline_config)
    split = offline.split_episodes(
        episodes,
        train_ratio=offline_config.train_ratio,
        val_ratio=offline_config.val_ratio,
        test_ratio=offline_config.test_ratio,
        seed=config.split_seed,
    )
    feature_bank_by_split, image_feature_dim, feature_metadata = build_multiview_feature_bank(split, config)
    obs_mean, obs_std, action_mean, action_std = compute_norm_stats(split["train"], feature_bank=feature_bank_by_split["train"])
    datasets, split_overrides = build_datasets(
        split,
        feature_bank_by_split=feature_bank_by_split,
        image_feature_dim=image_feature_dim,
        config=config,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    loaders = {
        split_name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split_name == "train"),
            num_workers=config.num_workers,
            pin_memory=config.device.startswith("cuda"),
        )
        for split_name, dataset in datasets.items()
    }
    model_config = RoboTwinDiffusionPolicyConfig(
        image_feature_dim=image_feature_dim,
        state_dim=16,
        action_dim=16,
        horizon=config.horizon,
        n_obs_steps=config.n_obs_steps,
        n_action_steps=config.n_action_steps,
        num_inference_steps=config.num_inference_steps,
    )
    model = RoboTwinDiffusionPolicy(model_config).to(device)
    model.set_normalization_stats(
        obs_mean=torch.from_numpy(obs_mean),
        obs_std=torch.from_numpy(obs_std),
        action_mean=torch.from_numpy(action_mean),
        action_std=torch.from_numpy(action_std),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(config.num_epochs):
        train_metrics = train_epoch(model, loaders["train"], optimizer, device)
        val_metrics = evaluate(model, loaders["val"], device)
        history.append(
            {
                "epoch": epoch + 1,
                **train_metrics,
                **{f"val_{key}": value for key, value in val_metrics.items() if not key.startswith("_")},
            }
        )
        if float(val_metrics["action_mse"]) < best_val:
            best_val = float(val_metrics["action_mse"])
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch {epoch + 1:02d}/{config.num_epochs} train_loss={train_metrics['train_loss']:.4f} "
            f"val_action_mse={float(val_metrics['action_mse']):.4f} val_diffusion_loss={float(val_metrics['diffusion_loss']):.4f}"
        )
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], device)
    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = PROJECT_ROOT / config.checkpoint_path
    export_robottwin_diffusion_policy_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        config=model_config,
        metadata={
            "tasks": list(config.tasks or []),
            "split_seed": config.split_seed,
            "train_seed": config.train_seed,
            "max_episodes_per_task": config.max_episodes_per_task,
            "image_size": config.image_size,
            "feature_metadata": feature_metadata,
        },
    )
    summary = {
        "model": "robottwin_diffusion_policy",
        "checkpoint_path": str(checkpoint_path),
        "config": asdict(config),
        "policy_config": asdict(model_config),
        "split_summary": offline.summarize_split(split),
        "split_overrides": split_overrides,
        "feature_metadata": feature_metadata,
        "test_metrics": {key: value for key, value in test_metrics.items() if not key.startswith("_")},
        "per_task": test_metrics.get("_per_task", {}),
        "history": history,
    }
    (output_dir / "robottwin_diffusion_policy_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
