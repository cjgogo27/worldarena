#!/usr/bin/env python3
from __future__ import annotations

# pyright: reportAny=false, reportArgumentType=false, reportExplicitAny=false, reportImplicitRelativeImport=false, reportMissingImports=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

import argparse
import copy
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import train_and_eval as offline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.chunked_policy import (  # noqa: E402
    ChunkedPolicyConfig,
    ChunkedRolloutPolicy,
    export_chunked_policy_checkpoint,
)


@dataclass(slots=True)
class TrainConfig:
    data_dir: str = "data/robottwin_hf"
    tasks: list[str] | None = None
    output_dir: str = "experiments/rollout_ckpts"
    max_episodes_per_task: int = 10
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    hidden_dim: int = 512
    chunk_size: int = 8
    image_size: int = 128
    split_seed: int = 2026
    train_seed: int = 42
    device: str = offline.DEFAULT_DEVICE
    backbone_model_path: str | None = None
    backbone_dtype: str = "float16"
    backbone_device: str | None = None
    num_workers: int = 0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a minimal multi-view chunked RoboTwin rollout policy.")
    parser.add_argument("--data-dir", default="data/robottwin_hf")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output-dir", default="experiments/rollout_ckpts")
    parser.add_argument("--max-episodes-per-task", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--device", default=offline.DEFAULT_DEVICE)
    parser.add_argument("--backbone-model-path", default=None)
    parser.add_argument("--backbone-dtype", default="float16")
    parser.add_argument("--backbone-device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def sample_has_full_obs(sample: Any) -> bool:
    return (
        sample.head_image is not None
        and sample.left_image is not None
        and sample.right_image is not None
        and sample.action is not None
        and np.asarray(sample.action).shape[-1] >= 16
    )


def extract_state(sample: Any) -> np.ndarray:
    if sample.state is not None and np.asarray(sample.state).shape[-1] >= 16:
        return np.asarray(sample.state, dtype=np.float32)[:16]
    return np.asarray(sample.action, dtype=np.float32)[:16]


def compute_stats(episodes: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in episodes:
        for sample in episode.samples:
            if not sample_has_full_obs(sample):
                continue
            states.append(extract_state(sample))
            actions.append(np.asarray(sample.action, dtype=np.float32)[:16])
    state_array = np.stack(states)
    action_array = np.stack(actions)
    state_mean = state_array.mean(axis=0).astype(np.float32)
    state_std = np.clip(state_array.std(axis=0).astype(np.float32), 1e-3, None)
    action_mean = action_array.mean(axis=0).astype(np.float32)
    action_std = np.clip(action_array.std(axis=0).astype(np.float32), 1e-3, None)
    return state_mean, state_std, action_mean, action_std


def make_record_key(episode: Any, timestep: int) -> str:
    return f"{episode.task_name}:{episode.episode_id}:{timestep}"


class ChunkedPolicyDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        episodes: list[Any],
        *,
        chunk_size: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        image_size: int,
    ):
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.state_mean = torch.from_numpy(state_mean).float()
        self.state_std = torch.from_numpy(state_std).float()
        self.action_mean = torch.from_numpy(action_mean).float()
        self.action_std = torch.from_numpy(action_std).float()
        self.records: list[tuple[Any, int]] = []
        for episode in episodes:
            for timestep, sample in enumerate(episode.samples):
                if sample_has_full_obs(sample):
                    self.records.append((episode, timestep))
        if not self.records:
            raise RuntimeError("No valid multi-view 16D RoboTwin samples found for chunked policy training.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, timestep = self.records[index]
        sample = episode.samples[timestep]
        state = torch.from_numpy(extract_state(sample))
        state_normalized = (state - self.state_mean) / self.state_std
        action_chunk = torch.zeros(self.chunk_size, 16, dtype=torch.float32)
        action_mask = torch.zeros(self.chunk_size, dtype=torch.float32)
        for chunk_offset in range(self.chunk_size):
            target_index = timestep + chunk_offset
            if target_index >= len(episode.samples):
                break
            target = episode.samples[target_index]
            if not sample_has_full_obs(target):
                break
            action_chunk[chunk_offset] = torch.from_numpy(np.asarray(target.action, dtype=np.float32)[:16])
            action_mask[chunk_offset] = 1.0
        normalized_chunk = (action_chunk - self.action_mean) / self.action_std
        return {
            "transition_id": make_record_key(episode, timestep),
            "state": state,
            "state_normalized": state_normalized,
            "action_chunk": action_chunk,
            "action_chunk_normalized": normalized_chunk,
            "action_mask": action_mask,
            "task_name": episode.task_name,
            "head_image": offline.preprocess_image(sample.head_image, self.image_size),
            "left_image": offline.preprocess_image(sample.left_image, self.image_size),
            "right_image": offline.preprocess_image(sample.right_image, self.image_size),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "transition_id": [item["transition_id"] for item in batch],
        "state": torch.stack([item["state"] for item in batch]),
        "state_normalized": torch.stack([item["state_normalized"] for item in batch]),
        "action_chunk": torch.stack([item["action_chunk"] for item in batch]),
        "action_chunk_normalized": torch.stack([item["action_chunk_normalized"] for item in batch]),
        "action_mask": torch.stack([item["action_mask"] for item in batch]),
        "task_name": [item["task_name"] for item in batch],
        "head_image": torch.stack([item["head_image"] for item in batch]),
        "left_image": torch.stack([item["left_image"] for item in batch]),
        "right_image": torch.stack([item["right_image"] for item in batch]),
    }


def build_feature_bank(
    loaders: dict[str, DataLoader[dict[str, Any]]],
    config: TrainConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], int]:
    backbone_config = offline.ExperimentConfig(
        data_dir=config.data_dir,
        tasks=list(config.tasks or []),
        max_episodes_per_task=config.max_episodes_per_task,
        batch_size=config.batch_size,
        image_size=config.image_size,
        backbone_model_path=config.backbone_model_path,
        backbone_dtype=config.backbone_dtype,
        backbone_device=config.backbone_device,
        device=config.device,
        split_seed=config.split_seed,
        seeds=[config.train_seed],
    )
    adapter = offline.build_backbone_adapter(backbone_config)
    bank: dict[str, dict[str, torch.Tensor]] = {split_name: {} for split_name in loaders}
    feature_dim = 0
    for split_name, loader in loaders.items():
        for batch in loader:
            flat_images = torch.cat([batch["head_image"], batch["left_image"], batch["right_image"]], dim=0).to(adapter.device)
            with torch.no_grad():
                features = adapter.encode_image(flat_images).detach().cpu().float()
            batch_size = batch["head_image"].shape[0]
            view_dim = int(features.shape[-1])
            feature_dim = view_dim * 3
            features = features.view(3, batch_size, view_dim).permute(1, 0, 2).reshape(batch_size, feature_dim)
            for idx, transition_id in enumerate(batch["transition_id"]):
                bank[split_name][transition_id] = {"feature": features[idx]}
    return bank, feature_dim


def attach_features(
    batch: dict[str, Any],
    feature_bank: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    features = torch.stack([feature_bank[transition_id]["feature"] for transition_id in batch["transition_id"]], dim=0)
    return features.to(device=device, dtype=torch.float32)


def masked_chunk_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_dim = F.mse_loss(predicted, target, reduction="none").mean(dim=-1)
    weighted = per_dim * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


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
    model: ChunkedRolloutPolicy,
    loader: DataLoader[dict[str, Any]],
    feature_bank: dict[str, dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    metric_rows: list[dict[str, float]] = []
    for batch in loader:
        features = attach_features(batch, feature_bank, device)
        state = batch["state_normalized"].to(device)
        target = batch["action_chunk_normalized"].to(device)
        mask = batch["action_mask"].to(device)
        optimizer.zero_grad()
        outputs = model(features, state)
        chunk_loss = masked_chunk_loss(outputs["action_chunk"], target, mask)
        first_loss = F.mse_loss(outputs["action"], target[:, 0, :])
        loss = chunk_loss + 0.5 * first_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = outputs["action"] * action_std.to(device) + action_mean.to(device)
            tgt = batch["action_chunk"][:, 0, :].to(device)
            metrics = compute_action_metrics(pred.detach().cpu().numpy(), tgt.detach().cpu().numpy())
        metric_rows.append({"train_loss": float(loss.item()), "train_chunk_loss": float(chunk_loss.item()), **metrics})
    return offline.average_metric_dicts(metric_rows)


def evaluate(
    model: ChunkedRolloutPolicy,
    loader: DataLoader[dict[str, Any]],
    feature_bank: dict[str, dict[str, torch.Tensor]],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    per_task_preds: dict[str, list[np.ndarray]] = defaultdict(list)
    per_task_targets: dict[str, list[np.ndarray]] = defaultdict(list)
    chunk_losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            features = attach_features(batch, feature_bank, device)
            state = batch["state_normalized"].to(device)
            target = batch["action_chunk_normalized"].to(device)
            mask = batch["action_mask"].to(device)
            outputs = model(features, state)
            chunk_losses.append(float(masked_chunk_loss(outputs["action_chunk"], target, mask).item()))
            pred = outputs["action"] * action_std.to(device) + action_mean.to(device)
            tgt = batch["action_chunk"][:, 0, :].to(device)
            pred_np = pred.detach().cpu().numpy()
            tgt_np = tgt.detach().cpu().numpy()
            predictions.append(pred_np)
            targets.append(tgt_np)
            for idx, task_name in enumerate(batch["task_name"]):
                per_task_preds[task_name].append(pred_np[idx])
                per_task_targets[task_name].append(tgt_np[idx])
    metrics: dict[str, Any] = compute_action_metrics(np.concatenate(predictions), np.concatenate(targets))
    metrics["chunk_loss"] = float(np.mean(chunk_losses))
    metrics["_per_task"] = {
        task_name: compute_action_metrics(np.stack(per_task_preds[task_name]), np.stack(per_task_targets[task_name]))
        for task_name in sorted(per_task_preds)
    }
    return metrics


def main() -> None:
    config = parse_args()
    offline.set_global_seed(config.train_seed)
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
    state_mean_np, state_std_np, action_mean_np, action_std_np = compute_stats(split["train"])
    datasets = {
        split_name: ChunkedPolicyDataset(
            split_episodes,
            chunk_size=config.chunk_size,
            state_mean=state_mean_np,
            state_std=state_std_np,
            action_mean=action_mean_np,
            action_std=action_std_np,
            image_size=config.image_size,
        )
        for split_name, split_episodes in split.items()
    }
    loaders = {
        split_name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split_name == "train"),
            num_workers=config.num_workers,
            pin_memory=config.device.startswith("cuda"),
            collate_fn=collate_batch,
        )
        for split_name, dataset in datasets.items()
    }
    feature_bank, image_feature_dim = build_feature_bank(loaders, config)
    model_config = ChunkedPolicyConfig(
        image_feature_dim=image_feature_dim,
        state_dim=16,
        action_dim=16,
        chunk_size=config.chunk_size,
        hidden_dim=config.hidden_dim,
    )
    device = torch.device(config.device)
    model = ChunkedRolloutPolicy(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    action_mean = torch.from_numpy(action_mean_np).float()
    action_std = torch.from_numpy(action_std_np).float()
    state_mean = torch.from_numpy(state_mean_np).float()
    state_std = torch.from_numpy(state_std_np).float()
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(config.num_epochs):
        train_metrics = train_epoch(model, loaders["train"], feature_bank["train"], optimizer, action_mean, action_std, device)
        val_metrics = evaluate(model, loaders["val"], feature_bank["val"], action_mean, action_std, device)
        history.append({"epoch": epoch + 1, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items() if not k.startswith("_")}})
        if float(val_metrics["action_mse"]) < best_val:
            best_val = float(val_metrics["action_mse"])
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch {epoch + 1:02d}/{config.num_epochs} train_loss={train_metrics['train_loss']:.4f} "
            f"val_action_mse={float(val_metrics['action_mse']):.4f} val_chunk_loss={float(val_metrics['chunk_loss']):.4f}"
        )
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], feature_bank["test"], action_mean, action_std, device)
    output_dir = PROJECT_ROOT / config.output_dir
    checkpoint_path = output_dir / "chunked_policy.pt"
    export_chunked_policy_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        config=model_config,
        action_mean=action_mean,
        action_std=action_std,
        state_mean=state_mean,
        state_std=state_std,
        metadata={
            "tasks": list(config.tasks or []),
            "split_seed": config.split_seed,
            "train_seed": config.train_seed,
            "max_episodes_per_task": config.max_episodes_per_task,
            "image_size": config.image_size,
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunked_policy_history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "image_feature_dim": image_feature_dim,
        "test_metrics": {k: v for k, v in test_metrics.items() if not k.startswith("_")},
        "per_task": test_metrics.get("_per_task", {}),
    }
    (output_dir / "chunked_policy_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
