#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
ACT_ROOT = PROJECT_ROOT / "third_party" / "act"
DETR_ROOT = ACT_ROOT / "detr"
sys.path.insert(0, str(DETR_ROOT))
sys.path.insert(0, str(ACT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from training.policy import ACTPolicy  # type: ignore  # noqa: E402
sys.argv = _ORIGINAL_ARGV


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def average_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = sorted({key for metric_dict in metric_dicts for key in metric_dict})
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts if key in metrics])) for key in keys}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(slots=True)
class ACTOriginalConfig:
    data_dir: str = "data/act_format"
    output_dir: str = "experiments/results"
    checkpoint_dir: str = "experiments/act_original"
    num_epochs: int = 20
    batch_size_train: int = 8
    batch_size_val: int = 8
    learning_rate: float = 1e-5
    lr_backbone: float = 1e-5
    num_queries: int = 100
    kl_weight: int = 10
    hidden_dim: int = 512
    dim_feedforward: int = 3200
    enc_layers: int = 4
    dec_layers: int = 7
    nheads: int = 8
    state_dim: int = 7
    action_dim: int = 7
    camera_names: list[str] = field(default_factory=lambda: ["head_cam"])
    backbone: str = "resnet18"
    seed: int = 42
    device: str = DEFAULT_DEVICE
    save_every: int = 5


def parse_args() -> ACTOriginalConfig:
    parser = argparse.ArgumentParser(description="Train/evaluate original ACT on RoboTwin fixed splits")
    parser.add_argument("--data-dir", default="data/act_format")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--checkpoint-dir", default="experiments/act_original")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size-train", type=int, default=8)
    parser.add_argument("--batch-size-val", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--camera-names", nargs="+", default=["head_cam"])
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()
    return ACTOriginalConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.num_epochs,
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_val,
        learning_rate=args.learning_rate,
        lr_backbone=args.lr_backbone,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        seed=args.seed,
        device=args.device,
        save_every=args.save_every,
    )


def list_episode_ids(split_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in sorted(split_dir.glob("episode_*.hdf5")):
        stem = path.stem
        ids.append(int(stem.split("_")[-1]))
    return ids


def load_split_metadata(data_dir: Path) -> dict[str, object]:
    return json.loads((data_dir / "metadata.json").read_text())


def compute_norm_stats(split_dir: Path, episode_ids: list[int]) -> dict[str, np.ndarray]:
    qpos_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    for episode_id in episode_ids:
        with h5py.File(split_dir / f"episode_{episode_id}.hdf5", "r") as root:
            qpos_chunks.append(root["/observations/qpos"][()].astype(np.float32))
            action_chunks.append(root["/action"][()].astype(np.float32))
    qpos = np.concatenate(qpos_chunks, axis=0)
    action = np.concatenate(action_chunks, axis=0)
    qpos_mean = qpos.mean(axis=0)
    qpos_std = np.clip(qpos.std(axis=0), 1e-2, None)
    action_mean = action.mean(axis=0)
    action_std = np.clip(action.std(axis=0), 1e-2, None)
    return {
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


class ACTSplitDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        split_dir: Path,
        episode_ids: list[int],
        camera_names: list[str],
        norm_stats: dict[str, np.ndarray],
        split_metadata: dict[str, object],
        num_queries: int,
    ):
        self.split_dir = split_dir
        self.episode_ids = episode_ids
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.num_queries = num_queries
        split_by_episode: dict[str, str] = {}
        split_meta = split_metadata.get("split_metadata", {})
        if isinstance(split_meta, dict):
            for split_name, payload in split_meta.items():
                if isinstance(payload, dict):
                    for item in payload.get("episode_ids", []):
                        if isinstance(item, str):
                            split_by_episode[item.split("/")[-1]] = split_name
        self.task_by_episode: dict[int, str] = {}
        for episode_id in episode_ids:
            key = f"episode_{episode_id}"
            for item, split_name in split_by_episode.items():
                if item == key:
                    task_name = next((part for part in key.split("/") if part != key), None)
                    _ = split_name
            self.task_by_episode[episode_id] = "unknown"
        self.samples: list[tuple[int, int]] = []
        for episode_id in episode_ids:
            with h5py.File(split_dir / f"episode_{episode_id}.hdf5", "r") as root:
                episode_len = int(root["/action"].shape[0])
            for start_ts in range(episode_len):
                self.samples.append((episode_id, start_ts))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        episode_id, start_ts = self.samples[index]
        dataset_path = self.split_dir / f"episode_{episode_id}.hdf5"
        with h5py.File(dataset_path, "r") as root:
            action_ds = root["/action"]
            episode_len = int(action_ds.shape[0])
            qpos = root["/observations/qpos"][start_ts].astype(np.float32)
            image_dict = {
                cam_name: root[f"/observations/images/{cam_name}"][start_ts]
                for cam_name in self.camera_names
            }
            action = action_ds[start_ts:start_ts + self.num_queries].astype(np.float32)
            action_len = int(action.shape[0])

        padded_action = np.zeros((self.num_queries, action.shape[-1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.num_queries, dtype=np.bool_)
        is_pad[action_len:] = True

        all_cam_images = np.stack([image_dict[cam_name] for cam_name in self.camera_names], axis=0)
        image_data = torch.from_numpy(all_cam_images).permute(0, 3, 1, 2).float() / 255.0
        qpos_data = torch.from_numpy((qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]).float()
        action_data = torch.from_numpy((padded_action - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]).float()
        is_pad_tensor = torch.from_numpy(is_pad)
        return image_data, qpos_data, action_data, is_pad_tensor


def build_policy_config(config: ACTOriginalConfig) -> dict[str, object]:
    return {
        "lr": config.learning_rate,
        "device": config.device,
        "num_queries": config.num_queries,
        "kl_weight": config.kl_weight,
        "hidden_dim": config.hidden_dim,
        "dim_feedforward": config.dim_feedforward,
        "lr_backbone": config.lr_backbone,
        "backbone": config.backbone,
        "enc_layers": config.enc_layers,
        "dec_layers": config.dec_layers,
        "nheads": config.nheads,
        "camera_names": config.camera_names,
        "policy_class": "ACT",
        "temporal_agg": False,
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
    }


def instantiate_act_policy(policy_config: dict[str, object]) -> ACTPolicy:
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        return ACTPolicy(policy_config)
    finally:
        sys.argv = original_argv


def forward_pass(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    policy: ACTPolicy,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image_data, qpos_data, action_data, is_pad = batch
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)


def compute_dict_mean(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: sum(item[key] for item in dicts) / len(dicts) for key in dicts[0]}


def train_policy(
    policy: ACTPolicy,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ACTOriginalConfig,
    device: torch.device,
) -> tuple[ACTPolicy, list[dict[str, float]], dict[str, np.ndarray], float]:
    best_state = copy.deepcopy(policy.state_dict())
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
        policy.train()
        train_epoch: list[dict[str, torch.Tensor]] = []
        for batch in train_loader:
            optimizer.zero_grad()
            forward_dict = forward_pass(batch, policy, device)
            forward_dict["loss"].backward()
            optimizer.step()
            train_epoch.append({key: value.detach() for key, value in forward_dict.items()})
        train_summary = {key: float(value.item()) for key, value in compute_dict_mean(train_epoch).items()}

        policy.eval()
        with torch.inference_mode():
            val_epoch: list[dict[str, torch.Tensor]] = []
            for batch in val_loader:
                val_epoch.append(forward_pass(batch, policy, device))
            val_summary = {key: float(value.item()) for key, value in compute_dict_mean(val_epoch).items()}

        if val_summary["loss"] < best_val_loss:
            best_val_loss = val_summary["loss"]
            best_state = copy.deepcopy(policy.state_dict())

        history.append({
            "epoch": float(epoch),
            "train_loss": train_summary["loss"],
            "train_l1": train_summary.get("l1", 0.0),
            "train_kl": train_summary.get("kl", 0.0),
            "val_loss": val_summary["loss"],
            "val_l1": val_summary.get("l1", 0.0),
            "val_kl": val_summary.get("kl", 0.0),
        })

        if epoch % config.save_every == 0 or epoch == config.num_epochs - 1:
            torch.save(policy.state_dict(), ckpt_dir / f"policy_epoch_{epoch}.ckpt")

    policy.load_state_dict(best_state)
    torch.save(policy.state_dict(), ckpt_dir / "policy_best.ckpt")
    return policy, history, best_state, best_val_loss


def evaluate_test_split(
    policy: ACTPolicy,
    split_dir: Path,
    episode_ids: list[int],
    camera_names: list[str],
    norm_stats: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    policy.eval()
    with torch.inference_mode():
        for episode_id in episode_ids:
            with h5py.File(split_dir / f"episode_{episode_id}.hdf5", "r") as root:
                qpos_seq = root["/observations/qpos"][()].astype(np.float32)
                action_seq = root["/action"][()].astype(np.float32)
                images = {cam: root[f"/observations/images/{cam}"][()] for cam in camera_names}
            for t in range(len(action_seq)):
                qpos = (qpos_seq[t] - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
                image = np.stack([images[cam][t] for cam in camera_names], axis=0)
                qpos_tensor = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
                image_tensor = torch.from_numpy(image).permute(0, 3, 1, 2).float().unsqueeze(0).to(device) / 255.0
                action_chunk = policy(qpos_tensor, image_tensor)
                pred = action_chunk[:, 0, :].squeeze(0).cpu().numpy()
                pred = pred * norm_stats["action_std"] + norm_stats["action_mean"]
                predictions.append(pred)
                targets.append(action_seq[t])
    return compute_action_metrics(np.stack(predictions), np.stack(targets))


def main() -> None:
    config = parse_args()
    set_global_seed(config.seed)
    device = torch.device(config.device)

    data_dir = PROJECT_ROOT / config.data_dir
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    metadata = load_split_metadata(data_dir)

    train_ids = list_episode_ids(train_dir)
    val_ids = list_episode_ids(val_dir)
    test_ids = list_episode_ids(test_dir)
    norm_stats = compute_norm_stats(train_dir, train_ids)

    train_dataset = ACTSplitDataset(train_dir, train_ids, config.camera_names, norm_stats, metadata, config.num_queries)
    val_dataset = ACTSplitDataset(val_dir, val_ids, config.camera_names, norm_stats, metadata, config.num_queries)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_train, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size_val, shuffle=False, num_workers=0)

    policy_config = build_policy_config(config)
    policy = instantiate_act_policy(policy_config).to(device)
    optimizer = policy.configure_optimizers()
    policy, history, _, best_val_loss = train_policy(policy, optimizer, train_loader, val_loader, config, device)

    test_metrics = evaluate_test_split(policy, test_dir, test_ids, config.camera_names, norm_stats, device)

    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "act_original",
        "data_dir": str(data_dir),
        "train_episodes": len(train_ids),
        "val_episodes": len(val_ids),
        "test_episodes": len(test_ids),
        "best_val_loss": best_val_loss,
        "metrics": test_metrics,
        "norm_stats_path": str((Path(config.checkpoint_dir) / "dataset_stats.pkl").resolve()),
        "config": asdict(config),
        "history": history,
    }
    with open(output_dir / "act_original_results.json", "w") as handle:
        json.dump(result, handle, indent=2)
    with open(Path(config.checkpoint_dir) / "dataset_stats.pkl", "wb") as handle:
        pickle.dump(norm_stats, handle)
    print(json.dumps({"metrics": test_metrics, "best_val_loss": best_val_loss}, indent=2))


if __name__ == "__main__":
    main()
