#!/usr/bin/env python3
"""Train and evaluate Diffusion Policy on RoboTwin fixed splits.

Uses processed/diffusion_policy/{train,val,test}.zarr which contain:
  - data/action: (T, 7) action sequence
  - data/state:  (T, 7) state observation
  - meta/episode_length: per-episode lengths

Metrics: action_mse, action_mae, action_r2 (same as ACT baseline).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
DP_ROOT = PROJECT_ROOT / "third_party" / "diffusion_policy"
sys.path.insert(0, str(DP_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

os.environ.setdefault("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

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
class DiffusionPolicyConfig:
    data_dir: str = "processed/diffusion_policy"
    output_dir: str = "experiments/results"
    checkpoint_dir: str = "experiments/diffusion_policy"
    num_epochs: int = 100
    batch_size_train: int = 256
    batch_size_val: int = 256
    learning_rate: float = 1e-4
    seed: int = 42
    device: str = DEFAULT_DEVICE
    save_every: int = 10

    # Diffusion Policy specific
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    obs_as_global_cond: bool = True
    action_dim: int = 7
    state_dim: int = 7
    num_inference_steps: int = 100
    down_dims: list[int] = field(default_factory=lambda: [256, 512, 1024])
    diffusion_step_embed_dim: int = 256
    n_groups: int = 8
    kernel_size: int = 5
    cond_predict_scale: bool = True
    pred_action_steps_only: bool = True


def parse_args() -> DiffusionPolicyConfig:
    parser = argparse.ArgumentParser(description="Train/evaluate Diffusion Policy on RoboTwin fixed splits")
    parser.add_argument("--data-dir", default="processed/diffusion_policy")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--checkpoint-dir", default="experiments/diffusion_policy")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size-train", type=int, default=256)
    parser.add_argument("--batch-size-val", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=8)
    args = parser.parse_args()
    return DiffusionPolicyConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.num_epochs,
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_val,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        save_every=args.save_every,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
    )


class ZarrDiffusionDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int = 2,
        pred_action_steps_only: bool = True,
        obs_as_global_cond: bool = True,
        action_dim: int = 7,
        seed: int = 42,
        max_episodes: int | None = None,
    ):
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.pred_action_steps_only = pred_action_steps_only
        self.obs_as_global_cond = obs_as_global_cond
        self.action_dim = action_dim
        self.seed = seed

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=["action", "state"])
        n_episodes = self.replay_buffer.n_episodes

        episode_mask = np.ones(n_episodes, dtype=bool)
        if max_episodes is not None and max_episodes < n_episodes:
            episode_mask = downsample_mask(episode_mask, max_n=max_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=["action", "state"],
            episode_mask=episode_mask,
        )

    def get_normalizer(self) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        all_actions = self.replay_buffer["action"]
        all_states = self.replay_buffer["state"]

        normalizer.fit(
            data={"action": all_actions, "obs": all_states},
            last_n_dims=1,
            mode="limits",
        )
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = {}
        for key, arr in data.items():
            torch_data[key] = torch.from_numpy(arr.copy()).float()
        return torch_data


class DiffusionUnetLowdimPolicyWrapper(torch.nn.Module):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        obs_as_global_cond: bool = True,
        pred_action_steps_only: bool = True,
        num_train_timesteps: int = 100,
        down_dims: list[int] = None,
        diffusion_step_embed_dim: int = 256,
        n_groups: int = 8,
        kernel_size: int = 5,
        cond_predict_scale: bool = True,
        oa_step_convention: bool = True,
    ):
        super().__init__()
        if down_dims is None:
            down_dims = [256, 512, 1024]

        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention

        if obs_as_global_cond:
            global_cond_dim = obs_dim * n_obs_steps
            input_dim = action_dim
        elif pred_action_steps_only:
            global_cond_dim = obs_dim * n_obs_steps
            input_dim = action_dim
        else:
            global_cond_dim = None
            input_dim = action_dim + obs_dim

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        model_device = next(self.model.parameters()).device
        self.normalizer.to(model_device)

    def compute_loss(self, batch, global_step=None) -> torch.Tensor:
        obs = batch["state"]
        action = batch["action"]

        nbatch = self.normalizer.normalize({"obs": obs, "action": action})
        nobs = nbatch["obs"]
        naction = nbatch["action"]

        if self.obs_as_global_cond:
            global_cond = nobs[:, : self.n_obs_steps, :].reshape(nobs.shape[0], -1)

            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = naction[:, start:end]
                condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
            else:
                trajectory = naction
                condition_mask = self.mask_generator(trajectory.shape)
                global_cond = None
        else:
            global_cond = None
            trajectory = torch.cat([naction, nobs], dim=-1)
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)
        target = noise
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = loss.mean()
        return loss

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor) -> torch.Tensor:
        device = obs.device
        dtype = obs.dtype
        B, T, Do = obs.shape

        nobs = self.normalizer["obs"].normalize(obs).to(device=device, dtype=dtype)

        if self.obs_as_global_cond:
            global_cond = nobs[:, : self.n_obs_steps, :].reshape(B, -1).to(device=device, dtype=dtype)

            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                shape = (B, self.n_action_steps, self.action_dim)
                cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                shape = (B, self.horizon, self.action_dim)
                cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            global_cond = None
            shape = (B, self.horizon, Do + self.action_dim)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, : self.n_obs_steps, self.action_dim :] = nobs[:, : self.n_obs_steps]
            cond_mask[:, : self.n_obs_steps, self.action_dim :] = True

        trajectory = torch.randn(size=cond_data.shape, device=device, dtype=dtype)
        self.noise_scheduler.set_timesteps(100)

        for t in self.noise_scheduler.timesteps:
            model_t = t.to(device) if isinstance(t, torch.Tensor) else t
            trajectory[cond_mask] = cond_data[cond_mask]
            model_output = self.model(trajectory, model_t, global_cond=global_cond)
            trajectory = self.noise_scheduler.step(model_output, t, trajectory).prev_sample

        trajectory[cond_mask] = cond_data[cond_mask]

        if self.pred_action_steps_only:
            naction_pred = trajectory
        else:
            start = self.n_obs_steps
            if self.oa_step_convention:
                start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            naction_pred = trajectory[:, start:end]

        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        return action_pred


def train_policy(
    policy: DiffusionUnetLowdimPolicyWrapper,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: DiffusionPolicyConfig,
    device: torch.device,
) -> tuple[DiffusionUnetLowdimPolicyWrapper, list[dict[str, float]]]:
    best_state = None
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
        policy.train()
        train_losses = []
        for batch in train_loader:
            obs = batch["state"].to(device)
            action = batch["action"].to(device)
            batch_data = {"state": obs, "action": action}

            optimizer.zero_grad()
            loss = policy.compute_loss(batch_data)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        policy.eval()
        val_losses = []
        with torch.inference_mode():
            for batch in val_loader:
                obs = batch["state"].to(device)
                action = batch["action"].to(device)
                batch_data = {"state": obs, "action": action}
                loss = policy.compute_loss(batch_data)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

        history.append({
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if epoch % config.save_every == 0 or epoch == config.num_epochs - 1:
            torch.save(policy.state_dict(), ckpt_dir / f"policy_epoch_{epoch}.ckpt")

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    if best_state is not None:
        policy.load_state_dict(best_state)
    torch.save(policy.state_dict(), ckpt_dir / "policy_best.ckpt")
    return policy, history


# =============================================================================
# Evaluation on test split
def evaluate_test_split(
    policy: DiffusionUnetLowdimPolicyWrapper,
    test_zarr_path: str,
    config: DiffusionPolicyConfig,
    device: torch.device,
) -> dict[str, float]:
    policy.eval()

    test_dataset = ZarrDiffusionDataset(
        zarr_path=test_zarr_path,
        horizon=config.horizon,
        pad_before=0,
        pad_after=0,
        n_obs_steps=config.n_obs_steps,
        pred_action_steps_only=config.pred_action_steps_only,
        obs_as_global_cond=config.obs_as_global_cond,
        action_dim=config.action_dim,
        seed=config.seed,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size_val,
        shuffle=False,
        num_workers=0,
    )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in test_loader:
            obs = batch["state"].to(device)
            action = batch["action"].to(device)
            B = obs.shape[0]

            action_pred = policy.predict_action(obs)
            target_action = action[:, : config.n_action_steps, :].cpu().numpy()
            pred_action = action_pred.cpu().numpy()

            for i in range(B):
                predictions.append(pred_action[i])
                targets.append(target_action[i])

    return compute_action_metrics(np.stack(predictions), np.stack(targets))


def main() -> None:
    import os

    config = parse_args()
    set_global_seed(config.seed)
    device = torch.device(config.device)

    data_dir = PROJECT_ROOT / config.data_dir
    train_zarr = str(data_dir / "train.zarr")
    val_zarr = str(data_dir / "val.zarr")
    test_zarr = str(data_dir / "test.zarr")

    print("=" * 60)
    print("Diffusion Policy Training")
    print("=" * 60)
    print(f"Train: {train_zarr}")
    print(f"Val:   {val_zarr}")
    print(f"Test:  {test_zarr}")
    print(f"Horizon: {config.horizon}, n_obs: {config.n_obs_steps}, n_action: {config.n_action_steps}")
    print(f"Device: {device}")
    print()

    train_dataset = ZarrDiffusionDataset(
        zarr_path=train_zarr,
        horizon=config.horizon,
        pad_before=0,
        pad_after=0,
        n_obs_steps=config.n_obs_steps,
        pred_action_steps_only=config.pred_action_steps_only,
        obs_as_global_cond=config.obs_as_global_cond,
        action_dim=config.action_dim,
        seed=config.seed,
    )

    val_dataset = ZarrDiffusionDataset(
        zarr_path=val_zarr,
        horizon=config.horizon,
        pad_before=0,
        pad_after=0,
        n_obs_steps=config.n_obs_steps,
        pred_action_steps_only=config.pred_action_steps_only,
        obs_as_global_cond=config.obs_as_global_cond,
        action_dim=config.action_dim,
        seed=config.seed,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:    {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_val,
        shuffle=False,
        num_workers=0,
    )

    normalizer = train_dataset.get_normalizer()

    policy = DiffusionUnetLowdimPolicyWrapper(
        action_dim=config.action_dim,
        obs_dim=config.state_dim,
        horizon=config.horizon,
        n_obs_steps=config.n_obs_steps,
        n_action_steps=config.n_action_steps,
        obs_as_global_cond=config.obs_as_global_cond,
        pred_action_steps_only=config.pred_action_steps_only,
        down_dims=config.down_dims,
        diffusion_step_embed_dim=config.diffusion_step_embed_dim,
        n_groups=config.n_groups,
        kernel_size=config.kernel_size,
        cond_predict_scale=config.cond_predict_scale,
    ).to(device)

    policy.set_normalizer(normalizer)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-6,
    )

    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    policy, history = train_policy(
        policy, optimizer, train_loader, val_loader, config, device
    )

    test_metrics = evaluate_test_split(policy, test_zarr, config, device)

    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "diffusion_policy",
        "data_dir": str(data_dir),
        "metrics": test_metrics,
        "config": asdict(config),
        "history": history,
    }
    with open(output_dir / "diffusion_policy_results.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(Path(config.checkpoint_dir) / "normalizer.pkl", "wb") as f:
        pickle.dump(normalizer.state_dict(), f)

    print()
    print("=" * 60)
    print("Test Metrics:")
    print("=" * 60)
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
