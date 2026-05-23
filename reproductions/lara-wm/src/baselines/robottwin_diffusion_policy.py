"""Shared RoboTwin diffusion-policy model and checkpoint helpers."""

from __future__ import annotations

# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportExplicitAny=false, reportMissingImports=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false

from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DP_ROOT = PROJECT_ROOT / "third_party" / "diffusion_policy"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


@dataclass(slots=True)
class RoboTwinDiffusionPolicyConfig:
    image_feature_dim: int
    state_dim: int = 16
    action_dim: int = 16
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    down_dims: list[int] = field(default_factory=lambda: [256, 512, 1024])
    diffusion_step_embed_dim: int = 256
    n_groups: int = 8
    kernel_size: int = 5
    cond_predict_scale: bool = True

    @property
    def obs_dim(self) -> int:
        return self.state_dim + self.image_feature_dim


class SimpleDDPMScheduler:
    def __init__(
        self,
        *,
        num_train_timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps)
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), self.alphas_cumprod[:-1]],
            dim=0,
        )
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1, dtype=torch.long)

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.timesteps = torch.linspace(
            self.config.num_train_timesteps - 1,
            0,
            num_inference_steps,
            dtype=torch.float32,
        ).round().long()

    def _expand(self, values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gathered = values.to(device=target.device)[timesteps.to(device=target.device)]
        return gathered.view(-1, *([1] * (target.ndim - 1)))

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._expand(self.alphas_cumprod.sqrt(), timesteps, clean)
        sqrt_one_minus = self._expand((1.0 - self.alphas_cumprod).sqrt(), timesteps, clean)
        return sqrt_alpha * clean + sqrt_one_minus * noise

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> SimpleNamespace:
        t = int(timestep)
        beta_t = self.betas[t].to(device=sample.device, dtype=sample.dtype)
        alpha_t = self.alphas[t].to(device=sample.device, dtype=sample.dtype)
        alpha_bar_t = self.alphas_cumprod[t].to(device=sample.device, dtype=sample.dtype)
        alpha_bar_prev = self.alphas_cumprod_prev[t].to(device=sample.device, dtype=sample.dtype)

        pred_original = (sample - (1.0 - alpha_bar_t).sqrt() * model_output) / alpha_bar_t.sqrt().clamp_min(1e-6)
        coef_x0 = beta_t * alpha_bar_prev.sqrt() / (1.0 - alpha_bar_t).clamp_min(1e-6)
        coef_xt = (1.0 - alpha_bar_prev) * alpha_t.sqrt() / (1.0 - alpha_bar_t).clamp_min(1e-6)
        mean = coef_x0 * pred_original + coef_xt * sample

        if t > 0:
            variance = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp_min(1e-6)) * beta_t
            prev_sample = mean + variance.clamp_min(1e-6).sqrt() * torch.randn_like(sample)
        else:
            prev_sample = mean
        return SimpleNamespace(prev_sample=prev_sample)


class RoboTwinDiffusionPolicy(nn.Module):
    def __init__(self, config: RoboTwinDiffusionPolicyConfig):
        super().__init__()
        self.config = config
        self.noise_scheduler = SimpleDDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
        )
        self.model = ConditionalUnet1D(
            input_dim=config.action_dim,
            local_cond_dim=None,
            global_cond_dim=config.obs_dim * config.n_obs_steps,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            cond_predict_scale=config.cond_predict_scale,
        )
        self.register_buffer("obs_mean", torch.zeros(config.obs_dim, dtype=torch.float32))
        self.register_buffer("obs_std", torch.ones(config.obs_dim, dtype=torch.float32))
        self.register_buffer("action_mean", torch.zeros(config.action_dim, dtype=torch.float32))
        self.register_buffer("action_std", torch.ones(config.action_dim, dtype=torch.float32))

    def set_normalization_stats(
        self,
        *,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        self.obs_mean.copy_(obs_mean.detach().to(dtype=torch.float32))
        self.obs_std.copy_(action_safe_std(obs_std.detach()).to(dtype=torch.float32))
        self.action_mean.copy_(action_mean.detach().to(dtype=torch.float32))
        self.action_std.copy_(action_safe_std(action_std.detach()).to(dtype=torch.float32))

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean.view(1, 1, -1)) / self.obs_std.view(1, 1, -1)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean.view(1, 1, -1)) / self.action_std.view(1, 1, -1)

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std.view(1, 1, -1) + self.action_mean.view(1, 1, -1)

    def compute_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        nobs = self.normalize_obs(obs)
        naction = self.normalize_action(action)
        global_cond = nobs[:, : self.config.n_obs_steps, :].reshape(nobs.shape[0], -1)
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        trajectory = naction[:, start:end]
        noise = torch.randn_like(trajectory)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor, *, num_inference_steps: int | None = None) -> torch.Tensor:
        device = obs.device
        dtype = obs.dtype
        batch_size = obs.shape[0]
        nobs = self.normalize_obs(obs).to(device=device, dtype=dtype)
        global_cond = nobs[:, : self.config.n_obs_steps, :].reshape(batch_size, -1)
        trajectory = torch.randn(
            size=(batch_size, self.config.n_action_steps, self.config.action_dim),
            device=device,
            dtype=dtype,
        )
        self.noise_scheduler.set_timesteps(num_inference_steps or self.config.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            step_t = int(timestep.item())
            model_output = self.model(trajectory, step_t, global_cond=global_cond)
            trajectory = self.noise_scheduler.step(model_output, step_t, trajectory).prev_sample
        return self.unnormalize_action(trajectory)


@dataclass(slots=True)
class RoboTwinDiffusionCheckpoint:
    model: RoboTwinDiffusionPolicy
    config: RoboTwinDiffusionPolicyConfig
    metadata: dict[str, Any]


def action_safe_std(std: torch.Tensor) -> torch.Tensor:
    return torch.clamp(std, min=1e-3)


def export_robottwin_diffusion_policy_checkpoint(
    *,
    checkpoint_path: Path,
    model: RoboTwinDiffusionPolicy,
    config: RoboTwinDiffusionPolicyConfig,
    metadata: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": "robottwin_diffusion_policy",
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "obs_mean": model.obs_mean.detach().cpu(),
        "obs_std": model.obs_std.detach().cpu(),
        "action_mean": model.action_mean.detach().cpu(),
        "action_std": model.action_std.detach().cpu(),
        "metadata": metadata,
    }
    torch.save(payload, checkpoint_path)


def load_robottwin_diffusion_policy_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> RoboTwinDiffusionCheckpoint:
    payload: dict[str, Any] = torch.load(Path(checkpoint_path), map_location=device)
    config = RoboTwinDiffusionPolicyConfig(**payload["config"])
    model = RoboTwinDiffusionPolicy(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return RoboTwinDiffusionCheckpoint(
        model=model,
        config=config,
        metadata=dict(payload.get("metadata", {})),
    )
