from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


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


class DirectPolicyRolloutModel(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, latent_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = FeatureProjector(feature_dim, latent_dim, hidden_dim)
        self.head = MLP(latent_dim, hidden_dim, action_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(features)
        action = self.head(latent)
        return {"latent": latent, "action": action}


class LatentNoRefineRolloutModel(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, latent_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = FeatureProjector(feature_dim, latent_dim, hidden_dim)
        self.decoder = MLP(latent_dim, hidden_dim, action_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(features)
        action = self.decoder(latent)
        return {"latent": latent, "action": action}


class LaraWMRolloutModel(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, latent_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = FeatureProjector(feature_dim, latent_dim, hidden_dim)
        self.action_decoder = MLP(latent_dim, hidden_dim, action_dim)
        self.transition = MLP(latent_dim + action_dim, hidden_dim, latent_dim)
        self.refiner = MLP(latent_dim * 3, hidden_dim, latent_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(features)
        action = self.action_decoder(latent)
        return {"latent": latent, "action": action}


@dataclass
class RolloutCheckpoint:
    model_name: str
    model: nn.Module
    action_mean: torch.Tensor
    action_std: torch.Tensor
    feature_dim: int
    latent_dim: int
    hidden_dim: int
    action_dim: int


def default_rollout_ckpt_path(project_root: Path, model_name: str) -> Path:
    return project_root / "experiments" / "rollout_ckpts" / f"{model_name.replace('-', '_')}.pt"


def maybe_load_rollout_checkpoint(checkpoint_path: Path | None, device: torch.device) -> RolloutCheckpoint | None:
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    payload: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    if "state_dict" not in payload or "model_name" not in payload:
        return None

    model_name = str(payload["model_name"])
    feature_dim = int(payload["feature_dim"])
    latent_dim = int(payload["latent_dim"])
    hidden_dim = int(payload["hidden_dim"])
    action_dim = int(payload["action_dim"])

    if model_name == "direct_policy":
        model = DirectPolicyRolloutModel(action_dim, hidden_dim, latent_dim, feature_dim)
    elif model_name == "latent_no_refine":
        model = LatentNoRefineRolloutModel(action_dim, hidden_dim, latent_dim, feature_dim)
    elif model_name == "lara-wm":
        model = LaraWMRolloutModel(action_dim, hidden_dim, latent_dim, feature_dim)
    else:
        return None

    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    return RolloutCheckpoint(
        model_name=model_name,
        model=model,
        action_mean=payload["action_mean"].to(device=device, dtype=torch.float32),
        action_std=payload["action_std"].to(device=device, dtype=torch.float32),
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
    )


def predict_denormalized_action(ckpt: RolloutCheckpoint, feature_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = ckpt.model(feature_tensor.to(dtype=torch.float32))
        action_norm = outputs["action"]
        return action_norm * ckpt.action_std + ckpt.action_mean
