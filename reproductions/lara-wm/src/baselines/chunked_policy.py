"""Minimal multi-view chunked rollout policy shared by train/deploy paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(slots=True)
class ChunkedPolicyConfig:
    image_feature_dim: int
    state_dim: int = 16
    action_dim: int = 16
    chunk_size: int = 8
    hidden_dim: int = 512
    dropout: float = 0.1


class ChunkedRolloutPolicy(nn.Module):
    def __init__(self, config: ChunkedPolicyConfig):
        super().__init__()
        self.config = config
        self.image_proj = nn.Sequential(
            nn.LayerNorm(config.image_feature_dim),
            nn.Linear(config.image_feature_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.action_head = nn.Linear(
            config.hidden_dim,
            config.chunk_size * config.action_dim,
        )

    def forward(self, image_features: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        image_hidden = self.image_proj(image_features)
        state_hidden = self.state_proj(state)
        fused = self.fusion(torch.cat([image_hidden, state_hidden], dim=-1))
        chunk_flat = self.action_head(fused)
        action_chunk = chunk_flat.view(-1, self.config.chunk_size, self.config.action_dim)
        return {
            "action": action_chunk[:, 0, :],
            "action_chunk": action_chunk,
        }


@dataclass(slots=True)
class ChunkedPolicyCheckpoint:
    model: ChunkedRolloutPolicy
    config: ChunkedPolicyConfig
    action_mean: torch.Tensor
    action_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    metadata: dict[str, Any]


def export_chunked_policy_checkpoint(
    *,
    checkpoint_path: Path,
    model: ChunkedRolloutPolicy,
    config: ChunkedPolicyConfig,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": "chunked_policy",
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "action_mean": action_mean.detach().cpu(),
        "action_std": action_std.detach().cpu(),
        "state_mean": state_mean.detach().cpu(),
        "state_std": state_std.detach().cpu(),
        "metadata": metadata,
    }
    torch.save(payload, checkpoint_path)


def load_chunked_policy_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> ChunkedPolicyCheckpoint:
    payload: dict[str, Any] = torch.load(Path(checkpoint_path), map_location=device)
    config = ChunkedPolicyConfig(**payload["config"])
    model = ChunkedRolloutPolicy(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return ChunkedPolicyCheckpoint(
        model=model,
        config=config,
        action_mean=payload["action_mean"].to(device=device, dtype=torch.float32),
        action_std=payload["action_std"].to(device=device, dtype=torch.float32),
        state_mean=payload["state_mean"].to(device=device, dtype=torch.float32),
        state_std=payload["state_std"].to(device=device, dtype=torch.float32),
        metadata=dict(payload.get("metadata", {})),
    )
