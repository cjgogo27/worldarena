# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportUntypedBaseClass=false, reportUnannotatedClassAttribute=false, reportDeprecated=false
"""Task-centric latent action encoder for LaRA-WM.

Encodes robot action trajectories into a compact latent representation using a
variational autoencoder (VAE). The default action feature width matches the
backbone adapter vision feature size (1536) so downstream multimodal modules
can align action features with backbone features when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_CONFIG_PATH = Path("/data/alice/cjtest/lara-wm/configs/config.yaml")
DEFAULT_FEATURE_DIM = 1536


@dataclass(frozen=True)
class LatentEncoderConfig:
    """Configuration for the latent action encoder."""

    action_dim: int = 7
    latent_dim: int = 32
    feature_dim: int = DEFAULT_FEATURE_DIM
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    kl_weight: float = 1.0
    collapse_variance_threshold: float = 1e-3
    collapse_kl_threshold: float = 1e-3

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> "LatentEncoderConfig":
        if not data:
            return cls()
        return cls(
            action_dim=int(data.get("action_dim", cls.action_dim)),
            latent_dim=int(data.get("latent_dim", cls.latent_dim)),
            feature_dim=int(data.get("feature_dim", cls.feature_dim)),
            hidden_dim=int(data.get("hidden_dim", cls.hidden_dim)),
            num_layers=int(data.get("num_layers", cls.num_layers)),
            dropout=float(data.get("dropout", cls.dropout)),
            kl_weight=float(data.get("kl_weight", cls.kl_weight)),
            collapse_variance_threshold=float(
                data.get("collapse_variance_threshold", cls.collapse_variance_threshold)
            ),
            collapse_kl_threshold=float(
                data.get("collapse_kl_threshold", cls.collapse_kl_threshold)
            ),
        )

    @classmethod
    def from_yaml(cls, config_path: Path | str = DEFAULT_CONFIG_PATH) -> "LatentEncoderConfig":
        path = Path(config_path)
        if not path.exists():
            return cls()

        import yaml

        config = yaml.safe_load(path.read_text()) or {}
        section = config.get("latent_encoder", config)
        return cls.from_dict(section)


@dataclass
class LatentEncoderOutput:
    """Structured output for VAE forward passes."""

    latent: Tensor
    mean: Tensor
    logvar: Tensor
    reconstruction: Tensor
    reconstruction_loss: Tensor
    kl_loss: Tensor
    total_loss: Tensor
    collapse_metrics: dict[str, Tensor]


class LatentActionEncoder(nn.Module):
    """VAE that compresses robot actions into task-centric latents.

    Supports inputs shaped as:
    - (batch, action_dim)
    - (batch, timesteps, action_dim)
    """

    def __init__(self, config: LatentEncoderConfig | None = None):
        super().__init__()
        self.config = config or LatentEncoderConfig()

        if self.config.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.config.action_dim <= 0:
            raise ValueError("action_dim must be positive")

        gru_dropout = self.config.dropout if self.config.num_layers > 1 else 0.0

        self.action_projection = nn.Sequential(
            nn.Linear(self.config.action_dim, self.config.feature_dim),
            nn.LayerNorm(self.config.feature_dim),
            nn.GELU(),
        )
        self.encoder = nn.GRU(
            input_size=self.config.feature_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.to_mean = nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        self.to_logvar = nn.Linear(self.config.hidden_dim, self.config.latent_dim)

        self.latent_projection = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.feature_dim),
            nn.LayerNorm(self.config.feature_dim),
            nn.GELU(),
        )
        self.decoder = nn.GRU(
            input_size=self.config.feature_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.decoder_init = nn.Linear(
            self.config.latent_dim,
            self.config.hidden_dim * self.config.num_layers,
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim),
        )

    @classmethod
    def from_config(cls, config: LatentEncoderConfig | None = None) -> "LatentActionEncoder":
        return cls(config=config or LatentEncoderConfig.from_yaml())

    def encode(
        self,
        actions: Tensor,
        sample: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sequence = self._ensure_sequence(actions)
        encoded = self.action_projection(sequence)
        _, hidden = self.encoder(encoded)
        summary = hidden[-1]
        mean = self.to_mean(summary)
        logvar = self.to_logvar(summary)
        latent = self.reparameterize(mean, logvar) if sample else mean
        return latent, mean, logvar

    def decode(self, latent: Tensor, sequence_length: int = 1) -> Tensor:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        repeated_latent = latent.unsqueeze(1).expand(-1, sequence_length, -1)
        decoder_input = self.latent_projection(repeated_latent)
        hidden = self.decoder_init(latent).view(
            latent.size(0),
            self.config.num_layers,
            self.config.hidden_dim,
        )
        hidden = hidden.permute(1, 0, 2).contiguous()
        decoded, _ = self.decoder(decoder_input, hidden)
        return self.reconstruction_head(decoded)

    def forward(self, actions: Tensor, sample: bool = True) -> LatentEncoderOutput:
        sequence = self._ensure_sequence(actions)
        latent, mean, logvar = self.encode(sequence, sample=sample)
        reconstruction = self.decode(latent, sequence_length=sequence.size(1))

        reconstruction_loss = F.mse_loss(reconstruction, sequence, reduction="mean")
        kl_loss = self.kl_divergence(mean, logvar).mean()
        total_loss = reconstruction_loss + (self.config.kl_weight * kl_loss)

        return LatentEncoderOutput(
            latent=latent,
            mean=mean,
            logvar=logvar,
            reconstruction=reconstruction,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            total_loss=total_loss,
            collapse_metrics=self.get_collapse_metrics(mean, logvar),
        )

    def get_collapse_metrics(self, mean: Tensor, logvar: Tensor) -> dict[str, Tensor]:
        latent_variance = mean.var(dim=0, unbiased=False)
        mean_kl = self.kl_divergence(mean, logvar).mean()
        posterior_std = torch.exp(0.5 * logvar)
        active_units = (latent_variance > self.config.collapse_variance_threshold).float().mean()
        variance_ratio = (latent_variance.mean() / self.config.collapse_variance_threshold).clamp(max=1.0)
        kl_ratio = (mean_kl / self.config.collapse_kl_threshold).clamp(max=1.0)
        collapse_score = 1.0 - (0.5 * variance_ratio + 0.5 * kl_ratio)
        is_collapsed = (
            (latent_variance.mean() < self.config.collapse_variance_threshold)
            | (mean_kl < self.config.collapse_kl_threshold)
        ).float()

        return {
            "latent_variance_mean": latent_variance.mean(),
            "latent_variance_min": latent_variance.min(),
            "active_units_ratio": active_units,
            "posterior_std_mean": posterior_std.mean(),
            "posterior_std_min": posterior_std.min(),
            "mean_abs_posterior_mean": mean.abs().mean(),
            "mean_kl": mean_kl,
            "collapse_score": collapse_score,
            "is_collapsed": is_collapsed,
        }

    @staticmethod
    def kl_divergence(mean: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)

    @staticmethod
    def reparameterize(mean: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    @staticmethod
    def _ensure_sequence(actions: Tensor) -> Tensor:
        if actions.dim() == 2:
            return actions.unsqueeze(1)
        if actions.dim() == 3:
            return actions
        raise ValueError(
            "Expected actions with shape (batch, action_dim) or (batch, timesteps, action_dim)"
        )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_FEATURE_DIM",
    "LatentActionEncoder",
    "LatentEncoderConfig",
    "LatentEncoderOutput",
]
