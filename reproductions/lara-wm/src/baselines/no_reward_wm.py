"""No-reward world model baseline for LaRA-WM."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..backbone.adapter import BackboneAdapter
from ..backbone.config import BackboneConfig

logger = logging.getLogger(__name__)


@dataclass
class NoRewardWMConfig:
    encoder_dim: int = 1536
    latent_dim: int = 1536
    hidden_dim: int = 1536
    action_dim: int = 7


class RSSM(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 1536,
        latent_dim: int = 1536,
        action_dim: int = 7,
        hidden_dim: int = 1536,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=encoder_dim + action_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)
        self.posterior_mean = nn.Linear(hidden_dim + encoder_dim, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim + encoder_dim, latent_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

    def prior(self, hidden: torch.Tensor):
        mean = self.prior_mean(hidden)
        logvar = self.prior_logvar(hidden)
        return mean, logvar

    def posterior(self, hidden: torch.Tensor, encoder_output: torch.Tensor):
        combined = torch.cat([hidden, encoder_output], dim=-1)
        mean = self.posterior_mean(combined)
        logvar = self.posterior_logvar(combined)
        return mean, logvar

    def recurrent(self, latent: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor):
        combined = torch.cat([latent, action], dim=-1)
        output, hidden = self.gru(combined.unsqueeze(1), hidden)
        return hidden.squeeze(0)

    def forward(
        self,
        encoder_output: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ):
        batch_size = encoder_output.shape[0]
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device=encoder_output.device)

        prior_mean, prior_logvar = self.prior(hidden.squeeze(0))
        posterior_mean, posterior_logvar = self.posterior(hidden.squeeze(0), encoder_output)

        std = torch.exp(0.5 * posterior_logvar)
        eps = torch.randn_like(std)
        latent = posterior_mean + eps * std

        kl = torch.sum(
            posterior_logvar - prior_logvar
            + (prior_logvar.exp() + (prior_mean - posterior_mean) ** 2)
            / posterior_logvar.exp()
            - 1,
            dim=-1,
        ).mean()

        hidden = self.recurrent(latent, action, hidden.squeeze(0)).unsqueeze(0)
        latent = self.latent_proj(latent)

        return latent, hidden, kl


class ObservationDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 1536,
        output_dim: int = 1536,
        hidden_dim: int = 1536,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class NoRewardWorldModel(nn.Module):
    def __init__(
        self,
        config: Optional[NoRewardWMConfig] = None,
        encoder: Optional[BackboneAdapter] = None,
    ):
        super().__init__()
        self.config = config or NoRewardWMConfig()
        self.encoder = encoder

        if self.encoder is None:
            self.encoder = BackboneAdapter.from_config()

        self.rssm = RSSM(
            encoder_dim=self.config.encoder_dim,
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        )

        self.obs_decoder = ObservationDecoder(
            latent_dim=self.config.hidden_dim,
            output_dim=self.config.encoder_dim,
            hidden_dim=self.config.hidden_dim,
        )

        logger.info(
            f"NoRewardWorldModel: latent_dim={self.config.latent_dim}, "
            f"encoder_dim={self.config.encoder_dim}"
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_image(obs)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.encode(obs)
        latent, hidden, _ = self.rssm(encoder_output, action, hidden)
        obs_predicted = self.obs_decoder(latent)
        return obs_predicted, hidden

    def predict(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(obs, action, hidden)

    def step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(obs, action)

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        obs_predicted, hidden = self.forward(obs, action)
        next_obs_encoded = self.encode(next_obs)
        mse_loss = torch.nn.functional.mse_loss(obs_predicted, next_obs_encoded)

        encoder_output = self.encode(obs)
        _, _, kl = self.rssm(encoder_output, action, hidden)

        return mse_loss + kl

    @property
    def has_reward_head(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (
            f"NoRewardWorldModel("
            f"latent_dim={self.config.latent_dim}, "
            f"encoder_dim={self.config.encoder_dim})"
        )


def create_no_reward_wm(
    latent_dim: int = 1536,
    hidden_dim: int = 1536,
    encoder: BackboneAdapter = None,
) -> NoRewardWorldModel:
    if encoder is None:
        encoder = BackboneAdapter.from_config()

    config = NoRewardWMConfig(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        encoder_dim=encoder.vision_dim,
    )
    return NoRewardWorldModel(config=config, encoder=encoder)


def create_no_reward_wm_from_config(
    config: Optional[NoRewardWMConfig] = None,
    backbone_config: Optional[BackboneConfig] = None,
) -> NoRewardWorldModel:
    if backbone_config is None:
        backbone_config = BackboneConfig.from_defaults()

    encoder = BackboneAdapter.from_config(backbone_config)

    if config is None:
        config = NoRewardWMConfig(
            latent_dim=1536,
            hidden_dim=1536,
            encoder_dim=encoder.vision_dim,
        )

    return NoRewardWorldModel(config=config, encoder=encoder)


def compare_with_reward_wm(
    wm_with_reward: nn.Module,
    wm_no_reward: nn.Module,
    test_batch: dict,
) -> dict:
    wm_with_reward.eval()
    wm_no_reward.eval()

    with torch.no_grad():
        obs_predicted_wr, _, _ = wm_with_reward(
            test_batch["obs"],
            test_batch["action"],
        )

        obs_predicted_nr, _ = wm_no_reward(
            test_batch["obs"],
            test_batch["action"],
        )

        obs_error_with = torch.nn.functional.mse_loss(
            obs_predicted_wr,
            test_batch["next_obs"],
        )
        obs_error_without = torch.nn.functional.mse_loss(
            obs_predicted_nr,
            test_batch["next_obs"],
        )

    return {
        "obs_error_with_reward": obs_error_with.item(),
        "obs_error_no_reward": obs_error_without.item(),
        "reward_helpful": obs_error_with.item() < obs_error_without.item(),
        "improvement_pct": (
            (obs_error_without.item() - obs_error_with.item())
            / obs_error_without.item() * 100
        ) if obs_error_without.item() > 0 else 0,
    }