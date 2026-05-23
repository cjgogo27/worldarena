"""Latent-space reward-observation world model for LaRA-WM.

The world model predicts latent observation transitions and reward outcomes from
latent actions. It supports both recurrent and transformer transition backbones
and can unroll N-step imagined trajectories for planning.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")
F = importlib.import_module("torch.nn.functional")


TransitionArchitecture = Literal["gru", "lstm", "transformer"]
StateLossType = Literal["mse", "smooth_l1"]


@dataclass
class WorldModelConfig:
    """Configuration for the latent-space world model."""

    latent_dim: int = 1536
    action_dim: int = 1536
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    architecture: TransitionArchitecture = "gru"
    num_attention_heads: int = 8
    ff_multiplier: int = 4
    max_rollout_horizon: int = 128
    state_loss: StateLossType = "mse"
    reward_output_dim: int = 1
    state_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    use_residual_transition: bool = True


class TransformerTransition(nn.Module):
    """Causal transformer used for autoregressive latent transitions."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.position_embedding = nn.Embedding(config.max_rollout_horizon, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * config.ff_multiplier,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

    def forward(self, tokens: Any) -> Any:
        """Encode a prefix of transition tokens with a causal mask."""
        seq_len = tokens.size(1)
        if seq_len > self.config.max_rollout_horizon:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_rollout_horizon="
                f"{self.config.max_rollout_horizon}"
            )

        positions = torch.arange(seq_len, device=tokens.device)
        hidden = tokens + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        return self.encoder(hidden, mask=causal_mask)


class WorldModel(nn.Module):
    """Predict latent next-state transitions and rewards from latent actions.

    Inputs operate entirely in latent space:
        (latent_state_t, latent_action_t) -> (latent_state_{t+1}, reward_t)
    """

    def __init__(
        self,
        config: WorldModelConfig | None = None,
        latent_encoder: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()

        inferred_latent_dim = self._infer_latent_dim(latent_encoder)
        if inferred_latent_dim is not None:
            self.config.latent_dim = inferred_latent_dim
            if self.config.action_dim == WorldModelConfig.action_dim:
                self.config.action_dim = inferred_latent_dim

        self.state_projector = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
        )
        self.action_projector = nn.Sequential(
            nn.Linear(self.config.action_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
        )
        self.transition_input = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        if self.config.architecture == "gru":
            self.sequence_model: nn.Module = nn.GRU(
                input_size=self.config.hidden_dim,
                hidden_size=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif self.config.architecture == "lstm":
            self.sequence_model = nn.LSTM(
                input_size=self.config.hidden_dim,
                hidden_size=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif self.config.architecture == "transformer":
            self.sequence_model = TransformerTransition(self.config)
        else:
            raise ValueError(f"Unsupported architecture: {self.config.architecture}")

        self.state_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim + self.config.latent_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.reward_output_dim),
        )

    @staticmethod
    def _infer_latent_dim(latent_encoder: Any | None) -> int | None:
        if latent_encoder is None:
            return None

        for attr in ("latent_dim", "output_dim", "vision_dim", "hidden_size", "_vision_dim"):
            value = getattr(latent_encoder, attr, None)
            if isinstance(value, int) and value > 0:
                return value

        encoder_config = getattr(latent_encoder, "config", None)
        if encoder_config is not None:
            for attr in ("latent_dim", "output_dim", "hidden_size"):
                value = getattr(encoder_config, attr, None)
                if isinstance(value, int) and value > 0:
                    return value

        return None

    def _format_sequence(self, tensor: Any, name: str) -> tuple[Any, bool]:
        if tensor.dim() == 2:
            return tensor.unsqueeze(1), True
        if tensor.dim() != 3:
            raise ValueError(f"{name} must have shape (B, D) or (B, T, D), got {tuple(tensor.shape)}")
        return tensor, False

    def _transition_token(self, latent_state: Any, latent_action: Any) -> Any:
        state_hidden = self.state_projector(latent_state)
        action_hidden = self.action_projector(latent_action)
        return self.transition_input(torch.cat([state_hidden, action_hidden], dim=-1))

    def _predict_step_recurrent(
        self,
        latent_state: Any,
        latent_action: Any,
        recurrent_state: Any | None,
    ) -> tuple[Any, Any, Any]:
        token = self._transition_token(latent_state, latent_action).unsqueeze(1)
        if recurrent_state is None:
            hidden_init = self.state_projector(latent_state).unsqueeze(0).repeat(self.config.num_layers, 1, 1)
            if self.config.architecture == "lstm":
                recurrent_state = (hidden_init, torch.zeros_like(hidden_init))
            else:
                recurrent_state = hidden_init

        output, recurrent_state = self.sequence_model(token, recurrent_state)
        transition_hidden = output[:, -1, :]
        next_latent_state = self.state_head(transition_hidden)
        if self.config.use_residual_transition:
            next_latent_state = next_latent_state + latent_state

        reward_input = torch.cat([transition_hidden, next_latent_state], dim=-1)
        reward_prediction = self.reward_head(reward_input)
        return next_latent_state, reward_prediction, recurrent_state

    def _predict_step_transformer(
        self,
        latent_state: Any,
        latent_action: Any,
        token_history: list[Any],
    ) -> tuple[Any, Any, list[Any]]:
        token_history.append(self._transition_token(latent_state, latent_action))
        encoded = self.sequence_model(torch.stack(token_history, dim=1))
        transition_hidden = encoded[:, -1, :]
        next_latent_state = self.state_head(transition_hidden)
        if self.config.use_residual_transition:
            next_latent_state = next_latent_state + latent_state

        reward_input = torch.cat([transition_hidden, next_latent_state], dim=-1)
        reward_prediction = self.reward_head(reward_input)
        return next_latent_state, reward_prediction, token_history

    def rollout(
        self,
        initial_latent_state: Any,
        latent_actions: Any,
        horizon: int | None = None,
    ) -> dict[str, Any]:
        """Imagine an N-step trajectory from an initial latent state.

        Args:
            initial_latent_state: Tensor shaped (B, D).
            latent_actions: Tensor shaped (B, T, A) or (B, A).
            horizon: Optional rollout limit. Defaults to full latent_actions length.
        """
        action_sequence, squeezed = self._format_sequence(latent_actions, "latent_actions")
        if initial_latent_state.dim() != 2:
            raise ValueError(
                f"initial_latent_state must have shape (B, D), got {tuple(initial_latent_state.shape)}"
            )

        total_steps = int(action_sequence.size(1))
        rollout_horizon = total_steps if horizon is None else horizon
        if rollout_horizon < 1:
            raise ValueError("horizon must be >= 1")
        if rollout_horizon > total_steps:
            raise ValueError(
                f"horizon={rollout_horizon} exceeds available action steps={total_steps}"
            )

        current_state = initial_latent_state
        predicted_states: list[Any] = []
        reward_predictions: list[Any] = []
        recurrent_state: Any | None = None
        token_history: list[Any] = []

        for step in range(rollout_horizon):
            action_t = action_sequence[:, step, :]
            if self.config.architecture == "transformer":
                next_state, reward_pred, token_history = self._predict_step_transformer(
                    current_state,
                    action_t,
                    token_history,
                )
            else:
                next_state, reward_pred, recurrent_state = self._predict_step_recurrent(
                    current_state,
                    action_t,
                    recurrent_state,
                )

            predicted_states.append(next_state)
            reward_predictions.append(reward_pred)
            current_state = next_state

        next_states = torch.stack(predicted_states, dim=1)
        rewards = torch.stack(reward_predictions, dim=1)

        if squeezed:
            next_states = next_states[:, 0, :]
            rewards = rewards[:, 0, :]

        return {
            "next_latent_states": next_states,
            "reward_predictions": rewards,
        }

    def forward(
        self,
        latent_state: Any,
        latent_action: Any,
        horizon: int | None = None,
    ) -> dict[str, Any]:
        """Predict latent next states and reward for one or more steps."""
        if latent_state.dim() == 3:
            initial_state = latent_state[:, 0, :]
        elif latent_state.dim() == 2:
            initial_state = latent_state
        else:
            raise ValueError(
                f"latent_state must have shape (B, D) or (B, T, D), got {tuple(latent_state.shape)}"
            )

        predictions = self.rollout(initial_state, latent_action, horizon=horizon)
        reward_predictions = predictions["reward_predictions"]
        if reward_predictions.size(-1) == 1:
            predictions["reward_predictions"] = reward_predictions.squeeze(-1)
        return predictions

    def compute_loss(
        self,
        latent_state: Any,
        latent_action: Any,
        target_next_latent_state: Any,
        target_reward: Any,
    ) -> dict[str, Any]:
        """Compute joint latent-transition and reward-prediction training loss."""
        target_states, squeezed_states = self._format_sequence(
            target_next_latent_state,
            "target_next_latent_state",
        )
        horizon = target_states.size(1)
        predictions = self.forward(latent_state, latent_action, horizon=horizon)

        predicted_states, _ = self._format_sequence(predictions["next_latent_states"], "predicted_next_latent_states")
        reward_predictions = predictions["reward_predictions"]

        if self.config.state_loss == "smooth_l1":
            state_loss = F.smooth_l1_loss(predicted_states, target_states)
        else:
            state_loss = F.mse_loss(predicted_states, target_states)

        reward_loss = self._compute_reward_loss(reward_predictions, target_reward)
        total_loss = (
            self.config.state_loss_weight * state_loss
            + self.config.reward_loss_weight * reward_loss
        )

        if squeezed_states and predicted_states.dim() == 3:
            predicted_states = predicted_states[:, 0, :]

        return {
            "loss": total_loss,
            "state_loss": state_loss,
            "reward_loss": reward_loss,
            "predicted_next_latent_states": predicted_states,
            "predicted_rewards": reward_predictions,
        }

    def _compute_reward_loss(
        self,
        reward_predictions: Any,
        target_reward: Any,
    ) -> Any:
        target = target_reward.to(reward_predictions.device)
        if target.dim() == 0:
            target = target.unsqueeze(0)

        if reward_predictions.dim() == 1:
            reward_predictions = reward_predictions.unsqueeze(-1)

        if target.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
            if reward_predictions.size(-1) == 1:
                target = target.float()
                if target.dim() < reward_predictions.dim():
                    target = target.unsqueeze(-1)
                return F.binary_cross_entropy_with_logits(reward_predictions, target)

            if target.dim() == reward_predictions.dim() and target.size(-1) == 1:
                target = target.squeeze(-1)
            return F.cross_entropy(
                reward_predictions.reshape(-1, reward_predictions.size(-1)),
                target.reshape(-1).long(),
            )

        target = target.float()
        if target.dim() < reward_predictions.dim():
            target = target.unsqueeze(-1)
        return F.mse_loss(reward_predictions, target)


def create_world_model(
    latent_dim: int | None = None,
    action_dim: int | None = None,
    architecture: TransitionArchitecture = "gru",
    latent_encoder: Any | None = None,
    **kwargs: Any,
) -> WorldModel:
    """Factory for creating a configurable latent-space world model."""
    config = WorldModelConfig(
        latent_dim=latent_dim or WorldModelConfig.latent_dim,
        action_dim=action_dim or latent_dim or WorldModelConfig.action_dim,
        architecture=architecture,
        **kwargs,
    )
    return WorldModel(config=config, latent_encoder=latent_encoder)
