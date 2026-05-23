"""Test-time latent refinement for LaRA-WM.

Optimizes a latent state before final action decoding by differentiating through an
action decoder and latent-space world model. Refinement is bounded by a fixed
number of optimization steps and automatically falls back to the original latent
when the optimization becomes unstable or fails to improve the objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_CONFIG_PATH = Path("/data/alice/cjtest/lara-wm/configs/config.yaml")


@dataclass(frozen=True)
class LatentRefinementConfig:
    """Configuration for gradient-based latent refinement."""

    enabled: bool = True
    steps: int = 5
    learning_rate: float = 1e-2
    reward_weight: float = 1.0
    observation_weight: float = 1.0
    latent_anchor_weight: float = 1e-2
    max_grad_norm: float = 10.0
    max_latent_shift: float = 5.0
    min_improvement: float = 1e-6
    divergence_tolerance: float = 5.0
    stop_on_non_finite: bool = True

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]] = None) -> "LatentRefinementConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            steps=int(data.get("steps", cls.steps)),
            learning_rate=float(data.get("learning_rate", data.get("lr", cls.learning_rate))),
            reward_weight=float(data.get("reward_weight", cls.reward_weight)),
            observation_weight=float(data.get("observation_weight", cls.observation_weight)),
            latent_anchor_weight=float(
                data.get("latent_anchor_weight", cls.latent_anchor_weight)
            ),
            max_grad_norm=float(data.get("max_grad_norm", cls.max_grad_norm)),
            max_latent_shift=float(data.get("max_latent_shift", cls.max_latent_shift)),
            min_improvement=float(data.get("min_improvement", cls.min_improvement)),
            divergence_tolerance=float(
                data.get("divergence_tolerance", cls.divergence_tolerance)
            ),
            stop_on_non_finite=bool(data.get("stop_on_non_finite", cls.stop_on_non_finite)),
        )

    @classmethod
    def from_yaml(
        cls,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
    ) -> "LatentRefinementConfig":
        path = Path(config_path)
        if not path.exists():
            return cls()

        import yaml

        config = yaml.safe_load(path.read_text()) or {}
        section = config.get("latent_refinement", config)
        return cls.from_dict(section)


@dataclass
class LatentRefinementResult:
    """Structured result from test-time latent optimization."""

    refined_latent: Tensor
    initial_latent: Tensor
    refined_action: Optional[Tensor]
    initial_action: Optional[Tensor]
    fallback_used: bool
    success: bool
    steps_taken: int
    initial_loss: float
    best_loss: float
    reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


class LatentRefiner:
    """Gradient-based latent optimizer with stability-aware fallback logic."""

    def __init__(self, config: Optional[LatentRefinementConfig] = None):
        self.config = config or LatentRefinementConfig()

    @classmethod
    def from_config(
        cls,
        config: Optional[LatentRefinementConfig] = None,
    ) -> "LatentRefiner":
        return cls(config=config or LatentRefinementConfig.from_yaml())

    def refine(
        self,
        latent: Tensor,
        world_model: nn.Module,
        action_decoder: nn.Module,
        target_next_observation: Optional[Tensor] = None,
        target_reward: Optional[Tensor] = None,
        horizon: Optional[int] = None,
    ) -> LatentRefinementResult:
        """Refine a latent before action decoding.

        Args:
            latent: Current latent state, shape ``(B, D)`` or ``(B, T, D)``.
            world_model: Module mapping ``(latent_state, latent_action)`` to reward
                and next latent observation predictions.
            action_decoder: Module used to decode the latent into actions during
                optimization.
            target_next_observation: Optional target next latent observation used for
                an observation-matching term.
            target_reward: Optional target reward used for a supervised reward term.
            horizon: Optional rollout horizon forwarded to the world model.
        """
        initial_latent = latent.detach().clone()
        initial_action = self._decode_action(action_decoder, initial_latent)

        if not self.config.enabled:
            return self._fallback_result(
                initial_latent=initial_latent,
                initial_action=initial_action,
                reason="refinement_disabled",
                initial_loss=0.0,
                best_loss=0.0,
                steps_taken=0,
            )

        if self.config.steps <= 0:
            return self._fallback_result(
                initial_latent=initial_latent,
                initial_action=initial_action,
                reason="non_positive_steps",
                initial_loss=0.0,
                best_loss=0.0,
                steps_taken=0,
            )

        if self.config.learning_rate <= 0:
            return self._fallback_result(
                initial_latent=initial_latent,
                initial_action=initial_action,
                reason="non_positive_learning_rate",
                initial_loss=0.0,
                best_loss=0.0,
                steps_taken=0,
            )

        refinement_latent = initial_latent.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([refinement_latent], lr=self.config.learning_rate)

        try:
            initial_loss_tensor, initial_metrics = self._objective(
                latent=initial_latent,
                initial_latent=initial_latent,
                world_model=world_model,
                action_decoder=action_decoder,
                target_next_observation=target_next_observation,
                target_reward=target_reward,
                horizon=horizon,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            return self._fallback_result(
                initial_latent=initial_latent,
                initial_action=initial_action,
                reason=f"initial_objective_failed:{exc}",
                initial_loss=float("inf"),
                best_loss=float("inf"),
                steps_taken=0,
            )

        initial_loss = float(initial_loss_tensor.detach().item())
        if not torch.isfinite(initial_loss_tensor):
            return self._fallback_result(
                initial_latent=initial_latent,
                initial_action=initial_action,
                reason="initial_loss_non_finite",
                initial_loss=initial_loss,
                best_loss=initial_loss,
                steps_taken=0,
            )

        best_latent = initial_latent.clone()
        best_action = initial_action.detach().clone() if initial_action is not None else None
        best_loss = initial_loss
        best_metrics = self._tensor_metrics_to_floats(initial_metrics)
        steps_taken = 0

        for step in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)

            try:
                loss, _ = self._objective(
                    latent=refinement_latent,
                    initial_latent=initial_latent,
                    world_model=world_model,
                    action_decoder=action_decoder,
                    target_next_observation=target_next_observation,
                    target_reward=target_reward,
                    horizon=horizon,
                )
            except Exception as exc:  # pragma: no cover - defensive path
                return self._fallback_result(
                    initial_latent=initial_latent,
                    initial_action=initial_action,
                    reason=f"objective_failed_at_step_{step}:{exc}",
                    initial_loss=initial_loss,
                    best_loss=best_loss,
                    steps_taken=steps_taken,
                )

            if not torch.isfinite(loss):
                return self._fallback_result(
                    initial_latent=initial_latent,
                    initial_action=initial_action,
                    reason=f"non_finite_loss_at_step_{step}",
                    initial_loss=initial_loss,
                    best_loss=best_loss,
                    steps_taken=steps_taken,
                )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([refinement_latent], self.config.max_grad_norm)
            if not torch.isfinite(grad_norm):
                return self._fallback_result(
                    initial_latent=initial_latent,
                    initial_action=initial_action,
                    reason=f"non_finite_grad_at_step_{step}",
                    initial_loss=initial_loss,
                    best_loss=best_loss,
                    steps_taken=steps_taken,
                )

            optimizer.step()
            steps_taken = step + 1

            if not torch.isfinite(refinement_latent).all():
                return self._fallback_result(
                    initial_latent=initial_latent,
                    initial_action=initial_action,
                    reason=f"latent_diverged_at_step_{step}",
                    initial_loss=initial_loss,
                    best_loss=best_loss,
                    steps_taken=steps_taken,
                )

            latent_shift = self._batch_norm(refinement_latent.detach() - initial_latent).mean()
            if float(latent_shift.item()) > self.config.max_latent_shift:
                return self._fallback_result(
                    initial_latent=initial_latent,
                    initial_action=initial_action,
                    reason=f"latent_shift_exceeded_at_step_{step}",
                    initial_loss=initial_loss,
                    best_loss=best_loss,
                    steps_taken=steps_taken,
                )

            with torch.no_grad():
                current_loss_tensor, current_metrics = self._objective(
                    latent=refinement_latent,
                    initial_latent=initial_latent,
                    world_model=world_model,
                    action_decoder=action_decoder,
                    target_next_observation=target_next_observation,
                    target_reward=target_reward,
                    horizon=horizon,
                )

            current_loss = float(current_loss_tensor.detach().item())
            if not torch.isfinite(current_loss_tensor):
                return self._fallback_result(
                    initial_latent=initial_latent,
                    initial_action=initial_action,
                    reason=f"non_finite_post_step_loss_at_step_{step}",
                    initial_loss=initial_loss,
                    best_loss=best_loss,
                    steps_taken=steps_taken,
                )

            if current_loss <= best_loss - self.config.min_improvement:
                best_loss = current_loss
                best_latent = refinement_latent.detach().clone()
                best_action = self._decode_action(action_decoder, best_latent)
                best_metrics = self._tensor_metrics_to_floats(current_metrics)

            allowed_loss = initial_loss + abs(initial_loss) * self.config.divergence_tolerance
            if current_loss > allowed_loss:
                return self._fallback_result(
                    initial_latent=initial_latent,
                    initial_action=initial_action,
                    reason=f"loss_diverged_at_step_{step}",
                    initial_loss=initial_loss,
                    best_loss=best_loss,
                    steps_taken=steps_taken,
                )

        if best_loss >= initial_loss - self.config.min_improvement:
            return self._fallback_result(
                initial_latent=initial_latent,
                initial_action=initial_action,
                reason="no_stable_improvement",
                initial_loss=initial_loss,
                best_loss=best_loss,
                steps_taken=steps_taken,
            )

        return LatentRefinementResult(
            refined_latent=best_latent,
            initial_latent=initial_latent,
            refined_action=best_action,
            initial_action=initial_action,
            fallback_used=False,
            success=True,
            steps_taken=steps_taken,
            initial_loss=initial_loss,
            best_loss=best_loss,
            reason="refinement_succeeded",
            metrics=best_metrics,
        )

    def refine_and_decode(
        self,
        latent: Tensor,
        world_model: nn.Module,
        action_decoder: nn.Module,
        target_next_observation: Optional[Tensor] = None,
        target_reward: Optional[Tensor] = None,
        horizon: Optional[int] = None,
    ) -> tuple[Tensor, LatentRefinementResult]:
        """Refine a latent and return the action decoded from the chosen latent."""
        result = self.refine(
            latent=latent,
            world_model=world_model,
            action_decoder=action_decoder,
            target_next_observation=target_next_observation,
            target_reward=target_reward,
            horizon=horizon,
        )
        action = result.refined_action
        if action is None:
            action = self._decode_action(action_decoder, result.refined_latent)
            result.refined_action = action
        return action, result

    def _objective(
        self,
        latent: Tensor,
        initial_latent: Tensor,
        world_model: nn.Module,
        action_decoder: nn.Module,
        target_next_observation: Optional[Tensor],
        target_reward: Optional[Tensor],
        horizon: Optional[int],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        action = action_decoder(latent)
        predictions = self._forward_world_model(world_model, latent, action, horizon=horizon)

        reward_prediction = self._extract_prediction(
            predictions,
            keys=("reward_predictions", "predicted_rewards", "reward", "rewards"),
        )
        observation_prediction = self._extract_prediction(
            predictions,
            keys=(
                "next_latent_states",
                "predicted_next_latent_states",
                "predicted_observation",
                "predicted_observations",
                "observation",
                "observations",
            ),
        )

        reward_term = latent.new_zeros(())
        if target_reward is not None and reward_prediction is not None:
            reward_target = target_reward.to(reward_prediction.device, reward_prediction.dtype)
            reward_term = F.mse_loss(reward_prediction, reward_target)
        elif reward_prediction is not None:
            reward_term = -reward_prediction.mean()

        observation_term = latent.new_zeros(())
        if target_next_observation is not None:
            if observation_prediction is None:
                raise ValueError("world model did not return an observation prediction")
            target_obs = target_next_observation.to(
                observation_prediction.device,
                observation_prediction.dtype,
            )
            observation_term = F.mse_loss(observation_prediction, target_obs)

        if reward_prediction is None and target_next_observation is None:
            raise ValueError(
                "latent refinement requires a reward prediction or target_next_observation"
            )

        anchor_term = F.mse_loss(latent, initial_latent)
        total_loss = (
            self.config.reward_weight * reward_term
            + self.config.observation_weight * observation_term
            + self.config.latent_anchor_weight * anchor_term
        )
        metrics = {
            "reward_term": reward_term.detach(),
            "observation_term": observation_term.detach(),
            "anchor_term": anchor_term.detach(),
            "decoded_action_norm": action.detach().norm(dim=-1).mean(),
            "latent_shift": self._batch_norm(latent.detach() - initial_latent).mean(),
        }
        if reward_prediction is not None:
            metrics["predicted_reward_mean"] = reward_prediction.detach().mean()
        if observation_prediction is not None:
            metrics["predicted_observation_norm"] = observation_prediction.detach().norm(dim=-1).mean()
        return total_loss, metrics

    @staticmethod
    def _decode_action(action_decoder: nn.Module, latent: Tensor) -> Tensor:
        return action_decoder(latent)

    @staticmethod
    def _forward_world_model(
        world_model: nn.Module,
        latent: Tensor,
        action: Tensor,
        horizon: Optional[int],
    ) -> Any:
        if horizon is None:
            return world_model(latent, action)
        return world_model(latent, action, horizon=horizon)

    @staticmethod
    def _extract_prediction(predictions: Any, keys: tuple[str, ...]) -> Optional[Tensor]:
        if isinstance(predictions, dict):
            for key in keys:
                value = predictions.get(key)
                if isinstance(value, Tensor):
                    return value
            return None

        if isinstance(predictions, (tuple, list)):
            for value in predictions:
                if isinstance(value, Tensor):
                    return value
        return predictions if isinstance(predictions, Tensor) else None

    @staticmethod
    def _tensor_metrics_to_floats(metrics: dict[str, Tensor]) -> dict[str, float]:
        return {key: float(value.detach().item()) for key, value in metrics.items()}

    @staticmethod
    def _batch_norm(tensor: Tensor) -> Tensor:
        return torch.linalg.vector_norm(tensor.reshape(tensor.size(0), -1), dim=-1)

    def _fallback_result(
        self,
        initial_latent: Tensor,
        initial_action: Optional[Tensor],
        reason: str,
        initial_loss: float,
        best_loss: float,
        steps_taken: int,
    ) -> LatentRefinementResult:
        return LatentRefinementResult(
            refined_latent=initial_latent,
            initial_latent=initial_latent,
            refined_action=initial_action,
            initial_action=initial_action,
            fallback_used=True,
            success=False,
            steps_taken=steps_taken,
            initial_loss=initial_loss,
            best_loss=best_loss,
            reason=reason,
            metrics={},
        )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "LatentRefinementConfig",
    "LatentRefinementResult",
    "LatentRefiner",
]
