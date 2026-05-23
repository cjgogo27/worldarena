"""Direct policy baseline for LaRA-WM.

Forward backbone features directly to actions (no latent space).

Architecture:
    image + state → BackboneAdapter → features → MLP → action

This baseline bypasses the latent space bottleneck by directly mapping
backbone-encoded features to action outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..backbone.adapter import BackboneAdapter
from ..eval.harness import PolicyHook

logger = logging.getLogger(__name__)


@dataclass
class DirectPolicyConfig:
    """Configuration for direct policy baseline.

    Attributes:
        backbone: Backbone configuration (uses BackboneConfig defaults).
        action_dim: Output action dimension (default 7 for 7DoF arm).
        hidden_dim: Hidden layer dimension in direct MLP.
        num_layers: Number of hidden layers.
        dropout: Dropout probability.
        use_state: Whether to include state features in input.
    """

    action_dim: int = 7
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    use_state: bool = True


class DirectPolicy(nn.Module):
    """Direct policy: backbone features → action (no latent space).

    Forward pass:
        image_features, state_features = backbone(images, states)
        combined = concat(image_features, state_features) if use_state
        action = mlp(combined)

    Args:
        config: DirectPolicyConfig instance.
        backbone_adapter: BackboneAdapter for feature extraction.
    """

    def __init__(
        self,
        config: Optional[DirectPolicyConfig] = None,
        backbone_adapter: Optional[BackboneAdapter] = None,
    ):
        super().__init__()
        self.config = config or DirectPolicyConfig()
        self.backbone_adapter = backbone_adapter

        # Infer feature dimensions
        vision_dim = (
            backbone_adapter.vision_dim
            if backbone_adapter is not None
            else 1536  # default from Qwen
        )
        state_dim = 14  # default state dimension

        self.use_state = self.config.use_state
        input_dim = vision_dim + (state_dim if self.use_state else 0)

        logger.info(
            f"DirectPolicy: input_dim={input_dim}, "
            f"action_dim={self.config.action_dim}"
        )

        # Build direct MLP: features -> action
        layers = []
        in_dim = input_dim
        for i in range(self.config.num_layers):
            layers.append(nn.Linear(in_dim, self.config.hidden_dim))
            layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = self.config.hidden_dim

        layers.append(nn.Linear(in_dim, self.config.action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass directly to actions.

        Args:
            images: Image tensor of shape (B, C, H, W) or (B, T, C, H, W)
            states: State tensor of shape (B, D) or (B, T, D)

        Returns:
            Action tensor of shape (B, action_dim) or (B, T, action_dim)
        """
        if self.backbone_adapter is None:
            raise RuntimeError("No backbone_adapter provided")

        # Encode features
        if images is not None:
            image_features = self.backbone_adapter.encode_image(images)
        else:
            B = states.shape[0] if states is not None else 1
            image_features = torch.zeros(
                B, self.backbone_adapter.vision_dim,
                device=states.device if states else "cpu"
            )

        if self.use_state and states is not None:
            state_features = self.backbone_adapter.encode_state(states)
            # Concatenate image + state features
            combined = torch.cat([image_features, state_features], dim=-1)
        else:
            combined = image_features

        # Handle sequence dimension
        squeeze = False
        if combined.dim() == 2:
            squeeze = False
        elif combined.dim() == 3:
            batch_size, seq_len, feat_dim = combined.shape
            combined = combined.reshape(batch_size * seq_len, feat_dim)
            actions = self.mlp(combined)
            return actions.reshape(batch_size, seq_len, -1)

        return self.mlp(combined)


class DirectPolicyHook(PolicyHook):
    """PolicyHook wrapper for DirectPolicy baseline.

    Compatible with EvaluationHarness interface.

    Usage:
        policy = DirectPolicyHook(backbone=backbone)
        harness = EvaluationHarness(policy=policy, env_factory=env_factory)
        results, metrics = harness.evaluate(instructions, num_episodes=100)
    """

    def __init__(
        self,
        backbone: Optional[BackboneAdapter] = None,
        config: Optional[DirectPolicyConfig] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.backbone = backbone
        self.config = config or DirectPolicyConfig()
        self.device = device

        self._policy: Optional[DirectPolicy] = None
        self._model: Optional[BackboneAdapter] = None
        self._action_fn = self._direct_action

    def get_model(self, config: dict) -> "DirectPolicyHook":
        """Initialize direct policy."""
        if self.backbone is None:
            self.backbone = BackboneAdapter.from_config()
            self.backbone.load()

        self._policy = DirectPolicy(
            config=self.config,
            backbone_adapter=self.backbone,
        )
        self._policy.to(self.device)
        self._policy.eval()

        self._model = self.backbone
        return self

    def eval(
        self, env: Any, model: "DirectPolicyHook", observation: dict
    ) -> np.ndarray:
        """Get action."""
        return self._direct_action(env, model, observation)

    def _direct_action(
        self, env: Any, model: "DirectPolicyHook", observation: dict
    ) -> np.ndarray:
        """Direct forward: observation → action."""
        image = observation.get("image")
        state = observation.get("state")

        with torch.no_grad():
            img_tensor = None
            if image is not None:
                img_tensor = (
                    torch.from_numpy(image)
                    .unsqueeze(0)
                    .to(model.device)
                )

            state_tensor = None
            if state is not None:
                state_tensor = (
                    torch.from_numpy(state)
                    .unsqueeze(0)
                    .float()
                    .to(model.device)
                )

            if model._policy is not None:
                action = model._policy(img_tensor, state_tensor)
                return action.cpu().numpy().squeeze(0)

        # Fallback: return zeros
        action_dim = env.action_space.shape[0] if hasattr(env, "action_space") else 7
        return np.zeros(action_dim)

    def reset_model(self, model: "DirectPolicyHook") -> None:
        """Reset model state between episodes."""
        pass


def create_direct_policy(
    backbone: Optional[BackboneAdapter] = None,
    config: Optional[DirectPolicyConfig] = None,
) -> tuple[DirectPolicy, BackboneAdapter]:
    """Factory function to create direct policy baseline.

    Args:
        backbone: BackboneAdapter instance (creates if None).
        config: DirectPolicyConfig instance.

    Returns:
        Tuple of (DirectPolicy, BackboneAdapter).
    """
    if backbone is None:
        backbone = BackboneAdapter.from_config()
        backbone.load()

    policy = DirectPolicy(
        config=config,
        backbone_adapter=backbone,
    )

    return policy, backbone