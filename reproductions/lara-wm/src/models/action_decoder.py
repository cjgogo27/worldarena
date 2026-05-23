"""Action decoder for LaRA-WM latent-to-joint action mapping.

Implements:
- Primary: MLP decoder from latent features → joint actions (7DoF)
- Secondary: EE pose decoder stub (stub only for v1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ActionDecoderConfig:
    """Configuration for action decoder.

    Attributes:
        latent_dim: Input latent dimension (from BackboneAdapter).
        action_dim: Output joint action dimension (default 7 for 7DoF arm).
        hidden_dim: Hidden layer dimension in MLP decoder.
        num_layers: Number of hidden layers in MLP.
        dropout: Dropout probability.
    """

    latent_dim: int = 1536  # Default from BackboneAdapter.vision_dim
    action_dim: int = 7  # Default 7DoF arm
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1


class MLPActionDecoder(nn.Module):
    """MLP decoder from latent features to joint actions.

    Architecture:
        latent → linear → relu → dropout → linear → relu → ... → linear → action_logits

    Args:
        config: ActionDecoderConfig instance.
    """

    def __init__(self, config: Optional[ActionDecoderConfig] = None):
        super().__init__()
        self.config = config or ActionDecoderConfig()

        # Build MLP layers
        layers = []
        in_dim = self.config.latent_dim
        for i in range(self.config.num_layers):
            layers.append(nn.Linear(in_dim, self.config.hidden_dim))
            layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = self.config.hidden_dim

        # Output projection
        layers.append(nn.Linear(in_dim, self.config.action_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent features to joint action logits.

        Args:
            latent: Latent feature tensor of shape (B, latent_dim) or (B, T, latent_dim)

        Returns:
            Action logits of shape (B, action_dim) or (B, T, action_dim)
        """
        # Handle sequence dimension
        squeeze = False
        if latent.dim() == 2:
            squeeze = False
        elif latent.dim() == 3:
            # (B, T, D) -> (B*T, D) -> (B*T, action_dim) -> (B, T, action_dim)
            B, T, D = latent.shape
            latent_flat = latent.reshape(B * T, D)
            actions = self.mlp(latent_flat)
            return actions.reshape(B, T, -1)

        return self.mlp(latent)


class EEPoseDecoder(nn.Module):
    """End-effector pose decoder stub.

    Returns zeros for v1 - full IK solver to be implemented in future version.

    Args:
        action_dim: EE pose dimension (default 7: 3 position + 4 quaternion).
    """

    def __init__(self, action_dim: int = 7):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to EE pose (stub returns zeros).

        Args:
            latent: Latent feature tensor of shape (B, latent_dim) or (B, T, latent_dim)

        Returns:
            EE pose tensor of shape (B, action_dim) or (B, T, action_dim)
        """
        # Return zeros with correct shape
        if latent.dim() == 2:
            batch_size = latent.shape[0]
            return torch.zeros(batch_size, self.action_dim, device=latent.device, dtype=latent.dtype)
        elif latent.dim() == 3:
            B, T, _ = latent.shape
            return torch.zeros(
                B, T, self.action_dim, device=latent.device, dtype=latent.dtype
            )
        else:
            raise ValueError(f"Invalid latent dimension: {latent.dim()}")


class ActionDecoder(nn.Module):
    """Dual-output action decoder for LaRA-WM.

    Provides:
    - Primary: Joint-space action decoder (MLP)
    - Secondary: EE pose decoder stub

    Usage:
        config = ActionDecoderConfig(latent_dim=1536, action_dim=7)
        decoder = ActionDecoder(config)

        # Encode observations
        image_features, state_features = backbone(images, states)
        latent = image_features  # Use vision features as latent

        # Decode to actions
        joint_actions = decoder.decode_joint(latent)  # Primary
        ee_pose = decoder.decode_ee(latent)  # Secondary (stub)
    """

    def __init__(
        self,
        config: Optional[ActionDecoderConfig] = None,
        backbone_adapter: Optional[nn.Module] = None,
    ):
        """Initialize action decoder.

        Args:
            config: ActionDecoderConfig instance.
            backbone_adapter: Optional BackboneAdapter to infer latent_dim.
        """
        super().__init__()
        self.config = config or ActionDecoderConfig()

        # Infer latent_dim from BackboneAdapter if provided
        if backbone_adapter is not None:
            if hasattr(backbone_adapter, "vision_dim"):
                self.config.latent_dim = backbone_adapter.vision_dim
                logger.info(
                    f"Using latent_dim from BackboneAdapter: {backbone_adapter.vision_dim}"
                )
            elif hasattr(backbone_adapter, "_vision_dim"):
                self.config.latent_dim = backbone_adapter._vision_dim
                logger.info(
                    f"Using latent_dim from BackboneAdapter: {backbone_adapter._vision_dim}"
                )

        # Initialize decoders
        self.joint_decoder = MLPActionDecoder(self.config)
        self.ee_decoder = EEPoseDecoder()

    def decode_joint(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to joint-space actions (primary output).

        Args:
            latent: Latent feature tensor

        Returns:
            Joint action logits
        """
        return self.joint_decoder(latent)

    def decode_ee(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to EE pose (secondary output, stub).

        Args:
            latent: Latent feature tensor

        Returns:
            EE pose tensor (zeros for v1)
        """
        return self.ee_decoder(latent)

    def forward(
        self, latent: torch.Tensor, return_ee: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass decoding latent to actions.

        Args:
            latent: Latent feature tensor of shape (B, latent_dim) or (B, T, latent_dim)
            return_ee: Whether to return EE pose as well

        Returns:
            If return_ee=False: Joint actions tensor
            If return_ee=True: Tuple of (joint_actions, ee_pose)
        """
        joint_actions = self.decode_joint(latent)
        if return_ee:
            ee_pose = self.decode_ee(latent)
            return joint_actions, ee_pose
        return joint_actions


def create_action_decoder(
    latent_dim: Optional[int] = None,
    action_dim: int = 7,
    backbone_adapter: Optional[nn.Module] = None,
) -> ActionDecoder:
    """Factory function to create action decoder.

    Args:
        latent_dim: Input latent dimension (overrides config default).
        action_dim: Output action dimension (default 7 for 7DoF).
        backbone_adapter: Optional BackboneAdapter to infer latent_dim.

    Returns:
        Configured ActionDecoder instance.
    """
    config = ActionDecoderConfig(
        latent_dim=latent_dim or 1536,
        action_dim=action_dim,
    )
    return ActionDecoder(config=config, backbone_adapter=backbone_adapter)