"""Latent No-Refine baseline for LaRA-WM.

This baseline tests whether latent representations from the vision backbone
alone (without refinement) can provide useful training signals for action prediction.

Key differences from full LaRA-WM:
- NO iterative refinement at inference time
- Direct latent → action mapping
- Tests the baseline contribution of latent encoder + world model

Usage:
    from lara_wm.src.baselines.latent_no_refine import LatentNoRefineBaseline
    
    baseline = LatentNoRefineBaseline(
        backbone_adapter=backbone,
        latent_dim=1536,
        action_dim=7,
    )
    
    # Training
    baseline.train(episodes)
    
    # Inference (NO refinement)
    action = baseline.predict(observation)  # Direct, no refinement loop
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from ..backbone.adapter import BackboneAdapter
from ..models.action_decoder import ActionDecoder, ActionDecoderConfig
from ..data.robottwin_reader import RoboTwinEpisode

logger = logging.getLogger(__name__)


@dataclass
class LatentNoRefineConfig:
    """Configuration for Latent No-Refine baseline.

    Attributes:
        latent_dim: Dimension of latent space from encoder.
        action_dim: Output action dimension (default 7 for 7DoF).
        hidden_dim: Hidden layer dimension in world model MLP.
        num_layers: Number of layers in world model.
        learning_rate: Learning rate for training.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        device: Device for computation.
    """

    latent_dim: int = 1536
    action_dim: int = 7
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    device: str = "cuda"


class WorldModelMLP(nn.Module):
    """Simple world model MLP that predicts next latent given current latent and action.

    This provides the training signal by predicting latent transitions,
    testing whether learning latent dynamics helps action prediction.

    Architecture:
        latent + action → MLP → next_latent prediction
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Input: latent + action
        layers = []
        in_dim = latent_dim + action_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output: predicted next latent
        layers.append(nn.Linear(in_dim, latent_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent state.

        Args:
            latent: Current latent tensor (B, latent_dim) or (B, T, latent_dim)
            action: Action tensor (B, action_dim) or (B, T, action_dim)

        Returns:
            Predicted next latent tensor
        """
        # Handle sequence dimension
        squeeze = False
        if latent.dim() == 2:
            squeeze = False
        elif latent.dim() == 3:
            B, T, D = latent.shape
            latent = latent.reshape(B * T, D)
            action = action.reshape(B * T, self.action_dim)

        # Concatenate latent and action
        combined = torch.cat([latent, action], dim=-1)
        predicted_next_latent = self.mlp(combined)

        if squeeze:
            predicted_next_latent = predicted_next_latent.squeeze(0)

        return predicted_next_latent


class LatentNoRefineBaseline(nn.Module):
    """Latent No-Refine baseline.

    Tests whether latent encoder + world model provides useful training signal,
    WITHOUT iterative refinement at inference time.

    Components:
    1. Latent Encoder: BackboneAdapter (vision features)
    2. World Model: MLP predicting next latent from (latent, action)
    3. Action Decoder: MLP mapping latent → actions

    Training:
        - World model: predict next_latent from (current_latent, action)
        - Action decoder: predict action from latent
        - Combined loss encourages latent space that supports both tasks

    Inference:
        - Encode observation to latent
        - Decode latent to action
        - NO refinement loop (unlike full LaRA-WM)

    Args:
        config: LatentNoRefineConfig instance
        backbone_adapter: BackboneAdapter for vision encoding
    """

    def __init__(
        self,
        config: Optional[LatentNoRefineConfig] = None,
        backbone_adapter: Optional[BackboneAdapter] = None,
    ):
        super().__init__()
        self.config = config or LatentNoRefineConfig()
        self.backbone = backbone_adapter

        # Initialize world model
        self.world_model = WorldModelMLP(
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

        # Initialize action decoder (from latent → action)
        decoder_config = ActionDecoderConfig(
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=2,
            dropout=self.config.dropout,
        )
        self.action_decoder = ActionDecoder(config=decoder_config)

        # Optimizer
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Training state
        self._is_fitted = False

    def _ensure_backbone(self) -> None:
        """Ensure backbone is loaded."""
        if self.backbone is None:
            self.backbone = BackboneAdapter.from_config()
        if self.backbone.model is None:
            self.backbone.load()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent features.

        Args:
            images: Image tensor (B, C, H, W) or (B, T, C, H, W)

        Returns:
            Latent features (B, latent_dim) or (B, T, latent_dim)
        """
        self._ensure_backbone()
        return self.backbone.encode_image(images)

    def predict(self, observation: Dict[str, Any]) -> np.ndarray:
        """Predict action from observation (NO refinement).

        Args:
            observation: Dict with 'image' and optionally 'state'

        Returns:
            Action array (action_dim,)
        """
        self.eval()
        with torch.no_grad():
            # Encode to latent
            image = observation.get("image")
            if image is not None:
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image)
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                image = image.to(self.config.device)
                latent = self.encode(image)
            else:
                # Fallback: zeros
                latent = torch.zeros(1, self.config.latent_dim, device=self.config.device)

            # Decode to action (NO refinement)
            action_logits = self.action_decoder.decode_joint(latent)
            action = action_logits.squeeze(0)

            # Convert to numpy
            if action.device.type == "cuda":
                action = action.cpu()
            return action.numpy()

    def forward(
        self,
        images: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            images: Image tensor (B, C, H, W)
            actions: Action tensor (B, action_dim) for world model training

        Returns:
            Dict with 'latent', 'action_logits', and optionally 'predicted_next_latent'
        """
        # Encode images to latent
        latent = self.encode(images)

        # Decode to action logits
        action_logits = self.action_decoder.decode_joint(latent)

        outputs = {
            "latent": latent,
            "action_logits": action_logits,
        }

        # World model prediction if actions provided
        if actions is not None:
            predicted_next = self.world_model(latent, actions)
            outputs["predicted_next_latent"] = predicted_next

        return outputs

    def compute_loss(
        self,
        images: torch.Tensor,
        target_actions: torch.Tensor,
        target_next_images: Optional[torch.Tensor] = None,
        lambda_action: float = 1.0,
        lambda_world_model: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss.

        Args:
            images: Current image observations (B, C, H, W)
            target_actions: Target actions (B, action_dim)
            target_next_images: Next image observations for world model (optional)
            lambda_action: Weight for action loss
            lambda_world_model: Weight for world model loss

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        outputs = self(images, actions=target_actions)

        # Action prediction loss
        action_loss = F.mse_loss(outputs["action_logits"], target_actions)

        loss_dict = {"action_loss": action_loss.item()}
        total_loss = lambda_action * action_loss

        # World model loss: predict next latent from (latent, action)
        if target_next_images is not None:
            with torch.no_grad():
                next_latent = self.encode(target_next_images)

            predicted_next = outputs["predicted_next_latent"]
            world_model_loss = F.mse_loss(predicted_next, next_latent)

            loss_dict["world_model_loss"] = world_model_loss.item()
            total_loss = total_loss + lambda_world_model * world_model_loss

        return total_loss, loss_dict

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        lambda_action: float = 1.0,
        lambda_world_model: float = 0.5,
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Dict with 'images', 'actions', optionally 'next_images'
            lambda_action: Weight for action loss
            lambda_world_model: Weight for world model loss

        Returns:
            Loss dict
        """
        self.train()

        images = batch["images"].to(self.config.device)
        actions = batch["actions"].to(self.config.device)
        next_images = batch.get("next_images")
        if next_images is not None:
            next_images = next_images.to(self.config.device)

        loss, loss_dict = self.compute_loss(
            images, actions, next_images, lambda_action, lambda_world_model
        )

        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict["total_loss"] = loss.item()
        return loss_dict

    def fit(
        self,
        episodes: List[RoboTwinEpisode],
        val_episodes: Optional[List[RoboTwinEpisode]] = None,
        lambda_action: float = 1.0,
        lambda_world_model: float = 0.5,
        save_path: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Train the baseline on episodes.

        Args:
            episodes: List of training episodes
            val_episodes: Optional validation episodes
            lambda_action: Weight for action loss
            lambda_world_model: Weight for world model loss
            save_path: Optional path to save checkpoint

        Returns:
            Dict of training curves
        """
        # Create dataset
        dataset = LatentNoRefineDataset(
            episodes,
            backbone=self.backbone,
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Setup optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
            )

        # Training loop
        self.train()
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.epochs):
            epoch_losses = []

            for batch in dataloader:
                loss_dict = self.train_step(batch, lambda_action, lambda_world_model)
                epoch_losses.append(loss_dict["total_loss"])

            avg_loss = np.mean(epoch_losses)
            history["train_loss"].append(avg_loss)

            # Validation
            if val_episodes is not None and len(val_episodes) > 0:
                val_loss = self._validate(val_episodes, lambda_action, lambda_world_model)
                history["val_loss"].append(val_loss)
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Train: {avg_loss:.4f} | Val: {val_loss:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} | Train: {avg_loss:.4f}")

            # Save checkpoint
            if save_path is not None and (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch+1}.pt")

        self._is_fitted = True
        return history

    def _validate(
        self,
        episodes: List[RoboTwinEpisode],
        lambda_action: float,
        lambda_world_model: float,
    ) -> float:
        """Validate on episodes."""
        self.eval()
        val_dataset = LatentNoRefineDataset(
            episodes,
            backbone=self.backbone,
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        losses = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(self.config.device)
                actions = batch["actions"].to(self.config.device)
                next_images = batch.get("next_images")
                if next_images is not None:
                    next_images = next_images.to(self.config.device)

                _, loss_dict = self.compute_loss(
                    images, actions, next_images, lambda_action, lambda_world_model
                )
                losses.append(loss_dict["total_loss"])

        return np.mean(losses)

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        self.eval()
        checkpoint = {
            "config": self.config,
            "state_dict": self.state_dict(),
            "is_fitted": self._is_fitted,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.config = checkpoint["config"]
        self.load_state_dict(checkpoint["state_dict"])
        self._is_fitted = checkpoint.get("is_fitted", False)
        logger.info(f"Loaded checkpoint from {path}")

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._is_fitted


class LatentNoRefineDataset(Dataset):
    """Dataset for Latent No-Refine training.

    Prepares (current_image, action, next_image) tuples from episodes.
    """

    def __init__(
        self,
        episodes: List[RoboTwinEpisode],
        backbone: Optional[BackboneAdapter] = None,
        latent_dim: int = 1536,
        action_dim: int = 7,
        max_steps_per_episode: int = 100,
    ):
        self.episodes = episodes
        self.backbone = backbone
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.max_steps = max_steps_per_episode

        # Build index of valid transitions
        self._samples: List[Dict[str, Any]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Build index of valid (episode, timestep) pairs."""
        for ep_idx, episode in enumerate(self.episodes):
            joint_states = episode.get("joint_states")
            if joint_states is None:
                continue

            T = min(len(joint_states), self.max_steps)
            for t in range(T - 1):  # Need current and next
                self._samples.append({"episode_idx": ep_idx, "timestep": t})

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self._samples[index]
        ep_idx = sample["episode_idx"]
        t = sample["timestep"]

        episode = self.episodes[ep_idx]

        # Get image observation (use agent_view or first image key)
        image_obs = episode.image_obs
        if image_obs:
            key = list(image_obs.keys())[0]
            image = image_obs[key][t]
        else:
            image = np.zeros((3, 224, 224), dtype=np.uint8)

        # Convert to tensor (C, H, W)
        if image.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
            image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float() / 255.0

        # Get action (joint_position)
        actions = episode.get("joint_position")
        if actions is not None:
            action = torch.from_numpy(actions[t])
        else:
            action = torch.zeros(self.action_dim)

        # Get next image for world model
        next_image_obs = episode.image_obs
        if next_image_obs:
            key = list(next_image_obs.keys())[0]
            next_image = next_image_obs[key][t + 1]
        else:
            next_image = np.zeros((3, 224, 224), dtype=np.uint8)

        if next_image.shape[-1] == 3:
            next_image = np.transpose(next_image, (2, 0, 1))
        next_image = torch.from_numpy(next_image).float() / 255.0

        return {
            "images": image,
            "actions": action,
            "next_images": next_image,
        }


def create_latent_no_refine_baseline(
    backbone_adapter: Optional[BackboneAdapter] = None,
    latent_dim: int = 1536,
    action_dim: int = 7,
    device: str = "cuda",
    **kwargs,
) -> LatentNoRefineBaseline:
    """Factory function to create Latent No-Refine baseline.

    Args:
        backbone_adapter: Optional BackboneAdapter (creates new if None)
        latent_dim: Latent dimension (default from vision backbone)
        action_dim: Action dimension (default 7 for 7DoF)
        device: Device for computation
        **kwargs: Additional config args

    Returns:
        Configured LatentNoRefineBaseline
    """
    config = LatentNoRefineConfig(
        latent_dim=latent_dim,
        action_dim=action_dim,
        device=device,
        **kwargs,
    )
    return LatentNoRefineBaseline(config=config, backbone_adapter=backbone_adapter)