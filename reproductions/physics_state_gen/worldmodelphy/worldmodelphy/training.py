"""Training and evaluation utilities for VideoPredictor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingMetrics:
    loss: list[float] = field(default_factory=list)
    reconstruction_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_reconstruction_loss: list[float] = field(default_factory=list)


def compute_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    B, T, C, H, W = pred.shape
    pred_flat = pred.view(B * T, C, H, W)
    target_flat = target.view(B * T, C, H, W)

    if loss_type == "mse":
        loss = nn.functional.mse_loss(pred_flat, target_flat)
    elif loss_type == "l1":
        loss = nn.functional.l1_loss(pred_flat, target_flat)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss


def _coerce_frames(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        frames = batch
    elif isinstance(batch, dict):
        if "frames" not in batch:
            raise KeyError("Expected key 'frames' in batch dictionary")
        frames = batch["frames"]
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("Empty batch sequence")
        frames = batch[0]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"Expected tensor frames after coercion, got {type(frames)!r}")

    if frames.ndim == 4:
        frames = frames.unsqueeze(2)
    if frames.ndim != 5:
        raise ValueError(f"Expected frames with 4 or 5 dims, got shape {tuple(frames.shape)}")
    return frames.float()


def training_step(
    model: nn.Module,
    frames: torch.Tensor,
    optimizer: optim.Optimizer,
    device: torch.device,
    teacher_forcing_ratio: float = 1.0,
    loss_type: str = "mse",
) -> tuple[float, float]:
    model.train()
    frames = frames.to(device)

    optimizer.zero_grad()

    if teacher_forcing_ratio > 0:
        pred_frames, _ = model.forward_with_teacher_forcing(
            frames,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
    else:
        pred_frames, _ = model(frames)

    loss = compute_reconstruction_loss(pred_frames, frames, loss_type=loss_type)
    loss.backward()
    optimizer.step()

    return loss.item(), loss.item()


def validation_step(
    model: nn.Module,
    frames: torch.Tensor,
    device: torch.device,
    loss_type: str = "mse",
) -> tuple[float, float]:
    model.eval()
    frames = frames.to(device)

    with torch.no_grad():
        pred_frames, _ = model(frames)
        loss = compute_reconstruction_loss(pred_frames, frames, loss_type=loss_type)

    return loss.item(), loss.item()


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    teacher_forcing_ratio: float = 1.0,
    loss_type: str = "mse",
) -> tuple[float, float]:
    total_loss = 0.0
    total_recon = 0.0
    num_batches = 0

    for batch in train_loader:
        frames = _coerce_frames(batch)
        loss, recon_loss = training_step(
            model=model,
            frames=frames,
            optimizer=optimizer,
            device=device,
            teacher_forcing_ratio=teacher_forcing_ratio,
            loss_type=loss_type,
        )
        total_loss += loss
        total_recon += recon_loss
        num_batches += 1

    return total_loss / num_batches, total_recon / num_batches


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_type: str = "mse",
) -> tuple[float, float]:
    total_loss = 0.0
    total_recon = 0.0
    num_batches = 0

    for batch in val_loader:
        frames = _coerce_frames(batch)
        loss, recon_loss = validation_step(
            model=model,
            frames=frames,
            device=device,
            loss_type=loss_type,
        )
        total_loss += loss
        total_recon += recon_loss
        num_batches += 1

    return total_loss / num_batches, total_recon / num_batches


def train_loop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    initial_teacher_forcing_ratio: float = 1.0,
    final_teacher_forcing_ratio: float = 0.0,
    teacher_forcing_decay_epochs: int | None = None,
    loss_type: str = "mse",
    log_interval: int = 1,
) -> TrainingMetrics:
    metrics = TrainingMetrics()

    decay_epochs = teacher_forcing_decay_epochs or num_epochs

    for epoch in range(num_epochs):
        tf_ratio = max(
            final_teacher_forcing_ratio,
            initial_teacher_forcing_ratio -
            (initial_teacher_forcing_ratio - final_teacher_forcing_ratio) *
            (epoch / decay_epochs),
        )

        train_loss, train_recon = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            teacher_forcing_ratio=tf_ratio,
            loss_type=loss_type,
        )

        val_loss, val_recon = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            loss_type=loss_type,
        )

        metrics.loss.append(train_loss)
        metrics.reconstruction_loss.append(train_recon)
        metrics.val_loss.append(val_loss)
        metrics.val_reconstruction_loss.append(val_recon)

        if (epoch + 1) % log_interval == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"TF: {tf_ratio:.2f} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

    return metrics


def generate_rollout(
    model: nn.Module,
    initial_frames: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    initial_frames = initial_frames.to(device)

    with torch.no_grad():
        generated = model.rollout(initial_frames, num_steps)

    return generated


def extract_hidden_states(
    model: nn.Module,
    frames: torch.Tensor,
    device: torch.device,
    return_all_layers: bool = False,
) -> torch.Tensor:
    model.eval()
    frames = frames.to(device)

    with torch.no_grad():
        hidden = model.get_hidden_states(frames, return_all_layers=return_all_layers)

    return hidden
