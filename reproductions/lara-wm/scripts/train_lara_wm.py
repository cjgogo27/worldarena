"""LaRA-WM Training Orchestration.

Train latent encoder, world model, and action decoder for latent robot action learning.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import yaml


from lara_wm.data.standardized_dataset import (
    StandardizedDataset,
    StandardizedEpisode,
    create_standardized_dataset,
)
from lara_wm.models.latent_encoder import LatentActionEncoder, LatentEncoderConfig
from lara_wm.models.world_model import WorldModel, WorldModelConfig
from lara_wm.models.action_decoder import ActionDecoder, ActionDecoderConfig


logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path("/data/alice/cjtest/lara-wm/configs/train.yaml")


@dataclass
class LaRAWMConfig:
    experiment_name: str = "lara_wm_default"
    seed: int = 42
    device: str = "cuda"

    dataset_path: str = "/data/alice/cjtest/lara-wm/data/robotwin/dataset"
    batch_size: int = 8
    num_workers: int = 4
    train_split: float = 0.9
    shuffle: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2

    latent_encoder: dict[str, Any] = field(default_factory=lambda: {
        "action_dim": 7,
        "latent_dim": 32,
        "feature_dim": 1536,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
        "kl_weight": 1.0,
    })

    world_model: dict[str, Any] = field(default_factory=lambda: {
        "latent_dim": 1536,
        "action_dim": 1536,
        "hidden_dim": 512,
        "num_layers": 2,
        "dropout": 0.1,
        "architecture": "gru",
        "state_loss_weight": 1.0,
        "reward_loss_weight": 1.0,
    })

    action_decoder: dict[str, Any] = field(default_factory=lambda: {
        "latent_dim": 1536,
        "action_dim": 7,
        "hidden_dim": 512,
        "num_layers": 2,
        "dropout": 0.1,
    })

    num_epochs: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 1000
    scheduler: str = "cosine"
    min_lr: float = 1.0e-6

    latent_encoder_weight: float = 1.0
    world_model_state_weight: float = 1.0
    world_model_reward_weight: float = 1.0
    action_decoder_weight: float = 1.0

    save_dir: str = "/data/alice/cjtest/lara-wm/experiments"
    save_every_n_epochs: int = 10
    save_best: bool = True
    metric_for_best: str = "total_loss"
    monitor_min: bool = True
    keep_n_checkpoints: int = 3
    save_optimizer: bool = True

    log_dir: str = "/data/alice/cjtest/lara-wm/logs"
    log_every_n_steps: int = 10
    tensorboard: bool = True
    wandb: bool = False

    validation_enabled: bool = True
    val_every_n_epochs: int = 5
    val_samples: int = 100

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "LaRAWMConfig":
        p = Path(path)
        if not p.exists():
            logger.warning(f"Config not found: {p}, using defaults")
            return cls()

        config = yaml.safe_load(p.read_text()) or {}

        data = config.get("data", {})
        training = config.get("training", {})
        checkpointing = config.get("checkpointing", {})
        logging_cfg = config.get("logging", {})
        validation = config.get("validation", {})

        return cls(
            experiment_name=config.get("experiment_name", cls.experiment_name),
            seed=config.get("seed", cls.seed),
            device=config.get("device", cls.device),
            dataset_path=data.get("dataset_path", cls.dataset_path),
            batch_size=data.get("batch_size", cls.batch_size),
            num_workers=data.get("num_workers", cls.num_workers),
            train_split=data.get("train_split", cls.train_split),
            shuffle=data.get("shuffle", cls.shuffle),
            pin_memory=data.get("pin_memory", cls.pin_memory),
            prefetch_factor=data.get("prefetch_factor", cls.prefetch_factor),
            latent_encoder=config.get("latent_encoder", cls.latent_encoder),
            world_model=config.get("world_model", cls.world_model),
            action_decoder=config.get("action_decoder", cls.action_decoder),
            num_epochs=training.get("num_epochs", cls.num_epochs),
            learning_rate=training.get("learning_rate", cls.learning_rate),
            weight_decay=training.get("weight_decay", cls.weight_decay),
            gradient_clip_norm=training.get("gradient_clip_norm", cls.gradient_clip_norm),
            warmup_steps=training.get("warmup_steps", cls.warmup_steps),
            scheduler=training.get("scheduler", cls.scheduler),
            min_lr=training.get("min_lr", cls.min_lr),
            latent_encoder_weight=training.get("latent_encoder_weight", cls.latent_encoder_weight),
            world_model_state_weight=training.get("world_model_state_weight", cls.world_model_state_weight),
            world_model_reward_weight=training.get("world_model_reward_weight", cls.world_model_reward_weight),
            action_decoder_weight=training.get("action_decoder_weight", cls.action_decoder_weight),
            save_dir=checkpointing.get("save_dir", cls.save_dir),
            save_every_n_epochs=checkpointing.get("save_every_n_epochs", cls.save_every_n_epochs),
            save_best=checkpointing.get("save_best", cls.save_best),
            metric_for_best=checkpointing.get("metric_for_best", cls.metric_for_best),
            monitor_min=checkpointing.get("monitor_min", cls.monitor_min),
            keep_n_checkpoints=checkpointing.get("keep_n_checkpoints", cls.keep_n_checkpoints),
            save_optimizer=checkpointing.get("save_optimizer", cls.save_optimizer),
            log_dir=logging_cfg.get("log_dir", cls.log_dir),
            log_every_n_steps=logging_cfg.get("log_every_n_steps", cls.log_every_n_steps),
            tensorboard=logging_cfg.get("tensorboard", cls.tensorboard),
            wandb=logging_cfg.get("wandb", cls.wandb),
            validation_enabled=validation.get("enabled", cls.validation_enabled),
            val_every_n_epochs=validation.get("val_every_n_epochs", cls.val_every_n_epochs),
            val_samples=validation.get("val_samples", cls.val_samples),
        )


@dataclass
class TrainingState:
    epoch: int
    global_step: int
    best_metric: float
    latent_encoder_state: dict[str, Any] | None = None
    world_model_state: dict[str, Any] | None = None
    action_decoder_state: dict[str, Any] | None = None
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    def __init__(self, save_dir: Path, keep_n: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n = keep_n
        self.checkpoints: list[Path] = []

    def save(self, state: TrainingState, is_best: bool = False, name: str = "checkpoint") -> Path:
        checkpoint = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
            "latent_encoder": state.latent_encoder_state,
            "world_model": state.world_model_state,
            "action_decoder": state.action_decoder_state,
            "optimizer": state.optimizer_state,
            "scheduler": state.scheduler_state,
            "config": state.config,
        }

        if is_best:
            path = self.save_dir / "best.pt"
        else:
            path = self.save_dir / f"{name}_epoch_{state.epoch:04d}.pt"

        torch.save(checkpoint, path)

        if not is_best:
            self.checkpoints.append(path)
            self._cleanup_old()

        logger.info(f"Checkpoint saved: {path}")
        return path

    def _cleanup_old(self) -> None:
        if len(self.checkpoints) <= self.keep_n:
            return

        to_remove = self.checkpoints[:-self.keep_n]
        for path in to_remove:
            if path.exists():
                path.unlink()
                logger.info(f"Removed old checkpoint: {path}")

        self.checkpoints = self.checkpoints[-self.keep_n:]

    def load_latest(self) -> dict[str, Any] | None:
        checkpoints = sorted(self.save_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        return torch.load(checkpoints[-1], map_location="cpu")

    def load_best(self) -> dict[str, Any] | None:
        best_path = self.save_dir / "best.pt"
        if not best_path.exists():
            return None
        return torch.load(best_path, map_location="cpu")


class LaRAWM:
    def __init__(self, config: Optional[LaRAWMConfig] = None):
        self.config = config or LaRAWMConfig()

        self.device = torch.device(self.config.device)

        self.latent_encoder: Optional[LatentActionEncoder] = None
        self.world_model: Optional[WorldModel] = None
        self.action_decoder: Optional[ActionDecoder] = None

        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        self.writer: Optional[SummaryWriter] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None

        self.global_step: int = 0
        self.epoch: int = 0
        self.best_metric: float = float("inf") if self.config.monitor_min else float("-inf")

    def setup(self) -> "LaRAWM":
        self._set_seed()
        self._build_models()
        self._build_dataloaders()
        self._build_optimizer()
        self._build_logging()
        self._build_checkpointing()

        logger.info(f"LaRA-WM initialized: {self.config.experiment_name}")
        return self

    def _set_seed(self) -> None:
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_models(self) -> None:
        latent_config = LatentEncoderConfig.from_dict(self.config.latent_encoder)
        self.latent_encoder = LatentActionEncoder(config=latent_config).to(self.device)

        wm_config = WorldModelConfig(
            latent_dim=self.config.world_model.get("latent_dim", 1536),
            action_dim=self.config.world_model.get("action_dim", 1536),
            hidden_dim=self.config.world_model.get("hidden_dim", 512),
            num_layers=self.config.world_model.get("num_layers", 2),
            dropout=self.config.world_model.get("dropout", 0.1),
            architecture=self.config.world_model.get("architecture", "gru"),
            state_loss_weight=self.config.world_model.get("state_loss_weight", 1.0),
            reward_loss_weight=self.config.world_model.get("reward_loss_weight", 1.0),
        )
        self.world_model = WorldModel(config=wm_config).to(self.device)

        decoder_config = ActionDecoderConfig(
            latent_dim=self.config.action_decoder.get("latent_dim", 1536),
            action_dim=self.config.action_decoder.get("action_dim", 7),
            hidden_dim=self.config.action_decoder.get("hidden_dim", 512),
            num_layers=self.config.action_decoder.get("num_layers", 2),
            dropout=self.config.action_decoder.get("dropout", 0.1),
        )
        self.action_decoder = ActionDecoder(config=decoder_config).to(self.device)

        logger.info(f"Models built: latent_encoder, world_model, action_decoder")

    def _build_dataloaders(self) -> None:
        dataset = create_standardized_dataset(self.config.dataset_path)
        total_len = len(dataset)
        train_len = int(total_len * self.config.train_split)
        val_len = total_len - train_len

        train_dataset, val_dataset = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(self.config.seed),
        )

        def collate_fn(episodes):
            if not episodes:
                return {}
            first = episodes[0].transformed
            actions_dict = {}
            for key in first["actions"]:
                stacked = np.stack([ep.transformed["actions"][key] for ep in episodes])
                actions_dict[key] = torch.from_numpy(stacked).float()
            states_dict = {}
            for key in first["states"]:
                stacked = np.stack([ep.transformed["states"][key] for ep in episodes])
                states_dict[key] = torch.from_numpy(stacked).float()
            images_dict = {}
            for key in first["images"]:
                stacked = np.stack([ep.transformed["images"][key] for ep in episodes])
                images_dict[key] = torch.from_numpy(stacked).float()
            rewards_stacked = np.stack([ep.transformed["rewards"] for ep in episodes])
            rewards_tensor = torch.from_numpy(rewards_stacked).float()
            return {"images": images_dict, "states": states_dict, "actions": actions_dict, "rewards": rewards_tensor}

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=collate_fn,
        )

        if self.config.validation_enabled:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                prefetch_factor=self.config.prefetch_factor,
                collate_fn=collate_fn,
            )

        logger.info(f"Dataloaders built: train={len(self.train_loader)}, val={len(self.val_loader) if self.val_loader else 0}")

    def _build_optimizer(self) -> None:
        params = list(self.latent_encoder.parameters())
        params += list(self.world_model.parameters())
        params += list(self.action_decoder.parameters())

        self.optimizer = optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.min_lr,
        )

        logger.info(f"Optimizer: AdamW(lr={self.config.learning_rate})")

    def _build_logging(self) -> None:
        if self.config.tensorboard:
            log_path = Path(self.config.log_dir) / self.config.experiment_name
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_path))
            logger.info(f"TensorBoard: {log_path}")

    def _build_checkpointing(self) -> None:
        save_path = Path(self.config.save_dir) / self.config.experiment_name
        save_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(
            save_dir=save_path,
            keep_n=self.config.keep_n_checkpoints,
        )
        logger.info(f"Checkpointing: {save_path}")

    def train_epoch(self) -> dict[str, float]:
        if self.latent_encoder is None or self.world_model is None or self.action_decoder is None:
            raise RuntimeError("Models not initialized. Call setup() first.")

        self.latent_encoder.train()
        self.world_model.train()
        self.action_decoder.train()

        metrics_total: dict[str, float] = {}
        num_batches = 0

        for batch in self.train_loader:
            images = batch["images"]
            states = batch["states"]
            actions = batch["actions"]
            rewards = batch.get("rewards", None)

            action_key = "joint_position" if "joint_position" in actions else list(actions.keys())[0]
            actions_tensor = actions[action_key].to(self.device)

            states = batch["states"]
            state_key = "joint_states" if "joint_states" in states else list(states.keys())[0]
            states_tensor = states[state_key].to(self.device)

            if rewards is not None:
                rewards_tensor = rewards.to(self.device)
            else:
                rewards_tensor = torch.zeros(actions_tensor.shape[0], device=self.device)

            self.optimizer.zero_grad()

            latent_output = self.latent_encoder(actions_tensor)
            latent = latent_output.latent

            wm_output = self.world_model(latent, latent)
            wm_loss = self.config.world_model_state_weight * wm_output.get("state_loss", torch.tensor(0.0))
            if wm_output.get("reward_loss") is not None:
                wm_loss = wm_loss + self.config.world_model_reward_weight * wm_output.get("reward_loss", torch.tensor(0.0))

            predicted_latent = wm_output.get("next_latent_states", latent)
            decoded_actions = self.action_decoder(predicted_latent)
            action_loss = nn.functional.mse_loss(decoded_actions, actions_tensor)

            total_encoder_loss = latent_output.total_loss * self.config.latent_encoder_weight
            total_decoder_loss = action_loss * self.config.action_decoder_weight

            total_loss = total_encoder_loss + wm_loss + total_decoder_loss

            total_loss.backward()

            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.latent_encoder.parameters(),
                    self.config.gradient_clip_norm,
                )
                torch.nn.utils.clip_grad_norm_(
                    self.world_model.parameters(),
                    self.config.gradient_clip_norm,
                )
                torch.nn.utils.clip_grad_norm_(
                    self.action_decoder.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                metrics = {
                    "total_loss": total_loss.item(),
                    "latent_encoder_loss": total_encoder_loss.item(),
                    "latent_recon_loss": latent_output.reconstruction_loss.item(),
                    "latent_kl_loss": latent_output.kl_loss.item(),
                    "world_model_loss": wm_loss.item(),
                    "action_decoder_loss": action_loss.item(),
                }
                for k, v in metrics.items():
                    metrics_total[k] = metrics_total.get(k, 0.0) + v

            num_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_metrics(metrics, prefix="train")

        for k in metrics_total:
            metrics_total[k] /= num_batches

        return metrics_total

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if not self.config.validation_enabled or self.val_loader is None:
            return {}

        self.latent_encoder.eval()
        self.world_model.eval()
        self.action_decoder.eval()

        metrics_total: dict[str, float] = {}
        num_batches = 0

        for batch in self.val_loader:
            images = batch["images"]
            states = batch["states"]
            actions = batch["actions"]
            rewards = batch.get("rewards", None)

            action_key = "joint_position" if "joint_position" in actions else list(actions.keys())[0]
            actions_tensor = actions[action_key].to(self.device)

            state_key = "joint_states" if "joint_states" in states else list(states.keys())[0]
            states_tensor = states[state_key].to(self.device)

            if rewards is not None:
                rewards_tensor = rewards.to(self.device)
            else:
                rewards_tensor = torch.zeros(actions_tensor.shape[0], device=self.device)

            latent_output = self.latent_encoder(actions_tensor)
            latent = latent_output.latent

            wm_output = self.world_model(latent, latent)
            wm_loss = self.config.world_model_state_weight * wm_output.get("state_loss", torch.tensor(0.0))

            predicted_latent = wm_output.get("next_latent_states", latent)
            decoded_actions = self.action_decoder(predicted_latent)
            action_loss = nn.functional.mse_loss(decoded_actions, actions_tensor)

            total_encoder_loss = latent_output.total_loss * self.config.latent_encoder_weight
            total_decoder_loss = action_loss * self.config.action_decoder_weight

            val_loss = total_encoder_loss + wm_loss + total_decoder_loss

            metrics = {
                "val_total_loss": val_loss.item(),
                "val_latent_encoder_loss": total_encoder_loss.item(),
                "val_latent_recon_loss": latent_output.reconstruction_loss.item(),
                "val_world_model_loss": wm_loss.item(),
                "val_action_decoder_loss": action_loss.item(),
            }

            for k, v in metrics.items():
                metrics_total[k] = metrics_total.get(k, 0.0) + v

            num_batches += 1

        for k in metrics_total:
            metrics_total[k] /= num_batches

        return metrics_total

    def _log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        if self.writer is None:
            return

        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(tag, v, self.global_step)

    def _save_checkpoint(self, is_best: bool = False) -> None:
        if self.checkpoint_manager is None:
            return

        state = TrainingState(
            epoch=self.epoch,
            global_step=self.global_step,
            best_metric=self.best_metric,
            latent_encoder_state=self.latent_encoder.state_dict() if self.latent_encoder else None,
            world_model_state=self.world_model.state_dict() if self.world_model else None,
            action_decoder_state=self.action_decoder.state_dict() if self.action_decoder else None,
            optimizer_state=self.optimizer.state_dict() if self.optimizer and self.config.save_optimizer else None,
            scheduler_state=self.scheduler.state_dict() if self.scheduler and self.config.save_optimizer else None,
            config=self.config.__dict__ if hasattr(self.config, "__dict__") else {},
        )

        self.checkpoint_manager.save(state, is_best=is_best)

    def train(self) -> dict[str, float]:
        self.setup()

        train_metrics: dict[str, float] = {}

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch()

            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} | "
                + " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
            )

            self._log_metrics(train_metrics, prefix="epoch")

            if self.config.validation_enabled and (epoch + 1) % self.config.val_every_n_epochs == 0:
                val_metrics = self.validate()
                logger.info(
                    f"Validation | "
                    + " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                )
                self._log_metrics(val_metrics, prefix="val")

                if self.config.save_best:
                    current_metric = val_metrics.get(
                        f"val_{self.config.metric_for_best}",
                        val_metrics.get(self.config.metric_for_best, float("inf")),
                    )
                    is_best = (
                        current_metric < self.best_metric
                        if self.config.monitor_min
                        else current_metric > self.best_metric
                    )
                    if is_best:
                        self.best_metric = current_metric
                        self._save_checkpoint(is_best=True)

            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(is_best=False)

        return train_metrics

    def load_checkpoint(self, checkpoint_path: Path | str) -> "LaRAWM":
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.latent_encoder and "latent_encoder" in checkpoint:
            self.latent_encoder.load_state_dict(checkpoint["latent_encoder"])
        if self.world_model and "world_model" in checkpoint:
            self.world_model.load_state_dict(checkpoint["world_model"])
        if self.action_decoder and "action_decoder" in checkpoint:
            self.action_decoder.load_state_dict(checkpoint["action_decoder"])

        if self.optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", self.best_metric)

        return self

    def close(self) -> None:
        if self.writer:
            self.writer.close()


def train_lara_wm(
    config_path: Path | str | None = None,
    resume: Path | str | None = None,
) -> dict[str, float]:
    """Main entrypoint for LaRA-WM training."""

    config = LaRAWMConfig.from_yaml(config_path or DEFAULT_CONFIG_PATH)

    model = LaRAWM(config)

    if resume:
        model.load_checkpoint(resume)

    try:
        metrics = model.train()
        logger.info("Training completed successfully")
        return metrics
    finally:
        model.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="LaRA-WM Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    train_lara_wm(config_path=args.config, resume=args.resume)