#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import train_and_eval as offline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train internal baselines and export deployable rollout checkpoints.")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["lara-wm", "direct_policy", "latent_no_refine"])
    parser.add_argument("--max-episodes-per-task", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=offline.DEFAULT_DEVICE)
    parser.add_argument("--backbone-device", default=None)
    parser.add_argument("--output-dir", default="experiments/rollout_ckpts")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--train-seed", type=int, default=42)
    return parser.parse_args()


def export_checkpoint(
    model_name: str,
    model: torch.nn.Module,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    feature_dim: int,
    config: offline.ExperimentConfig,
    output_dir: Path,
    train_seed: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "action_mean": action_mean.cpu(),
        "action_std": action_std.cpu(),
        "feature_dim": feature_dim,
        "latent_dim": config.latent_dim,
        "hidden_dim": config.hidden_dim,
        "action_dim": int(action_mean.numel()),
        "tasks": list(config.tasks),
        "split_seed": config.split_seed,
        "train_seed": train_seed,
    }
    ckpt_path = output_dir / f"{model_name.replace('-', '_')}.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


def main() -> None:
    args = parse_args()
    config = offline.ExperimentConfig(
        tasks=args.tasks,
        max_episodes_per_task=args.max_episodes_per_task,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device=args.device,
        backbone_device=args.backbone_device,
        split_seed=args.split_seed,
        seeds=[args.train_seed],
    )
    offline.set_global_seed(args.train_seed)

    episodes = offline.load_real_episodes(config)
    split = offline.split_episodes(
        episodes,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.split_seed,
    )
    loaders, action_mean, action_std, feature_bank, feature_metadata = offline.build_dataloaders(split, config)
    feature_dim = int(feature_metadata["feature_dim"])
    device = torch.device(config.device)

    results: dict[str, dict[str, float]] = {}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        print(f"[rollout-ckpt] training {model_name} on tasks={args.tasks}")
        model = offline.instantiate_model(model_name, action_mean.numel(), feature_dim, config).to(device)
        trained_model, history = offline.train_and_select(
            model=model,
            model_name=model_name,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            train_feature_bank=feature_bank["train"],
            val_feature_bank=feature_bank["val"],
            action_mean=action_mean,
            action_std=action_std,
            config=config,
            device=device,
        )
        test_metrics = offline.evaluate_model(
            trained_model,
            loaders["test"],
            model_name,
            action_mean,
            action_std,
            device,
            feature_bank["test"],
        )
        ckpt_path = export_checkpoint(model_name, trained_model, action_mean, action_std, feature_dim, config, out_dir, args.train_seed)
        results[model_name] = {
            "action_mse": float(test_metrics["action_mse"]),
            "action_mae": float(test_metrics["action_mae"]),
            "action_r2": float(test_metrics["action_r2"]),
            "checkpoint_path": str(ckpt_path),
        }
        (out_dir / f"{model_name.replace('-', '_')}_history.json").write_text(json.dumps(history, indent=2))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
