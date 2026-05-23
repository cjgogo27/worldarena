from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import ARTIFACTS_ROOT, REPORT_ROOT, RUNS_ROOT
from .data import (
    MotionType,
    generate_split_dataset,
    save_frame_strip,
    save_frames_as_gif,
    save_metadata,
)
from .eval import fit_linear_probe, rollout_metrics, save_json
from .model import create_bottleneck_model, create_model, create_model_local
from .training import extract_hidden_states, train_loop


@dataclass
class ExperimentConfig:
    motion_type: MotionType
    context_len: int
    label: str
    model_kind: str = "gru"
    num_train: int = 96
    num_val: int = 24
    num_test: int = 24
    num_ood: int = 24
    epochs: int = 8
    batch_size: int = 16
    hidden_size: int = 128
    latent_dim: int = 64
    learning_rate: float = 1e-3
    state_dim: int = 12
    attn_window_size: int = 4


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: list[tuple[torch.Tensor, dict[str, Any]]], window_len: int):
        self.windows: list[torch.Tensor] = []
        for frames, _ in samples:
            frames = frames.unsqueeze(1)
            if frames.shape[0] <= window_len:
                self.windows.append(frames)
                continue
            for start in range(0, frames.shape[0] - window_len + 1):
                self.windows.append(frames[start : start + window_len])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"frames": self.windows[index]}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _experiment_name(cfg: ExperimentConfig) -> str:
    return f"{cfg.motion_type.value}_{cfg.model_kind}_{cfg.label}"


def _metadata_positions(metadata: dict[str, Any]) -> np.ndarray:
    if "positions_x" in metadata and "positions_y" in metadata:
        return np.stack(
            [np.asarray(metadata["positions_x"], dtype=np.float32), np.asarray(metadata["positions_y"], dtype=np.float32)],
            axis=1,
        )
    if "positions_x_1" in metadata and "positions_x_2" in metadata:
        x1 = np.asarray(metadata["positions_x_1"], dtype=np.float32)
        y1 = np.asarray(metadata["positions_y_1"], dtype=np.float32)
        x2 = np.asarray(metadata["positions_x_2"], dtype=np.float32)
        y2 = np.asarray(metadata["positions_y_2"], dtype=np.float32)
        return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0], axis=1)
    raise KeyError("Unsupported metadata positions format")


def _metadata_targets(metadata: dict[str, Any]) -> np.ndarray:
    if all(key in metadata for key in ["velocities_x", "velocities_y", "accelerations_x", "accelerations_y"]):
        return np.stack(
            [
                np.asarray(metadata["velocities_x"], dtype=np.float32),
                np.asarray(metadata["velocities_y"], dtype=np.float32),
                np.asarray(metadata["accelerations_x"], dtype=np.float32),
                np.asarray(metadata["accelerations_y"], dtype=np.float32),
            ],
            axis=1,
        )
    if all(key in metadata for key in ["velocities_x_1", "velocities_y_1", "velocities_x_2", "velocities_y_2"]):
        vx = (np.asarray(metadata["velocities_x_1"], dtype=np.float32) + np.asarray(metadata["velocities_x_2"], dtype=np.float32)) / 2.0
        vy = (np.asarray(metadata["velocities_y_1"], dtype=np.float32) + np.asarray(metadata["velocities_y_2"], dtype=np.float32)) / 2.0
        ax = (np.asarray(metadata["accelerations_x_1"], dtype=np.float32) + np.asarray(metadata["accelerations_x_2"], dtype=np.float32)) / 2.0
        ay = (np.asarray(metadata["accelerations_y_1"], dtype=np.float32) + np.asarray(metadata["accelerations_y_2"], dtype=np.float32)) / 2.0
        return np.stack([vx, vy, ax, ay], axis=1)
    raise KeyError("Unsupported metadata target format")


def _build_model(cfg: ExperimentConfig) -> torch.nn.Module:
    if cfg.model_kind == "gru":
        return create_model(hidden_size=cfg.hidden_size, latent_dim=cfg.latent_dim)
    if cfg.model_kind == "local":
        return create_model_local(
            hidden_size=cfg.hidden_size,
            latent_dim=cfg.latent_dim,
            attn_window_size=cfg.attn_window_size,
        )
    if cfg.model_kind == "bottleneck":
        return create_bottleneck_model(embed_dim=cfg.latent_dim, state_dim=cfg.state_dim)
    raise ValueError(f"Unknown model kind: {cfg.model_kind}")


def _extract_probe_arrays(
    model: torch.nn.Module,
    samples: list[tuple[torch.Tensor, dict[str, Any]]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for frames, metadata in samples:
        batch = frames.unsqueeze(0).unsqueeze(2).to(device)
        hidden = extract_hidden_states(model, batch, device)[0].detach().cpu().numpy()
        target = _metadata_targets(metadata)
        xs.append(hidden)
        ys.append(target)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _evaluate_rollouts(
    model: torch.nn.Module,
    samples: list[tuple[torch.Tensor, dict[str, Any]]],
    context_len: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    agg = {"frame_mse": 0.0, "trajectory_mse": 0.0, "velocity_mse": 0.0, "acceleration_mse": 0.0}
    sample_artifact: dict[str, Any] = {}
    count = 0
    model.eval()
    for idx, (frames, metadata) in enumerate(samples):
        context = frames[:context_len].unsqueeze(0).unsqueeze(2).to(device)
        target = frames[context_len:].unsqueeze(1)
        if target.shape[0] == 0:
            continue
        with torch.no_grad():
            pred = model.rollout(context, num_steps=target.shape[0])[0].cpu()
        metrics = rollout_metrics(pred, target)
        for key, value in metrics.items():
            agg[key] += value
        if idx == 0:
            sample_artifact = {
                "context": frames[:context_len],
                "target": frames[context_len:],
                "pred": pred.squeeze(1),
                "metadata": metadata,
            }
        count += 1
    if count == 0:
        return {key: float("nan") for key in agg}, sample_artifact
    return {key: value / count for key, value in agg.items()}, sample_artifact


def _save_sample_artifacts(base_dir: Path, sample: dict[str, Any], prefix: str) -> None:
    if not sample:
        return
    combined = torch.cat([sample["context"], sample["target"], sample["pred"]], dim=0)
    save_frames_as_gif(combined, base_dir / f"{prefix}.gif")
    save_frame_strip(combined, base_dir / f"{prefix}_strip.png")
    save_metadata(sample["metadata"], base_dir / f"{prefix}_metadata.json")


def _plot_loss_curve(metrics: dict[str, list[float]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(metrics["loss"], label="train")
    ax.plot(metrics["val_loss"], label="val")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mse")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(cfg: ExperimentConfig, device: torch.device) -> dict[str, Any]:
    exp_name = _experiment_name(cfg)
    run_dir = RUNS_ROOT / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_split_dataset(
        cfg.motion_type,
        num_train=cfg.num_train,
        num_val=cfg.num_val,
        num_test=cfg.num_test,
        num_ood=cfg.num_ood,
    )

    train_loader = DataLoader(WindowDataset(dataset["train"], cfg.context_len), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(dataset["val"], cfg.context_len), batch_size=cfg.batch_size, shuffle=False)

    model = _build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    history = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=cfg.epochs,
        initial_teacher_forcing_ratio=1.0,
        final_teacher_forcing_ratio=0.5,
        teacher_forcing_decay_epochs=cfg.epochs,
        log_interval=1,
    )

    test_metrics, test_sample = _evaluate_rollouts(model, dataset["test"], cfg.context_len, device)
    ood_metrics, ood_sample = _evaluate_rollouts(model, dataset["ood"], cfg.context_len, device)

    train_x, train_y = _extract_probe_arrays(model, dataset["train"], device)
    test_x, test_y = _extract_probe_arrays(model, dataset["test"], device)
    ood_x, ood_y = _extract_probe_arrays(model, dataset["ood"], device)
    probe_test = fit_linear_probe(train_x, train_y, test_x, test_y)
    probe_ood = fit_linear_probe(train_x, train_y, ood_x, ood_y)

    _save_sample_artifacts(run_dir, test_sample, "test_rollout")
    _save_sample_artifacts(run_dir, ood_sample, "ood_rollout")
    _plot_loss_curve(asdict(history), run_dir / "loss_curve.png", title=exp_name)

    result = {
        "experiment": exp_name,
        "config": asdict(cfg),
        "history": asdict(history),
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "probe_test": probe_test,
        "probe_ood": probe_ood,
    }
    save_json(result, run_dir / "metrics.json")
    torch.save(model.state_dict(), run_dir / "model.pt")
    return result


def _copy_report_assets(results: list[dict[str, Any]]) -> None:
    assets_root = REPORT_ROOT / "assets"
    figures_dir = assets_root / "figures"
    tables_dir = assets_root / "tables"
    videos_dir = assets_root / "videos"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for result in results:
        exp = result["experiment"]
        run_dir = RUNS_ROOT / exp
        rows.append(
            {
                "experiment": exp,
                **result["test_metrics"],
                **{f"ood_{k}": v for k, v in result["ood_metrics"].items()},
                **result["probe_test"],
                **{f"ood_{k}": v for k, v in result["probe_ood"].items()},
            }
        )
        if (run_dir / "loss_curve.png").exists():
            target = figures_dir / f"fig_loss_curve_{exp}.png"
            target.write_bytes((run_dir / "loss_curve.png").read_bytes())
        for src_name, dst_name in [
            ("test_rollout.gif", f"gif_test_{exp}.gif"),
            ("ood_rollout.gif", f"gif_ood_{exp}.gif"),
        ]:
            src = run_dir / src_name
            if src.exists():
                (videos_dir / dst_name).write_bytes(src.read_bytes())

    _write_csv(rows, tables_dir / "table_quantitative_results.csv")

    if results:
        fig, ax = plt.subplots(figsize=(8, 4))
        names = [r["experiment"] for r in results]
        id_vals = [r["test_metrics"]["trajectory_mse"] for r in results]
        ood_vals = [r["ood_metrics"]["trajectory_mse"] for r in results]
        x = np.arange(len(names))
        ax.bar(x - 0.18, id_vals, width=0.36, label="ID traj MSE")
        ax.bar(x + 0.18, ood_vals, width=0.36, label="OOD traj MSE")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.legend()
        ax.set_title("Trajectory generalization across experiments")
        fig.tight_layout()
        fig.savefig(figures_dir / "fig_ood_failures.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        probe_vals = [r["probe_test"]["probe_r2_mean"] for r in results]
        ax.bar(names, probe_vals)
        ax.set_ylabel("mean probe R2")
        ax.set_title("Latent probe quality")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figures_dir / "fig_probe_accuracy.png", dpi=220)
        plt.close(fig)

        primary = RUNS_ROOT / results[0]["experiment"]
        for src_name, dst_name in [
            ("test_rollout_strip.png", "fig_id_generation_comparison.png"),
            ("ood_rollout_strip.png", "fig_latent_pca.png"),
            ("test_rollout.gif", "gif_id_samples.gif"),
            ("ood_rollout.gif", "gif_ood_breakdown.gif"),
        ]:
            src = primary / src_name
            if src.exists():
                out_dir = videos_dir if dst_name.endswith(".gif") else figures_dir
                (out_dir / dst_name).write_bytes(src.read_bytes())


def _update_report(results: list[dict[str, Any]]) -> None:
    report_path = REPORT_ROOT / "report.md"
    if not report_path.exists() or not results:
        return
    best_probe = max(results, key=lambda r: r["probe_test"]["probe_r2_mean"])
    best_ood = min(results, key=lambda r: r["ood_metrics"]["trajectory_mse"])
    summary = (
        f"### Summary of Findings\n\n"
        f"Best latent probe quality came from **{best_probe['experiment']}** with mean probe R² = "
        f"{best_probe['probe_test']['probe_r2_mean']:.3f}. "
        f"Best OOD trajectory generalization came from **{best_ood['experiment']}** with OOD trajectory MSE = "
        f"{best_ood['ood_metrics']['trajectory_mse']:.3f}.\n"
    )
    text = report_path.read_text(encoding="utf-8")
    text = text.replace("*(Populate after experimental runs.)*", summary)
    text = text.replace("../assets/fig_id_generation_comparison.png", "./assets/figures/fig_id_generation_comparison.png")
    text = text.replace("../assets/gif_id_samples.gif", "./assets/videos/gif_id_samples.gif")
    text = text.replace("../assets/fig_ood_failures.png", "./assets/figures/fig_ood_failures.png")
    text = text.replace("../assets/gif_ood_breakdown.gif", "./assets/videos/gif_ood_breakdown.gif")
    text = text.replace("../assets/fig_latent_pca.png", "./assets/figures/fig_latent_pca.png")
    report_path.write_text(text, encoding="utf-8")


def run_all() -> None:
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"Running worldmodelphy on device={device}")

    configs = [
        ExperimentConfig(MotionType.CIRCULAR, context_len=4, label="short", model_kind="gru"),
        ExperimentConfig(MotionType.CIRCULAR, context_len=16, label="long", model_kind="gru"),
        ExperimentConfig(MotionType.PROJECTILE, context_len=4, label="short", model_kind="gru"),
        ExperimentConfig(MotionType.PROJECTILE, context_len=16, label="long", model_kind="gru"),
        ExperimentConfig(MotionType.BOUNCE, context_len=4, label="short", model_kind="gru"),
        ExperimentConfig(MotionType.BOUNCE, context_len=16, label="long", model_kind="gru"),
        ExperimentConfig(MotionType.PENDULUM, context_len=4, label="short", model_kind="gru"),
        ExperimentConfig(MotionType.PENDULUM, context_len=16, label="long", model_kind="gru"),
        ExperimentConfig(MotionType.TWO_BODY, context_len=4, label="short", model_kind="gru"),
        ExperimentConfig(MotionType.TWO_BODY, context_len=16, label="long", model_kind="gru"),
        ExperimentConfig(MotionType.CIRCULAR, context_len=4, label="short", model_kind="local"),
        ExperimentConfig(MotionType.PROJECTILE, context_len=4, label="short", model_kind="local"),
        ExperimentConfig(MotionType.PENDULUM, context_len=4, label="short", model_kind="local"),
        ExperimentConfig(MotionType.CIRCULAR, context_len=4, label="short", model_kind="bottleneck"),
        ExperimentConfig(MotionType.PROJECTILE, context_len=4, label="short", model_kind="bottleneck"),
        ExperimentConfig(MotionType.PENDULUM, context_len=4, label="short", model_kind="bottleneck"),
    ]

    results = [run_experiment(cfg, device) for cfg in configs]
    save_json({"results": results}, ARTIFACTS_ROOT / "summary.json")
    _copy_report_assets(results)
    _update_report(results)

    print(json.dumps({"completed": [r["experiment"] for r in results]}, indent=2))
