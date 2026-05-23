from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedBaseClass=false, reportUnannotatedClassAttribute=false, reportAny=false, reportExplicitAny=false, reportImplicitStringConcatenation=false

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from torch import nn


@dataclass
class Config:
    n_steps: int = 60
    batch_size_1d: int = 256
    batch_size_2d: int = 256
    epochs_1d: int = 700
    epochs_2d: int = 520
    lr: float = 1e-3
    hidden_small: int = 24
    hidden_large: int = 96
    train_size_1d: int = 72
    holdout_size_1d: int = 512
    train_size_2d: int = 140
    holdout_size_2d: int = 2400
    eval_points_1d: int = 512
    eval_points_2d: int = 180
    summary_seeds: tuple[int, ...] = (0, 1, 2, 3)


class VelocityMLP1D(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


class FourierVelocityMLP1D(nn.Module):
    def __init__(self, hidden: int = 96, n_bands: int = 8) -> None:
        super().__init__()
        self.register_buffer("bands", 2.0 ** torch.arange(n_bands, dtype=torch.float32))
        feat_dim = 2 + 4 * n_bands
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t], dim=-1)
        freq = self.bands[None, :]
        x_proj = x * np.pi * freq
        t_proj = t * np.pi * freq
        feats = torch.cat(
            [
                xt,
                torch.sin(x_proj),
                torch.cos(x_proj),
                torch.sin(t_proj),
                torch.cos(t_proj),
            ],
            dim=-1,
        )
        return self.net(feats)


class VectorFieldMLP2D(nn.Module):
    def __init__(self, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_map_1d(u: np.ndarray) -> np.ndarray:
    return (1.35 * u).astype(np.float32)


def target_low_freq_map_1d(u: np.ndarray) -> np.ndarray:
    y = 1.15 * u + 0.42 * np.sin(np.pi * u)
    return y.astype(np.float32)


def target_high_freq_map_1d(u: np.ndarray) -> np.ndarray:
    y = 0.95 * u + 0.24 * np.sin(5.0 * np.pi * u) + 0.10 * np.sin(11.0 * np.pi * u)
    return y.astype(np.float32)


def sample_circle(n: int, radius: float = 1.0) -> np.ndarray:
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    pts = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    return pts.astype(np.float32)


def sample_flower_target_2d(n: int) -> np.ndarray:
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    radius = 1.0 + 0.34 * np.cos(5.0 * theta) + 0.12 * np.cos(10.0 * theta)
    radius += np.random.normal(0.0, 0.03, size=n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pts = np.stack([x, y], axis=1)
    pts += np.random.normal(0.0, 0.015, size=pts.shape)
    return pts.astype(np.float32)


def build_bridge_batch_1d(source: np.ndarray, target: np.ndarray, n_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tt = np.random.randint(0, n_steps, size=len(source)).astype(np.float32)
    tt = tt / float(n_steps - 1)
    x_t = (1.0 - tt) * source + tt * target
    v = target - source
    return (
        torch.from_numpy(x_t).unsqueeze(-1),
        torch.from_numpy(tt).unsqueeze(-1),
        torch.from_numpy(v).unsqueeze(-1),
    )


def build_bridge_batch_2d(source: np.ndarray, target: np.ndarray, n_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tt = np.random.randint(0, n_steps, size=len(source)).astype(np.float32)
    tt = tt / float(n_steps - 1)
    x_t = (1.0 - tt[:, None]) * source + tt[:, None] * target
    v = target - source
    return torch.from_numpy(x_t), torch.from_numpy(tt).unsqueeze(-1), torch.from_numpy(v)


def train_1d_case(
    cfg: Config,
    case_name: str,
    model_kind: str,
    target_fn: Any,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    train_u = np.sort(np.random.uniform(-1.0, 1.0, size=cfg.train_size_1d).astype(np.float32))
    holdout_u = np.linspace(-1.0, 1.0, cfg.holdout_size_1d, dtype=np.float32)
    train_source = source_map_1d(train_u)
    train_target = target_fn(train_u)
    holdout_source = source_map_1d(holdout_u)
    holdout_target = target_fn(holdout_u)

    if model_kind == "mlp_small":
        model: nn.Module = VelocityMLP1D(hidden=cfg.hidden_small)
    elif model_kind == "mlp_large":
        model = VelocityMLP1D(hidden=cfg.hidden_large)
    elif model_kind == "fourier":
        model = FourierVelocityMLP1D(hidden=cfg.hidden_large)
    else:
        raise ValueError(f"unknown model_kind={model_kind}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    losses: list[float] = []

    for ep in range(cfg.epochs_1d):
        idx = np.random.randint(0, len(train_u), size=cfg.batch_size_1d)
        source = train_source[idx]
        target = train_target[idx]
        x_t, t_t, v_t = build_bridge_batch_1d(source, target, cfg.n_steps)

        pred = model(x_t, t_t)
        loss = loss_fn(pred, v_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

        if (ep + 1) % 200 == 0:
            print(f"[1d:{case_name}:{model_kind}] epoch {ep + 1}/{cfg.epochs_1d} loss={losses[-1]:.6f}")

    traj = rollout_1d(model, holdout_source, cfg.n_steps)
    generated = traj[-1]
    metrics = compute_1d_metrics(
        generated=generated,
        train_source=train_source,
        train_target=train_target,
        holdout_source=holdout_source,
        holdout_target=holdout_target,
        train_u=train_u,
        holdout_u=holdout_u,
    )
    return {
        "case": case_name,
        "model_kind": model_kind,
        "seed": seed,
        "config": asdict(cfg),
        "losses": losses,
        "train_u": train_u,
        "holdout_u": holdout_u,
        "train_source": train_source,
        "holdout_source": holdout_source,
        "train_target": train_target,
        "holdout_target": holdout_target,
        "trajectory": traj,
        "generated": generated,
        "metrics": metrics,
    }


def rollout_1d(model: nn.Module, x_init: np.ndarray, n_steps: int) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(x_init.astype(np.float32)).unsqueeze(-1)
    traj = [x.squeeze(-1).numpy().copy()]
    dt = 1.0 / float(n_steps - 1)
    with torch.no_grad():
        for i in range(n_steps - 1):
            t = torch.full((x.shape[0], 1), i / float(n_steps - 1), dtype=torch.float32)
            v = model(x, t)
            x = x + dt * v
            traj.append(x.squeeze(-1).numpy().copy())
    return np.stack(traj, axis=0)


def train_2d_case(cfg: Config, seed: int) -> dict[str, Any]:
    set_seed(seed)
    train_target = sample_flower_target_2d(cfg.train_size_2d)
    holdout_target = sample_flower_target_2d(cfg.holdout_size_2d)
    model = VectorFieldMLP2D(hidden=cfg.hidden_large)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    losses: list[float] = []

    for ep in range(cfg.epochs_2d):
        source = sample_circle(cfg.batch_size_2d)
        idx = np.random.randint(0, len(train_target), size=cfg.batch_size_2d)
        target = train_target[idx]
        x_t, t_t, v_t = build_bridge_batch_2d(source, target, cfg.n_steps)

        pred = model(x_t, t_t)
        loss = loss_fn(pred, v_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

        if (ep + 1) % 200 == 0:
            print(f"[2d:seed={seed}] epoch {ep + 1}/{cfg.epochs_2d} loss={losses[-1]:.6f}")

    x0 = sample_circle(cfg.eval_points_2d)
    traj = rollout_2d(model, x0, cfg.n_steps)
    generated = traj[-1]
    metrics = compute_2d_metrics(generated, train_target, holdout_target)
    return {
        "seed": seed,
        "config": asdict(cfg),
        "losses": losses,
        "train_target": train_target,
        "holdout_target": holdout_target,
        "trajectory": traj,
        "generated": generated,
        "metrics": metrics,
        "model": model,
    }


def rollout_2d(model: nn.Module, x_init: np.ndarray, n_steps: int) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(x_init.astype(np.float32))
    traj = [x.numpy().copy()]
    dt = 1.0 / float(n_steps - 1)
    with torch.no_grad():
        for i in range(n_steps - 1):
            t = torch.full((x.shape[0], 1), i / float(n_steps - 1), dtype=torch.float32)
            v = model(x, t)
            x = x + dt * v
            traj.append(x.numpy().copy())
    return np.stack(traj, axis=0)


def nearest_distance_1d(points: np.ndarray, refs: np.ndarray) -> np.ndarray:
    diff = np.abs(points[:, None] - refs[None, :])
    return np.min(diff, axis=1)


def nearest_distance_2d(points: np.ndarray, refs: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - refs[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return np.min(dist, axis=1)


def compute_1d_metrics(
    generated: np.ndarray,
    train_source: np.ndarray,
    train_target: np.ndarray,
    holdout_source: np.ndarray,
    holdout_target: np.ndarray,
    train_u: np.ndarray,
    holdout_u: np.ndarray,
) -> dict[str, float]:
    train_pred = np.interp(train_u, holdout_u, generated)
    residual = generated - holdout_target
    gen_spec = np.abs(np.fft.rfft(generated - np.mean(generated)))
    hold_spec = np.abs(np.fft.rfft(holdout_target - np.mean(holdout_target)))
    high_start = max(2, int(0.35 * len(hold_spec)))
    pred_smooth = np.std(np.diff(generated, n=2))
    target_smooth = np.std(np.diff(holdout_target, n=2))
    train_error = train_pred - train_target
    holdout_error = residual
    _ = train_source, holdout_source
    return {
        "train_endpoint_mse": float(np.mean(train_error * train_error)),
        "holdout_endpoint_mse": float(np.mean(holdout_error * holdout_error)),
        "generalization_gap": float(
            np.mean(holdout_error * holdout_error) / (np.mean(train_error * train_error) + 1e-8)
        ),
        "spectrum_error": float(np.mean(np.abs(gen_spec - hold_spec)) / (np.mean(hold_spec) + 1e-6)),
        "high_freq_capture_ratio": float(
            np.sum(gen_spec[high_start:]) / (np.sum(hold_spec[high_start:]) + 1e-6)
        ),
        "smoothness_ratio": float(pred_smooth / (target_smooth + 1e-6)),
        "endpoint_l1": float(np.mean(np.abs(generated - holdout_target))),
    }


def compute_2d_metrics(generated: np.ndarray, train_target: np.ndarray, holdout_target: np.ndarray) -> dict[str, float]:
    train_nn = nearest_distance_2d(generated, train_target)
    hold_nn = nearest_distance_2d(generated, holdout_target)
    gen_r = np.sqrt(np.sum(generated * generated, axis=1))
    hold_r = np.sqrt(np.sum(holdout_target * holdout_target, axis=1))
    gen_r_hist, _ = np.histogram(gen_r, bins=80, range=(0.45, 1.65), density=True)
    hold_r_hist, _ = np.histogram(hold_r, bins=80, range=(0.45, 1.65), density=True)
    gen_theta = np.mod(np.arctan2(generated[:, 1], generated[:, 0]), 2.0 * np.pi)
    hold_theta = np.mod(np.arctan2(holdout_target[:, 1], holdout_target[:, 0]), 2.0 * np.pi)
    gen_ang_hist, _ = np.histogram(gen_theta, bins=72, range=(0.0, 2.0 * np.pi), density=True)
    hold_ang_hist, _ = np.histogram(hold_theta, bins=72, range=(0.0, 2.0 * np.pi), density=True)
    return {
        "holdout_chamfer_like": float(np.mean(hold_nn)),
        "memorization_ratio": float(np.mean(train_nn) / (np.mean(hold_nn) + 1e-6)),
        "radial_profile_l1": float(np.mean(np.abs(gen_r_hist - hold_r_hist))),
        "angular_spectrum_error": float(np.mean(np.abs(np.abs(np.fft.rfft(gen_ang_hist)) - np.abs(np.fft.rfft(hold_ang_hist))))),
        "final_radius_mean": float(np.mean(gen_r)),
        "final_radius_std": float(np.std(gen_r)),
    }


def plot_1d_summary(results: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    order = [
        ("low_freq", "mlp_small", "Low-frequency target / small MLP"),
        ("high_freq", "mlp_small", "High-frequency target / small MLP"),
        ("high_freq", "fourier", "High-frequency target / Fourier MLP"),
    ]

    for col, (case_name, model_kind, title) in enumerate(order):
        result = next(r for r in results if r["case"] == case_name and r["model_kind"] == model_kind)
        traj = result["trajectory"]
        generated = result["generated"]
        holdout_u = result["holdout_u"]
        holdout = result["holdout_target"]
        train_u = result["train_u"]
        train_target = result["train_target"]
        losses = result["losses"]

        idx = np.linspace(0, traj.shape[1] - 1, 24, dtype=int)
        t_axis = np.linspace(0.0, 1.0, traj.shape[0])
        for i in idx:
            axes[0, col].plot(t_axis, traj[:, i], alpha=0.35)
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("time")
        axes[0, col].set_ylabel("x")

        axes[1, col].plot(holdout_u, holdout, linewidth=2.0, label="target map")
        axes[1, col].plot(holdout_u, generated, linewidth=1.8, label="predicted map")
        axes[1, col].scatter(train_u, train_target, s=10, alpha=0.55, label="train anchors")
        axes[1, col].set_title(
            f"holdout_mse={result['metrics']['holdout_endpoint_mse']:.4f}  hf={result['metrics']['high_freq_capture_ratio']:.2f}"
        )
        if col == 0:
            axes[1, col].legend(loc="upper left")

        axes[2, col].plot(losses)
        axes[2, col].set_title(
            f"gap={result['metrics']['generalization_gap']:.2f}  smooth={result['metrics']['smoothness_ratio']:.2f}"
        )
        axes[2, col].set_xlabel("epoch")
        axes[2, col].set_ylabel("MSE")

    fig.suptitle("Demo3 / Flow Matching Generalization - Hypothesis A: spectral bias and smoothing")
    fig.tight_layout()
    path = out_dir / "spectral_bias_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_2d_summary(results: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for idx, result in enumerate(results[:4]):
        ax = axes.flat[idx]
        traj = result["trajectory"]
        generated = result["generated"]
        holdout = result["holdout_target"]
        for i in range(min(18, traj.shape[1])):
            ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.25, linewidth=0.8)
        ax.scatter(holdout[:, 0], holdout[:, 1], s=8, alpha=0.08, c="#888888")
        ax.scatter(generated[:, 0], generated[:, 1], s=15, alpha=0.8, c="#1f77b4")
        ax.set_title(
            f"seed {result['seed']} | chamfer={result['metrics']['holdout_chamfer_like']:.3f} | memo={result['metrics']['memorization_ratio']:.2f}"
        )
        ax.axis("equal")

    chamfers = [r["metrics"]["holdout_chamfer_like"] for r in results]
    memos = [r["metrics"]["memorization_ratio"] for r in results]
    spectral = [r["metrics"]["angular_spectrum_error"] for r in results]
    seed_labels = [str(r["seed"]) for r in results]
    ax_bar = axes[1, 1]
    x = np.arange(len(results))
    w = 0.25
    ax_bar.bar(x - w, chamfers, width=w, label="holdout_chamfer")
    ax_bar.bar(x, memos, width=w, label="memorization_ratio")
    ax_bar.bar(x + w, spectral, width=w, label="angular_spectrum_err")
    ax_bar.set_xticks(x, seed_labels)
    ax_bar.set_title("seed-to-seed metric spread")
    ax_bar.legend(fontsize=8)

    ax_text = axes[1, 2]
    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.96,
        "Randomness summary\n"
        f"holdout_chamfer mean±std = {np.mean(chamfers):.4f} ± {np.std(chamfers):.4f}\n"
        f"memorization_ratio mean±std = {np.mean(memos):.4f} ± {np.std(memos):.4f}\n"
        f"angular_spectrum_err mean±std = {np.mean(spectral):.4f} ± {np.std(spectral):.4f}\n",
        va="top",
        fontsize=10,
    )

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.02,
        0.96,
        "Hypothesis B:\nidentical train data + architecture\nstill gives visible sample and metric spread across seeds.",
        va="top",
        fontsize=12,
    )

    fig.suptitle("Demo3 / Flow Matching Generalization - Hypothesis B: training randomness")
    fig.tight_layout()
    path = out_dir / "randomness_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_vector_field(result: dict[str, Any], out_dir: Path) -> Path:
    model: nn.Module = result["model"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    grid_x = np.linspace(-1.65, 1.65, 17)
    grid_y = np.linspace(-1.65, 1.65, 17)
    xx, yy = np.meshgrid(grid_x, grid_y)
    coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)

    with torch.no_grad():
        for ax, t_value in zip(axes, [0.1, 0.5, 0.9]):
            t = torch.full((coords.shape[0], 1), t_value, dtype=torch.float32)
            vel = model(torch.from_numpy(coords), t).numpy()
            ax.quiver(coords[:, 0], coords[:, 1], vel[:, 0], vel[:, 1], angles="xy", scale_units="xy", scale=6.0, alpha=0.8)
            ax.scatter(result["holdout_target"][:, 0], result["holdout_target"][:, 1], s=8, alpha=0.08, c="#888888")
            ax.set_title(f"vector field at t={t_value:.1f}")
            ax.axis("equal")
            ax.set_xlim(-1.75, 1.75)
            ax.set_ylim(-1.75, 1.75)

    fig.suptitle("Demo3 / 2D vector fields for the flower target")
    fig.tight_layout()
    path = out_dir / "vector_field_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_combined_summary(results_1d: list[dict[str, Any]], results_2d: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    names = [
        "low_freq / small MLP",
        "high_freq / small MLP",
        "high_freq / Fourier MLP",
    ]
    spec_err = [r["metrics"]["spectrum_error"] for r in results_1d]
    hf_ratio = [r["metrics"]["high_freq_capture_ratio"] for r in results_1d]
    x = np.arange(len(results_1d))
    axes[0, 0].bar(x, spec_err, color=["#54a24b", "#e45756", "#4c78a8"])
    axes[0, 0].set_xticks(x, names, rotation=15, ha="right")
    axes[0, 0].set_title("1D spectrum error")

    axes[0, 1].bar(x, hf_ratio, color=["#54a24b", "#e45756", "#4c78a8"])
    axes[0, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_xticks(x, names, rotation=15, ha="right")
    axes[0, 1].set_title("1D high-frequency energy ratio")

    seed_labels = [str(r["seed"]) for r in results_2d]
    chamfers = [r["metrics"]["holdout_chamfer_like"] for r in results_2d]
    memos = [r["metrics"]["memorization_ratio"] for r in results_2d]
    x2 = np.arange(len(results_2d))
    axes[1, 0].plot(x2, chamfers, marker="o", label="holdout_chamfer")
    axes[1, 0].plot(x2, memos, marker="s", label="memorization_ratio")
    axes[1, 0].set_xticks(x2, seed_labels)
    axes[1, 0].set_title("2D seed sensitivity")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        "Takeaways\n"
        "• Small MLP matches low-frequency targets but smooths high-frequency comb structure.\n"
        "• Fourier features recover more high-frequency energy and more modes.\n"
        "• Re-running the same 2D setup across seeds changes coverage and memorization proxies.\n"
        "• Artifacts are saved under demos/demo3_flow_generalization/outputs.",
        va="top",
        fontsize=11,
    )

    fig.suptitle("Demo3 / Combined Flow Matching Generalization Summary")
    fig.tight_layout()
    path = out_dir / "combined_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def export_animation_2d(name: str, traj: np.ndarray, target_points: np.ndarray, out_dir: Path) -> tuple[Path, Path | None]:
    stem = name.lower().replace(" ", "_")
    gif_path = out_dir / f"{stem}.gif"
    mp4_path = out_dir / f"{stem}.mp4"

    all_x = np.concatenate([traj[:, :, 0].reshape(-1), target_points[:, 0]])
    all_y = np.concatenate([traj[:, :, 1].reshape(-1), target_points[:, 1]])
    pad = 0.2

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.set_xlim(float(np.min(all_x) - pad), float(np.max(all_x) + pad))
    ax.set_ylim(float(np.min(all_y) - pad), float(np.max(all_y) + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(name)
    ax.scatter(target_points[:, 0], target_points[:, 1], s=12, alpha=0.15, c="#888888", label="holdout target")
    current = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=22, c="#1f77b4", label="current")
    time_text = ax.text(0.03, 0.95, "t=0.00", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def update(frame_idx: int) -> list[Any]:
        xy = np.column_stack([traj[frame_idx, :, 0], traj[frame_idx, :, 1]])
        current.set_offsets(xy)
        time_text.set_text(f"t={frame_idx / max(traj.shape[0] - 1, 1):.2f}")
        return [current, time_text]

    ani = animation.FuncAnimation(fig, update, frames=traj.shape[0], interval=70, blit=True)
    ani.save(gif_path, writer=animation.PillowWriter(fps=14))

    saved_mp4: Path | None = None
    if animation.writers.is_available("ffmpeg"):
        ani.save(mp4_path, writer=animation.FFMpegWriter(fps=14, bitrate=1800))
        saved_mp4 = mp4_path

    plt.close(fig)
    return gif_path, saved_mp4


def json_ready(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for result in results:
        output.append(
            {
                "case": result.get("case"),
                "model_kind": result.get("model_kind"),
                "seed": result["seed"],
                "metrics": result["metrics"],
                "final_loss": float(result["losses"][-1]),
            }
        )
    return output


def main() -> None:
    cfg = Config()
    root = Path(__file__).resolve().parent
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Demo3: Flow Matching generalization suite...")

    one_d_results = [
        train_1d_case(cfg, "low_freq", "mlp_small", target_low_freq_map_1d, seed=11),
        train_1d_case(cfg, "high_freq", "mlp_small", target_high_freq_map_1d, seed=11),
        train_1d_case(cfg, "high_freq", "fourier", target_high_freq_map_1d, seed=11),
    ]

    two_d_results = [train_2d_case(cfg, seed) for seed in cfg.summary_seeds]

    spectral_fig = plot_1d_summary(one_d_results, out_dir)
    randomness_fig = plot_2d_summary(two_d_results, out_dir)
    vector_field_fig = plot_vector_field(two_d_results[0], out_dir)
    combined_fig = plot_combined_summary(one_d_results, two_d_results, out_dir)
    gif_path, mp4_path = export_animation_2d(
        "Flower Morph Seed 0",
        two_d_results[0]["trajectory"],
        two_d_results[0]["holdout_target"],
        out_dir,
    )

    randomness_summary = {
        "holdout_chamfer_mean": float(np.mean([r["metrics"]["holdout_chamfer_like"] for r in two_d_results])),
        "holdout_chamfer_std": float(np.std([r["metrics"]["holdout_chamfer_like"] for r in two_d_results])),
        "memorization_ratio_mean": float(np.mean([r["metrics"]["memorization_ratio"] for r in two_d_results])),
        "memorization_ratio_std": float(np.std([r["metrics"]["memorization_ratio"] for r in two_d_results])),
        "angular_spectrum_error_mean": float(np.mean([r["metrics"]["angular_spectrum_error"] for r in two_d_results])),
        "angular_spectrum_error_std": float(np.std([r["metrics"]["angular_spectrum_error"] for r in two_d_results])),
    }

    metrics = {
        "demo": "demo3_flow_generalization",
        "summary": {
            "hypothesis_a": "FM with a small plain MLP fits low-frequency targets more easily and smooths out high-frequency comb structure; Fourier features recover more high-frequency detail.",
            "hypothesis_b": "With fixed data and architecture, different seeds still change 2D flower coverage and memorization/generalization proxies.",
        },
        "config": asdict(cfg),
        "one_d_spectral_bias": json_ready(one_d_results),
        "two_d_randomness_runs": json_ready(two_d_results),
        "two_d_randomness_summary": randomness_summary,
        "artifacts": {
            "spectral_bias_summary": str(spectral_fig),
            "randomness_summary": str(randomness_fig),
            "vector_field_summary": str(vector_field_fig),
            "combined_summary": str(combined_fig),
            "animation_gif": str(gif_path),
            "animation_mp4": str(mp4_path) if mp4_path is not None else None,
        },
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Demo3 done.")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")
    print(f"Saved outputs in: {out_dir}")


if __name__ == "__main__":
    main()
