from __future__ import annotations

import json
from collections.abc import Callable
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
    n_particles: int = 100
    n_steps: int = 60
    epochs: int = 1400
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 42
    save_every: int = 200


class VelocityMLP(nn.Module):
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
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


def mixture_sample(n: int, left_mu: float = -1.6, right_mu: float = 1.6) -> np.ndarray:
    half = n // 2
    x1 = np.random.normal(left_mu, 0.18, size=half)
    x2 = np.random.normal(right_mu, 0.18, size=n - half)
    x = np.concatenate([x1, x2])
    np.random.shuffle(x)
    return x


def fixed_spikes(n: int, points: tuple[float, ...] = (-2.0, -1.2, -0.1, 1.0, 2.2)) -> np.ndarray:
    idx = np.random.randint(0, len(points), size=n)
    return np.array([points[i] for i in idx], dtype=np.float32)


def build_training_pairs(x0: np.ndarray, x1: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.random.randint(0, n_steps, size=len(x0))
    t = ts / float(n_steps - 1)
    x_t = (1.0 - t) * x0 + t * x1
    v = x1 - x0
    return x_t.astype(np.float32), t.astype(np.float32), v.astype(np.float32)


def simulate_trajectory(model: VelocityMLP, x_init: np.ndarray, n_steps: int) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(x_init.astype(np.float32)).unsqueeze(-1)
    traj = [x.squeeze(-1).numpy().copy()]
    dt = 1.0 / (n_steps - 1)
    with torch.no_grad():
        for i in range(n_steps - 1):
            t = torch.full((x.shape[0], 1), i / (n_steps - 1), dtype=torch.float32)
            v = model(x, t)
            x = x + dt * v
            traj.append(x.squeeze(-1).numpy().copy())
    return np.stack(traj, axis=0)


def train_case(case_name: str, x1_sampler: Callable[[int], np.ndarray]) -> dict[str, Any]:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = VelocityMLP()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    losses: list[float] = []

    for ep in range(cfg.epochs):
        x0 = np.random.uniform(-3.0, 3.0, size=cfg.batch_size)
        x1 = x1_sampler(cfg.batch_size)
        x_t, t, v = build_training_pairs(x0, x1, cfg.n_steps)

        x_t_t = torch.from_numpy(x_t).unsqueeze(-1)
        t_t = torch.from_numpy(t).unsqueeze(-1)
        v_t = torch.from_numpy(v).unsqueeze(-1)

        pred = model(x_t_t, t_t)
        loss = loss_fn(pred, v_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

        if (ep + 1) % cfg.save_every == 0:
            print(f"[{case_name}] epoch {ep + 1}/{cfg.epochs} loss={losses[-1]:.6f}")

    x_init = np.random.uniform(-3.0, 3.0, size=cfg.n_particles)
    traj = simulate_trajectory(model, x_init, cfg.n_steps)

    final_points = traj[-1]
    target_points = x1_sampler(4000)

    return {
        "config": asdict(cfg),
        "losses": losses,
        "trajectory": traj,
        "final_points": final_points,
        "target_points": target_points,
    }


def plot_case(name: str, data: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    traj: np.ndarray = data["trajectory"]
    losses: list[float] = data["losses"]
    final_points: np.ndarray = data["final_points"]
    target_points: np.ndarray = data["target_points"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sample_idx = np.linspace(0, traj.shape[1] - 1, 18, dtype=int)
    t_axis = np.linspace(0, 1, traj.shape[0])
    for idx in sample_idx:
        axes[0, 0].plot(t_axis, traj[:, idx], alpha=0.5)
    axes[0, 0].set_title(f"{name}: particle trajectories")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("x")

    axes[0, 1].plot(losses)
    axes[0, 1].set_title("training loss")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("MSE")

    axes[1, 0].hist(target_points, bins=40, alpha=0.6, label="target", density=True)
    axes[1, 0].hist(final_points, bins=40, alpha=0.6, label="generated", density=True)
    axes[1, 0].legend()
    axes[1, 0].set_title("distribution comparison")

    jitter = np.random.uniform(-0.03, 0.03, size=len(final_points))
    axes[1, 1].scatter(final_points, jitter, s=18, alpha=0.8, label="generated")
    jitter2 = np.random.uniform(-0.03, 0.03, size=len(target_points[:300]))
    axes[1, 1].scatter(target_points[:300], jitter2 + 0.08, s=10, alpha=0.45, label="target")
    axes[1, 1].set_yticks([])
    axes[1, 1].set_title("final particle locations")
    axes[1, 1].legend(loc="upper right")

    fig.suptitle(f"Demo1 / 1D Particle Flow - {name}")
    fig.tight_layout()

    png_path = out_dir / f"{name.lower().replace(' ', '_')}.png"
    fig.savefig(png_path, dpi=170)
    plt.close(fig)

    def peak_count(points: np.ndarray, bins: int = 60) -> int:
        hist, _ = np.histogram(points, bins=bins)
        count = 0
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > np.mean(hist):
                count += 1
        return count

    metrics = {
        "case": name,
        "final_mean": float(np.mean(final_points)),
        "final_std": float(np.std(final_points)),
        "target_mean": float(np.mean(target_points)),
        "target_std": float(np.std(target_points)),
        "estimated_peak_count_generated": int(peak_count(final_points)),
        "estimated_peak_count_target": int(peak_count(target_points)),
        "image": str(png_path),
    }
    return metrics


def export_animation(
    name: str,
    traj: np.ndarray,
    target_points: np.ndarray,
    out_dir: Path,
) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name.lower().replace(" ", "_")
    gif_path = out_dir / f"{stem}.gif"
    mp4_path = out_dir / f"{stem}.mp4"

    x_min = float(min(np.min(traj), np.min(target_points)) - 0.5)
    x_max = float(max(np.max(traj), np.max(target_points)) + 0.5)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.25, 0.25)
    ax.set_yticks([])
    ax.set_title(f"{name} animation")

    target_y = np.full_like(target_points, 0.12)
    ax.scatter(target_points, target_y, s=9, alpha=0.25, c="#888888", label="target")
    current = ax.scatter(traj[0], np.zeros_like(traj[0]), s=18, c="#1f77b4", label="particles")
    time_text = ax.text(0.02, 0.9, "t=0.00", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def update(frame_idx: int) -> list[Any]:
        pts = np.column_stack([traj[frame_idx], np.zeros_like(traj[frame_idx])])
        current.set_offsets(pts)
        t = frame_idx / max(traj.shape[0] - 1, 1)
        time_text.set_text(f"t={t:.2f}")
        return [current, time_text]

    ani = animation.FuncAnimation(fig, update, frames=traj.shape[0], interval=70, blit=True)
    ani.save(gif_path, writer=animation.PillowWriter(fps=14))

    saved_mp4: Path | None = None
    if animation.writers.is_available("ffmpeg"):
        ani.save(mp4_path, writer=animation.FFMpegWriter(fps=14, bitrate=1800))
        saved_mp4 = mp4_path

    plt.close(fig)
    return gif_path, saved_mp4


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "outputs"

    print("Running Demo1: 1D particle flow / spectral bias check...")
    smooth_data = train_case("smooth_mixture", mixture_sample)
    spike_data = train_case("fixed_spikes", fixed_spikes)

    smooth_metrics = plot_case("Smooth Mixture Target", smooth_data, out_dir)
    spike_metrics = plot_case("Discrete Spike Target", spike_data, out_dir)

    smooth_gif, smooth_mp4 = export_animation(
        "Smooth Mixture Target",
        smooth_data["trajectory"],
        smooth_data["target_points"],
        out_dir,
    )
    spike_gif, spike_mp4 = export_animation(
        "Discrete Spike Target",
        spike_data["trajectory"],
        spike_data["target_points"],
        out_dir,
    )

    smooth_metrics["animation_gif"] = str(smooth_gif)
    smooth_metrics["animation_mp4"] = str(smooth_mp4) if smooth_mp4 is not None else None
    spike_metrics["animation_gif"] = str(spike_gif)
    spike_metrics["animation_mp4"] = str(spike_mp4) if spike_mp4 is not None else None

    metrics = {
        "demo": "demo1_1d_particle_flow",
        "summary": "Compare smooth bimodal target vs discrete spike target to expose spectral bias tendencies.",
        "results": [smooth_metrics, spike_metrics],
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Demo1 done.")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")
    print(f"Saved images in: {out_dir}")


if __name__ == "__main__":
    main()
