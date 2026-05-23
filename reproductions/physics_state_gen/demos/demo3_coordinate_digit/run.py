from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from torch import nn
from typing import Any


@dataclass
class Config:
    n_points: int = 220
    n_steps: int = 70
    epochs: int = 1800
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 11
    hidden: int = 96


class VectorFieldMLP(nn.Module):
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
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


def sample_ellipse_perimeter(n: int, center: tuple[float, float], rx: float, ry: float) -> np.ndarray:
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    x = center[0] + rx * np.cos(theta)
    y = center[1] + ry * np.sin(theta)
    return np.stack([x, y], axis=1).astype(np.float32)


def sample_digit_eight_points(n: int) -> np.ndarray:
    upper_n = n // 2
    lower_n = n - upper_n

    upper = sample_ellipse_perimeter(upper_n, center=(0.0, 0.72), rx=0.55, ry=0.43)
    lower = sample_ellipse_perimeter(lower_n, center=(0.0, -0.72), rx=0.72, ry=0.53)

    pts = np.concatenate([upper, lower], axis=0)
    pts += np.random.normal(0.0, 0.025, size=pts.shape).astype(np.float32)
    return pts


def sort_by_angle_around_centroid(points: np.ndarray) -> np.ndarray:
    c = np.mean(points, axis=0, keepdims=True)
    shifted = points - c
    ang = np.arctan2(shifted[:, 1], shifted[:, 0])
    idx = np.argsort(ang)
    return points[idx]


def make_pair(n: int) -> tuple[np.ndarray, np.ndarray]:
    src = np.random.uniform(-1.25, 1.25, size=(n, 2)).astype(np.float32)
    tgt = sample_digit_eight_points(n)
    return sort_by_angle_around_centroid(src), sort_by_angle_around_centroid(tgt)


def train_model(cfg: Config) -> tuple[VectorFieldMLP, list[float]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = VectorFieldMLP(hidden=cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    losses: list[float] = []

    for ep in range(cfg.epochs):
        x0 = np.zeros((cfg.batch_size, 2), dtype=np.float32)
        x1 = np.zeros((cfg.batch_size, 2), dtype=np.float32)

        fill = 0
        while fill < cfg.batch_size:
            s, t = make_pair(cfg.n_points)
            take = min(cfg.n_points, cfg.batch_size - fill)
            x0[fill : fill + take] = s[:take]
            x1[fill : fill + take] = t[:take]
            fill += take

        ts = np.random.randint(0, cfg.n_steps, size=cfg.batch_size)
        tt = (ts / (cfg.n_steps - 1)).astype(np.float32)

        x_t = (1.0 - tt[:, None]) * x0 + tt[:, None] * x1
        v = x1 - x0

        x_t_t = torch.from_numpy(x_t)
        t_t = torch.from_numpy(tt).unsqueeze(-1)
        v_t = torch.from_numpy(v)

        pred = model(x_t_t, t_t)
        loss = loss_fn(pred, v_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

        if (ep + 1) % 300 == 0:
            print(f"[demo3] epoch {ep + 1}/{cfg.epochs} loss={losses[-1]:.6f}")

    return model, losses


def rollout(model: VectorFieldMLP, x_init: np.ndarray, n_steps: int) -> np.ndarray:
    model.eval()
    dt = 1.0 / (n_steps - 1)
    x = torch.from_numpy(x_init.astype(np.float32))
    traj = [x.numpy().copy()]
    with torch.no_grad():
        for i in range(n_steps - 1):
            t = torch.full((x.shape[0], 1), i / (n_steps - 1), dtype=torch.float32)
            v = model(x, t)
            x = x + dt * v
            traj.append(x.numpy().copy())
    return np.stack(traj, axis=0)


def path_smoothness(traj: np.ndarray) -> float:
    vel = np.diff(traj, axis=0)
    acc = np.diff(vel, axis=0)
    return float(np.mean(np.linalg.norm(acc, axis=-1)))


def export_animation(name: str, traj: np.ndarray, target_points: np.ndarray, out_dir: Path) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name.lower().replace(" ", "_")
    gif_path = out_dir / f"{stem}.gif"
    mp4_path = out_dir / f"{stem}.mp4"

    all_x = np.concatenate([traj[:, :, 0].reshape(-1), target_points[:, 0]])
    all_y = np.concatenate([traj[:, :, 1].reshape(-1), target_points[:, 1]])
    pad = 0.25

    fig, ax = plt.subplots(figsize=(6.7, 6.7))
    ax.set_xlim(float(np.min(all_x) - pad), float(np.max(all_x) + pad))
    ax.set_ylim(float(np.min(all_y) - pad), float(np.max(all_y) + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(name)

    ax.scatter(target_points[:, 0], target_points[:, 1], s=12, alpha=0.18, c="#888888", label="target")
    current = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=16, c="#1f77b4", label="current")
    time_text = ax.text(0.02, 0.95, "t=0.00", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def update(frame_idx: int) -> list[Any]:
        frame_xy = np.column_stack([traj[frame_idx, :, 0], traj[frame_idx, :, 1]])
        current.set_offsets(frame_xy)
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
    cfg = Config()
    root = Path(__file__).resolve().parent
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Demo3: coordinate digit generation (digit 8 point cloud)...")
    model, losses = train_model(cfg)

    x0, x1_target = make_pair(cfg.n_points)
    traj = rollout(model, x0, cfg.n_steps)
    x1_pred = traj[-1]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    trace_ids = np.linspace(0, cfg.n_points - 1, 30, dtype=int)
    for i in trace_ids:
        axes[0, 0].plot(traj[:, i, 0], traj[:, i, 1], alpha=0.4)
    axes[0, 0].set_title("point trajectories")
    axes[0, 0].axis("equal")

    axes[0, 1].scatter(x0[:, 0], x0[:, 1], s=14, alpha=0.7, label="start(noise)")
    axes[0, 1].scatter(x1_target[:, 0], x1_target[:, 1], s=14, alpha=0.7, label="target(8)")
    axes[0, 1].legend()
    axes[0, 1].set_title("source vs target")
    axes[0, 1].axis("equal")

    axes[1, 0].scatter(x1_target[:, 0], x1_target[:, 1], s=14, alpha=0.65, label="target")
    axes[1, 0].scatter(x1_pred[:, 0], x1_pred[:, 1], s=14, alpha=0.65, marker="x", label="generated")
    axes[1, 0].legend()
    axes[1, 0].set_title("final generated point cloud")
    axes[1, 0].axis("equal")

    axes[1, 1].plot(losses)
    axes[1, 1].set_title("training loss")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("MSE")

    fig.suptitle("Demo3 / Coordinate Digit (8) Point Cloud")
    fig.tight_layout()
    img_path = out_dir / "coordinate_digit_8.png"
    fig.savefig(img_path, dpi=170)
    plt.close(fig)

    gif_path, mp4_path = export_animation("Coordinate Digit 8", traj, x1_target, out_dir)

    mse = float(np.mean((x1_pred - x1_target) ** 2))
    metrics = {
        "demo": "demo3_coordinate_digit",
        "config": asdict(cfg),
        "final_mse": mse,
        "trajectory_smoothness": path_smoothness(traj),
        "image": str(img_path),
        "animation_gif": str(gif_path),
        "animation_mp4": str(mp4_path) if mp4_path is not None else None,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Demo3 done.")
    print(f"Saved image: {img_path}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
