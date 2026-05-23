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
    n_points: int = 100
    n_steps: int = 70
    epochs: int = 1800
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 7
    hidden: int = 96


def sample_circle(n: int, radius: float = 1.1) -> np.ndarray:
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack([x, y], axis=1).astype(np.float32)


def sample_triangle_perimeter(n: int, scale: float = 1.55) -> np.ndarray:
    v = np.array(
        [
            [0.0, 1.0],
            [-0.8660254, -0.5],
            [0.8660254, -0.5],
        ],
        dtype=np.float32,
    )
    v = v * scale

    edge_ids = np.random.randint(0, 3, size=n)
    u = np.random.uniform(0.0, 1.0, size=n).astype(np.float32)

    pts = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        a = v[edge_ids[i]]
        b = v[(edge_ids[i] + 1) % 3]
        pts[i] = (1.0 - u[i]) * a + u[i] * b
    return pts


def sort_by_angle(points: np.ndarray) -> np.ndarray:
    ang = np.arctan2(points[:, 1], points[:, 0])
    idx = np.argsort(ang)
    return points[idx]


def make_matched_pair(n: int) -> tuple[np.ndarray, np.ndarray]:
    c = sort_by_angle(sample_circle(n))
    t = sort_by_angle(sample_triangle_perimeter(n))
    return c, t


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
            c, t = make_matched_pair(cfg.n_points)
            take = min(cfg.n_points, cfg.batch_size - fill)
            x0[fill : fill + take] = c[:take]
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
            print(f"[demo2] epoch {ep + 1}/{cfg.epochs} loss={losses[-1]:.6f}")

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


def crossing_ratio(traj: np.ndarray) -> float:
    start_x = traj[0, :, 0]
    end_x = traj[-1, :, 0]
    inv = 0
    total = 0
    n = len(start_x)
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if (start_x[i] - start_x[j]) * (end_x[i] - end_x[j]) < 0:
                inv += 1
    return float(inv / max(total, 1))


def export_animation(name: str, traj: np.ndarray, target_points: np.ndarray, out_dir: Path) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name.lower().replace(" ", "_")
    gif_path = out_dir / f"{stem}.gif"
    mp4_path = out_dir / f"{stem}.mp4"

    all_x = np.concatenate([traj[:, :, 0].reshape(-1), target_points[:, 0]])
    all_y = np.concatenate([traj[:, :, 1].reshape(-1), target_points[:, 1]])
    pad = 0.25

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlim(float(np.min(all_x) - pad), float(np.max(all_x) + pad))
    ax.set_ylim(float(np.min(all_y) - pad), float(np.max(all_y) + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(name)

    ax.scatter(target_points[:, 0], target_points[:, 1], s=14, alpha=0.2, c="#888888", label="target")
    current = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=20, c="#1f77b4", label="current")
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

    print("Running Demo2: 2D circle -> triangle morph...")
    model, losses = train_model(cfg)

    x0, x1_target = make_matched_pair(cfg.n_points)
    traj = rollout(model, x0, cfg.n_steps)
    x1_pred = traj[-1]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for i in range(min(cfg.n_points, 26)):
        axes[0, 0].plot(traj[:, i, 0], traj[:, i, 1], alpha=0.45)
    axes[0, 0].set_title("particle paths")
    axes[0, 0].axis("equal")

    axes[0, 1].scatter(x0[:, 0], x0[:, 1], s=20, label="start(circle)")
    axes[0, 1].scatter(x1_target[:, 0], x1_target[:, 1], s=20, label="target(triangle)")
    axes[0, 1].set_title("start vs target")
    axes[0, 1].legend()
    axes[0, 1].axis("equal")

    axes[1, 0].scatter(x1_target[:, 0], x1_target[:, 1], s=25, label="target")
    axes[1, 0].scatter(x1_pred[:, 0], x1_pred[:, 1], s=25, marker="x", label="pred")
    axes[1, 0].set_title("final comparison")
    axes[1, 0].legend()
    axes[1, 0].axis("equal")

    axes[1, 1].plot(losses)
    axes[1, 1].set_title("training loss")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("MSE")

    fig.suptitle("Demo2 / 2D Geometric Point Cloud Morph")
    fig.tight_layout()
    img_path = out_dir / "circle_to_triangle.png"
    fig.savefig(img_path, dpi=170)
    plt.close(fig)

    gif_path, mp4_path = export_animation("Circle To Triangle", traj, x1_target, out_dir)

    mse = float(np.mean((x1_pred - x1_target) ** 2))
    metrics = {
        "demo": "demo2_2d_shape_morph",
        "config": asdict(cfg),
        "final_mse": mse,
        "crossing_ratio_x_order": crossing_ratio(traj),
        "image": str(img_path),
        "animation_gif": str(gif_path),
        "animation_mp4": str(mp4_path) if mp4_path is not None else None,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Demo2 done.")
    print(f"Saved image: {img_path}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
