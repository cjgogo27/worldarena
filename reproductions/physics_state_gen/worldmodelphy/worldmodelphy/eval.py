from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def frames_to_centroids(frames: torch.Tensor) -> np.ndarray:
    if frames.ndim == 4:
        frames = frames.squeeze(1)
    frames_np = frames.detach().cpu().numpy()
    centroids: list[list[float]] = []
    for frame in frames_np:
        ys, xs = np.nonzero(frame > 0.1)
        if len(xs) == 0:
            centroids.append([np.nan, np.nan])
        else:
            centroids.append([float(xs.mean()), float(ys.mean())])
    return np.asarray(centroids, dtype=np.float32)


def _finite_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)


def trajectory_mse(pred_xy: np.ndarray, target_xy: np.ndarray) -> float:
    mask = _finite_mask(pred_xy, target_xy)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sum((pred_xy[mask] - target_xy[mask]) ** 2, axis=-1)))


def derivative_mse(pred_xy: np.ndarray, target_xy: np.ndarray, order: int = 1) -> float:
    pred = pred_xy.copy()
    target = target_xy.copy()
    for _ in range(order):
        pred = np.diff(pred, axis=0)
        target = np.diff(target, axis=0)
    if len(pred) == 0:
        return 0.0
    return trajectory_mse(pred, target)


def rollout_metrics(pred_frames: torch.Tensor, target_frames: torch.Tensor) -> dict[str, float]:
    pred_xy = frames_to_centroids(pred_frames)
    target_xy = frames_to_centroids(target_frames)
    return {
        "frame_mse": float(torch.mean((pred_frames - target_frames) ** 2).item()),
        "trajectory_mse": trajectory_mse(pred_xy, target_xy),
        "velocity_mse": derivative_mse(pred_xy, target_xy, order=1),
        "acceleration_mse": derivative_mse(pred_xy, target_xy, order=2),
    }


def fit_linear_probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, float]:
    ones_train = np.ones((train_x.shape[0], 1), dtype=train_x.dtype)
    ones_test = np.ones((test_x.shape[0], 1), dtype=test_x.dtype)
    x_train = np.concatenate([train_x, ones_train], axis=1)
    x_test = np.concatenate([test_x, ones_test], axis=1)
    coef, *_ = np.linalg.lstsq(x_train, train_y, rcond=None)
    pred = x_test @ coef
    mse = np.mean((pred - test_y) ** 2, axis=0)
    var = np.var(test_y, axis=0)
    r2 = np.empty_like(mse)
    constant_mask = var < 1e-8
    r2[constant_mask] = (mse[constant_mask] < 1e-8).astype(np.float32)
    r2[~constant_mask] = 1.0 - (mse[~constant_mask] / var[~constant_mask])
    names = ["vx", "vy", "ax", "ay"][: test_y.shape[1]]
    result = {f"probe_r2_{name}": float(r2[idx]) for idx, name in enumerate(names)}
    result.update({f"probe_mse_{name}": float(mse[idx]) for idx, name in enumerate(names)})
    result["probe_r2_mean"] = float(np.mean(r2))
    return result


def save_json(data: dict[str, Any], path: str | Path) -> None:
    def _default(obj: Any) -> Any:
        if hasattr(obj, "value"):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default)
