"""Synthetic video dataset generation for physics trajectory experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

Image = None
_pil_available = False

try:
    from PIL import Image as PILImage
    Image = PILImage
    _pil_available = True
except ImportError:
    pass


class MotionType(Enum):
    CIRCULAR = "circular"
    PROJECTILE = "projectile"
    BOUNCE = "bounce"
    PENDULUM = "pendulum"
    TWO_BODY = "two_body"


@dataclass(frozen=True)
class CircularConfig:
    radius: float = 0.25
    angular_velocity: float = 0.15
    center_x: float = 0.5
    center_y: float = 0.5
    object_radius: int = 4
    object_value: int = 255


@dataclass(frozen=True)
class ProjectileConfig:
    initial_x: float = 0.1
    initial_y: float = 0.8
    vx: float = 2.0
    vy: float = -4.0
    gravity: float = 0.15
    object_radius: int = 4
    object_value: int = 255


@dataclass(frozen=True)
class BounceConfig:
    """Bouncing object that bounces off walls/floor with restitution."""
    initial_x: float = 0.2
    initial_y: float = 0.2
    vx: float = 1.5
    vy: float = 2.0
    gravity: float = 0.15
    restitution: float = 0.8  # Bounciness (1.0 = perfectly elastic)
    object_radius: int = 4
    object_value: int = 255


@dataclass(frozen=True)
class PendulumConfig:
    """Simple pendulum swinging in a plane."""
    length: float = 0.35  # Length as fraction of frame height
    gravity: float = 0.2
    initial_angle: float = 0.5  # Initial angle in radians
    initial_angular_vel: float = 0.0
    pivot_x: float = 0.5  # Pivot point as fraction of frame
    pivot_y: float = 0.15
    object_radius: int = 4
    object_value: int = 255


@dataclass(frozen=True)
class TwoBodyConfig:
    """Two-body system orbiting a shared center of mass."""
    orbit_radius_1: float = 0.12  # Inner object orbit radius
    orbit_radius_2: float = 0.25  # Outer object orbit radius (further out)
    angular_velocity: float = 0.15
    center_x: float = 0.5
    center_y: float = 0.5
    mass_ratio: float = 2.0  # mass_2 / mass_1 (outer is heavier)
    object_radius_1: int = 3
    object_radius_2: int = 5
    object_value_1: int = 255
    object_value_2: int = 200  # Slightly dimmer for outer object


@dataclass
class FrameConfig:
    width: int = 64
    height: int = 64
    num_frames: int = 32
    background_value: int = 0


@dataclass
class DatasetConfig:
    motion_type: MotionType
    frame: FrameConfig = field(default_factory=FrameConfig)
    split: str = "train"
    seed: int = 42
    motion: CircularConfig | ProjectileConfig | BounceConfig | PendulumConfig | TwoBodyConfig = field(
        default_factory=CircularConfig
    )
    context_frames: int | None = None


CIRCULAR_TRAIN_CONFIG = DatasetConfig(
    motion_type=MotionType.CIRCULAR,
    frame=FrameConfig(),
    split="train",
    seed=42,
    motion=CircularConfig(radius=0.2, angular_velocity=0.1),
)

CIRCULAR_VAL_CONFIG = DatasetConfig(
    motion_type=MotionType.CIRCULAR,
    frame=FrameConfig(),
    split="val",
    seed=43,
    motion=CircularConfig(radius=0.2, angular_velocity=0.1),
)

CIRCULAR_TEST_CONFIG = DatasetConfig(
    motion_type=MotionType.CIRCULAR,
    frame=FrameConfig(),
    split="test",
    seed=44,
    motion=CircularConfig(radius=0.2, angular_velocity=0.1),
)

PROJECTILE_TRAIN_CONFIG = DatasetConfig(
    motion_type=MotionType.PROJECTILE,
    frame=FrameConfig(),
    split="train",
    seed=42,
    motion=ProjectileConfig(
        initial_x=0.1, initial_y=0.8, vx=2.0, vy=-4.0, gravity=0.15
    ),
)

PROJECTILE_VAL_CONFIG = DatasetConfig(
    motion_type=MotionType.PROJECTILE,
    frame=FrameConfig(),
    split="val",
    seed=43,
    motion=ProjectileConfig(
        initial_x=0.1, initial_y=0.8, vx=2.0, vy=-4.0, gravity=0.15
    ),
)

PROJECTILE_TEST_CONFIG = DatasetConfig(
    motion_type=MotionType.PROJECTILE,
    frame=FrameConfig(),
    split="test",
    seed=44,
    motion=ProjectileConfig(
        initial_x=0.1, initial_y=0.8, vx=2.0, vy=-4.0, gravity=0.15
    ),
)

CIRCULAR_OOD_CONFIG = DatasetConfig(
    motion_type=MotionType.CIRCULAR,
    frame=FrameConfig(),
    split="ood",
    seed=999,
    motion=CircularConfig(
        radius=0.4,
        angular_velocity=0.3,
        center_x=0.3,
        center_y=0.7,
    ),
)

PROJECTILE_OOD_CONFIG = DatasetConfig(
    motion_type=MotionType.PROJECTILE,
    frame=FrameConfig(),
    split="ood",
    seed=999,
    motion=ProjectileConfig(
        initial_x=0.05,
        initial_y=0.9,
        vx=3.5,
        vy=-6.0,
        gravity=0.05,
    ),
)

BOUNCE_TRAIN_CONFIG = DatasetConfig(
    motion_type=MotionType.BOUNCE,
    frame=FrameConfig(),
    split="train",
    seed=42,
    motion=BounceConfig(
        initial_x=0.2, initial_y=0.2, vx=1.5, vy=2.0, gravity=0.15, restitution=0.8
    ),
)

BOUNCE_VAL_CONFIG = DatasetConfig(
    motion_type=MotionType.BOUNCE,
    frame=FrameConfig(),
    split="val",
    seed=43,
    motion=BounceConfig(
        initial_x=0.2, initial_y=0.2, vx=1.5, vy=2.0, gravity=0.15, restitution=0.8
    ),
)

BOUNCE_TEST_CONFIG = DatasetConfig(
    motion_type=MotionType.BOUNCE,
    frame=FrameConfig(),
    split="test",
    seed=44,
    motion=BounceConfig(
        initial_x=0.2, initial_y=0.2, vx=1.5, vy=2.0, gravity=0.15, restitution=0.8
    ),
)

BOUNCE_OOD_CONFIG = DatasetConfig(
    motion_type=MotionType.BOUNCE,
    frame=FrameConfig(),
    split="ood",
    seed=999,
    motion=BounceConfig(
        initial_x=0.1,
        initial_y=0.1,
        vx=2.5,
        vy=3.0,
        gravity=0.05,
        restitution=0.95,
    ),
)

PENDULUM_TRAIN_CONFIG = DatasetConfig(
    motion_type=MotionType.PENDULUM,
    frame=FrameConfig(),
    split="train",
    seed=42,
    motion=PendulumConfig(
        length=0.35, gravity=0.2, initial_angle=0.5, initial_angular_vel=0.0
    ),
)

PENDULUM_VAL_CONFIG = DatasetConfig(
    motion_type=MotionType.PENDULUM,
    frame=FrameConfig(),
    split="val",
    seed=43,
    motion=PendulumConfig(
        length=0.35, gravity=0.2, initial_angle=0.5, initial_angular_vel=0.0
    ),
)

PENDULUM_TEST_CONFIG = DatasetConfig(
    motion_type=MotionType.PENDULUM,
    frame=FrameConfig(),
    split="test",
    seed=44,
    motion=PendulumConfig(
        length=0.35, gravity=0.2, initial_angle=0.5, initial_angular_vel=0.0
    ),
)

PENDULUM_OOD_CONFIG = DatasetConfig(
    motion_type=MotionType.PENDULUM,
    frame=FrameConfig(),
    split="ood",
    seed=999,
    motion=PendulumConfig(
        length=0.5, gravity=0.1, initial_angle=0.8, initial_angular_vel=0.3
    ),
)

TWO_BODY_TRAIN_CONFIG = DatasetConfig(
    motion_type=MotionType.TWO_BODY,
    frame=FrameConfig(),
    split="train",
    seed=42,
    motion=TwoBodyConfig(
        orbit_radius_1=0.12,
        orbit_radius_2=0.25,
        angular_velocity=0.15,
        mass_ratio=2.0,
    ),
)

TWO_BODY_VAL_CONFIG = DatasetConfig(
    motion_type=MotionType.TWO_BODY,
    frame=FrameConfig(),
    split="val",
    seed=43,
    motion=TwoBodyConfig(
        orbit_radius_1=0.12,
        orbit_radius_2=0.25,
        angular_velocity=0.15,
        mass_ratio=2.0,
    ),
)

TWO_BODY_TEST_CONFIG = DatasetConfig(
    motion_type=MotionType.TWO_BODY,
    frame=FrameConfig(),
    split="test",
    seed=44,
    motion=TwoBodyConfig(
        orbit_radius_1=0.12,
        orbit_radius_2=0.25,
        angular_velocity=0.15,
        mass_ratio=2.0,
    ),
)

TWO_BODY_OOD_CONFIG = DatasetConfig(
    motion_type=MotionType.TWO_BODY,
    frame=FrameConfig(),
    split="ood",
    seed=999,
    motion=TwoBodyConfig(
        orbit_radius_1=0.2,
        orbit_radius_2=0.4,
        angular_velocity=0.25,
        mass_ratio=5.0,
    ),
)


def _generate_circular_sequence(
    config: DatasetConfig,
) -> tuple[Tensor, dict[str, Any]]:
    fc = config.frame
    cc = cast(CircularConfig, config.motion)

    num_frames = fc.num_frames
    height = fc.height
    width = fc.width

    center_x = cc.center_x * width
    center_y = cc.center_y * height

    positions_x = np.zeros(num_frames)
    positions_y = np.zeros(num_frames)
    velocities_x = np.zeros(num_frames)
    velocities_y = np.zeros(num_frames)
    accelerations_x = np.zeros(num_frames)
    accelerations_y = np.zeros(num_frames)

    for t in range(num_frames):
        angle = cc.angular_velocity * t
        positions_x[t] = center_x + cc.radius * width * np.cos(angle)
        positions_y[t] = center_y + cc.radius * height * np.sin(angle)

    for t in range(num_frames):
        if t == 0:
            velocities_x[t] = positions_x[1] - positions_x[0]
            velocities_y[t] = positions_y[1] - positions_y[0]
        elif t == num_frames - 1:
            velocities_x[t] = positions_x[t] - positions_x[t - 1]
            velocities_y[t] = positions_y[t] - positions_y[t - 1]
        else:
            velocities_x[t] = (positions_x[t + 1] - positions_x[t - 1]) / 2
            velocities_y[t] = (positions_y[t + 1] - positions_y[t - 1]) / 2

    for t in range(num_frames):
        if t == 0:
            accelerations_x[t] = velocities_x[1] - velocities_x[0]
            accelerations_y[t] = velocities_y[1] - velocities_y[0]
        elif t == num_frames - 1:
            accelerations_x[t] = velocities_x[t] - velocities_x[t - 1]
            accelerations_y[t] = velocities_y[t] - velocities_y[t - 1]
        else:
            accelerations_x[t] = (velocities_x[t + 1] - velocities_x[t - 1]) / 2
            accelerations_y[t] = (velocities_y[t + 1] - velocities_y[t - 1]) / 2

    frames = np.zeros((num_frames, height, width), dtype=np.uint8)

    for t in range(num_frames):
        frame = frames[t]
        frame[:] = fc.background_value

        ox = int(positions_x[t])
        oy = int(positions_y[t])
        r = cc.object_radius

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    px, py = ox + dx, oy + dy
                    if 0 <= px < width and 0 <= py < height:
                        frame[py, px] = cc.object_value

    frames_tensor = torch.from_numpy(frames).float() / 255.0

    metadata = {
        "motion_type": MotionType.CIRCULAR.value,
        "split": config.split,
        "seed": config.seed,
        "positions_x": positions_x.tolist(),
        "positions_y": positions_y.tolist(),
        "velocities_x": velocities_x.tolist(),
        "velocities_y": velocities_y.tolist(),
        "accelerations_x": accelerations_x.tolist(),
        "accelerations_y": accelerations_y.tolist(),
        "params": {
            "radius": cc.radius,
            "angular_velocity": cc.angular_velocity,
            "center_x": cc.center_x,
            "center_y": cc.center_y,
            "object_radius": cc.object_radius,
            "object_value": cc.object_value,
        },
        "frame_config": {
            "width": fc.width,
            "height": fc.height,
            "num_frames": fc.num_frames,
            "background_value": fc.background_value,
        },
    }

    return frames_tensor, metadata


def _generate_projectile_sequence(
    config: DatasetConfig,
) -> tuple[Tensor, dict[str, Any]]:
    fc = config.frame
    pc = cast(ProjectileConfig, config.motion)

    num_frames = fc.num_frames
    height = fc.height
    width = fc.width

    positions_x = np.zeros(num_frames)
    positions_y = np.zeros(num_frames)
    velocities_x = np.zeros(num_frames)
    velocities_y = np.zeros(num_frames)
    accelerations_x = np.zeros(num_frames)
    accelerations_y = np.zeros(num_frames)

    init_x = pc.initial_x * width
    init_y = pc.initial_y * height

    for t in range(num_frames):
        positions_x[t] = init_x + pc.vx * t
        positions_y[t] = init_y + pc.vy * t + 0.5 * pc.gravity * t * t

    for t in range(num_frames):
        velocities_x[t] = pc.vx
        velocities_y[t] = pc.vy + pc.gravity * t

    accelerations_x[:] = 0
    accelerations_y[:] = pc.gravity

    frames = np.zeros((num_frames, height, width), dtype=np.uint8)

    for t in range(num_frames):
        frame = frames[t]
        frame[:] = fc.background_value

        ox = int(positions_x[t])
        oy = int(positions_y[t])
        r = pc.object_radius

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    px, py = ox + dx, oy + dy
                    if 0 <= px < width and 0 <= py < height:
                        frame[py, px] = pc.object_value

    frames_tensor = torch.from_numpy(frames).float() / 255.0

    metadata = {
        "motion_type": MotionType.PROJECTILE.value,
        "split": config.split,
        "seed": config.seed,
        "positions_x": positions_x.tolist(),
        "positions_y": positions_y.tolist(),
        "velocities_x": velocities_x.tolist(),
        "velocities_y": velocities_y.tolist(),
        "accelerations_x": accelerations_x.tolist(),
        "accelerations_y": accelerations_y.tolist(),
        "params": {
            "initial_x": pc.initial_x,
            "initial_y": pc.initial_y,
            "vx": pc.vx,
            "vy": pc.vy,
            "gravity": pc.gravity,
            "object_radius": pc.object_radius,
            "object_value": pc.object_value,
        },
        "frame_config": {
            "width": fc.width,
            "height": fc.height,
            "num_frames": fc.num_frames,
            "background_value": fc.background_value,
        },
    }

    return frames_tensor, metadata


def _generate_bounce_sequence(
    config: DatasetConfig,
) -> tuple[Tensor, dict[str, Any]]:
    fc = config.frame
    bc = cast(BounceConfig, config.motion)

    num_frames = fc.num_frames
    height = fc.height
    width = fc.width

    positions_x = np.zeros(num_frames)
    positions_y = np.zeros(num_frames)
    velocities_x = np.zeros(num_frames)
    velocities_y = np.zeros(num_frames)
    accelerations_x = np.zeros(num_frames)
    accelerations_y = np.zeros(num_frames)

    px = bc.initial_x * width
    py = bc.initial_y * height
    vx = bc.vx
    vy = bc.vy

    for t in range(num_frames):
        positions_x[t] = px
        positions_y[t] = py

        vx, px = _bounce_step(vx, px, bc.restitution, width, bc.object_radius)
        vy, py = _bounce_step(vy, py, bc.restitution, height, bc.object_radius)
        vy -= bc.gravity

    for t in range(num_frames):
        if t == 0:
            velocities_x[t] = positions_x[1] - positions_x[0]
            velocities_y[t] = positions_y[1] - positions_y[0]
        elif t == num_frames - 1:
            velocities_x[t] = positions_x[t] - positions_x[t - 1]
            velocities_y[t] = positions_y[t] - positions_y[t - 1]
        else:
            velocities_x[t] = (positions_x[t + 1] - positions_x[t - 1]) / 2
            velocities_y[t] = (positions_y[t + 1] - positions_y[t - 1]) / 2

    for t in range(num_frames):
        if t == 0:
            accelerations_x[t] = velocities_x[1] - velocities_x[0]
            accelerations_y[t] = velocities_y[1] - velocities_y[0]
        elif t == num_frames - 1:
            accelerations_x[t] = velocities_x[t] - velocities_x[t - 1]
            accelerations_y[t] = velocities_y[t] - velocities_y[t - 1]
        else:
            accelerations_x[t] = (velocities_x[t + 1] - velocities_x[t - 1]) / 2
            accelerations_y[t] = (velocities_y[t + 1] - velocities_y[t - 1]) / 2

    frames = np.zeros((num_frames, height, width), dtype=np.uint8)

    for t in range(num_frames):
        frame = frames[t]
        frame[:] = fc.background_value

        ox = int(positions_x[t])
        oy = int(positions_y[t])
        r = bc.object_radius

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    px_draw, py_draw = ox + dx, oy + dy
                    if 0 <= px_draw < width and 0 <= py_draw < height:
                        frame[py_draw, px_draw] = bc.object_value

    frames_tensor = torch.from_numpy(frames).float() / 255.0

    metadata = {
        "motion_type": MotionType.BOUNCE.value,
        "split": config.split,
        "seed": config.seed,
        "positions_x": positions_x.tolist(),
        "positions_y": positions_y.tolist(),
        "velocities_x": velocities_x.tolist(),
        "velocities_y": velocities_y.tolist(),
        "accelerations_x": accelerations_x.tolist(),
        "accelerations_y": accelerations_y.tolist(),
        "params": {
            "initial_x": bc.initial_x,
            "initial_y": bc.initial_y,
            "vx": bc.vx,
            "vy": bc.vy,
            "gravity": bc.gravity,
            "restitution": bc.restitution,
            "object_radius": bc.object_radius,
            "object_value": bc.object_value,
        },
        "frame_config": {
            "width": fc.width,
            "height": fc.height,
            "num_frames": fc.num_frames,
            "background_value": fc.background_value,
        },
    }

    return frames_tensor, metadata


def _bounce_step(
    v: float, p: float, restitution: float, bound: int, r: int
) -> tuple[float, float]:
    margin = r
    new_p = p + v
    new_v = v
    if new_p < margin:
        new_p = margin
        new_v = -v * restitution
    elif new_p >= bound - margin:
        new_p = bound - margin
        new_v = -v * restitution
    return new_v, new_p


def _generate_pendulum_sequence(
    config: DatasetConfig,
) -> tuple[Tensor, dict[str, Any]]:
    fc = config.frame
    pc = cast(PendulumConfig, config.motion)

    num_frames = fc.num_frames
    height = fc.height
    width = fc.width

    pivot_x = pc.pivot_x * width
    pivot_y = pc.pivot_y * height
    length = pc.length * height

    positions_x = np.zeros(num_frames)
    positions_y = np.zeros(num_frames)
    angles = np.zeros(num_frames)
    angular_velocities = np.zeros(num_frames)
    angular_accels = np.zeros(num_frames)

    angle = pc.initial_angle
    omega = pc.initial_angular_vel

    for t in range(num_frames):
        angles[t] = angle
        angular_velocities[t] = omega

        alpha = -(pc.gravity / length) * np.sin(angle)
        angular_accels[t] = alpha

        angle += omega
        omega += alpha

    for t in range(num_frames):
        positions_x[t] = pivot_x + length * np.sin(angles[t])
        positions_y[t] = pivot_y + length * np.cos(angles[t])

    velocities_x = np.zeros(num_frames)
    velocities_y = np.zeros(num_frames)
    accelerations_x = np.zeros(num_frames)
    accelerations_y = np.zeros(num_frames)

    for t in range(num_frames):
        if t == 0:
            velocities_x[t] = positions_x[1] - positions_x[0]
            velocities_y[t] = positions_y[1] - positions_y[0]
        elif t == num_frames - 1:
            velocities_x[t] = positions_x[t] - positions_x[t - 1]
            velocities_y[t] = positions_y[t] - positions_y[t - 1]
        else:
            velocities_x[t] = (positions_x[t + 1] - positions_x[t - 1]) / 2
            velocities_y[t] = (positions_y[t + 1] - positions_y[t - 1]) / 2

    for t in range(num_frames):
        if t == 0:
            accelerations_x[t] = velocities_x[1] - velocities_x[0]
            accelerations_y[t] = velocities_y[1] - velocities_y[0]
        elif t == num_frames - 1:
            accelerations_x[t] = velocities_x[t] - velocities_x[t - 1]
            accelerations_y[t] = velocities_y[t] - velocities_y[t - 1]
        else:
            accelerations_x[t] = (velocities_x[t + 1] - velocities_x[t - 1]) / 2
            accelerations_y[t] = (velocities_y[t + 1] - velocities_y[t - 1]) / 2

    frames = np.zeros((num_frames, height, width), dtype=np.uint8)

    for t in range(num_frames):
        frame = frames[t]
        frame[:] = fc.background_value

        ox = int(positions_x[t])
        oy = int(positions_y[t])
        r = pc.object_radius

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    px_draw, py_draw = ox + dx, oy + dy
                    if 0 <= px_draw < width and 0 <= py_draw < height:
                        frame[py_draw, px_draw] = pc.object_value

    frames_tensor = torch.from_numpy(frames).float() / 255.0

    metadata = {
        "motion_type": MotionType.PENDULUM.value,
        "split": config.split,
        "seed": config.seed,
        "positions_x": positions_x.tolist(),
        "positions_y": positions_y.tolist(),
        "velocities_x": velocities_x.tolist(),
        "velocities_y": velocities_y.tolist(),
        "accelerations_x": accelerations_x.tolist(),
        "accelerations_y": accelerations_y.tolist(),
        "angles": angles.tolist(),
        "angular_velocities": angular_velocities.tolist(),
        "angular_accelerations": angular_accels.tolist(),
        "params": {
            "length": pc.length,
            "gravity": pc.gravity,
            "initial_angle": pc.initial_angle,
            "initial_angular_vel": pc.initial_angular_vel,
            "pivot_x": pc.pivot_x,
            "pivot_y": pc.pivot_y,
            "object_radius": pc.object_radius,
            "object_value": pc.object_value,
        },
        "frame_config": {
            "width": fc.width,
            "height": fc.height,
            "num_frames": fc.num_frames,
            "background_value": fc.background_value,
        },
    }

    return frames_tensor, metadata


def _generate_two_body_sequence(
    config: DatasetConfig,
) -> tuple[Tensor, dict[str, Any]]:
    fc = config.frame
    tbc = cast(TwoBodyConfig, config.motion)

    num_frames = fc.num_frames
    height = fc.height
    width = fc.width

    center_x = tbc.center_x * width
    center_y = tbc.center_y * height

    orbit_r1 = tbc.orbit_radius_1 * min(width, height)
    orbit_r2 = tbc.orbit_radius_2 * min(width, height)

    positions_x_1 = np.zeros(num_frames)
    positions_y_1 = np.zeros(num_frames)
    positions_x_2 = np.zeros(num_frames)
    positions_y_2 = np.zeros(num_frames)
    velocities_x_1 = np.zeros(num_frames)
    velocities_y_1 = np.zeros(num_frames)
    velocities_x_2 = np.zeros(num_frames)
    velocities_y_2 = np.zeros(num_frames)
    accelerations_x_1 = np.zeros(num_frames)
    accelerations_y_1 = np.zeros(num_frames)
    accelerations_x_2 = np.zeros(num_frames)
    accelerations_y_2 = np.zeros(num_frames)

    for t in range(num_frames):
        angle = tbc.angular_velocity * t
        positions_x_1[t] = center_x + orbit_r1 * np.cos(angle)
        positions_y_1[t] = center_y + orbit_r1 * np.sin(angle)
        positions_x_2[t] = center_x + orbit_r2 * np.cos(angle + np.pi)
        positions_y_2[t] = center_y + orbit_r2 * np.sin(angle + np.pi)

    for t in range(num_frames):
        if t == 0:
            velocities_x_1[t] = positions_x_1[1] - positions_x_1[0]
            velocities_y_1[t] = positions_y_1[1] - positions_y_1[0]
            velocities_x_2[t] = positions_x_2[1] - positions_x_2[0]
            velocities_y_2[t] = positions_y_2[1] - positions_y_2[0]
        elif t == num_frames - 1:
            velocities_x_1[t] = positions_x_1[t] - positions_x_1[t - 1]
            velocities_y_1[t] = positions_y_1[t] - positions_y_1[t - 1]
            velocities_x_2[t] = positions_x_2[t] - positions_x_2[t - 1]
            velocities_y_2[t] = positions_y_2[t] - positions_y_2[t - 1]
        else:
            velocities_x_1[t] = (positions_x_1[t + 1] - positions_x_1[t - 1]) / 2
            velocities_y_1[t] = (positions_y_1[t + 1] - positions_y_1[t - 1]) / 2
            velocities_x_2[t] = (positions_x_2[t + 1] - positions_x_2[t - 1]) / 2
            velocities_y_2[t] = (positions_y_2[t + 1] - positions_y_2[t - 1]) / 2

    for t in range(num_frames):
        if t == 0:
            accelerations_x_1[t] = velocities_x_1[1] - velocities_x_1[0]
            accelerations_y_1[t] = velocities_y_1[1] - velocities_y_1[0]
            accelerations_x_2[t] = velocities_x_2[1] - velocities_x_2[0]
            accelerations_y_2[t] = velocities_y_2[1] - velocities_y_2[0]
        elif t == num_frames - 1:
            accelerations_x_1[t] = velocities_x_1[t] - velocities_x_1[t - 1]
            accelerations_y_1[t] = velocities_y_1[t] - velocities_y_1[t - 1]
            accelerations_x_2[t] = velocities_x_2[t] - velocities_x_2[t - 1]
            accelerations_y_2[t] = velocities_y_2[t] - velocities_y_2[t - 1]
        else:
            accelerations_x_1[t] = (velocities_x_1[t + 1] - velocities_x_1[t - 1]) / 2
            accelerations_y_1[t] = (velocities_y_1[t + 1] - velocities_y_1[t - 1]) / 2
            accelerations_x_2[t] = (velocities_x_2[t + 1] - velocities_x_2[t - 1]) / 2
            accelerations_y_2[t] = (velocities_y_2[t + 1] - velocities_y_2[t - 1]) / 2

    frames = np.zeros((num_frames, height, width), dtype=np.uint8)

    for t in range(num_frames):
        frame = frames[t]
        frame[:] = fc.background_value

        for ox, oy, r, val in [
            (int(positions_x_1[t]), int(positions_y_1[t]), tbc.object_radius_1, tbc.object_value_1),
            (int(positions_x_2[t]), int(positions_y_2[t]), tbc.object_radius_2, tbc.object_value_2),
        ]:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        px_draw, py_draw = ox + dx, oy + dy
                        if 0 <= px_draw < width and 0 <= py_draw < height:
                            frame[py_draw, px_draw] = val

    frames_tensor = torch.from_numpy(frames).float() / 255.0

    metadata = {
        "motion_type": MotionType.TWO_BODY.value,
        "split": config.split,
        "seed": config.seed,
        "positions_x_1": positions_x_1.tolist(),
        "positions_y_1": positions_y_1.tolist(),
        "positions_x_2": positions_x_2.tolist(),
        "positions_y_2": positions_y_2.tolist(),
        "velocities_x_1": velocities_x_1.tolist(),
        "velocities_y_1": velocities_y_1.tolist(),
        "velocities_x_2": velocities_x_2.tolist(),
        "velocities_y_2": velocities_y_2.tolist(),
        "accelerations_x_1": accelerations_x_1.tolist(),
        "accelerations_y_1": accelerations_y_1.tolist(),
        "accelerations_x_2": accelerations_x_2.tolist(),
        "accelerations_y_2": accelerations_y_2.tolist(),
        "params": {
            "orbit_radius_1": tbc.orbit_radius_1,
            "orbit_radius_2": tbc.orbit_radius_2,
            "angular_velocity": tbc.angular_velocity,
            "center_x": tbc.center_x,
            "center_y": tbc.center_y,
            "mass_ratio": tbc.mass_ratio,
            "object_radius_1": tbc.object_radius_1,
            "object_radius_2": tbc.object_radius_2,
            "object_value_1": tbc.object_value_1,
            "object_value_2": tbc.object_value_2,
        },
        "frame_config": {
            "width": fc.width,
            "height": fc.height,
            "num_frames": fc.num_frames,
            "background_value": fc.background_value,
        },
    }

    return frames_tensor, metadata


def generate_sequence(config: DatasetConfig) -> tuple[Tensor, dict[str, Any]]:
    if config.motion_type == MotionType.CIRCULAR:
        return _generate_circular_sequence(config)
    elif config.motion_type == MotionType.PROJECTILE:
        return _generate_projectile_sequence(config)
    elif config.motion_type == MotionType.BOUNCE:
        return _generate_bounce_sequence(config)
    elif config.motion_type == MotionType.PENDULUM:
        return _generate_pendulum_sequence(config)
    elif config.motion_type == MotionType.TWO_BODY:
        return _generate_two_body_sequence(config)
    else:
        raise ValueError(f"Unknown motion type: {config.motion_type}")


def generate_dataset(
    config: DatasetConfig,
    num_samples: int = 100,
    seed_offset: int = 0,
) -> list[tuple[Tensor, dict[str, Any]]]:
    samples = []

    for i in range(num_samples):
        sample_config = DatasetConfig(
            motion_type=config.motion_type,
            frame=config.frame,
            split=config.split,
            seed=config.seed + seed_offset + i,
            motion=config.motion,
            context_frames=config.context_frames,
        )
        frames, metadata = generate_sequence(sample_config)
        samples.append((frames, metadata))

    return samples


def generate_split_dataset(
    motion_type: MotionType,
    num_train: int = 80,
    num_val: int = 10,
    num_test: int = 10,
    num_ood: int = 10,
    ood_config: DatasetConfig | None = None,
) -> dict[str, list[tuple[Tensor, dict[str, Any]]]]:
    if motion_type == MotionType.CIRCULAR:
        train_cfg = CIRCULAR_TRAIN_CONFIG
        val_cfg = CIRCULAR_VAL_CONFIG
        test_cfg = CIRCULAR_TEST_CONFIG
        ood_cfg = ood_config or CIRCULAR_OOD_CONFIG
    elif motion_type == MotionType.PROJECTILE:
        train_cfg = PROJECTILE_TRAIN_CONFIG
        val_cfg = PROJECTILE_VAL_CONFIG
        test_cfg = PROJECTILE_TEST_CONFIG
        ood_cfg = ood_config or PROJECTILE_OOD_CONFIG
    elif motion_type == MotionType.BOUNCE:
        train_cfg = BOUNCE_TRAIN_CONFIG
        val_cfg = BOUNCE_VAL_CONFIG
        test_cfg = BOUNCE_TEST_CONFIG
        ood_cfg = ood_config or BOUNCE_OOD_CONFIG
    elif motion_type == MotionType.PENDULUM:
        train_cfg = PENDULUM_TRAIN_CONFIG
        val_cfg = PENDULUM_VAL_CONFIG
        test_cfg = PENDULUM_TEST_CONFIG
        ood_cfg = ood_config or PENDULUM_OOD_CONFIG
    elif motion_type == MotionType.TWO_BODY:
        train_cfg = TWO_BODY_TRAIN_CONFIG
        val_cfg = TWO_BODY_VAL_CONFIG
        test_cfg = TWO_BODY_TEST_CONFIG
        ood_cfg = ood_config or TWO_BODY_OOD_CONFIG
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")

    dataset = {
        "train": generate_dataset(train_cfg, num_train),
        "val": generate_dataset(val_cfg, num_val),
        "test": generate_dataset(test_cfg, num_test),
        "ood": generate_dataset(ood_cfg, num_ood),
    }

    return dataset


def get_context_frames(frames: Tensor, context_len: int) -> Tensor:
    if context_len is None or context_len >= frames.shape[0]:
        return frames
    return frames[:context_len]


def get_future_frames(frames: Tensor, context_len: int) -> Tensor:
    if context_len is None or context_len >= frames.shape[0]:
        return frames[0:0]
    return frames[context_len:]


def split_into_context_target(
    frames: Tensor, context_len: int
) -> tuple[Tensor, Tensor]:
    return get_context_frames(frames, context_len), get_future_frames(frames, context_len)


def save_frames_as_gif(frames: Tensor, path: str | Path, fps: int = 8) -> None:
    if not _pil_available:
        raise ImportError("PIL required for saving GIFs")

    if frames.dtype == torch.float32 or frames.dtype == torch.float64:
        frames_np = (frames.numpy() * 255).astype(np.uint8)
    else:
        frames_np = frames.numpy().astype(np.uint8)

    if frames_np.ndim == 3:
        num_frames, height, width = frames_np.shape
        images = []
        for i in range(num_frames):
            assert Image is not None
            img = Image.fromarray(frames_np[i], mode="L")
            images.append(img)
    else:
        assert Image is not None
        img = Image.fromarray(frames_np, mode="L")
        path_str = str(path).rsplit(".gif", 1)[0] + ".png"
        img.save(path_str)
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def save_frame_strip(
    frames: Tensor, path: str | Path, max_width: int = 1024
) -> None:
    if not _pil_available:
        raise ImportError("PIL required for saving frame strips")

    if frames.dtype == torch.float32 or frames.dtype == torch.float64:
        frames_np = (frames.numpy() * 255).astype(np.uint8)
    else:
        frames_np = frames.numpy().astype(np.uint8)

    num_frames, height, width = frames_np.shape

    # Scale down if needed
    if max_width < num_frames * width:
        scale = max_width / (num_frames * width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Use PIL for resizing
        scaled_frames = []
        for i in range(num_frames):
            assert Image is not None
            img = Image.fromarray(frames_np[i], mode="L")
            img = img.resize((new_width, new_height), Image.BILINEAR)
            scaled_frames.append(np.array(img))
        frames_np = np.stack(scaled_frames, axis=0)
        _, height, width = frames_np.shape

    # Concatenate horizontally
    strip = np.concatenate(frames_np, axis=1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert Image is not None
    Image.fromarray(strip, mode="L").save(path)


def save_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, list) and any(
            isinstance(x, np.ndarray) for x in value
        ):
            serializable[key] = [
                x.tolist() if isinstance(x, np.ndarray) else x for x in value
            ]
        else:
            serializable[key] = value

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def save_sample(
    frames: Tensor,
    metadata: dict[str, Any],
    base_path: str | Path,
    sample_idx: int = 0,
    save_gif: bool = True,
    save_strip: bool = True,
) -> None:
    base_path = Path(base_path)

    if save_gif:
        gif_path = base_path / f"sample_{sample_idx:04d}.gif"
        save_frames_as_gif(frames, gif_path)

    if save_strip:
        strip_path = base_path / f"sample_{sample_idx:04d}_strip.png"
        save_frame_strip(frames, strip_path)

    meta_path = base_path / f"sample_{sample_idx:04d}_metadata.json"
    save_metadata(metadata, meta_path)


def compute_dataset_stats(
    samples: list[tuple[Tensor, dict[str, Any]]]
) -> dict[str, Any]:
    num_samples = len(samples)
    if num_samples == 0:
        return {"num_samples": 0}

    first_frames, first_metadata = samples[0]
    num_frames, height, width = first_frames.shape

    motion_types = set()
    splits = set()

    for frames, metadata in samples:
        motion_types.add(metadata.get("motion_type", "unknown"))
        splits.add(metadata.get("split", "unknown"))

    return {
        "num_samples": num_samples,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "motion_types": list(motion_types),
        "splits": list(splits),
    }


def print_dataset_summary(dataset: dict[str, list[tuple[Tensor, dict[str, Any]]]]) -> None:
    print("Dataset Summary")
    print("=" * 40)

    for split, samples in dataset.items():
        stats = compute_dataset_stats(samples)
        print(f"\n{split.upper()} ({stats['num_samples']} samples)")
        print(f"  Resolution: {stats['width']}x{stats['height']}")
        print(f"  Frames: {stats['num_frames']}")
        print(f"  Motion types: {stats['motion_types']}")


def demo() -> None:
    print("Generating demo dataset...")

    dataset = generate_split_dataset(
        motion_type=MotionType.CIRCULAR,
        num_train=5,
        num_val=2,
        num_test=2,
        num_ood=2,
    )

    print_dataset_summary(dataset)

    config = CIRCULAR_TRAIN_CONFIG
    frames, metadata = generate_sequence(config)

    print(f"\nGenerated sample:")
    print(f"  Frames shape: {frames.shape}")
    print(f"  Motion type: {metadata['motion_type']}")
    print(f"  Position range X: {min(metadata['positions_x']):.1f} - {max(metadata['positions_x']):.1f}")
    print(f"  Position range Y: {min(metadata['positions_y']):.1f} - {max(metadata['positions_y']):.1f}")


if __name__ == "__main__":
    demo()
