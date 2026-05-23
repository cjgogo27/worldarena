"""Deployable multi-view chunked rollout policy for RoboTwin."""

# pyright: reportAny=false, reportAttributeAccessIssue=false, reportExplicitAny=false, reportImplicitRelativeImport=false, reportMissingImports=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

try:
    from ..backbone.adapter import BackboneAdapter
    from ..backbone.config import BackboneConfig
    from ..baselines.chunked_policy import load_chunked_policy_checkpoint
except ImportError:  # pragma: no cover
    from src.backbone.adapter import BackboneAdapter
    from src.backbone.config import BackboneConfig
    from src.baselines.chunked_policy import load_chunked_policy_checkpoint

logger = logging.getLogger(__name__)

_model: Optional[dict[str, Any]] = None
_config: dict[str, Any] = {}


def _extract_rgb(camera_payload: Any) -> np.ndarray | None:
    if isinstance(camera_payload, dict):
        rgb = camera_payload.get("rgb")
        if rgb is not None:
            return np.asarray(rgb)
    if isinstance(camera_payload, np.ndarray):
        return np.asarray(camera_payload)
    if hasattr(camera_payload, "cpu"):
        return camera_payload.cpu().numpy()
    return None


def encode_obs(observation: dict[str, Any]) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    nested = observation.get("observation")
    if isinstance(nested, dict):
        for camera_name in ("head_camera", "left_camera", "right_camera"):
            rgb = _extract_rgb(nested.get(camera_name))
            if rgb is not None:
                obs[camera_name] = rgb

    for camera_name in ("head_camera", "left_camera", "right_camera"):
        if camera_name not in obs:
            rgb = _extract_rgb(observation.get(camera_name))
            if rgb is not None:
                obs[camera_name] = rgb

    joint_action = observation.get("joint_action")
    if isinstance(joint_action, dict) and "vector" in joint_action:
        obs["state"] = np.asarray(joint_action["vector"], dtype=np.float32)
    elif "state" in observation:
        obs["state"] = np.asarray(observation["state"], dtype=np.float32)

    if "instruction" in observation:
        obs["instruction"] = observation["instruction"]
    return obs


def get_model(usr_args: Optional[dict[str, Any]] = None) -> "ChunkedPolicyWrapper":
    global _model, _config
    usr_args = usr_args or {}
    config_path = usr_args.get("config")
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "chunked_policy.yaml"

    if isinstance(config_path, (str, Path)) and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as handle:
            _config = yaml.safe_load(handle) or {}
    elif isinstance(config_path, dict):
        _config = dict(config_path)
    else:
        _config = {}

    for key, value in usr_args.items():
        if key != "config":
            _config[key] = value

    model_path = _config.get("model_path")
    if isinstance(model_path, str):
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            _config["model_path"] = str((Path(__file__).parent.parent.parent / model_path_obj).resolve())

    policy = ChunkedPolicyWrapper(
        device=str(_config.get("device", "cuda")),
        model_path=_config.get("model_path"),
        config=_config,
    )
    _model = {"policy": policy}
    return policy


def eval(task_env: Any, model: "ChunkedPolicyWrapper", observation: dict[str, Any]) -> None:
    global _model
    if _model is None:
        _model = {"policy": model}

    obs = encode_obs(observation)
    obs["instruction"] = task_env.get_instruction()
    actions = model.get_action(obs)
    for action in actions:
        task_env.take_action(action, action_type="qpos")
        observation = task_env.get_obs()


def reset_model(model: "ChunkedPolicyWrapper") -> None:
    model.reset()


class ChunkedPolicyWrapper:
    def __init__(
        self,
        *,
        backbone: BackboneAdapter | None = None,
        device: str = "cuda",
        model_path: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.device = torch.device(device)
        self.model_path = model_path
        self.config = config or {}
        self._backbone = backbone
        self._checkpoint = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        if self.model_path is None:
            default_path = Path(__file__).parent.parent.parent / "experiments" / "rollout_ckpts" / "chunked_policy.pt"
            self.model_path = str(default_path)
        if self._backbone is None:
            backbone_cfg = BackboneConfig.from_defaults()
            backbone_cfg.device = str(self.device)
            backbone_cfg.dtype = str(self.config.get("backbone_dtype", backbone_cfg.dtype))
            self._backbone = BackboneAdapter.from_config(backbone_cfg)
        self._checkpoint = load_chunked_policy_checkpoint(self.model_path, self.device)
        self._initialized = True

    @property
    def backbone(self) -> BackboneAdapter:
        self._lazy_init()
        assert self._backbone is not None
        return self._backbone

    @property
    def checkpoint(self):
        self._lazy_init()
        assert self._checkpoint is not None
        return self._checkpoint

    def _prepare_image_tensor(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = torch.from_numpy(image)
        if image_tensor.ndim == 3 and image_tensor.shape[-1] in (1, 3):
            image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor.float() / 255.0

    def _encode_multiview(self, obs: dict[str, Any]) -> torch.Tensor:
        current_images = []
        fallback = obs.get("head_camera")
        for camera_name in ("head_camera", "left_camera", "right_camera"):
            image = obs.get(camera_name, fallback)
            if image is None:
                raise RuntimeError(f"Missing required camera observation: {camera_name}")
            current_images.append(self._prepare_image_tensor(np.asarray(image)))
        image_batch = torch.stack(current_images, dim=0).to(self.device)
        features = self.backbone.encode_image(image_batch).to(dtype=torch.float32)
        return features.reshape(1, -1)

    def _prepare_state(self, state: np.ndarray | None) -> tuple[torch.Tensor, np.ndarray]:
        state_dim = int(self.checkpoint.config.state_dim)
        current_state = np.zeros(state_dim, dtype=np.float32)
        if state is not None:
            flat = np.asarray(state, dtype=np.float32).reshape(-1)
            current_state[: min(state_dim, flat.shape[0])] = flat[:state_dim]
        state_tensor = torch.from_numpy(current_state).to(self.device)
        state_tensor = (state_tensor - self.checkpoint.state_mean) / self.checkpoint.state_std
        return state_tensor.unsqueeze(0), current_state

    def _format_rollout_action(self, predicted_action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        left_arm_dim = int(self.config.get("left_arm_dim", 7))
        right_arm_dim = int(self.config.get("right_arm_dim", 7))
        total_dim = left_arm_dim + 1 + right_arm_dim + 1
        rollout_action = np.zeros(total_dim, dtype=np.float32)
        rollout_action[: min(total_dim, current_state.shape[0])] = current_state[:total_dim]
        rollout_action[: min(total_dim, predicted_action.shape[0])] = predicted_action[:total_dim]
        return rollout_action

    def get_action(self, obs: dict[str, Any]) -> list[np.ndarray]:
        image_features = self._encode_multiview(obs)
        state_tensor, current_state = self._prepare_state(obs.get("state"))
        with torch.no_grad():
            outputs = self.checkpoint.model(image_features, state_tensor)
            chunk = outputs["action_chunk"]
            chunk = chunk * self.checkpoint.action_std.view(1, 1, -1) + self.checkpoint.action_mean.view(1, 1, -1)
        action_horizon = int(self.config.get("action_horizon", 2))
        predicted_chunk = chunk[0, :action_horizon].detach().cpu().numpy()
        return [self._format_rollout_action(action, current_state) for action in predicted_chunk]

    def reset(self) -> None:
        return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chunked RoboTwin rollout policy")
    parser.add_argument("--config", default="configs/chunked_policy.yaml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    _ = get_model({"config": args.config, "model_path": args.model_path, "device": args.device})
    logger.info("chunked policy initialized")


if __name__ == "__main__":
    main()
