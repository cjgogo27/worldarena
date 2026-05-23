"""Deployable RoboTwin diffusion-policy wrapper with multi-camera 16D rollout."""

from __future__ import annotations

# pyright: reportAny=false, reportAttributeAccessIssue=false, reportDeprecated=false, reportExplicitAny=false, reportMissingImports=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false

import logging
from collections import deque
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ..backbone.adapter import BackboneAdapter
    from ..backbone.config import BackboneConfig
    from ..baselines.robottwin_diffusion_policy import load_robottwin_diffusion_policy_checkpoint
except ImportError:  # pragma: no cover
    from src.backbone.adapter import BackboneAdapter
    from src.backbone.config import BackboneConfig
    from src.baselines.robottwin_diffusion_policy import load_robottwin_diffusion_policy_checkpoint

logger = logging.getLogger(__name__)

_model: dict[str, Any] | None = None
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


def get_model(usr_args: dict[str, Any] | None = None) -> "DiffusionPolicyWrapper":
    global _model, _config
    usr_args = usr_args or {}
    config_path = usr_args.get("config")
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "robottwin_diffusion_policy.yaml"
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
    requested_device = str(_config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    policy = DiffusionPolicyWrapper(
        device=requested_device,
        model_path=_config.get("model_path"),
        config=_config,
    )
    _model = {"policy": policy}
    return policy


def eval(task_env: Any, model: "DiffusionPolicyWrapper", observation: dict[str, Any]) -> None:
    global _model
    if _model is None:
        _model = {"policy": model}
    obs = encode_obs(observation)
    obs["instruction"] = task_env.get_instruction()
    actions = model.get_action(obs)
    for action in actions:
        task_env.take_action(action, action_type="qpos")
        observation = task_env.get_obs()
        model.update_obs(encode_obs(observation))


def reset_model(model: "DiffusionPolicyWrapper") -> None:
    model.reset()


class DiffusionPolicyWrapper:
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
        self._obs_history: deque[np.ndarray] = deque(maxlen=int(self.config.get("n_obs_steps", 2)))

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        if self.model_path is None:
            default_path = Path(__file__).parent.parent.parent / "experiments" / "rollout_ckpts" / "robottwin_diffusion_policy.pt"
            self.model_path = str(default_path)
        if self._backbone is None:
            backbone_cfg = BackboneConfig.from_defaults()
            backbone_cfg.device = str(self.device)
            backbone_cfg.dtype = str(self.config.get("backbone_dtype", backbone_cfg.dtype))
            self._backbone = BackboneAdapter.from_config(backbone_cfg)
        self._checkpoint = load_robottwin_diffusion_policy_checkpoint(self.model_path, self.device)
        self._obs_history = deque(maxlen=int(self._checkpoint.config.n_obs_steps))
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
        return features.reshape(-1)

    def _extract_state(self, obs: dict[str, Any]) -> np.ndarray:
        state_dim = int(self.checkpoint.config.state_dim)
        current_state = np.zeros(state_dim, dtype=np.float32)
        state = obs.get("state")
        if state is not None:
            flat = np.asarray(state, dtype=np.float32).reshape(-1)
            current_state[: min(state_dim, flat.shape[0])] = flat[:state_dim]
        return current_state

    def _build_obs_vector(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        current_state = self._extract_state(obs)
        image_feature = self._encode_multiview(obs).detach().cpu().numpy().astype(np.float32)
        return np.concatenate([current_state, image_feature], axis=0), current_state

    def update_obs(self, obs: dict[str, Any]) -> None:
        obs_vector, _ = self._build_obs_vector(obs)
        self._obs_history.append(obs_vector)

    def _make_history_tensor(self, obs_vector: np.ndarray) -> torch.Tensor:
        history = list(self._obs_history)
        if not history:
            history = [obs_vector]
        while len(history) < int(self.checkpoint.config.n_obs_steps):
            history.insert(0, history[0])
        obs_history = np.stack(history[-int(self.checkpoint.config.n_obs_steps):], axis=0).astype(np.float32)
        return torch.from_numpy(obs_history).unsqueeze(0).to(self.device)

    def _format_rollout_action(self, predicted_action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        left_arm_dim = int(self.config.get("left_arm_dim", 7))
        right_arm_dim = int(self.config.get("right_arm_dim", 7))
        total_dim = left_arm_dim + 1 + right_arm_dim + 1
        rollout_action = np.zeros(total_dim, dtype=np.float32)
        rollout_action[: min(total_dim, current_state.shape[0])] = current_state[:total_dim]
        rollout_action[: min(total_dim, predicted_action.shape[0])] = predicted_action[:total_dim]
        return rollout_action

    def get_action(self, obs: dict[str, Any]) -> list[np.ndarray]:
        obs_vector, current_state = self._build_obs_vector(obs)
        self._obs_history.append(obs_vector)
        obs_history = self._make_history_tensor(obs_vector)
        with torch.no_grad():
            action_chunk = self.checkpoint.model.predict_action(
                obs_history,
                num_inference_steps=int(self.config.get("num_inference_steps", self.checkpoint.config.num_inference_steps)),
            )
        predicted = action_chunk[0].detach().cpu().numpy()
        action_horizon = min(int(self.config.get("action_horizon", self.checkpoint.config.n_action_steps)), predicted.shape[0])
        return [self._format_rollout_action(action, current_state) for action in predicted[:action_horizon]]

    def reset(self) -> None:
        self._obs_history.clear()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RoboTwin diffusion policy deploy wrapper")
    parser.add_argument("--config", default="configs/robottwin_diffusion_policy.yaml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    _ = get_model({"config": args.config, "model_path": args.model_path, "device": args.device})
    logger.info("diffusion policy initialized")


if __name__ == "__main__":
    main()
