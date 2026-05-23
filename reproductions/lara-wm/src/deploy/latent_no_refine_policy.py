"""Deployable RoboTwin policy entrypoint for LatentNoRefine baseline.

Integrates the LatentNoRefine baseline with the RoboTwin Your_Policy
interface for compatibility with eval_policy.py.

Key differences from LaRA-WM:
- NO iterative refinement at inference time
- Direct latent → action mapping
- World model used for training signal only

Usage:
    python src/deploy/latent_no_refine_policy.py --config configs/latent_no_refine.yaml
    
    # Or with RoboTwin eval harness:
    python script/eval_policy.py --config policy/LatentNoRefine/deploy_policy.yml \
        --task_name <task> --task_config <config> --ckpt_setting <ckpt>

Interface (RoboTwin Your_Policy compatible):
    - encode_obs(observation): Post-process observation
    - get_model(usr_args): Initialize LatentNoRefine model
    - eval(task_env, model, observation): Get actions from policy
    - reset_model(model): Reset model state between episodes
"""

# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false, reportImplicitRelativeImport=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnannotatedClassAttribute=false, reportMissingTypeArgument=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportDeprecated=false, reportUnusedCallResult=false

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

try:
    from ..backbone.adapter import BackboneAdapter
    from ..models.action_decoder import ActionDecoder, ActionDecoderConfig
    from ..baselines.latent_no_refine import LatentNoRefineBaseline, LatentNoRefineConfig
    from .rollout_ckpt_utils import default_rollout_ckpt_path, maybe_load_rollout_checkpoint, predict_denormalized_action
except ImportError:  # pragma: no cover - direct script execution fallback
    from src.backbone.adapter import BackboneAdapter
    from src.models.action_decoder import ActionDecoder, ActionDecoderConfig
    from src.baselines.latent_no_refine import LatentNoRefineBaseline, LatentNoRefineConfig
    from src.deploy.rollout_ckpt_utils import default_rollout_ckpt_path, maybe_load_rollout_checkpoint, predict_denormalized_action

logger = logging.getLogger(__name__)


# ============================================================================
# Global model state (RoboTwin pattern)
# ============================================================================

_model: Optional[Dict[str, Any]] = None
_config: Dict[str, Any] = {}


# ============================================================================
# Observation encoding (RoboTwin interface)
# ============================================================================

def encode_obs(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process observation for LatentNoRefine pipeline.
    
    Args:
        observation: Raw observation dict with keys like
            'image', 'state', 'proprio', 'instruction', etc.
            
    Returns:
        Encoded observation dict
    """
    obs = {}
    
    nested_observation = observation.get('observation')
    if isinstance(nested_observation, dict):
        head_camera = nested_observation.get('head_camera', {})
        head_rgb = head_camera.get('rgb')
        if head_rgb is not None:
            obs['image'] = np.asarray(head_rgb)

    if 'image' in observation and 'image' not in obs:
        img = observation['image']
        if isinstance(img, np.ndarray):
            obs['image'] = img
        elif hasattr(img, 'cpu'):
            obs['image'] = img.cpu().numpy()

    joint_action = observation.get('joint_action')
    if isinstance(joint_action, dict) and 'vector' in joint_action:
        obs['state'] = np.asarray(joint_action['vector'], dtype=np.float32)

    if 'state' in observation and 'state' not in obs:
        obs['state'] = observation['state']
    if 'proprio' in observation:
        obs['proprio'] = observation['proprio']
        
    # Pass instruction
    if 'instruction' in observation:
        obs['instruction'] = observation['instruction']
        
    # Pass any additional fields
    for key, val in observation.items():
        if key not in obs:
            obs[key] = val
            
    return obs


# ============================================================================
# Model initialization (RoboTwin interface)
# ============================================================================

def get_model(usr_args: Optional[Dict[str, Any]] = None) -> "LatentNoRefinePolicy":
    """Initialize LatentNoRefine model for inference.
    
    Args:
        usr_args: User arguments dict with keys:
            - config: Path to config YAML or dict
            - model_path: Optional checkpoint path
            - device: Device to use ('cuda' or 'cpu')
            - freeze_backbone: Whether to freeze backbone
            - latent_dim, hidden_dim, etc.: Model hyperparameters
            
    Returns:
        LatentNoRefinePolicy instance ready for inference
    """
    global _model, _config
    
    if usr_args is None:
        usr_args = {}
    
    # Load config
    config_path = usr_args.get('config')
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'configs' / 'latent_no_refine.yaml'
    
    if isinstance(config_path, (str, Path)):
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                _config = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Config not found: {config_path}, using defaults")
            _config = {}
    elif isinstance(config_path, dict):
        _config = config_path
    else:
        _config = {}
    
    # Merge user args into config
    for key, val in usr_args.items():
        if key not in ('config',):
            _config[key] = val
            
    # Override with explicit args
    device = usr_args.get('device', _config.get('device', 'cuda'))
    model_path = usr_args.get('model_path', _config.get('model_path'))
    freeze_backbone = usr_args.get('freeze_backbone', False)
    
    # Initialize model components
    policy = LatentNoRefinePolicy(
        device=device,
        model_path=model_path,
        freeze_backbone=freeze_backbone,
        config=_config,
    )
    
    _model = {'policy': policy}
    return policy


# ============================================================================
# Policy evaluation (RoboTwin interface)
# ============================================================================

def eval(task_env: Any, model: "LatentNoRefinePolicy", observation: dict[str, Any]) -> None:
    """Execute policy for one step.
    
    This function is called by the RoboTwin eval harness. It processes
    the observation, queries the model, and takes actions in the env.
    
    Args:
        task_env: RoboTwin task environment with get_obs(), take_action(), 
                  get_instruction() methods
        model: LatentNoRefinePolicy instance
        observation: Current observation dict
    """
    global _model
    
    if _model is None:
        _model = {'policy': model}
    
    # Encode observation
    obs = encode_obs(observation)
    instruction = task_env.get_instruction()
    obs['instruction'] = instruction
    
    # First frame: update observation cache if empty
    if len(model.obs_cache) == 0:
        model.update_obs(obs)
    
    # Get actions from policy
    actions = model.get_action(obs)
    
    # Execute each action step
    for action in actions:
        task_env.take_action(action, action_type='qpos')
        observation = task_env.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


# ============================================================================
# Model reset (RoboTwin interface)
# ============================================================================

def reset_model(model: "LatentNoRefinePolicy") -> None:
    """Reset model state at the beginning of each episode.
    
    Args:
        model: LatentNoRefinePolicy instance
    """
    if hasattr(model, 'reset'):
        model.reset()
    if hasattr(model, 'obs_cache'):
        model.obs_cache.clear()


# ============================================================================
# LatentNoRefine Policy wrapper
# ============================================================================

class LatentNoRefinePolicy:
    """LatentNoRefine policy compatible with RoboTwin interface.
    
    Wraps LatentNoRefine's backbone + action decoder pipeline
    to provide actions for robotic manipulation tasks.
    Unlike LaRA-WM, this does NOT use iterative refinement.
    
    Key characteristics:
        - Direct latent → action mapping (no refinement loop)
        - World model used during training only (for latent dynamics)
        - Simpler inference than LaRA-WM
    
    Attributes:
        obs_cache: Sliding window of recent observations
        device: Device for inference
    """
    
    def __init__(
        self,
        backbone: Optional[BackboneAdapter] = None,
        action_decoder: Optional[ActionDecoder] = None,
        device: str = "cuda",
        model_path: Optional[str] = None,
        freeze_backbone: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.device = torch.device(device)
        self.config = config or {}
        self.freeze_backbone = freeze_backbone
        self.model_path = model_path
        self.obs_cache: list[dict] = []
        self._initialized = False
        
        # Components (initialized lazily)
        self._backbone = backbone
        self._action_decoder = action_decoder
        self._rollout_ckpt = None
        
        # Baseline for latent→action prediction
        self._baseline: Optional[LatentNoRefineBaseline] = None
        
    def _lazy_init(self) -> None:
        """Lazy initialization of model components."""
        if self._initialized:
            return
            
        # Initialize backbone
        if self._backbone is None:
            backbone_cfg = self.config.get('backbone', {})
            device_str = backbone_cfg.get('device', 'cuda')
            
            if self.model_path:
                from ..backbone.config import BackboneConfig
                cfg = BackboneConfig(primary_path=self.model_path)
                self._backbone = BackboneAdapter(
                    model_path=cfg.get_working_path() or self.model_path,
                    model_name="policy_backbone",
                    device=device_str,
                    dtype=backbone_cfg.get('dtype', 'float16'),
                )
            else:
                self._backbone = BackboneAdapter.from_config()

        assert self._backbone is not None
        vision_dim = self._backbone.vision_dim

        rollout_ckpt_path = None
        if self.model_path:
            candidate = Path(self.model_path)
            if candidate.suffix in {".pt", ".pth", ".ckpt"}:
                rollout_ckpt_path = candidate
        if rollout_ckpt_path is None:
            rollout_ckpt_path = default_rollout_ckpt_path(Path(__file__).parent.parent.parent, "latent_no_refine")
        self._rollout_ckpt = maybe_load_rollout_checkpoint(rollout_ckpt_path, self.device)
        if self._rollout_ckpt is not None:
            self._initialized = True
            return
        
        # Initialize action decoder
        if self._action_decoder is None:
            ad_config = self.config.get('action_decoder', {})
            latent_dim = ad_config.get('latent_dim', self.config.get('latent_dim', vision_dim))
            if latent_dim != vision_dim:
                logger.warning(
                    "action_decoder.latent_dim=%s does not match backbone vision_dim=%s; using vision_dim for RoboTwin rollout compatibility.",
                    latent_dim,
                    vision_dim,
                )
                latent_dim = vision_dim
            action_dim = ad_config.get('action_dim', 7)
            hidden_dim = ad_config.get('hidden_dim', 512)
            dropout = ad_config.get('dropout', 0.1)
            
            self._action_decoder = ActionDecoder(
                config=ActionDecoderConfig(
                    latent_dim=latent_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    num_layers=2,
                    dropout=dropout,
                )
            ).to(self.device)
        
        # Create baseline wrapper for latent → action prediction
        if self._baseline is None:
            baseline_config = LatentNoRefineConfig(
                latent_dim=vision_dim,
                action_dim=7,
                hidden_dim=self.config.get('hidden_dim', 512),
                num_layers=self.config.get('num_layers', 3),
                dropout=self.config.get('dropout', 0.1),
                device=str(self.device),
            )
            self._baseline = LatentNoRefineBaseline(
                config=baseline_config,
                backbone_adapter=self._backbone,
            )
            # Copy decoder weights to baseline if available
            if self._action_decoder is not None:
                self._baseline.action_decoder = self._action_decoder
            self._baseline.to(self.device)
            self._baseline.eval()
            
        self._initialized = True
        
    @property
    def backbone(self) -> BackboneAdapter:
        self._lazy_init()
        assert self._backbone is not None
        return self._backbone
    
    def update_obs(self, obs: Dict[str, Any]) -> None:
        """Add observation to cache.
        
        Args:
            obs: Encoded observation dict
        """
        self.obs_cache.append(obs)
        
        # Limit cache size
        max_cache = self.config.get('max_obs_cache', 32)
        if len(self.obs_cache) > max_cache:
            self.obs_cache.pop(0)

    def _prepare_image_tensor(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image)
        else:
            image_tensor = image

        if image_tensor.ndim == 3 and image_tensor.shape[-1] in (1, 3):
            image_tensor = image_tensor.permute(2, 0, 1)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device, dtype=torch.float32) / 255.0

    def _prepare_rollout_action(self, predicted_joint_action: np.ndarray, current_state: np.ndarray | None) -> np.ndarray:
        left_arm_dim = int(self.config.get('left_arm_dim', 7))
        right_arm_dim = int(self.config.get('right_arm_dim', 7))
        total_dim = left_arm_dim + 1 + right_arm_dim + 1

        if current_state is not None and current_state.shape[0] >= total_dim:
            rollout_action = np.asarray(current_state[:total_dim], dtype=np.float32).copy()
        else:
            rollout_action = np.zeros(total_dim, dtype=np.float32)

        predicted = np.asarray(predicted_joint_action, dtype=np.float32).reshape(-1)
        copy_dim = min(total_dim, predicted.shape[0])
        rollout_action[:copy_dim] = predicted[:copy_dim]
        return rollout_action
            
    def get_action(
        self,
        obs: Optional[Dict[str, Any]] = None,
        num_action_steps: int = 1,
    ) -> List[np.ndarray]:
        """Get action from current observation (NO refinement).
        
        Unlike LaRA-WM, this uses direct latent → action mapping
        without iterative refinement loop.
        
        Args:
            obs: Current observation (uses obs_cache if None)
            num_action_steps: Number of action steps to generate
            
        Returns:
            List of action arrays
        """
        if obs is not None:
            self.update_obs(obs)
            
        if not self.obs_cache:
            left_arm_dim = int(self.config.get('left_arm_dim', 7))
            right_arm_dim = int(self.config.get('right_arm_dim', 7))
            return [np.zeros(left_arm_dim + 1 + right_arm_dim + 1, dtype=np.float32)]
            
        self._lazy_init()
        
        if self._rollout_ckpt is not None:
            with torch.no_grad():
                current_obs = self.obs_cache[-1]
                current_state = current_obs.get('state')
                if 'image' in current_obs:
                    img_tensor = self._prepare_image_tensor(current_obs['image'])
                    feature_tensor = self.backbone.encode_image(img_tensor).to(self.device, dtype=torch.float32)
                else:
                    feature_tensor = torch.zeros(1, self._rollout_ckpt.feature_dim, device=self.device, dtype=torch.float32)
                predicted_joint = predict_denormalized_action(self._rollout_ckpt, feature_tensor).squeeze(0).detach().cpu().numpy()
                rollout_action = self._prepare_rollout_action(predicted_joint, current_state)
                return [rollout_action]

        assert self._action_decoder is not None
        decoder_param = next(self._action_decoder.parameters())
        decoder_device = decoder_param.device
        decoder_dtype = decoder_param.dtype
        
        with torch.no_grad():
            current_obs = self.obs_cache[-1]
            current_state = current_obs.get('state')

            if 'image' in current_obs:
                img_tensor = self._prepare_image_tensor(current_obs['image'])
                current_latent = self.backbone.encode_image(img_tensor)
            else:
                current_latent = torch.zeros(1, self.backbone.vision_dim, device=self.device)

            current_latent = current_latent.to(device=decoder_device, dtype=decoder_dtype)

            # NO refinement loop - direct latent → action
            actions = []
            for _ in range(num_action_steps):
                predicted_joint = self._action_decoder.decode_joint(current_latent).squeeze(0).detach().cpu().numpy()
                rollout_action = self._prepare_rollout_action(predicted_joint, current_state)
                actions.append(rollout_action)
                
            return actions
        
    def reset(self) -> None:
        """Reset policy state."""
        self.obs_cache.clear()


# ============================================================================
# Checkpoint loading
# ============================================================================

def load_checkpoint(checkpoint_path: Path, device: str = "cuda") -> LatentNoRefinePolicy:
    """Load a trained LatentNoRefine checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Loaded LatentNoRefinePolicy
        
    Raises:
        FileNotFoundError: If checkpoint does not exist
        RuntimeError: If checkpoint format is invalid
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    policy = LatentNoRefinePolicy(device=device)
    policy._lazy_init()
    
    if policy._baseline is None:
        raise RuntimeError(f"Failed to initialize baseline from {checkpoint_path}")
    
    if 'state_dict' in checkpoint:
        policy._baseline.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        policy._baseline.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise RuntimeError(f"Invalid checkpoint format at {checkpoint_path}")
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return policy


# ============================================================================
# Main entrypoint
# ============================================================================

def main():
    """CLI entrypoint for standalone inference."""
    parser = argparse.ArgumentParser(
        description='LatentNoRefine RoboTwin policy entrypoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/latent_no_refine.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Freeze backbone during inference'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize model
    logger.info(f"Initializing LatentNoRefine policy on {args.device}")
    usr_args = {
        'config': args.config,
        'device': args.device,
        'model_path': args.model_path,
        'freeze_backbone': args.freeze_backbone,
    }
    _ = get_model(usr_args)
    
    logger.info("Model initialized successfully")
    logger.info(f"Policy ready for inference (NO refinement)")
    logger.info("To use with RoboTwin eval harness:")
    logger.info("  python script/eval_policy.py --config policy/LatentNoRefine/deploy_policy.yml")


if __name__ == '__main__':
    main()
