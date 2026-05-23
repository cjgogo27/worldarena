"""Evaluation harness for LaRA-WM using RoboTwin policy hooks.

Integrates with RoboTwin's eval framework while supporting LaRA-WM's
BackboneAdapter policy interface.

Usage:
    from lara_wm.src.eval.harness import EvaluationHarness
    
    # Single episode
    harness = EvaluationHarness(policy, env_config)
    result = harness.run_episode(instruction)
    
    # Batch eval
    results = harness.run_batch(instructions, num_episodes=100)
    
    # Metrics
    metrics = harness.get_metrics(results)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..backbone.adapter import BackboneAdapter

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Result from a single episode."""
    success: bool
    return_value: float
    episode_length: int
    instruction: str
    seed: int
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    success_rate: float
    mean_return: float
    std_return: float
    mean_episode_length: float
    std_episode_length: float
    num_episodes: int
    num_successes: int


class PolicyHook:
    """Policy hook interface compatible with RoboTwin eval pattern.
    
    Wraps LaRA-WM's BackboneAdapter to provide RoboTwin-style hooks:
    - get_model(): Initialize/return model
    - eval(): Get action from observation
    - reset_model(): Reset model state between episodes
    """
    
    def __init__(
        self,
        backbone: Optional[BackboneAdapter] = None,
        action_fn: Optional[Callable] = None,
        device: str = "cuda",
    ):
        self.backbone = backbone
        self.action_fn = action_fn or self._default_action
        self.device = device
        self._model: Optional[Any] = None
        
    def get_model(self, config: Dict[str, Any]) -> "PolicyHook":
        """Initialize model (RoboTwin hook)."""
        if self.backbone is None:
            self.backbone = BackboneAdapter.from_config()
            self.backbone.load()
        self._model = self.backbone
        return self
        
    def eval(self, env: Any, model: "PolicyHook", observation: Dict) -> np.ndarray:
        """Get action from observation (RoboTwin hook).
        
        Args:
            env: Environment instance
            model: PolicyHook instance
            observation: Dict with 'image', 'state', 'instruction'
            
        Returns:
            Action array
        """
        return self.action_fn(env, model, observation)
    
    def reset_model(self, model: "PolicyHook") -> None:
        pass
    
    def _default_action(
        self, env: Any, model: "PolicyHook", observation: Dict
    ) -> np.ndarray:
        image = observation.get("image")
        state = observation.get("state")
        
        with torch.no_grad():
            if image is not None:
                img_tensor = torch.from_numpy(image).unsqueeze(0).to(model.device)
                model.backbone.encode_image(img_tensor)
                
            if state is not None:
                state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(model.device)
                model.backbone.encode_state(state_tensor)
                
        action_dim = env.action_space.shape[0] if hasattr(env, "action_space") else 14
        return np.zeros(action_dim)


class EvaluationHarness:
    """Evaluation harness for LaRA-WM policies.
    
    Integrates RoboTwin's eval_policy pattern with LaRA-WM's
    BackboneAdapter for policy inference.
    
    Args:
        policy: PolicyHook instance or callable action function
        env_factory: Callable that returns a new environment instance
        device: Device for model inference
        max_steps: Maximum steps per episode
    """
    
    def __init__(
        self,
        policy: Optional[PolicyHook] = None,
        env_factory: Optional[Callable] = None,
        device: str = "cuda",
        max_steps: int = 300,
    ):
        self.policy = policy or PolicyHook()
        self.env_factory = env_factory
        self.device = device
        self.max_steps = max_steps
        self._config: Dict[str, Any] = {}
        
    def configure(self, config: Dict[str, Any]) -> "EvaluationHarness":
        """Apply configuration.
        
        Args:
            config: Dict with env and eval settings
            
        Returns:
            Self for chaining
        """
        self._config = config
        return self
        
    def run_episode(
        self,
        instruction: str,
        seed: int = 0,
        env: Optional[Any] = None,
    ) -> EpisodeResult:
        """Run single episode.
        
        Args:
            instruction: Language instruction for the task
            seed: Random seed for reproducibility
            env: Optional pre-created environment (uses factory if None)
            
        Returns:
            EpisodeResult with success, return, length
        """
        if env is None:
            if self.env_factory is None:
                raise ValueError("No env_factory provided and no env given")
            env = self.env_factory()
            
        env.reset(seed=seed)
        env.set_instruction(instruction=instruction)
        
        model = self.policy.get_model(self._config)
        self.policy.reset_model(model)
        
        total_return = 0.0
        episode_length = 0
        success = False
        info = {}
        
        while episode_length < self.max_steps:
            observation = env.get_obs()
            observation["instruction"] = instruction
            
            action = self.policy.eval(env, model, observation)
            
            obs, reward, done, step_info = env.step(action)
            total_return += reward
            episode_length += 1
            
            if hasattr(env, "eval_success") and env.eval_success:
                success = True
                info = step_info
                break
            if done:
                break
                
        if hasattr(env, "close_env"):
            env.close_env()
            
        return EpisodeResult(
            success=success,
            return_value=total_return,
            episode_length=episode_length,
            instruction=instruction,
            seed=seed,
            info=info,
        )
        
    def run_batch(
        self,
        instructions: List[str],
        num_episodes: int = 100,
        seeds: Optional[List[int]] = None,
    ) -> List[EpisodeResult]:
        """Run batch evaluation.
        
        Args:
            instructions: List of instructions to evaluate
            num_episodes: Total number of episodes to run
            seeds: Optional list of seeds (generated if not provided)
            
        Returns:
            List of EpisodeResult for each episode
        """
        results = []
        
        if seeds is None:
            seeds = list(range(num_episodes))
            
        if len(instructions) == 0:
            instructions = ["complete the task"]
            
        logger.info(f"Running batch eval: {num_episodes} episodes")
        
        for i in range(num_episodes):
            seed = seeds[i] if i < len(seeds) else i
            instruction = instructions[i % len(instructions)]
            
            try:
                result = self.run_episode(instruction=instruction, seed=seed)
                results.append(result)
                
                success_count = sum(1 for r in results if r.success)
                logger.info(
                    f"Episode {i+1}/{num_episodes} | "
                    f"Success: {success_count}/{len(results)} "
                    f"({100*success_count/len(results):.1f}%)"
                )
                
            except Exception as e:
                logger.warning(f"Episode {i} failed: {e}")
                results.append(EpisodeResult(
                    success=False,
                    return_value=0.0,
                    episode_length=0,
                    instruction=instruction,
                    seed=seed,
                    info={"error": str(e)},
                ))
                
        return results
        
    @staticmethod
    def get_metrics(results: List[EpisodeResult]) -> EvalMetrics:
        """Compute aggregated metrics from episode results.
        
        Args:
            results: List of EpisodeResult
            
        Returns:
            EvalMetrics with aggregated statistics
        """
        if not results:
            return EvalMetrics(
                success_rate=0.0,
                mean_return=0.0,
                std_return=0.0,
                mean_episode_length=0.0,
                std_episode_length=0.0,
                num_episodes=0,
                num_successes=0,
            )
            
        returns = [r.return_value for r in results]
        lengths = [r.episode_length for r in results]
        successes = [r.success for r in results]
        
        return EvalMetrics(
            success_rate=np.mean(successes),
            mean_return=np.mean(returns),
            std_return=np.std(returns) if len(returns) > 1 else 0.0,
            mean_episode_length=np.mean(lengths),
            std_episode_length=np.std(lengths) if len(lengths) > 1 else 0.0,
            num_episodes=len(results),
            num_successes=sum(successes),
        )
        
    def evaluate(
        self,
        instructions: List[str],
        num_episodes: int = 100,
    ) -> Tuple[List[EpisodeResult], EvalMetrics]:
        """Run evaluation and return results with metrics.
        
        Convenience method combining run_batch and get_metrics.
        
        Args:
            instructions: Instructions to evaluate
            num_episodes: Number of episodes
            
        Returns:
            Tuple of (results, metrics)
        """
        results = self.run_batch(instructions, num_episodes)
        metrics = self.get_metrics(results)
        return results, metrics


def create_harness(
    policy: Optional[PolicyHook] = None,
    env_factory: Optional[Callable] = None,
    config_path: Optional[Path] = None,
    **kwargs,
) -> EvaluationHarness:
    """Factory function to create evaluation harness.
    
    Args:
        policy: Optional PolicyHook (creates default if None)
        env_factory: Environment factory callable
        config_path: Optional path to config file
        **kwargs: Additional harness arguments
        
    Returns:
        Configured EvaluationHarness
    """
    harness = EvaluationHarness(
        policy=policy,
        env_factory=env_factory,
        **kwargs,
    )
    
    if config_path is not None and config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        harness.configure(config)
        
    return harness