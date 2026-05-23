#!/usr/bin/env python3
"""Experiment runner and batch evaluation for LaRA-WM baselines.

This script runs all baseline comparisons and generates aggregated results
with comparison tables.

Baselines:
- direct_policy: Forward backbone features directly to actions
- latent_no_refine: Latent encoder + world model WITHOUT iterative refinement
- no_reward_wm: World model WITHOUT reward prediction head

Usage:
    # Run all baselines with default settings
    python scripts/run_experiments.py

    # Run specific baseline
    python scripts/run_experiments.py --baseline direct_policy

    # Run with custom arguments
    python scripts/run_experiments.py --num-episodes 50 --seeds 0,1,2,3,4
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baselines.direct_policy import DirectPolicyHook, DirectPolicyConfig
from baselines.latent_no_refine import LatentNoRefineBaseline, LatentNoRefineConfig, create_latent_no_refine_baseline
from baselines.no_reward_wm import NoRewardWorldModel, NoRewardWMConfig, create_no_reward_wm
from eval.harness import EvaluationHarness, EpisodeResult, EvalMetrics, PolicyHook

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

BASELINES = ["direct_policy", "latent_no_refine", "no_reward_wm"]

@dataclass
class ExperimentConfig:
    """Configuration for experiment runs."""
    
    # Evaluation settings
    num_episodes: int = 100
    max_steps: int = 300
    seeds: Optional[List[int]] = None
    
    # Model settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim: int = 1536
    action_dim: int = 7
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("experiments/results"))
    save_results: bool = True
    verbose: bool = True


@dataclass
class BaselineResult:
    """Results from a single baseline experiment."""
    
    baseline_name: str
    metrics: EvalMetrics
    episode_results: List[EpisodeResult]
    config: Dict[str, Any]
    run_time_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Environment Factory (mock for testing)
# ============================================================================

class MockEnv:
    """Mock environment for testing experiment runner.
    
    In production, replace with actual RoboTwin environment.
    """
    
    def __init__(
        self,
        action_dim: int = 7,
        state_dim: int = 14,
        image_shape: Tuple[int, int, int] = (3, 224, 224),
    ):
        self.action_space = type(
            "Space", (), {"shape": (action_dim,)}
        )()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.image_shape = image_shape
        self._state = None
        self._episode_length = 0
        self._max_episode_length = 300
        self._done = False
        self._total_reward = 0.0
        self.eval_success = False
        self._instruction = ""
        
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self._state = np.random.randn(self.state_dim).astype(np.float32)
        self._episode_length = 0
        self._done = False
        self._total_reward = 0.0
        
        # Simulate success for ~50-70% of episodes (random)
        self._target_success = np.random.random() > 0.35
        self._steps_to_success = np.random.randint(10, 100) if self._target_success else -1
        
        return {"state": self._state}
    
    def set_instruction(self, instruction: str):
        self._instruction = instruction
    
    def get_obs(self) -> Dict[str, Any]:
        image = np.random.randint(
            0, 255, self.image_shape, dtype=np.uint8
        )
        return {
            "image": image,
            "state": self._state.copy() if self._state is not None else None,
            "instruction": self._instruction,
        }
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        self._episode_length += 1
        
        # Apply action (slight state change)
        if self._state is not None and action is not None:
            self._state = self._state + np.random.randn(self.state_dim).astype(np.float32) * 0.01
        
        # Check success condition
        if self._target_success and self._episode_length >= self._steps_to_success:
            self.eval_success = True
            reward = 1.0
            self._done = True
        elif self._episode_length >= self._max_episode_length:
            self._done = True
            reward = 0.0
        else:
            reward = np.random.random() * 0.1  # Small step reward
            
        self._total_reward += reward
        
        step_info = {
            "episode_length": self._episode_length,
            "cumulative_reward": self._total_reward,
        }
        
        obs = self.get_obs()
        return obs, reward, self._done, step_info
    
    def close_env(self):
        pass


def create_env_factory(action_dim: int = 7, state_dim: int = 14):
    """Create environment factory callable."""
    def factory():
        return MockEnv(action_dim=action_dim, state_dim=state_dim)
    return factory


# ============================================================================
# Baseline Evaluators
# ============================================================================

class BaselineEvaluator:
    """Base class for evaluating baselines."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        env_factory,
    ):
        self.config = config
        self.env_factory = env_factory
        self.device = config.device
        
    def evaluate(
        self,
        num_episodes: int,
        seeds: Optional[List[int]] = None,
    ) -> Tuple[List[EpisodeResult], EvalMetrics]:
        """Run evaluation and return results."""
        raise NotImplementedError


class DirectPolicyEvaluator(BaselineEvaluator):
    """Evaluator for direct_policy baseline."""
    
    def evaluate(
        self,
        num_episodes: int,
        seeds: Optional[List[int]] = None,
    ) -> Tuple[List[EpisodeResult], EvalMetrics]:
        """Evaluate direct policy baseline."""
        
        policy_config = DirectPolicyConfig(
            action_dim=self.config.action_dim,
            hidden_dim=512,
            num_layers=2,
            dropout=0.1,
        )
        
        policy = DirectPolicyHook(
            config=policy_config,
            device=self.device,
        )
        
        harness = EvaluationHarness(
            policy=policy,
            env_factory=self.env_factory,
            device=self.device,
            max_steps=self.config.max_steps,
        )
        
        # Run evaluation
        instructions = [f"complete task {i}" for i in range(num_episodes)]
        results, metrics = harness.evaluate(instructions, num_episodes)
        
        return results, metrics


class LatentNoRefineEvaluator(BaselineEvaluator):
    """Evaluator for latent_no_refine baseline."""
    
    def evaluate(
        self,
        num_episodes: int,
        seeds: Optional[List[int]] = None,
    ) -> Tuple[List[EpisodeResult], EvalMetrics]:
        """Evaluate latent no-refine baseline."""
        
        from backbone.adapter import BackboneAdapter
        
        # Create mock backbone for testing
        # In production: backbone = BackboneAdapter.from_config(); backbone.load()
        class MockBackbone:
            vision_dim = 1536
            
            def __init__(self):
                self.model = None
                
            def encode_image(self, x):
                B = x.shape[0] if x.dim() > 3 else 1
                return torch.randn(B, 1536, device=x.device)
            
            def encode_state(self, x):
                B = x.shape[0] if x.dim() > 1 else 1
                return torch.randn(B, 512, device=x.device)
        
        backbone = MockBackbone()
        
        config = LatentNoRefineConfig(
            latent_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
            hidden_dim=512,
            num_layers=3,
            batch_size=32,
            epochs=10,  # Reduced for testing
            device=self.device,
        )
        
        baseline = LatentNoRefineBaseline(
            config=config,
            backbone_adapter=backbone,
        )
        
        # Create simple policy hook for harness
        class LatentNoRefineHook(PolicyHook):
            def __init__(self, baseline):
                self.baseline = baseline
                self._model = baseline
                self.device = self.baseline.config.device
                self._action_fn = self._predict_action
                
            def get_model(self, config):
                return self
                
            def _predict_action(self, env, model, observation):
                # Get action from baseline
                action = self.baseline.predict(observation)
                return action
                
            def reset_model(self, model):
                pass
        
        policy = LatentNoRefineHook(baseline)
        
        harness = EvaluationHarness(
            policy=policy,
            env_factory=self.env_factory,
            device=self.device,
            max_steps=self.config.max_steps,
        )
        
        instructions = [f"complete task {i}" for i in range(num_episodes)]
        results, metrics = harness.evaluate(instructions, num_episodes)
        
        return results, metrics


class NoRewardWMEvaluator(BaselineEvaluator):
    """Evaluator for no_reward_wm baseline."""
    
    def evaluate(
        self,
        num_episodes: int,
        seeds: Optional[List[int]] = None,
    ) -> Tuple[List[EpisodeResult], EvalMetrics]:
        """Evaluate no_reward_wm baseline."""
        
        from backbone.adapter import BackboneAdapter
        
        # Create mock encoder for testing
        class MockEncoder:
            vision_dim = 1536
            
            def encode_image(self, x):
                B = x.shape[0] if x.dim() > 3 else 1
                return torch.randn(B, 1536, device=x.device)
        
        encoder = MockEncoder()
        
        config = NoRewardWMConfig(
            encoder_dim=self.config.latent_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
        )
        
        wm = NoRewardWorldModel(config=config, encoder=encoder)
        
        # Create simple policy hook
        class NoRewardWMHook(PolicyHook):
            def __init__(self, wm):
                self.wm = wm
                self._model = wm
                self.device = "cpu"
                self.hidden = None
                self._action_fn = self._step_action
                
            def get_model(self, config):
                self.hidden = torch.zeros(1, 1, 1536)
                return self
                
            def _step_action(self, env, model, observation):
                # Simple random action (WM would normally predict)
                action = np.random.randn(7) * 0.5
                self.hidden = torch.randn(1, 1, 1536)
                return action
                
            def reset_model(self, model):
                self.hidden = torch.zeros(1, 1, 1536)
        
        policy = NoRewardWMHook(wm)
        
        harness = EvaluationHarness(
            policy=policy,
            env_factory=self.env_factory,
            device="cpu",
            max_steps=self.config.max_steps,
        )
        
        instructions = [f"complete task {i}" for i in range(num_episodes)]
        results, metrics = harness.evaluate(instructions, num_episodes)
        
        return results, metrics


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """Main experiment runner for LaRA-WM baselines."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: Dict[str, BaselineResult] = {}
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_evaluator(self, baseline_name: str) -> BaselineEvaluator:
        """Get evaluator for baseline."""
        env_factory = create_env_factory(
            action_dim=self.config.action_dim,
            state_dim=14,
        )
        
        if baseline_name == "direct_policy":
            return DirectPolicyEvaluator(self.config, env_factory)
        elif baseline_name == "latent_no_refine":
            return LatentNoRefineEvaluator(self.config, env_factory)
        elif baseline_name == "no_reward_wm":
            return NoRewardWMEvaluator(self.config, env_factory)
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")
    
    def run_baseline(
        self,
        baseline_name: str,
        num_episodes: Optional[int] = None,
        seeds: Optional[List[int]] = None,
    ) -> BaselineResult:
        """Run evaluation for a single baseline."""
        
        num_episodes = num_episodes or self.config.num_episodes
        seeds = seeds or self.config.seeds
        
        logger.info(f"Running baseline: {baseline_name}")
        logger.info(f"  Episodes: {num_episodes}")
        logger.info(f"  Device: {self.config.device}")
        
        start_time = time.time()
        
        evaluator = self._get_evaluator(baseline_name)
        episode_results, metrics = evaluator.evaluate(
            num_episodes=num_episodes,
            seeds=seeds,
        )
        
        run_time = time.time() - start_time
        
        result = BaselineResult(
            baseline_name=baseline_name,
            metrics=metrics,
            episode_results=episode_results,
            config={
                "num_episodes": num_episodes,
                "max_steps": self.config.max_steps,
                "device": self.config.device,
                "latent_dim": self.config.latent_dim,
                "action_dim": self.config.action_dim,
            },
            run_time_seconds=run_time,
        )
        
        logger.info(f"  Completed in {run_time:.1f}s")
        logger.info(f"  Success Rate: {metrics.success_rate:.1%}")
        logger.info(f"  Mean Return: {metrics.mean_return:.3f} ± {metrics.std_return:.3f}")
        logger.info(f"  Mean Episode Length: {metrics.mean_episode_length:.1f} ± {metrics.std_episode_length:.1f}")
        
        self.results[baseline_name] = result
        return result
    
    def run_all_baselines(
        self,
        baselines: Optional[List[str]] = None,
    ) -> Dict[str, BaselineResult]:
        """Run all baselines."""
        
        baselines = baselines or BASELINES
        
        logger.info(f"Running {len(baselines)} baselines: {baselines}")
        logger.info("=" * 60)
        
        for baseline_name in baselines:
            result = self.run_baseline(baseline_name)
            self.results[baseline_name] = result
        
        logger.info("=" * 60)
        logger.info("All baselines completed")
        
        return self.results


# ============================================================================
# Results Aggregation & Comparison Tables
# ============================================================================

def aggregate_results(
    results: Dict[str, BaselineResult],
) -> Dict[str, Any]:
    """Aggregate results from all baselines."""
    
    aggregated = {
        "baselines": {},
        "summary": {
            "num_baselines": len(results),
            "total_episodes": sum(
                len(r.episode_results) for r in results.values()
            ),
        },
    }
    
    for name, result in results.items():
        metrics = result.metrics
        aggregated["baselines"][name] = {
            "success_rate": metrics.success_rate,
            "mean_return": metrics.mean_return,
            "std_return": metrics.std_return,
            "mean_episode_length": metrics.mean_episode_length,
            "std_episode_length": metrics.std_episode_length,
            "num_episodes": metrics.num_episodes,
            "num_successes": metrics.num_successes,
            "run_time_seconds": result.run_time_seconds,
        }
    
    # Find best baseline for each metric
    if results:
        success_rates = {
            name: r.metrics.success_rate 
            for name, r in results.items()
        }
        best_success = max(success_rates, key=success_rates.get)
        aggregated["summary"]["best_success_rate"] = best_success
        
        returns = {
            name: r.metrics.mean_return 
            for name, r in results.items()
        }
        best_return = max(returns, key=returns.get)
        aggregated["summary"]["best_mean_return"] = best_return
    
    return aggregated


def generate_comparison_table(
    results: Dict[str, BaselineResult],
    format: str = "ascii",
) -> str:
    """Generate comparison table for baselines."""
    
    if not results:
        return "No results to compare"
    
    # Sort baselines by success rate
    sorted_names = sorted(
        results.keys(), 
        key=lambda x: results[x].metrics.success_rate,
        reverse=True,
    )
    
    if format == "ascii":
        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append("BASELINE COMPARISON RESULTS")
        lines.append("=" * 80)
        lines.append("")
        
        # Header
        lines.append(
            f"{'Baseline':<20} {'Success Rate':>12} {'Mean Return':>14} "
            f"{'Std Return':>12} {'Ep Length':>12} {'Time (s)':>10}"
        )
        lines.append("-" * 80)
        
        # Rows
        for name in sorted_names:
            r = results[name]
            m = r.metrics
            lines.append(
                f"{name:<20} {m.success_rate:>11.1%} {m.mean_return:>13.3f} "
                f"{m.std_return:>11.3f} {m.mean_episode_length:>11.1f} "
                f"{r.run_time_seconds:>9.1f}"
            )
        
        lines.append("-" * 80)
        lines.append("")
        
        # Summary
        best = sorted_names[0]
        lines.append(f"Best (Success Rate): {best}")
        lines.append("")
        
        return "\n".join(lines)
    
    elif format == "markdown":
        lines = []
        lines.append("## Baseline Comparison Results")
        lines.append("")
        lines.append("| Baseline | Success Rate | Mean Return | Std Return | Ep Length | Time (s) |")
        lines.append(
            "|---------|----------:|---------:|---------:|--------:|--------:|"
        )
        
        for name in sorted_names:
            r = results[name]
            m = r.metrics
            lines.append(
                f"| {name} | {m.success_rate:.1%} | "
                f"{m.mean_return:.3f} | {m.std_return:.3f} | "
                f"{m.mean_episode_length:.1f} | {r.run_time_seconds:.1f} |"
            )
        
        lines.append("")
        lines.append(f"**Best (Success Rate)**: {sorted_names[0]}")
        lines.append("")
        
        return "\n".join(lines)
    
    elif format == "json":
        return json.dumps(aggregate_results(results), indent=2)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def save_results(
    results: Dict[str, BaselineResult],
    output_dir: Path,
    experiment_name: Optional[str] = None,
) -> Path:
    """Save results to files."""
    
    experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    aggregated = aggregate_results(results)
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    # Save ASCII table
    table_path = output_dir / "comparison.txt"
    with open(table_path, "w") as f:
        f.write(generate_comparison_table(results, format="ascii"))
    
    # Save markdown table
    md_path = output_dir / "comparison.md"
    with open(md_path, "w") as f:
        f.write(generate_comparison_table(results, format="markdown"))
    
    logger.info(f"Results saved to: {output_dir}")
    
    return output_dir


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Run LaRA-WM baseline experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--baseline",
        choices=BASELINES,
        help="Specific baseline to run (default: all)",
    )
    
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes per baseline (default: 100)",
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated seeds (default: 0,1,2,...)",
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Max steps per episode (default: 300)",
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (default: cuda if available)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Output directory (default: experiments/results)",
    )
    
    parser.add_argument(
        "--format",
        choices=["ascii", "markdown", "json"],
        default="ascii",
        help="Output format for comparison table (default: ascii)",
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results to files",
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Parse seeds
    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create config
    config = ExperimentConfig(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seeds=seeds,
        device=device,
        output_dir=args.output_dir,
        save_results=not args.no_save,
        verbose=args.verbose,
    )
    
    # Determine baselines to run
    baselines = [args.baseline] if args.baseline else BASELINES
    
    logger.info("LaRA-WM Experiment Runner")
    logger.info("=" * 60)
    logger.info(f"Baselines: {baselines}")
    logger.info(f"Episodes: {config.num_episodes}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 60)
    
    # Run experiments
    runner = ExperimentRunner(config)
    results = runner.run_all_baselines(baselines)
    
    # Print comparison table
    print(generate_comparison_table(results, format=args.format))
    
    # Save results
    if config.save_results:
        save_results(results, config.output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())