#!/usr/bin/env python
# pyright: reportAny=false, reportConstantRedefinition=false, reportExplicitAny=false, reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportImplicitStringConcatenation=false
"""Run RoboTwin native success-rate evaluation for LaRA-WM policies."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROBOTWIN_ROOT = Path("/data/alice/cjtest/AgentCode_Baseline/RoboTwin")


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    policy_module: str | None
    policy_config: str | None
    runnable: bool
    status: str
    notes: str


def baseline_registry() -> dict[str, BaselineSpec]:
    return {
        "lara_wm": BaselineSpec(
            name="lara_wm",
            policy_module="src.deploy.robotwin_policy",
            policy_config=str(PROJECT_ROOT / "configs" / "lara_wm.yaml"),
            runnable=True,
            status="ready",
            notes="Uses the existing LaRA-WM RoboTwin deploy wrapper through RoboTwin's native eval harness.",
        ),
        "direct_policy": BaselineSpec(
            name="direct_policy",
            policy_module="src.deploy.direct_policy_deploy",
            policy_config=str(PROJECT_ROOT / "configs" / "direct_policy.yaml"),
            runnable=True,
            status="ready",
            notes="Direct policy baseline wraps backbone features directly to actions via MLP (no latent space).",
        ),
        "diffusion_policy": BaselineSpec(
            name="diffusion_policy",
            policy_module="src.deploy.diffusion_policy_deploy",
            policy_config=str(PROJECT_ROOT / "configs" / "robottwin_diffusion_policy.yaml"),
            runnable=True,
            status="ready",
            notes="RoboTwin-compatible 16D multi-camera diffusion policy rollout wrapper.",
        ),
        "latent_no_refine": BaselineSpec(
            name="latent_no_refine",
            policy_module="src.deploy.latent_no_refine_policy",
            policy_config=str(PROJECT_ROOT / "configs" / "latent_no_refine.yaml"),
            runnable=True,
            status="ready",
            notes="Uses LatentNoRefine baseline with direct latent→action mapping (no refinement). Checkpoint may need training.",
        ),
        "chunked_policy": BaselineSpec(
            name="chunked_policy",
            policy_module="src.deploy.chunked_policy_deploy",
            policy_config=str(PROJECT_ROOT / "configs" / "chunked_policy.yaml"),
            runnable=True,
            status="ready",
            notes="Multi-camera + 16D state rollout head that predicts short 16D action chunks for RoboTwin native success-rate eval.",
        ),
        "diffusion_policy": BaselineSpec(
            name="diffusion_policy",
            policy_module="src.deploy.diffusion_policy_deploy",
            policy_config=str(PROJECT_ROOT / "configs" / "robottwin_diffusion_policy.yaml"),
            runnable=True,
            status="ready",
            notes="16D multi-camera Diffusion Policy wrapper with observation history and multi-step action rollout aligned to RoboTwin deploy assumptions.",
        ),
        "no_reward_wm": BaselineSpec(
            name="no_reward_wm",
            policy_module=None,
            policy_config=None,
            runnable=False,
            status="missing_wrapper",
            notes="World-model baseline exists offline only; rollout wrapper/action head integration is still missing.",
        ),
        "act": BaselineSpec(
            name="act",
            policy_module=None,
            policy_config=None,
            runnable=False,
            status="missing_wrapper",
            notes="ACT training/evaluation exists offline, but this repo does not yet expose a RoboTwin-native deploy wrapper for the trained baseline.",
        ),
    }


@contextlib.contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_robottwin_eval_module(robottwin_root: Path) -> Any:
    script_path = robottwin_root / "script" / "eval_policy.py"
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(robottwin_root))
    sys.path.insert(0, str(script_path.parent))
    spec = importlib.util.spec_from_file_location("robottwin_native_eval_policy", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load RoboTwin eval harness from {script_path}")
    module = importlib.util.module_from_spec(spec)
    with pushd(robottwin_root):
        spec.loader.exec_module(module)
    patch_mplib_batch_fallback()
    return module


def patch_mplib_batch_fallback() -> None:
    try:
        from envs.robot.planner import MplibPlanner
    except Exception:
        return

    if not hasattr(MplibPlanner, "_lara_original_plan_path"):
        MplibPlanner._lara_original_plan_path = MplibPlanner.plan_path

        def plan_path_with_curobo_fallback(
            self: Any,
            now_qpos: Any,
            target_pose: Any,
            use_point_cloud: bool = False,
            use_attach: bool = False,
            arms_tag: str | None = None,
            log: bool = True,
        ) -> Any:
            if getattr(self, "planner_type", None) not in {"mplib_RRT", "mplib_screw"}:
                original_planner_type = self.planner_type
                self.planner_type = "mplib_RRT"
                try:
                    return self._lara_original_plan_path(
                        now_qpos,
                        target_pose,
                        use_point_cloud=use_point_cloud,
                        use_attach=use_attach,
                        arms_tag=arms_tag,
                        log=log,
                    )
                finally:
                    self.planner_type = original_planner_type

            return self._lara_original_plan_path(
                now_qpos,
                target_pose,
                use_point_cloud=use_point_cloud,
                use_attach=use_attach,
                arms_tag=arms_tag,
                log=log,
            )

        MplibPlanner.plan_path = plan_path_with_curobo_fallback

    if hasattr(MplibPlanner, "plan_batch"):
        return

    def plan_batch(self: Any, curr_joint_pos: Any, target_gripper_pose_list: list[Any], constraint_pose: Any = None, arms_tag: str | None = None) -> dict[str, Any]:
        import numpy as np

        statuses: list[str] = []
        positions: list[Any] = []
        velocities: list[Any] = []

        for target_pose in target_gripper_pose_list:
            result = self.plan_path(
                curr_joint_pos,
                target_pose,
                use_point_cloud=False,
                use_attach=False,
                arms_tag=arms_tag,
                log=False,
            )
            status = "Success" if result.get("status") == "Success" else "Failure"
            statuses.append(status)
            positions.append(result.get("position"))
            velocities.append(result.get("velocity"))

        response: dict[str, Any] = {"status": np.array(statuses, dtype=object)}
        if any(status == "Success" for status in statuses):
            response["position"] = np.array(positions, dtype=object)
            response["velocity"] = np.array(velocities, dtype=object)
        return response

    MplibPlanner.plan_batch = plan_batch


def get_camera_dims(robottwin_eval: Any, args: dict[str, Any]) -> None:
    with open(robottwin_eval.CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as handle:
        camera_config = yaml.load(handle.read(), Loader=yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]


def prepare_eval_args(robottwin_eval: Any, task_name: str, task_config: str, ckpt_setting: str, baseline: BaselineSpec) -> tuple[dict[str, Any], dict[str, Any], Any]:
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as handle:
        eval_args = yaml.load(handle.read(), Loader=yaml.FullLoader)

    eval_args = copy.deepcopy(eval_args)
    eval_args["task_name"] = task_name
    eval_args["task_config"] = task_config
    eval_args["ckpt_setting"] = ckpt_setting
    eval_args["policy_name"] = baseline.policy_module
    eval_args["eval_mode"] = True

    embodiment_type = eval_args.get("embodiment")
    with open(Path(robottwin_eval.CONFIGS_PATH) / "_embodiment_config.yml", "r", encoding="utf-8") as handle:
        embodiment_types = yaml.load(handle.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name: str) -> str:
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file configured for {name}")
        return robot_file

    if len(embodiment_type) == 1:
        eval_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        eval_args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        eval_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        eval_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        eval_args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        eval_args["embodiment_dis"] = embodiment_type[2]
        eval_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should contain 1 or 3 entries")

    eval_args["left_embodiment_config"] = robottwin_eval.get_embodiment_config(eval_args["left_robot_file"])
    eval_args["right_embodiment_config"] = robottwin_eval.get_embodiment_config(eval_args["right_robot_file"])
    get_camera_dims(robottwin_eval, eval_args)

    task_env = robottwin_eval.class_decorator(task_name)
    usr_args = {
        "task_name": task_name,
        "task_config": task_config,
        "ckpt_setting": ckpt_setting,
        "policy_name": baseline.policy_module,
        "instruction_type": None,
        "config": baseline.policy_config,
        "left_arm_dim": len(eval_args["left_embodiment_config"]["arm_joints_name"][0]),
        "right_arm_dim": len(eval_args["right_embodiment_config"]["arm_joints_name"][1]),
    }
    return usr_args, eval_args, task_env


def choose_rollout_instruction(robottwin_eval: Any, task_env: Any, task_name: str, instruction_type: str) -> str:
    episode_info = getattr(task_env, "info", None)
    info_payload = episode_info.get("info") if isinstance(episode_info, dict) else None
    if callable(getattr(robottwin_eval, "generate_episode_descriptions", None)) and info_payload is not None:
        results = robottwin_eval.generate_episode_descriptions(task_name, [info_payload], 1)
        if results and isinstance(results[0], dict):
            candidates = results[0].get(instruction_type)
            if isinstance(candidates, list) and candidates:
                return str(candidates[0])

    return task_name.replace("_", " ")


def run_task_success_eval_without_expert_check(
    robottwin_eval: Any,
    task_name: str,
    task_env: Any,
    eval_args: dict[str, Any],
    model: Any,
    start_seed: int,
    test_num: int,
    instruction_type: str,
) -> int:
    print(f"\033[34mTask Name: {task_name}\033[0m")
    print(f"\033[34mPolicy Name: {eval_args['policy_name']}\033[0m")
    print("Curobo unavailable; running RoboTwin policy loop without expert seed precheck.")

    task_env.suc = 0
    task_env.test_num = 0
    now_id = 0
    now_seed = start_seed
    clear_cache_freq = int(eval_args["clear_cache_freq"])
    max_setup_retries = int(eval_args.get("max_setup_retries", 20))

    eval_func = robottwin_eval.eval_function_decorator(eval_args["policy_name"], "eval")
    reset_func = robottwin_eval.eval_function_decorator(eval_args["policy_name"], "reset_model")

    while task_env.test_num < test_num:
        render_freq = eval_args["render_freq"]
        eval_args["render_freq"] = 0
        succ = False
        setup_ok = False
        attempt = 0

        try:
            while attempt < max_setup_retries and not setup_ok:
                try:
                    task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **eval_args)
                    setup_ok = True
                except Exception as exc:
                    if exc.__class__.__name__ == "UnStableError":
                        attempt += 1
                        print(
                            f"Skipping unstable seed {now_seed} for task {task_name} "
                            f"(attempt {attempt}/{max_setup_retries}): {exc}"
                        )
                        try:
                            task_env.close_env(clear_cache=False)
                        except Exception:
                            pass
                        now_seed += 1
                        continue
                    raise

            if not setup_ok:
                raise RuntimeError(
                    f"Exceeded max_setup_retries={max_setup_retries} for task {task_name} starting from seed {start_seed}"
                )

            instruction = choose_rollout_instruction(robottwin_eval, task_env, task_name, instruction_type)
            task_env.set_instruction(instruction=instruction)

            reset_func(model)
            while task_env.take_action_cnt < task_env.step_lim:
                observation = task_env.get_obs()
                eval_func(task_env, model, observation)
                if task_env.eval_success:
                    succ = True
                    break
        finally:
            eval_args["render_freq"] = render_freq

        if succ:
            task_env.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        task_env.close_env(clear_cache=((task_env.test_num + 1) % clear_cache_freq == 0))
        if task_env.render_freq:
            task_env.viewer.close()

        task_env.test_num += 1
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{eval_args['policy_name']}\033[0m | \033[92m{eval_args['task_config']}\033[0m | \033[91m{eval_args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{task_env.suc}/{task_env.test_num}\033[0m => \033[95m{round(task_env.suc / task_env.test_num * 100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return int(task_env.suc)


def run_task_success_eval(
    robottwin_eval: Any,
    baseline: BaselineSpec,
    task_name: str,
    task_config: str,
    ckpt_setting: str,
    instruction_type: str,
    seed: int,
    test_num: int,
) -> dict[str, Any]:
    usr_args, eval_args, task_env = prepare_eval_args(
        robottwin_eval=robottwin_eval,
        task_name=task_name,
        task_config=task_config,
        ckpt_setting=ckpt_setting,
        baseline=baseline,
    )
    usr_args["instruction_type"] = instruction_type
    usr_args["seed"] = seed

    get_model = robottwin_eval.eval_function_decorator(baseline.policy_module, "get_model")
    model = get_model(usr_args)
    start_seed = 100000 * (1 + seed)
    try:
        from envs.robot.planner import CUROBO_AVAILABLE
    except Exception:
        CUROBO_AVAILABLE = True

    if CUROBO_AVAILABLE:
        _, successes = robottwin_eval.eval_policy(
            task_name,
            task_env,
            eval_args,
            model,
            start_seed,
            test_num=test_num,
            video_size=None,
            instruction_type=instruction_type,
        )
    else:
        successes = run_task_success_eval_without_expert_check(
            robottwin_eval=robottwin_eval,
            task_name=task_name,
            task_env=task_env,
            eval_args=eval_args,
            model=model,
            start_seed=start_seed,
            test_num=test_num,
            instruction_type=instruction_type,
        )

    success_rate = successes / test_num if test_num else 0.0
    return {
        "task_name": task_name,
        "task_config": task_config,
        "ckpt_setting": ckpt_setting,
        "seed": seed,
        "test_num": test_num,
        "successes": successes,
        "success_rate": success_rate,
        "sr": success_rate,
        "metrics": {
            "success": {
                "sr": success_rate,
                "success_rate": success_rate,
                "success_count": successes,
                "num_episodes": test_num,
            },
            "id": None,
            "ood": None,
            "adaptation": None,
        },
    }


def write_outputs(output_dir: Path, baseline: BaselineSpec, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "baseline": asdict(baseline),
        "args": {
            "task_names": args.task_name,
            "task_config": args.task_config,
            "ckpt_setting": args.ckpt_setting,
            "instruction_type": args.instruction_type,
            "seed": args.seed,
            "test_num": args.test_num,
            "robottwin_root": str(args.robottwin_root),
        },
        "baseline_registry": {name: asdict(spec) for name, spec in baseline_registry().items()},
        "results": results,
    }
    json_path = output_dir / "robotwin_success_eval.json"
    json_path.write_text(json.dumps(payload, indent=2))

    lines = [
        "# RoboTwin Success Evaluation",
        "",
        f"- Baseline: `{baseline.name}`",
        f"- RoboTwin harness: `{args.robottwin_root / 'script' / 'eval_policy.py'}`",
        f"- Instruction type: `{args.instruction_type}`",
        f"- Task config: `{args.task_config}`",
        f"- Episodes per task: `{args.test_num}`",
        "",
        "## Results",
        "",
        "| Task | Successes | Episodes | SR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result['task_name']} | {result['successes']} | {result['test_num']} | {result['success_rate']:.3f} |"
        )

    lines.extend([
        "",
        "## Baseline availability",
        "",
        "| Baseline | Status | Runnable now | Notes |",
        "| --- | --- | --- | --- |",
    ])
    for name, spec in baseline_registry().items():
        lines.append(f"| {name} | {spec.status} | {'yes' if spec.runnable else 'no'} | {spec.notes} |")

    (output_dir / "robotwin_success_eval.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="lara_wm", choices=sorted(baseline_registry()))
    parser.add_argument("--task-name", nargs="+", required=False, default=[])
    parser.add_argument("--task-config", default="default")
    parser.add_argument("--ckpt-setting", default="deploy")
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "rollout_success")
    parser.add_argument("--robottwin-root", type=Path, default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument("--list-baselines", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = baseline_registry()

    if args.list_baselines:
        for name, spec in registry.items():
            print(f"{name}: status={spec.status}, runnable={spec.runnable} :: {spec.notes}")
        return 0

    if not args.task_name:
        raise SystemExit("--task-name is required unless --list-baselines is used")

    baseline = registry[args.baseline]
    if not baseline.runnable or baseline.policy_module is None or baseline.policy_config is None:
        raise SystemExit(f"Baseline '{baseline.name}' is not rollout-runnable yet: {baseline.notes}")

    robottwin_root = args.robottwin_root.resolve()
    robottwin_eval = load_robottwin_eval_module(robottwin_root)

    results: list[dict[str, Any]] = []
    with pushd(robottwin_root):
        for task_name in args.task_name:
            result = run_task_success_eval(
                robottwin_eval=robottwin_eval,
                baseline=baseline,
                task_name=task_name,
                task_config=args.task_config,
                ckpt_setting=args.ckpt_setting,
                instruction_type=args.instruction_type,
                seed=args.seed,
                test_num=args.test_num,
            )
            results.append(result)
            print(
                f"{task_name}: {result['successes']}/{result['test_num']} success "
                f"(SR={result['success_rate']:.3f})"
            )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / baseline.name / args.task_config / timestamp
    write_outputs(output_dir, baseline, results, args)
    print(f"Saved rollout success artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
