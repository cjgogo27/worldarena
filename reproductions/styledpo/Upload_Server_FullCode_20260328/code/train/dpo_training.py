#!/usr/bin/env python3
"""Minimal DPO training entry for chosen/rejected preference pairs."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PairExample:
    split: str
    group_id: str
    chosen_score: float
    rejected_score: float
    weight: float


class ScorePolicy(nn.Module):
    """Policy mapping scalar score to scalar log-prob proxy."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return self.weight * scores + self.bias


def load_config(config_path: Optional[str]) -> Dict:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def config_get(cfg: Dict, keys: List[str], default):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal DPO scorer from preference pairs")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--train_data", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints")
    parser.add_argument("--logging_dir", type=str, default="results/logs")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=50)
    return parser.parse_args()


def expand_examples(groups: List[Dict], split: str) -> List[PairExample]:
    examples: List[PairExample] = []
    for g in groups:
        g_split = g.get("split", "train")
        if split != "all" and g_split != split:
            continue
        chosen = g.get("chosen", {})
        rejected_list = g.get("rejected_list", [])
        m = max(1, len(rejected_list))
        weight = float(1.0 / m)
        chosen_score = float(chosen.get("score", 0.0))
        for rej in rejected_list:
            examples.append(
                PairExample(
                    split=g_split,
                    group_id=g.get("group_id", ""),
                    chosen_score=chosen_score,
                    rejected_score=float(rej.get("score", 0.0)),
                    weight=weight,
                )
            )
    return examples


def sample_batch(examples: List[PairExample], batch_size: int, device: torch.device):
    idx = torch.randint(low=0, high=len(examples), size=(batch_size,))
    chosen = torch.tensor([examples[i].chosen_score for i in idx.tolist()], dtype=torch.float32, device=device)
    rejected = torch.tensor([examples[i].rejected_score for i in idx.tolist()], dtype=torch.float32, device=device)
    weights = torch.tensor([examples[i].weight for i in idx.tolist()], dtype=torch.float32, device=device)
    return chosen, rejected, weights


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config if args.config else None)

    train_data = args.train_data or config_get(cfg, ["data", "train_data_path"], "results/pair_benchmark_build/preference_pairs.json")
    output_dir = Path(args.output_dir or config_get(cfg, ["training", "output_dir"], "results/checkpoints"))
    logging_dir = Path(args.logging_dir or config_get(cfg, ["training", "logging_dir"], "results/logs"))
    max_steps = args.max_steps if args.max_steps > 0 else int(config_get(cfg, ["training", "max_steps"], 5))
    batch_size = max(1, args.batch_size)
    beta = float(args.beta if args.beta > 0 else config_get(cfg, ["dpo", "beta"], 0.1))
    lr = float(args.learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = logging_dir / "training.log"

    set_seed(args.seed)

    with open(train_data, "r", encoding="utf-8") as f:
        groups = json.load(f)

    examples = expand_examples(groups, args.split)
    if not examples:
        raise RuntimeError(f"No training examples found for split={args.split} in {train_data}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScorePolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: List[Dict] = []
    with train_log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"[{now_iso()}] start dpo training\n")
        logf.write(f"train_data={train_data} split={args.split} examples={len(examples)}\n")
        logf.write(f"device={device} beta={beta} lr={lr} max_steps={max_steps} batch_size={batch_size}\n")

        for step in range(1, max_steps + 1):
            chosen, rejected, weights = sample_batch(examples, batch_size=min(batch_size, len(examples)), device=device)

            pi_chosen = model(chosen)
            pi_rejected = model(rejected)
            ref_chosen = chosen
            ref_rejected = rejected

            logits = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
            loss_per = -F.logsigmoid(logits)
            loss = (loss_per * weights).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            margin = (chosen - rejected).mean().item()
            pi_margin = (pi_chosen - pi_rejected).mean().item()
            row = {
                "step": step,
                "loss": float(loss.item()),
                "margin": float(margin),
                "policy_margin": float(pi_margin),
                "w": float(model.weight.detach().cpu().item()),
                "b": float(model.bias.detach().cpu().item()),
            }
            history.append(row)
            logf.write(json.dumps(row, ensure_ascii=False) + "\n")

            if step % args.save_every == 0 or step == max_steps:
                ckpt_path = output_dir / f"checkpoint_step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "history_tail": history[-20:],
                        "train_data": train_data,
                        "split": args.split,
                    },
                    ckpt_path,
                )
                logf.write(f"saved_checkpoint={ckpt_path.as_posix()}\n")

        logf.write(f"[{now_iso()}] end dpo training\n")

    metrics = {
        "generated_at": now_iso(),
        "train_data": train_data,
        "split": args.split,
        "num_examples": len(examples),
        "max_steps": max_steps,
        "batch_size": batch_size,
        "beta": beta,
        "learning_rate": lr,
        "device": str(device),
        "final": history[-1],
        "history": history,
        "checkpoint": (output_dir / f"checkpoint_step_{max_steps}.pt").as_posix(),
        "training_log": train_log_path.as_posix(),
    }
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"WROTE {metrics_path}")
    print(f"FINAL loss={metrics['final']['loss']:.6f} w={metrics['final']['w']:.6f} b={metrics['final']['b']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
