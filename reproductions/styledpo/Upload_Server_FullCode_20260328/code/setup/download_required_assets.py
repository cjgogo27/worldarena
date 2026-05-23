#!/usr/bin/env python3
"""Download and verify required assets for Style-DPO server runs.

Usage example:
python code/setup/download_required_assets.py \
  --output_root models \
  --bagel_repo ByteDance-Seed/BAGEL-7B-MoT \
  --clip_repo openai/clip-vit-large-patch14 \
    --vlm_repo Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from huggingface_hub import snapshot_download


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _download_repo(repo_id: str, local_dir: Path, token: str | None = None) -> Dict:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt", "*.model"],
    )
    return {
        "repo_id": repo_id,
        "local_dir": str(local_dir.resolve()),
        "timestamp": _now(),
    }


def _verify_bagel_dir(model_dir: Path) -> Dict:
    required = [
        "llm_config.json",
        "vit_config.json",
        "ae.safetensors",
        "ema.safetensors",
    ]
    missing = [name for name in required if not (model_dir / name).exists()]
    return {"required_files": required, "missing": missing, "ok": len(missing) == 0}


def _verify_clip_dir(model_dir: Path) -> Dict:
    required_any = ["config.json", "preprocessor_config.json"]
    missing = [name for name in required_any if not (model_dir / name).exists()]
    return {"required_files": required_any, "missing": missing, "ok": len(missing) == 0}


def _collect_file_count(target_dir: Path) -> int:
    return sum(1 for _ in target_dir.rglob("*") if _.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download required model assets for Style-DPO")
    parser.add_argument("--output_root", type=str, default="models", help="Root directory for downloaded models")
    parser.add_argument("--bagel_repo", type=str, default="ByteDance-Seed/BAGEL-7B-MoT", help="HF repo id for BAGEL")
    parser.add_argument("--clip_repo", type=str, default="openai/clip-vit-large-patch14", help="HF repo id for CLIP")
    parser.add_argument(
        "--vlm_repo",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="HF repo id for scorer model (default: Qwen/Qwen3.5-9B)",
    )
    parser.add_argument("--hf_token", type=str, default="", help="HF token (optional; can also use HF_TOKEN env)")
    parser.add_argument("--skip_bagel", action="store_true")
    parser.add_argument("--skip_clip", action="store_true")
    parser.add_argument("--skip_vlm", action="store_true")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="results/setup/assets_manifest_20260328.json",
        help="Manifest output path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN") or None

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {
        "created_at": _now(),
        "output_root": str(output_root.resolve()),
        "downloads": {},
        "verification": {},
        "errors": [],
    }

    try:
        if not args.skip_bagel:
            bagel_dir = output_root / "BAGEL-7B-MoT"
            results["downloads"]["bagel"] = _download_repo(args.bagel_repo, bagel_dir, token)
            results["verification"]["bagel"] = _verify_bagel_dir(bagel_dir)
            results["verification"]["bagel"]["file_count"] = _collect_file_count(bagel_dir)

        if not args.skip_clip:
            clip_dir = output_root / "clip-vit-large-patch14"
            results["downloads"]["clip"] = _download_repo(args.clip_repo, clip_dir, token)
            results["verification"]["clip"] = _verify_clip_dir(clip_dir)
            results["verification"]["clip"]["file_count"] = _collect_file_count(clip_dir)

        if args.vlm_repo and not args.skip_vlm:
            vlm_name = args.vlm_repo.split("/")[-1]
            vlm_dir = output_root / vlm_name
            results["downloads"]["vlm"] = _download_repo(args.vlm_repo, vlm_dir, token)
            results["verification"]["vlm"] = {
                "ok": (vlm_dir / "config.json").exists(),
                "required_files": ["config.json"],
                "missing": [] if (vlm_dir / "config.json").exists() else ["config.json"],
                "file_count": _collect_file_count(vlm_dir),
            }

    except Exception as exc:  # pragma: no cover
        results["errors"].append(str(exc))

    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Manifest written to: {manifest_path}")

    has_errors = len(results["errors"]) > 0
    failed_checks: List[str] = []
    for key, val in results.get("verification", {}).items():
        if not val.get("ok", False):
            failed_checks.append(key)

    if has_errors or failed_checks:
        print("Asset setup completed with issues.")
        if has_errors:
            print("Errors:", results["errors"])
        if failed_checks:
            print("Failed checks:", failed_checks)
        return 1

    print("Asset setup completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
