#!/usr/bin/env python3
"""Validate generated BAGEL content-style benchmark and emit benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate content-style benchmark")
    parser.add_argument("--benchmark_dir", type=str, default="benchmark/bagel_content_style_benchmark")
    parser.add_argument("--style_prompts", type=str, default="resources/styles/style_prompts_40_v1.json")
    parser.add_argument("--output_dir", type=str, default="results/benchmark_build")
    parser.add_argument("--required_content_count", type=int, default=17)
    parser.add_argument("--required_style_count", type=int, default=40)
    return parser.parse_args()


def load_style_ids(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    prompts = obj.get("prompts", {}) if isinstance(obj, dict) else {}
    if not isinstance(prompts, dict):
        return []
    return sorted(prompts.keys())


def safe_name(text: str) -> str:
    import re

    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return s or "style"


def main() -> int:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    style_prompts = Path(args.style_prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style_ids = load_style_ids(style_prompts)
    expected_style_files = {safe_name(s): s for s in style_ids}

    content_dirs = sorted([p for p in benchmark_dir.iterdir() if p.is_dir() and p.name.startswith("content_")]) if benchmark_dir.exists() else []

    details: List[Dict] = []
    mismatch: List[Dict] = []

    for c in content_dirs:
        pngs = sorted([p for p in c.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
        styles_present = {p.stem for p in pngs}
        missing = sorted(set(expected_style_files.keys()) - styles_present)
        extra = sorted(styles_present - set(expected_style_files.keys()))
        entry = {
            "content_id": c.name,
            "image_count": len(pngs),
            "missing_style_count": len(missing),
            "extra_style_count": len(extra),
            "missing_styles": [expected_style_files[m] for m in missing],
            "extra_style_safe": extra,
        }
        details.append(entry)
        if len(missing) > 0 or len(extra) > 0:
            mismatch.append(entry)

    content_count = len(content_dirs)
    expected_total = content_count * len(style_ids)
    total_images = sum(x["image_count"] for x in details)
    pass_gate = (
        content_count >= args.required_content_count
        and len(style_ids) == args.required_style_count
        and len(mismatch) == 0
        and total_images == expected_total
    )

    manifest = {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "benchmark_dir": benchmark_dir.as_posix(),
        "style_prompts": style_prompts.as_posix(),
        "content_count": content_count,
        "required_content_count": args.required_content_count,
        "style_count": len(style_ids),
        "required_style_count": args.required_style_count,
        "total_images": total_images,
        "expected_total_images": expected_total,
        "pass_gate": pass_gate,
        "mismatch_count": len(mismatch),
        "details": details,
    }
    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Content-Style Benchmark Check Report",
        f"generated_at: {manifest['generated_at']}",
        f"benchmark_dir: {benchmark_dir.as_posix()}",
        f"content_count: {content_count} (required >= {args.required_content_count})",
        f"style_count: {len(style_ids)} (required {args.required_style_count})",
        f"total_images: {total_images} (expected {expected_total})",
        f"mismatch_count: {len(mismatch)}",
        f"pass_gate: {pass_gate}",
        "",
        "Per-content summary:",
    ]
    for d in details:
        lines.append(
            f"- {d['content_id']}: images={d['image_count']} missing={d['missing_style_count']} extra={d['extra_style_count']}"
        )
    report_path = output_dir / "benchmark_check_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"WROTE {manifest_path}")
    print(f"WROTE {report_path}")
    print(f"PASS {pass_gate}")
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
