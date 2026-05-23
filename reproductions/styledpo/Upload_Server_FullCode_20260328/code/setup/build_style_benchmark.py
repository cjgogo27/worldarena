#!/usr/bin/env python3
"""Build and validate style benchmark manifest/report for 40x10 gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build style benchmark check artifacts")
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="benchmark/style_benchmark_40x10",
        help="Input benchmark directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/benchmark_build",
        help="Output directory for manifest/report",
    )
    parser.add_argument("--expected_styles", type=int, default=40)
    parser.add_argument("--expected_images_per_style", type=int, default=10)
    return parser.parse_args()


def collect_style_info(style_dir: Path) -> Dict:
    images = sorted([p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    return {
        "style_id": style_dir.name,
        "image_count": len(images),
        "images": [str(p.as_posix()) for p in images],
    }


def main() -> int:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style_dirs: List[Path] = []
    if benchmark_dir.exists():
        style_dirs = sorted([p for p in benchmark_dir.iterdir() if p.is_dir()])

    styles = [collect_style_info(p) for p in style_dirs]
    mismatches = [
        {
            "style_id": s["style_id"],
            "actual": s["image_count"],
            "expected": args.expected_images_per_style,
        }
        for s in styles
        if s["image_count"] != args.expected_images_per_style
    ]

    style_count = len(styles)
    total_images = sum(s["image_count"] for s in styles)
    expected_total = args.expected_styles * args.expected_images_per_style
    pass_gate = (
        style_count == args.expected_styles
        and total_images == expected_total
        and len(mismatches) == 0
    )

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_path": benchmark_dir.as_posix(),
        "expected_styles": args.expected_styles,
        "expected_images_per_style": args.expected_images_per_style,
        "expected_total_images": expected_total,
        "style_count": style_count,
        "total_images": total_images,
        "pass_gate": pass_gate,
        "mismatch_styles": mismatches,
        "styles": styles,
    }

    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "Style Benchmark 40x10 Check Report",
        f"generated_at: {manifest['generated_at']}",
        f"source_path: {manifest['source_path']}",
        f"style_count: {style_count} (expected {args.expected_styles})",
        f"total_images: {total_images} (expected {expected_total})",
        f"images_per_style_expected: {args.expected_images_per_style}",
        f"mismatch_style_count: {len(mismatches)}",
        f"pass_gate: {pass_gate}",
        "",
        "Per-style counts:",
    ]
    for s in styles:
        report_lines.append(f"- {s['style_id']}: {s['image_count']}")
    if mismatches:
        report_lines.extend(["", "Mismatch details:"])
        for m in mismatches:
            report_lines.append(
                f"- {m['style_id']}: {m['actual']} (expected {m['expected']})"
            )

    report_path = output_dir / "benchmark_check_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"WROTE {manifest_path}")
    print(f"WROTE {report_path}")
    print(f"PASS {pass_gate}")
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
