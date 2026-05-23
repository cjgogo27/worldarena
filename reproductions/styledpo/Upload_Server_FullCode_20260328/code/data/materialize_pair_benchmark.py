#!/usr/bin/env python3
"""Materialize preference-pair benchmark images for easy visual inspection."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize preference pairs into image folders")
    parser.add_argument("--pairs_file", type=str, default="results/pair_benchmark_build/preference_pairs.json")
    parser.add_argument("--output_dir", type=str, default="results/pair_benchmark_build/pair_images")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--preview_count", type=int, default=24)
    parser.add_argument("--thumb_size", type=int, default=192)
    return parser.parse_args()


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return cleaned.strip("_")[:180] or "group"


def ensure_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return

    rel = os.path.relpath(src.as_posix(), dst.parent.as_posix())
    try:
        dst.symlink_to(rel)
    except OSError:
        shutil.copy2(src, dst)


def build_preview(groups: List[Dict], out_path: Path, count: int, thumb: int) -> None:
    picked = groups[: max(0, min(count, len(groups)))]
    if not picked:
        return

    cols = 4
    row_height = thumb + 36
    margin = 12
    width = cols * thumb + (cols + 1) * margin
    height = len(picked) * row_height + margin
    canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    for row_i, g in enumerate(picked):
        y0 = margin + row_i * row_height
        draw.text((margin, y0), f"{g.get('split','?')} | {g.get('style_id','?')} | {g.get('content_id','?')}", fill=(240, 240, 240))

        paths = [g.get("chosen", {}).get("image_path", "")]
        paths.extend([r.get("image_path", "") for r in g.get("rejected_list", [])[:3]])
        labels = ["chosen", "rej1", "rej2", "rej3"]

        for col_i in range(cols):
            x = margin + col_i * (thumb + margin)
            y = y0 + 18
            draw.rectangle([x - 1, y - 1, x + thumb + 1, y + thumb + 1], outline=(90, 90, 90), width=1)
            if col_i < len(paths) and paths[col_i]:
                try:
                    img = Image.open(paths[col_i]).convert("RGB")
                    img.thumbnail((thumb, thumb))
                    paste_x = x + (thumb - img.width) // 2
                    paste_y = y + (thumb - img.height) // 2
                    canvas.paste(img, (paste_x, paste_y))
                except Exception:
                    draw.text((x + 8, y + thumb // 2 - 6), "load_err", fill=(220, 120, 120))
            draw.text((x + 4, y + thumb - 14), labels[col_i], fill=(255, 230, 140))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    args = parse_args()
    pairs_file = Path(args.pairs_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: List[Dict] = json.loads(pairs_file.read_text(encoding="utf-8"))

    index_rows: List[Dict] = []
    split_counts: Dict[str, int] = {}

    for g in groups:
        split = g.get("split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
        group_id = g.get("group_id", "group")
        group_dir = output_dir / split / safe_name(group_id)
        group_dir.mkdir(parents=True, exist_ok=True)

        chosen = g.get("chosen", {})
        chosen_path = Path(chosen.get("image_path", ""))
        if chosen_path.exists():
            chosen_dst = group_dir / f"chosen{chosen_path.suffix.lower() or '.jpg'}"
            ensure_link_or_copy(chosen_path, chosen_dst, args.mode)

        rejected_list = g.get("rejected_list", [])
        for i, r in enumerate(rejected_list, start=1):
            r_path = Path(r.get("image_path", ""))
            if not r_path.exists():
                continue
            r_dst = group_dir / f"rejected_{i}{r_path.suffix.lower() or '.jpg'}"
            ensure_link_or_copy(r_path, r_dst, args.mode)

        meta_path = group_dir / "meta.json"
        meta_path.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")

        index_rows.append(
            {
                "split": split,
                "group_id": group_id,
                "group_dir": group_dir.as_posix(),
                "content_id": g.get("content_id", ""),
                "style_id": g.get("style_id", ""),
                "rejected_count": len(rejected_list),
            }
        )

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "generated_at": now_iso(),
        "pairs_file": pairs_file.as_posix(),
        "output_dir": output_dir.as_posix(),
        "mode": args.mode,
        "group_count": len(groups),
        "split_counts": split_counts,
        "index_file": index_path.as_posix(),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    preview_path = output_dir / "preview_contact_sheet.jpg"
    build_preview(groups, preview_path, args.preview_count, args.thumb_size)

    print(f"WROTE {index_path}")
    print(f"WROTE {summary_path}")
    print(f"WROTE {preview_path}")
    print(f"GROUPS {len(groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
