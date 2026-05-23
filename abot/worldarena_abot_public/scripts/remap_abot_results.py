from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap ABot batch outputs to WorldArena flat episode filenames")
    parser.add_argument("--manifest", required=True, help="Source WorldArena manifest JSON")
    parser.add_argument("--results-json", required=True, help="ABot inference results.json")
    parser.add_argument("--target-dir", required=True, help="Flat output directory like episode1.mp4, episode2.mp4")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    prompt_index = {(item["image"], item["prompt"]): item for item in manifest}
    mapped = 0
    skipped = 0

    for row in results:
        if row.get("status") != "success":
            skipped += 1
            continue
        key = (row.get("video"), row.get("prompt"))
        item = prompt_index.get(key)
        if item is None:
            skipped += 1
            continue
        src = Path(row["output_video"])
        if not src.exists():
            skipped += 1
            continue
        dst = target_dir / item["output_video"]
        if args.copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
        mapped += 1

    print(json.dumps({"mapped": mapped, "skipped": skipped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
