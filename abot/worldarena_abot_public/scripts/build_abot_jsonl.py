from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert WorldArena manifest to ABot batch JSONL")
    parser.add_argument("--manifest", required=True, help="Input manifest JSON from build_worldarena_manifests.py")
    parser.add_argument("--output", required=True, help="Output JSONL path for ABot inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for item in manifest:
            row = {
                "video": item["image"],
                "prompt": item["prompt"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(manifest)} rows to {output_path}")


if __name__ == "__main__":
    main()
