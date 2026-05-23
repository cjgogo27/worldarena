#!/usr/bin/env python3
"""
Build manifests for the two old instruction versions (instructions, instructions_1)
so they can be used with batch_predict_i2v_worldarena.py.

Each manifest part file contains 250 entries with:
  - prompt: instruction text
  - image: absolute path to first frame PNG
  - output_video: fixed_scene_task_episode{N}.mp4
  - task_name: fixed_scene_task
  - episode_id: episode{N}

Output: test_dataset/manifests_{instructions,instructions_1}/
"""

import json
from pathlib import Path

TEST_DATASET = Path("/data/alice/cjtest/VideoX-Fun/test_dataset")
FIRST_FRAME_DIR = TEST_DATASET / "raw/first_frame/fixed_scene_task"
INSTRUCTIONS_OLD = TEST_DATASET / "raw/instructions_old"
OUTPUT_BASE = TEST_DATASET

PART_SIZE = 250

def build_manifest(instruction_version: str) -> list[dict]:

    instr_dir = INSTRUCTIONS_OLD / instruction_version / "fixed_scene_task"
    if not instr_dir.exists():
        raise FileNotFoundError(f"Instruction dir not found: {instr_dir}")

    episodes = []
    for fpath in sorted(instr_dir.glob("episode*.json")):
        stem = fpath.stem
        try:
            ep_num = int(stem.replace("episode", ""))
        except ValueError:
            continue
        episodes.append((ep_num, fpath))

    episodes.sort(key=lambda x: x[0])
    print(f"[{instruction_version}] Found {len(episodes)} instruction files")

    entries = []
    for ep_num, instr_path in episodes:
        with open(instr_path) as f:
            data = json.load(f)
        prompt = data.get("instruction", "")
        if not prompt:
            print(f"  WARNING: empty instruction in {instr_path.name}")
            continue

        frame_path = FIRST_FRAME_DIR / f"episode{ep_num}.png"
        if not frame_path.exists():
            matches = list(FIRST_FRAME_DIR.glob(f"episode{ep_num}.*"))
            if matches:
                frame_path = matches[0]
            else:
                print(f"  SKIP: no first frame image for episode{ep_num}")
                continue

        entries.append({
            "prompt": prompt,
            "image": str(frame_path.resolve()),
            "output_video": f"fixed_scene_task_episode{ep_num}.mp4",
            "task_name": "fixed_scene_task",
            "episode_id": f"episode{ep_num}",
        })

    print(f"[{instruction_version}] Built {len(entries)} valid entries")
    return entries


def write_part_files(entries: list[dict], output_dir: Path, prefix: str = "part"):

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(entries)
    num_parts = (total + PART_SIZE - 1) // PART_SIZE

    for i in range(num_parts):
        start = i * PART_SIZE
        end = min(start + PART_SIZE, total)
        part_entries = entries[start:end]
        part_path = output_dir / f"{prefix}_{i:02d}.json"
        with open(part_path, "w", encoding="utf-8") as f:
            json.dump(part_entries, f, ensure_ascii=False, indent=2)
        ep_range = f"{part_entries[0]['episode_id']}..{part_entries[-1]['episode_id']}"
        print(f"  Wrote {part_path.name}: {len(part_entries)} entries ({ep_range})")


def main():
    for version in ["instructions", "instructions_1"]:
        print(f"\n{'='*60}")
        print(f"Building manifest for: {version}")
        print('='*60)

        entries = build_manifest(version)
        output_dir = OUTPUT_BASE / f"manifests_{version}"
        write_part_files(entries, output_dir)

        # Also write combined manifest (all entries) for convenience
        combined_path = OUTPUT_BASE / f"manifests_{version}_all.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"  Wrote combined: {combined_path.name} ({len(entries)} entries)")

    print("\nDone!")


if __name__ == "__main__":
    main()
