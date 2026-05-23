from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

ROOT = Path("/data/alice/cjtest")
OUT = ROOT / "world_model_video_pack_aligned"


def frame_count(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return max(1, total)


def extract_frames(video: Path, frames_dir: Path, count: int = 5) -> list[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    total = frame_count(video)
    if count <= 1:
        indices = [0]
    else:
        indices = sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})
    rels: list[str] = []
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    for i, idx in enumerate(indices, start=1):
        out = frames_dir / f"frame_{i:02d}_idx{idx:04d}.jpg"
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {idx} from {video}")
        if not cv2.imwrite(str(out), frame):
            raise RuntimeError(f"Could not write frame: {out}")
        rels.append(str(out.relative_to(OUT)))
    capture.release()
    return rels


instructions = {
    "episode10": "Lift the red beverage can with curved top, put it in the blue plasticbox for storing things, then handle the smooth white can with accents similarly.",
    "episode102": "Lift the blue pack holding playingcards using the left arm and transfer it outward.",
    "episode106": "Lift the glazed bowl with curved sides and position it neatly over the ceramic bowl.",
    "episode105": "With the right arm, grab the portable white fan and position it on the Silver mat facing the robot.",
}


entries = [
    {
        "category": "01_gigaworld",
        "label": "giga_world_smoke",
        "episode": None,
        "quality": "reference",
        "source_path": ROOT / "model_repros/giga-world-0/run_outputs/video_gr1_run1/0.mp4",
        "note": "GigaWorld README smoke output; not a WorldArena episode, included as project reference video.",
    },
    {
        "category": "02_abot_physworld_repro",
        "label": "abot_physworld_standalone_smoke_franka_pick",
        "episode": None,
        "quality": "good",
        "source_path": ROOT / "model_repros/ABot-PhysWorld/outputs/smoke/franka_pick_0_generated.mp4",
        "note": "Only true standalone ABot-PhysWorld repro output found locally. Generated with public ABot checkpoint from demo Franka image/prompt, not WorldArena-aligned.",
    },
]

for ep in ["episode10", "episode102"]:
    num = ep.removeprefix("episode")
    entries.append(
        {
            "category": "02_abot_physworld_repro",
            "label": f"abot_worldarena_hq_aux_{ep}",
            "episode": ep,
            "instruction": instructions[ep],
            "quality": "good",
            "source_path": ROOT / f"model_repros/worldarena_abot_public/outputs/test10_hq_raw/fixed_scene_task_{ep}_generated.mp4",
            "note": "Supplementary ABot WorldArena HQ reproduction output used to provide 3 good ABot repro examples; source is not standalone ABot-PhysWorld smoke.",
        }
    )

groups = [
    ("03_abot_worldarena_good", "abot_worldarena_good", "model_repros/worldarena_abot_public/outputs/test10_t1_flat/{ep}.mp4", ["episode10", "episode102", "episode106"], "good"),
    ("04_abot_worldarena_bad", "abot_worldarena_bad", "model_repros/worldarena_abot_public/outputs/test10_t1_flat/{ep}.mp4", ["episode105"], "bad/common"),
    ("05_wan21_good", "wan21_good", "model_repros/worldarena_wan_public/outputs/test10_t1_flat/{ep}.mp4", ["episode10", "episode102", "episode106"], "good"),
    ("06_wan21_bad", "wan21_bad", "model_repros/worldarena_wan_public/outputs/test10_t1_flat/{ep}.mp4", ["episode105"], "bad/common"),
    ("07_sft_wan_good", "sft_wan_good", "VideoX-Fun/eval_ckpt300_test10/{ep}.mp4", ["episode10", "episode102", "episode106"], "good"),
    ("08_sft_wan_bad", "sft_wan_bad", "VideoX-Fun/eval_ckpt300_test10/{ep}.mp4", ["episode105"], "bad/common"),
    ("09_sft_wan_seedvr_good", "sft_seedvr_good", "VideoX-Fun/eval_ckpt_latest_test1000_seedvr/fixed_scene_task_{ep}.mp4", ["episode10", "episode102", "episode106"], "good"),
    ("10_sft_wan_seedvr_bad", "sft_seedvr_bad", "VideoX-Fun/eval_ckpt_latest_test1000_seedvr/fixed_scene_task_{ep}.mp4", ["episode105"], "bad/common"),
]

for category, prefix, template, episodes, quality in groups:
    for ep in episodes:
        entries.append(
            {
                "category": category,
                "label": f"{prefix}_{ep}",
                "episode": ep,
                "instruction": instructions[ep],
                "quality": quality,
                "source_path": ROOT / template.format(ep=ep),
                "note": "Aligned WorldArena episode/instruction selected to match the other ABot/Wan/SFT/SFT(seedvr) groups.",
            }
        )


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    manifest = []
    missing = []
    for entry in entries:
        source = Path(entry["source_path"])
        exists = source.exists()
        item_dir = OUT / entry["category"] / entry["label"]
        item_dir.mkdir(parents=True, exist_ok=True)
        packaged_video = None
        frames: list[str] = []
        if exists:
            dst = item_dir / source.name
            shutil.copy2(source, dst)
            packaged_video = str(dst.relative_to(OUT))
            frames = extract_frames(dst, item_dir / "frames", 5)
        else:
            missing.append(str(source))
        record = dict(entry)
        record["source_path"] = str(source)
        record["exists"] = exists
        record["packaged_video"] = packaged_video
        record["frames"] = frames
        manifest.append(record)

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Aligned World Model Project Video Pack\n\n"
        "This rebuilt package prioritizes the same WorldArena tasks/instructions across ABot, Wan2.1, VideoX-Fun SFT, and VideoX-Fun SFT(seedvr).\n\n"
        "Aligned good episodes: `episode10`, `episode102`, `episode106`.\n\n"
        "Aligned bad/common episode: `episode105`. It is a weak SFT/SFT(seedvr) case and exists in all four WorldArena groups; Wan may not be weak on this exact episode, so see `MANIFEST.json` for notes.\n\n"
        "ABot-PhysWorld standalone note: only one true standalone local repro output was found (`franka_pick_0_generated.mp4`). The other two entries in `02_abot_physworld_repro` are clearly marked ABot WorldArena HQ auxiliary repro outputs used to provide three good examples.\n\n"
        "Each video includes roughly five extracted frames under its `frames/` directory.\n",
        encoding="utf-8",
    )
    if missing:
        raise SystemExit("Missing sources:\n" + "\n".join(missing))
    print(json.dumps({"output": str(OUT), "entries": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
