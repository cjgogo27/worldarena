from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path("/data/alice/cjtest")
SRC = ROOT / "world_model_video_pack_aligned"
OUT = ROOT / "world_model_video_pack_aligned_parts"

PARTS = {
    "part1_gigaworld_abot_repro.zip": ["01_gigaworld", "02_abot_physworld_repro", "README.md", "MANIFEST.json"],
    "part2_abot_wan_worldarena.zip": ["03_abot_worldarena_good", "04_abot_worldarena_bad", "05_wan21_good", "06_wan21_bad", "README.md", "MANIFEST.json"],
    "part3_sft_raw_seedvr.zip": ["07_sft_wan_good", "08_sft_wan_bad", "09_sft_wan_seedvr_good", "10_sft_wan_seedvr_bad", "README.md", "MANIFEST.json"],
}


def add_path(zf: zipfile.ZipFile, path: Path) -> None:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                zf.write(child, child.relative_to(ROOT))
    else:
        zf.write(path, path.relative_to(ROOT))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name, rels in PARTS.items():
        target = OUT / name
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel in rels:
                add_path(zf, SRC / rel)
        print(target)


if __name__ == "__main__":
    main()
