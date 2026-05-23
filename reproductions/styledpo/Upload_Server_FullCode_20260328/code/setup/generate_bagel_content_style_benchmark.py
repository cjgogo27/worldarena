#!/usr/bin/env python3
"""Generate BAGEL content-style benchmark from content images and style prompts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Avoid OpenBLAS thread-runtime conflicts that can stall long generation jobs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return s or "style"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BAGEL benchmark images for content x styles")
    parser.add_argument("--content_dir", type=str, default="resources/datasets/omnistyle/content")
    parser.add_argument("--style_prompts", type=str, default="resources/styles/style_prompts_40_v1.json")
    parser.add_argument("--style_ref_dir", type=str, default="benchmark/style_benchmark_40x10")
    parser.add_argument("--output_dir", type=str, default="benchmark/bagel_content_style_benchmark")
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT")
    parser.add_argument("--mode", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--content_start", type=int, default=0, help="Start index (inclusive) over sorted content files")
    parser.add_argument("--content_end", type=int, default=0, help="End index (exclusive) over sorted content files; 0 means until end")
    parser.add_argument("--max_contents", type=int, default=17)
    parser.add_argument("--max_styles", type=int, default=0)
    parser.add_argument("--seed_base", type=int, default=20260329)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=3.5)
    parser.add_argument("--cfg_img_scale_retry_step", type=float, default=0.5)
    parser.add_argument("--num_timesteps", type=int, default=8)
    parser.add_argument("--style_ref_index", type=int, default=1, help="1-based index for style reference image selection")
    parser.add_argument(
        "--strict_content_lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject strict content-preservation instruction into every style prompt",
    )
    parser.add_argument(
        "--content_similarity_method",
        type=str,
        default="edge",
        choices=["edge", "clip"],
        help="Similarity metric used for content-preservation gate",
    )
    parser.add_argument(
        "--content_similarity_clip_model_dir",
        type=str,
        default="models/clip-vit-large-patch14",
        help="CLIP model dir used for source-vs-stylized content similarity guard",
    )
    parser.add_argument(
        "--content_similarity_threshold",
        type=float,
        default=0.80,
        help="Minimum CLIP image-image similarity in [0,1] to accept generated image",
    )
    parser.add_argument(
        "--content_similarity_max_retries",
        type=int,
        default=2,
        help="Retry count when generated image fails content similarity threshold",
    )
    parser.add_argument(
        "--save_triplets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-result bundle: content image + style reference image + stylized image",
    )
    parser.add_argument("--triplets_dirname", type=str, default="_triplets")
    parser.add_argument("--use_content_size", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def load_style_prompts(path: Path) -> List[Tuple[str, str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    prompts = obj.get("prompts", {}) if isinstance(obj, dict) else {}
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError(f"Invalid style prompt file: {path}")
    return sorted([(k, v) for k, v in prompts.items()], key=lambda x: x[0])


def list_content_images(content_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = sorted([p for p in content_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return files


def list_image_files(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts])


def lock_style_prompt(style_prompt: str, strict_content_lock: bool) -> str:
    if not strict_content_lock:
        return style_prompt
    return (
        "STYLE TRANSFER ONLY. Preserve content exactly.\n"
        "Mandatory constraints:\n"
        "1) Keep identical objects, object count, identities, positions, poses, proportions, and semantic meaning.\n"
        "2) Keep identical camera viewpoint, perspective, depth ordering, composition, and layout.\n"
        "3) Do NOT add, remove, replace, move, crop, deform, or hallucinate any scene element.\n"
        "4) Allowed edits: style rendering only (palette, texture, brushwork, line quality, lighting mood).\n"
        f"Target style: {style_prompt}\n"
        "Output one stylized image that keeps the original content unchanged."
    )


def build_style_reference_map(style_ref_dir: Path, style_ids: List[str]) -> Dict[str, List[Path]]:
    refs: Dict[str, List[Path]] = {}
    if not style_ref_dir.exists():
        return refs

    for d in sorted([x for x in style_ref_dir.iterdir() if x.is_dir()]):
        images = list_image_files(d)
        if not images:
            continue
        refs[d.name] = images
        refs[safe_name(d.name)] = images

    # Ensure all style ids have lookup keys when names are sanitized.
    for sid in style_ids:
        sid_safe = safe_name(sid)
        if sid not in refs and sid_safe in refs:
            refs[sid] = refs[sid_safe]
    return refs


def pick_style_reference(
    style_id: str,
    style_safe: str,
    style_ref_map: Dict[str, List[Path]],
    style_ref_index: int,
) -> Optional[Path]:
    candidates = style_ref_map.get(style_id) or style_ref_map.get(style_safe) or []
    if not candidates:
        return None
    idx = max(1, style_ref_index) - 1
    idx = idx % len(candidates)
    return candidates[idx]


def save_triplet_bundle(
    triplets_root: Path,
    style_safe: str,
    content_img: Image.Image,
    style_ref_path: Optional[Path],
    stylized_output_path: Path,
    metadata: Dict,
) -> Dict[str, str]:
    bundle_dir = triplets_root / style_safe
    bundle_dir.mkdir(parents=True, exist_ok=True)

    content_copy = bundle_dir / "content.png"
    style_copy = bundle_dir / "style.png"
    stylized_copy = bundle_dir / "stylized.png"
    meta_copy = bundle_dir / "meta.json"

    content_img.save(content_copy)
    if style_ref_path is not None and style_ref_path.exists():
        with Image.open(style_ref_path) as style_src:
            style_src.convert("RGB").save(style_copy)
    if stylized_output_path.exists():
        shutil.copy2(stylized_output_path, stylized_copy)
    meta_copy.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "bundle_dir": bundle_dir.as_posix(),
        "content_image": content_copy.as_posix(),
        "style_image": style_copy.as_posix() if style_copy.exists() else "",
        "stylized_image": stylized_copy.as_posix() if stylized_copy.exists() else "",
        "meta": meta_copy.as_posix(),
    }


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    bagel_main_dir = project_root / "code" / "repos" / "bagel-main"
    import sys

    sys.path.insert(0, str(bagel_main_dir))
    from cli_inferencer import init_model, pil_img2rgb, set_seed  # type: ignore

    content_dir = project_root / args.content_dir
    style_prompts_path = project_root / args.style_prompts
    style_ref_dir = project_root / args.style_ref_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = load_style_prompts(style_prompts_path)
    if args.max_styles > 0:
        styles = styles[: args.max_styles]
    all_content_files = list_content_images(content_dir)
    if not all_content_files:
        raise RuntimeError(f"No content images found in {content_dir}")

    content_start = max(0, args.content_start)
    requested_end = len(all_content_files) if args.content_end <= 0 else min(args.content_end, len(all_content_files))
    if content_start >= requested_end:
        raise RuntimeError(
            f"Invalid content range: start={content_start}, end={requested_end}, total={len(all_content_files)}"
        )
    content_files = all_content_files[content_start:requested_end]
    if args.max_contents > 0:
        content_files = content_files[: args.max_contents]
    if not content_files:
        raise RuntimeError(
            f"No content selected after range slicing: start={content_start}, end={requested_end}, max={args.max_contents}"
        )
    content_end_effective = content_start + len(content_files)

    style_index = [
        {
            "style_id": sid,
            "style_safe": safe_name(sid),
            "prompt": prompt,
        }
        for sid, prompt in styles
    ]
    (output_dir / "style_index.json").write_text(
        json.dumps({"generated_at": now_iso(), "styles": style_index}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    style_ref_map = build_style_reference_map(style_ref_dir, [sid for sid, _ in styles])

    inferencer = init_model(str(project_root / args.model_path), mode=args.mode)

    clip_model = None
    clip_processor = None
    clip_device = None
    if args.content_similarity_threshold > 0 and args.content_similarity_method == "clip":
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
        import torch
        import torch.nn.functional as F

        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = CLIPModel.from_pretrained(
            str(project_root / args.content_similarity_clip_model_dir),
            local_files_only=True,
        ).to(clip_device)
        clip_processor = CLIPProcessor.from_pretrained(
            str(project_root / args.content_similarity_clip_model_dir),
            local_files_only=True,
        )
        clip_model.eval()

        def content_similarity_score(src_img: Image.Image, out_img: Image.Image) -> float:
            assert clip_model is not None and clip_processor is not None and clip_device is not None
            inputs = clip_processor(images=[src_img, out_img], return_tensors="pt")
            inputs = {k: v.to(clip_device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = clip_model.get_image_features(**inputs)
                feats = F.normalize(feats, dim=-1)
                sim = (feats[0] * feats[1]).sum()
                sim = (sim + 1.0) / 2.0
            return float(sim.detach().cpu().item())
    elif args.content_similarity_threshold > 0:

        def _edge_map(img: Image.Image, side: int = 384) -> np.ndarray:
            arr = np.asarray(img.resize((side, side), Image.BICUBIC).convert("L"), dtype=np.float32) / 255.0
            gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
            gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
            grad = gx + gy
            thr = float(np.mean(grad) + 0.5 * np.std(grad))
            edges = (grad >= thr).astype(np.float32)
            return edges

        def content_similarity_score(src_img: Image.Image, out_img: Image.Image) -> float:
            e1 = _edge_map(src_img)
            e2 = _edge_map(out_img)
            inter = float(np.sum(e1 * e2))
            denom = float(np.sum(e1) + np.sum(e2) - inter)
            if denom <= 1e-6:
                return 1.0
            return max(0.0, min(1.0, inter / denom))
    else:

        def content_similarity_score(src_img: Image.Image, out_img: Image.Image) -> float:
            return 1.0

    records: List[Dict] = []
    failures: List[Dict] = []
    generated = 0
    skipped = 0

    for c_idx, content_path in enumerate(content_files, start=content_start + 1):
        content_id = f"content_{c_idx:04d}"
        with Image.open(content_path) as content_src:
            content_img = pil_img2rgb(content_src.convert("RGB"))
        content_out_dir = output_dir / content_id
        content_out_dir.mkdir(parents=True, exist_ok=True)
        triplets_root = content_out_dir / args.triplets_dirname
        if args.save_triplets:
            triplets_root.mkdir(parents=True, exist_ok=True)

        record = {
            "content_id": content_id,
            "content_path": content_path.as_posix(),
            "images": [],
        }

        for s_idx, (style_id, style_prompt) in enumerate(styles, start=1):
            style_safe = safe_name(style_id)
            out_path = content_out_dir / f"{style_safe}.png"
            style_ref_path = pick_style_reference(style_id, style_safe, style_ref_map, args.style_ref_index)
            prompt_used = lock_style_prompt(style_prompt, args.strict_content_lock)

            if args.skip_existing and out_path.exists():
                with Image.open(out_path) as existing_img:
                    existing_sim = content_similarity_score(content_img, existing_img.convert("RGB"))
                if existing_sim >= args.content_similarity_threshold:
                    skipped += 1
                    bundle_info = {}
                    if args.save_triplets:
                        bundle_info = save_triplet_bundle(
                            triplets_root=triplets_root,
                            style_safe=style_safe,
                            content_img=content_img,
                            style_ref_path=style_ref_path,
                            stylized_output_path=out_path,
                            metadata={
                                "content_id": content_id,
                                "content_path": content_path.as_posix(),
                                "style_id": style_id,
                                "style_safe": style_safe,
                                "style_prompt": style_prompt,
                                "prompt_used": prompt_used,
                                "style_ref_path": style_ref_path.as_posix() if style_ref_path else "",
                                "status": "skipped_existing",
                                "content_similarity": existing_sim,
                            },
                        )
                    record["images"].append(
                        {
                            "style_id": style_id,
                            "style_safe": style_safe,
                            "prompt": style_prompt,
                            "prompt_used": prompt_used,
                            "output_path": out_path.as_posix(),
                            "style_ref_path": style_ref_path.as_posix() if style_ref_path else "",
                            "content_similarity": existing_sim,
                            "status": "skipped_existing",
                            **bundle_info,
                        }
                    )
                    continue

            base_seed = args.seed_base + c_idx * 1000 + s_idx
            image_shapes = content_img.size[::-1] if args.use_content_size else (512, 512)
            try:
                accepted = False
                accepted_img = None
                accepted_seed = None
                accepted_cfg_img_scale = None
                accepted_similarity = 0.0
                attempts = max(0, args.content_similarity_max_retries) + 1

                for retry_id in range(attempts):
                    cur_seed = base_seed + retry_id * 100_000
                    cur_cfg_img_scale = args.cfg_img_scale + retry_id * args.cfg_img_scale_retry_step
                    set_seed(cur_seed)
                    result = inferencer(
                        image=content_img,
                        text=prompt_used,
                        understanding_output=False,
                        cfg_text_scale=args.cfg_text_scale,
                        cfg_img_scale=cur_cfg_img_scale,
                        num_timesteps=args.num_timesteps,
                        think=False,
                        image_shapes=image_shapes,
                    )
                    if result.get("image") is None:
                        continue
                    cur_img = result["image"]
                    cur_similarity = content_similarity_score(content_img, cur_img)
                    if cur_similarity >= args.content_similarity_threshold:
                        accepted = True
                        accepted_img = cur_img
                        accepted_seed = cur_seed
                        accepted_cfg_img_scale = cur_cfg_img_scale
                        accepted_similarity = cur_similarity
                        break

                if not accepted or accepted_img is None or accepted_seed is None or accepted_cfg_img_scale is None:
                    raise RuntimeError(
                        f"content_lock_failed(threshold={args.content_similarity_threshold}, attempts={attempts})"
                    )

                accepted_img.save(out_path)
                generated += 1
                bundle_info = {}
                if args.save_triplets:
                    bundle_info = save_triplet_bundle(
                        triplets_root=triplets_root,
                        style_safe=style_safe,
                        content_img=content_img,
                        style_ref_path=style_ref_path,
                        stylized_output_path=out_path,
                        metadata={
                            "content_id": content_id,
                            "content_path": content_path.as_posix(),
                            "style_id": style_id,
                            "style_safe": style_safe,
                            "style_prompt": style_prompt,
                            "prompt_used": prompt_used,
                            "style_ref_path": style_ref_path.as_posix() if style_ref_path else "",
                            "seed": accepted_seed,
                            "cfg_img_scale": accepted_cfg_img_scale,
                            "content_similarity": accepted_similarity,
                            "status": "generated",
                        },
                    )
                record["images"].append(
                    {
                        "style_id": style_id,
                        "style_safe": style_safe,
                        "prompt": style_prompt,
                        "prompt_used": prompt_used,
                        "output_path": out_path.as_posix(),
                        "style_ref_path": style_ref_path.as_posix() if style_ref_path else "",
                        "seed": accepted_seed,
                        "cfg_img_scale": accepted_cfg_img_scale,
                        "content_similarity": accepted_similarity,
                        "status": "generated",
                        **bundle_info,
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "content_id": content_id,
                        "content_path": content_path.as_posix(),
                        "style_id": style_id,
                        "style_safe": style_safe,
                        "prompt_used": prompt_used,
                        "style_ref_path": style_ref_path.as_posix() if style_ref_path else "",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=2),
                    }
                )
                record["images"].append(
                    {
                        "style_id": style_id,
                        "style_safe": style_safe,
                        "prompt": style_prompt,
                        "prompt_used": prompt_used,
                        "output_path": out_path.as_posix(),
                        "style_ref_path": style_ref_path.as_posix() if style_ref_path else "",
                        "seed": base_seed,
                        "status": "failed",
                    }
                )

        records.append(record)

    failures_path = output_dir / "generation_failures.jsonl"
    with failures_path.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": now_iso(),
        "content_dir": content_dir.as_posix(),
        "style_prompts": style_prompts_path.as_posix(),
        "style_ref_dir": style_ref_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "model_path": str(project_root / args.model_path),
        "mode": args.mode,
        "content_start": content_start,
        "content_end_requested": requested_end,
        "content_end_effective": content_end_effective,
        "max_contents": args.max_contents,
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "cfg_img_scale_retry_step": args.cfg_img_scale_retry_step,
        "num_timesteps": args.num_timesteps,
        "strict_content_lock": args.strict_content_lock,
        "content_similarity_method": args.content_similarity_method,
        "content_similarity_clip_model_dir": str(project_root / args.content_similarity_clip_model_dir),
        "content_similarity_threshold": args.content_similarity_threshold,
        "content_similarity_max_retries": args.content_similarity_max_retries,
        "save_triplets": args.save_triplets,
        "triplets_dirname": args.triplets_dirname,
        "style_ref_index": args.style_ref_index,
        "use_content_size": args.use_content_size,
        "content_count": len(content_files),
        "style_count": len(styles),
        "expected_images": len(content_files) * len(styles),
        "generated_images": generated,
        "skipped_images": skipped,
        "failed_images": len(failures),
        "records": records,
        "failures_file": failures_path.as_posix(),
    }
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"WROTE {manifest_path}")
    print(f"WROTE {failures_path}")
    print(
        "SUMMARY "
        f"content={len(content_files)} style={len(styles)} expected={manifest['expected_images']} "
        f"generated={generated} skipped={skipped} failed={len(failures)}"
    )
    return 0 if generated > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
