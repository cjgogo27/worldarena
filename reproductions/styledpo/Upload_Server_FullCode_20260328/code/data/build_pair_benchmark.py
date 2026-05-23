#!/usr/bin/env python3
"""Build preference-pair benchmark artifacts following protocol v1."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPModel,
    CLIPProcessor,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _as_feature_tensor(obj: torch.Tensor) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "text_embeds") and isinstance(obj.text_embeds, torch.Tensor):
        return obj.text_embeds
    if hasattr(obj, "image_embeds") and isinstance(obj.image_embeds, torch.Tensor):
        return obj.image_embeds
    if hasattr(obj, "pooler_output") and isinstance(obj.pooler_output, torch.Tensor):
        return obj.pooler_output
    if hasattr(obj, "last_hidden_state") and isinstance(obj.last_hidden_state, torch.Tensor):
        return obj.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unsupported feature output type: {type(obj)}")


@dataclass(frozen=True)
class Sample:
    content_id: str
    style_id: str
    image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preference-pair benchmark")
    parser.add_argument("--benchmark_dir", type=str, default="benchmark/style_benchmark_40x10")
    parser.add_argument("--output_dir", type=str, default="results/pair_benchmark_build")
    parser.add_argument("--clip_model_dir", type=str, default="models/clip-vit-large-patch14")
    parser.add_argument("--vlm_model_dir", type=str, default="models/Qwen3.5-9B")
    parser.add_argument("--prompt_template_id", type=str, default="style_transfer_v1")
    parser.add_argument("--delta_min", type=float, default=0.02)
    parser.add_argument("--vlm_weight", type=float, default=0.7)
    parser.add_argument("--image_batch_size", type=int, default=32)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_content_id(file_name: str) -> str:
    stem = Path(file_name).stem
    match = re.search(r"(?i)image[_-]?(\d+)", stem)
    if match:
        return str(int(match.group(1)))
    match = re.search(r"(\d+)", stem)
    if match:
        return str(int(match.group(1)))
    return stem.lower()


def discover_samples(benchmark_dir: Path) -> Tuple[List[Sample], List[str], List[str]]:
    samples: List[Sample] = []
    styles: List[str] = []
    contents: set[str] = set()

    for style_dir in sorted([p for p in benchmark_dir.iterdir() if p.is_dir()]):
        styles.append(style_dir.name)
        for image_path in sorted(style_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            content_id = extract_content_id(image_path.name)
            contents.add(content_id)
            samples.append(
                Sample(
                    content_id=content_id,
                    style_id=style_dir.name,
                    image_path=image_path.as_posix(),
                )
            )

    def _content_sort_key(x: str) -> Tuple[int, str]:
        return (0, f"{int(x):08d}") if x.isdigit() else (1, x)

    return samples, sorted(styles), sorted(contents, key=_content_sort_key)


def split_content_ids(content_ids: List[str]) -> Dict[str, List[str]]:
    n = len(content_ids)
    n_train = max(1, int(round(n * 0.8)))
    n_val = max(1, int(round(n * 0.1))) if n >= 3 else max(0, n - n_train)
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    train = content_ids[:n_train]
    val = content_ids[n_train : n_train + n_val]
    test = content_ids[n_train + n_val :]
    if not test and content_ids:
        test = [content_ids[-1]]
        if test[0] in train:
            train = train[:-1]
        elif test[0] in val:
            val = val[:-1]
    return {"train": train, "val": val, "test": test}


def load_clip_features(
    clip_model_dir: str,
    image_paths: List[str],
    styles: List[str],
    image_batch_size: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(clip_model_dir, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_dir, local_files_only=True)
    model.eval()

    prompts = [f"in {s} style" for s in styles]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = _as_feature_tensor(model.get_text_features(**text_inputs))
        text_features = F.normalize(text_features, dim=-1)
    text_feature_map = {style: text_features[i].detach().cpu() for i, style in enumerate(styles)}

    image_feature_map: Dict[str, torch.Tensor] = {}
    for i in range(0, len(image_paths), image_batch_size):
        batch_paths = image_paths[i : i + image_batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        image_inputs = processor(images=images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        with torch.no_grad():
            image_features = _as_feature_tensor(model.get_image_features(**image_inputs))
            image_features = F.normalize(image_features, dim=-1)
        for j, p in enumerate(batch_paths):
            image_feature_map[p] = image_features[j].detach().cpu()
        for img in images:
            img.close()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return image_feature_map, text_feature_map


def load_vlm_style_embeddings(vlm_model_dir: str, styles: List[str]) -> Dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(vlm_model_dir, local_files_only=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        vlm_model_dir,
        local_files_only=True,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quant if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    device = next(model.parameters()).device

    style_embeds: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for style in styles:
            prompt = f"Style descriptor: {style}."
            tokens = tokenizer(prompt, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = model(**tokens, output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states[-1].mean(dim=1).float().cpu()
            hidden = F.normalize(hidden, dim=-1)
            style_embeds[style] = hidden[0]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return style_embeds


def clip_score(
    image_feature_map: Dict[str, torch.Tensor],
    text_feature_map: Dict[str, torch.Tensor],
    image_path: str,
    target_style: str,
) -> float:
    image_f = image_feature_map[image_path]
    text_f = text_feature_map[target_style]
    cosine = float(torch.dot(image_f, text_f).item())
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def vlm_style_score(
    style_embed_map: Dict[str, torch.Tensor],
    target_style: str,
    candidate_style: str,
) -> float:
    a = style_embed_map[target_style]
    b = style_embed_map[candidate_style]
    cosine = float(torch.dot(a, b).item())
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def iter_group_keys(split_map: Dict[str, List[str]], styles: List[str]) -> Iterable[Tuple[str, str, str]]:
    for split_name, content_ids in split_map.items():
        for content_id in content_ids:
            for style_id in styles:
                yield split_name, content_id, style_id


def main() -> int:
    args = parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, styles, content_ids = discover_samples(benchmark_dir)
    sample_map: Dict[Tuple[str, str], Sample] = {(s.content_id, s.style_id): s for s in samples}
    split_map = split_content_ids(content_ids)

    all_image_paths = [s.image_path for s in samples]
    image_feature_map, text_feature_map = load_clip_features(
        args.clip_model_dir,
        all_image_paths,
        styles,
        args.image_batch_size,
    )
    style_embed_map = load_vlm_style_embeddings(args.vlm_model_dir, styles)

    groups: List[Dict] = []
    failures: List[Dict] = []
    fallback_12 = 0
    fallback_11 = 0
    target_13 = 0

    for split_name, content_id, target_style in iter_group_keys(split_map, styles):
        candidates: List[Dict] = []
        for candidate_style in styles:
            sample = sample_map.get((content_id, candidate_style))
            if sample is None:
                continue
            c_score = clip_score(image_feature_map, text_feature_map, sample.image_path, target_style)
            v_score = vlm_style_score(style_embed_map, target_style, candidate_style)
            final = args.vlm_weight * v_score + (1.0 - args.vlm_weight) * c_score
            candidates.append(
                {
                    "image_path": sample.image_path,
                    "score": float(final),
                    "seed": f"{candidate_style}_{content_id}",
                    "source_style": candidate_style,
                    "clip_score": float(c_score),
                    "vlm_score": float(v_score),
                }
            )

        if len(candidates) < 2:
            failures.append(
                {
                    "split": split_name,
                    "content_id": content_id,
                    "style_id": target_style,
                    "reason": "candidate_count_lt_2",
                    "candidate_count": len(candidates),
                }
            )
            continue

        candidates.sort(key=lambda x: x["score"], reverse=True)
        chosen = candidates[0]

        valid_rejected = [
            c
            for c in reversed(candidates[1:])
            if chosen["score"] > c["score"] and (chosen["score"] - c["score"]) >= args.delta_min
        ]

        if len(valid_rejected) >= 3:
            rejected_list = valid_rejected[:3]
            target_13 += 1
        elif len(valid_rejected) == 2:
            rejected_list = valid_rejected[:2]
            fallback_12 += 1
        elif len(valid_rejected) == 1:
            rejected_list = valid_rejected[:1]
            fallback_11 += 1
        else:
            failures.append(
                {
                    "split": split_name,
                    "content_id": content_id,
                    "style_id": target_style,
                    "reason": "no_valid_rejected_after_delta",
                    "chosen_score": chosen["score"],
                    "delta_min": args.delta_min,
                }
            )
            continue

        m = len(rejected_list)
        prompt = f"Transform input image into {target_style} style."
        group_id = f"{split_name}|{content_id}|{target_style}|{args.prompt_template_id}"
        groups.append(
            {
                "schema_version": "1.0",
                "split": split_name,
                "group_id": group_id,
                "content_id": content_id,
                "style_id": target_style,
                "prompt_template_id": args.prompt_template_id,
                "prompt": prompt,
                "chosen": chosen,
                "rejected_list": rejected_list,
                "pair_weight": float(1.0 / m),
                "group_negatives": m,
            }
        )

    split_sets = {k: set(v) for k, v in split_map.items()}
    leakage = bool((split_sets["train"] & split_sets["val"]) or (split_sets["train"] & split_sets["test"]) or (split_sets["val"] & split_sets["test"]))

    groups_by_split = {
        "train": sum(1 for g in groups if g["split"] == "train"),
        "val": sum(1 for g in groups if g["split"] == "val"),
        "test": sum(1 for g in groups if g["split"] == "test"),
    }
    expanded_pairs = sum(len(g["rejected_list"]) for g in groups)
    avg_margin = 0.0
    margin_count = 0
    for g in groups:
        c = g["chosen"]["score"]
        for r in g["rejected_list"]:
            avg_margin += c - r["score"]
            margin_count += 1
    avg_margin = avg_margin / margin_count if margin_count else 0.0

    pref_path = output_dir / "preference_pairs.json"
    pref_path.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")

    fail_path = output_dir / "pair_build_failures.jsonl"
    with fail_path.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "benchmark_dir": benchmark_dir.as_posix(),
        "clip_model_dir": args.clip_model_dir,
        "vlm_model_dir": args.vlm_model_dir,
        "prompt_template_id": args.prompt_template_id,
        "delta_min": args.delta_min,
        "vlm_weight": args.vlm_weight,
        "split_content_ids": split_map,
        "split_leakage_detected": leakage,
        "style_count": len(styles),
        "content_count": len(content_ids),
        "groups_total": len(groups),
        "groups_by_split": groups_by_split,
        "expanded_pairs": expanded_pairs,
        "ratio_stats": {
            "target_1_3": target_13,
            "fallback_1_2": fallback_12,
            "fallback_1_1": fallback_11,
            "failed_groups": len(failures),
        },
        "files": {
            "preference_pairs": pref_path.as_posix(),
            "pair_build_failures": fail_path.as_posix(),
            "pair_quality_report": (output_dir / "pair_quality_report.txt").as_posix(),
        },
    }
    manifest_path = output_dir / "pair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "Preference Pair Benchmark Quality Report",
        f"generated_at: {manifest['generated_at']}",
        f"benchmark_dir: {benchmark_dir.as_posix()}",
        f"style_count: {len(styles)}",
        f"content_count: {len(content_ids)}",
        f"groups_total: {len(groups)}",
        f"groups_by_split: {groups_by_split}",
        f"expanded_pairs: {expanded_pairs}",
        f"delta_min: {args.delta_min}",
        f"avg_margin: {avg_margin:.6f}",
        f"split_leakage_detected: {leakage}",
        f"ratio_1_3: {target_13}",
        f"fallback_1_2: {fallback_12}",
        f"fallback_1_1: {fallback_11}",
        f"failed_groups: {len(failures)}",
    ]
    report_path = output_dir / "pair_quality_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"WROTE {pref_path}")
    print(f"WROTE {manifest_path}")
    print(f"WROTE {report_path}")
    print(f"WROTE {fail_path}")
    print(f"GROUPS {len(groups)} EXPANDED_PAIRS {expanded_pairs}")
    return 0 if len(groups) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
