#!/usr/bin/env python3
"""Build preference pairs from generated content-style benchmark images."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, CLIPModel, CLIPProcessor


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return s or "style"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pair benchmark from content-style benchmark")
    parser.add_argument("--benchmark_dir", type=str, default="benchmark/bagel_content_style_benchmark")
    parser.add_argument("--style_prompts", type=str, default="resources/styles/style_prompts_40_v1.json")
    parser.add_argument("--output_dir", type=str, default="results/pair_benchmark_build")
    parser.add_argument("--clip_model_dir", type=str, default="models/clip-vit-large-patch14")
    parser.add_argument("--vlm_model_dir", type=str, default="models/Qwen3.5-9B")
    parser.add_argument("--prompt_template_id", type=str, default="style_transfer_40_v1")
    parser.add_argument("--delta_min", type=float, default=0.02)
    parser.add_argument("--vlm_weight", type=float, default=0.7)
    parser.add_argument("--image_batch_size", type=int, default=64)
    parser.add_argument("--max_contents", type=int, default=17)
    return parser.parse_args()


def load_style_prompts(path: Path) -> Dict[str, str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    prompts = obj.get("prompts", {}) if isinstance(obj, dict) else {}
    if not isinstance(prompts, dict):
        raise ValueError(f"Invalid style prompts file: {path}")
    return dict(sorted(prompts.items(), key=lambda x: x[0]))


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
    return {"train": train, "val": val, "test": test}


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


def load_clip_features(
    clip_model_dir: str,
    image_paths: List[str],
    style_prompts: Dict[str, str],
    image_batch_size: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(clip_model_dir, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_dir, local_files_only=True)
    model.eval()

    style_ids = list(style_prompts.keys())
    text_inputs = processor(text=[style_prompts[s] for s in style_ids], return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = _as_feature_tensor(model.get_text_features(**text_inputs))
        text_features = F.normalize(text_features, dim=-1)
    text_map = {s: text_features[i].detach().cpu() for i, s in enumerate(style_ids)}

    image_map: Dict[str, torch.Tensor] = {}
    for i in range(0, len(image_paths), image_batch_size):
        batch = image_paths[i : i + image_batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        image_inputs = processor(images=images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        with torch.no_grad():
            feats = _as_feature_tensor(model.get_image_features(**image_inputs))
            feats = F.normalize(feats, dim=-1)
        for j, p in enumerate(batch):
            image_map[p] = feats[j].detach().cpu()
        for im in images:
            im.close()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return image_map, text_map


def load_vlm_style_embeddings(vlm_model_dir: str, style_prompts: Dict[str, str]) -> Dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(vlm_model_dir, local_files_only=True, trust_remote_code=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")

    base_kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        base_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            vlm_model_dir,
            quantization_config=quant if torch.cuda.is_available() else None,
            **base_kwargs,
        )
    except Exception:
        # Fallback when bitsandbytes or quantized loading is unavailable/incompatible.
        model = AutoModelForCausalLM.from_pretrained(
            vlm_model_dir,
            **base_kwargs,
        )
    model.eval()
    device = next(model.parameters()).device

    embeds: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for style_id, prompt in style_prompts.items():
            tokens = tokenizer(f"Style description: {prompt}", return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model(**tokens, output_hidden_states=True, use_cache=False)
            hid = out.hidden_states[-1].mean(dim=1).float().cpu()
            embeds[style_id] = F.normalize(hid, dim=-1)[0]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeds


def discover_samples(benchmark_dir: Path, style_prompts: Dict[str, str], max_contents: int) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    style_safe_to_id = {safe_name(s): s for s in style_prompts.keys()}
    content_dirs = sorted([p for p in benchmark_dir.iterdir() if p.is_dir() and p.name.startswith("content_")])
    if max_contents > 0:
        content_dirs = content_dirs[:max_contents]

    sample_map: Dict[str, Dict[str, str]] = {}
    for c in content_dirs:
        style_map: Dict[str, str] = {}
        for img in sorted(c.glob("*.png")):
            style_id = style_safe_to_id.get(img.stem)
            if style_id:
                style_map[style_id] = img.as_posix()
        sample_map[c.name] = style_map
    return [c.name for c in content_dirs], sample_map


def main() -> int:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style_prompts = load_style_prompts(Path(args.style_prompts))

    content_ids, sample_map = discover_samples(benchmark_dir, style_prompts, args.max_contents)
    split_map = split_content_ids(content_ids)

    all_image_paths = []
    for c in content_ids:
        all_image_paths.extend(sample_map.get(c, {}).values())

    image_feature_map, text_feature_map = load_clip_features(
        args.clip_model_dir,
        all_image_paths,
        style_prompts,
        args.image_batch_size,
    )
    style_embed_map = load_vlm_style_embeddings(args.vlm_model_dir, style_prompts)

    groups: List[Dict] = []
    failures: List[Dict] = []
    ratio_13 = 0
    fallback_12 = 0
    fallback_11 = 0

    for split, cids in split_map.items():
        for content_id in cids:
            style_to_path = sample_map.get(content_id, {})
            for target_style, target_prompt in style_prompts.items():
                chosen_path = style_to_path.get(target_style)
                if not chosen_path:
                    failures.append(
                        {
                            "split": split,
                            "content_id": content_id,
                            "style_id": target_style,
                            "reason": "chosen_missing",
                        }
                    )
                    continue

                chosen_clip = float(torch.dot(image_feature_map[chosen_path], text_feature_map[target_style]).item())
                chosen_clip = max(0.0, min(1.0, (chosen_clip + 1.0) / 2.0))
                chosen_vlm = 1.0
                chosen_score = args.vlm_weight * chosen_vlm + (1.0 - args.vlm_weight) * chosen_clip

                candidates: List[Dict] = []
                for cand_style, cand_path in style_to_path.items():
                    if cand_style == target_style:
                        continue
                    clip_cos = float(torch.dot(image_feature_map[cand_path], text_feature_map[target_style]).item())
                    clip_score = max(0.0, min(1.0, (clip_cos + 1.0) / 2.0))
                    vlm_cos = float(torch.dot(style_embed_map[target_style], style_embed_map[cand_style]).item())
                    vlm_score = max(0.0, min(1.0, (vlm_cos + 1.0) / 2.0))
                    final = args.vlm_weight * vlm_score + (1.0 - args.vlm_weight) * clip_score
                    candidates.append(
                        {
                            "image_path": cand_path,
                            "score": float(final),
                            "seed": f"{cand_style}_{content_id}",
                            "source_style": cand_style,
                            "clip_score": float(clip_score),
                            "vlm_score": float(vlm_score),
                        }
                    )

                candidates.sort(key=lambda x: x["score"])
                valid_rejected = [c for c in candidates if (chosen_score > c["score"]) and (chosen_score - c["score"] >= args.delta_min)]

                if len(valid_rejected) >= 3:
                    rejected = valid_rejected[:3]
                    ratio_13 += 1
                elif len(valid_rejected) == 2:
                    rejected = valid_rejected[:2]
                    fallback_12 += 1
                elif len(valid_rejected) == 1:
                    rejected = valid_rejected[:1]
                    fallback_11 += 1
                else:
                    failures.append(
                        {
                            "split": split,
                            "content_id": content_id,
                            "style_id": target_style,
                            "reason": "no_valid_rejected",
                            "chosen_score": chosen_score,
                        }
                    )
                    continue

                m = len(rejected)
                group_id = f"{split}|{content_id}|{target_style}|{args.prompt_template_id}"
                groups.append(
                    {
                        "schema_version": "1.0",
                        "split": split,
                        "group_id": group_id,
                        "content_id": content_id,
                        "style_id": target_style,
                        "prompt_template_id": args.prompt_template_id,
                        "prompt": target_prompt,
                        "chosen": {
                            "image_path": chosen_path,
                            "score": float(chosen_score),
                            "seed": f"{target_style}_{content_id}",
                            "source_style": target_style,
                            "clip_score": float(chosen_clip),
                            "vlm_score": float(chosen_vlm),
                        },
                        "rejected_list": rejected,
                        "pair_weight": float(1.0 / m),
                        "group_negatives": m,
                    }
                )

    pref_path = output_dir / "preference_pairs.json"
    pref_path.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")

    fail_path = output_dir / "pair_build_failures.jsonl"
    with fail_path.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_sets = {k: set(v) for k, v in split_map.items()}
    leakage = bool((split_sets["train"] & split_sets["val"]) or (split_sets["train"] & split_sets["test"]) or (split_sets["val"] & split_sets["test"]))

    groups_by_split = {
        "train": sum(1 for g in groups if g["split"] == "train"),
        "val": sum(1 for g in groups if g["split"] == "val"),
        "test": sum(1 for g in groups if g["split"] == "test"),
    }
    expanded_pairs = sum(len(g["rejected_list"]) for g in groups)
    margins = [g["chosen"]["score"] - r["score"] for g in groups for r in g["rejected_list"]]
    avg_margin = float(sum(margins) / len(margins)) if margins else 0.0

    manifest = {
        "schema_version": "1.0",
        "generated_at": now_iso(),
        "benchmark_dir": benchmark_dir.as_posix(),
        "style_prompts": args.style_prompts,
        "clip_model_dir": args.clip_model_dir,
        "vlm_model_dir": args.vlm_model_dir,
        "prompt_template_id": args.prompt_template_id,
        "delta_min": args.delta_min,
        "vlm_weight": args.vlm_weight,
        "split_content_ids": split_map,
        "split_leakage_detected": leakage,
        "style_count": len(style_prompts),
        "content_count": len(content_ids),
        "groups_total": len(groups),
        "groups_by_split": groups_by_split,
        "expanded_pairs": expanded_pairs,
        "ratio_stats": {
            "target_1_3": ratio_13,
            "fallback_1_2": fallback_12,
            "fallback_1_1": fallback_11,
            "failed_groups": len(failures),
        },
        "files": {
            "preference_pairs": pref_path.as_posix(),
            "pair_manifest": (output_dir / "pair_manifest.json").as_posix(),
            "pair_quality_report": (output_dir / "pair_quality_report.txt").as_posix(),
            "pair_build_failures": fail_path.as_posix(),
        },
    }
    manifest_path = output_dir / "pair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "Preference Pair Benchmark Quality Report",
        f"generated_at: {manifest['generated_at']}",
        f"benchmark_dir: {benchmark_dir.as_posix()}",
        f"style_count: {len(style_prompts)}",
        f"content_count: {len(content_ids)}",
        f"groups_total: {len(groups)}",
        f"groups_by_split: {groups_by_split}",
        f"expanded_pairs: {expanded_pairs}",
        f"delta_min: {args.delta_min}",
        f"avg_margin: {avg_margin:.6f}",
        f"split_leakage_detected: {leakage}",
        f"ratio_1_3: {ratio_13}",
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
