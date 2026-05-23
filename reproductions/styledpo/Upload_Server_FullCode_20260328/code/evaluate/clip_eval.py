#!/usr/bin/env python3
"""Compute CLIP metrics for chosen images in preference-pair benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP evaluation for preference pairs")
    parser.add_argument("--pairs_file", type=str, default="results/pair_benchmark_build/preference_pairs.json")
    parser.add_argument("--clip_model_dir", type=str, default="models/clip-vit-large-patch14")
    parser.add_argument("--output_csv", type=str, default="results/evaluations/clip_scores.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.pairs_file, "r", encoding="utf-8") as f:
        groups = json.load(f)

    rows: List[Dict] = []
    for g in groups:
        rows.append(
            {
                "group_id": g.get("group_id", ""),
                "split": g.get("split", ""),
                "style_id": g.get("style_id", ""),
                "prompt": g.get("prompt", ""),
                "image_path": g.get("chosen", {}).get("image_path", ""),
            }
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.clip_model_dir, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model_dir, local_files_only=True)
    model.eval()

    scores: List[float] = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        texts = [r["prompt"] for r in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = _as_feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            text_features = _as_feature_tensor(model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ))
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            sim = (image_features * text_features).sum(dim=-1)
            sim = ((sim + 1.0) / 2.0).detach().float().cpu().tolist()
        scores.extend(sim)
        for img in images:
            img.close()

    for i, score in enumerate(scores):
        rows[i]["clip_score"] = float(score)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group_id", "split", "style_id", "prompt", "image_path", "clip_score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    agg = {
        "generated_at": now_iso(),
        "pairs_file": args.pairs_file,
        "count": len(rows),
        "clip_mean": float(sum(scores) / len(scores) if scores else 0.0),
        "clip_min": float(min(scores) if scores else 0.0),
        "clip_max": float(max(scores) if scores else 0.0),
        "output_csv": output_csv.as_posix(),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"WROTE {output_csv}")
    print(f"WROTE {summary_path}")
    print(f"CLIP_MEAN {agg['clip_mean']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
