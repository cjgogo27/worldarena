#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import T5Tokenizer

CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOTS = [os.path.dirname(CURRENT_FILE_PATH), os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)), os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))]
for project_root in PROJECT_ROOTS:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
from videox_fun.pipeline import WanI2VPipeline
from videox_fun.utils import register_auto_device_hook, safe_enable_group_offload
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import filter_kwargs, get_image_to_video_latent, save_videos_grid


NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch VideoX-Fun Wan2.1 I2V inference for WorldArena samples")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--video-length", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--lora-weight", type=float, default=0.55)
    parser.add_argument("--gpu-memory-mode", default="none")
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--enable-teacache", action="store_true")
    parser.add_argument("--teacache-threshold", type=float, default=0.20)
    return parser.parse_args()


def resolve_image_path(image_path: str, dataset_root: str | None) -> str:
    p = Path(image_path)
    if p.is_absolute() and p.exists():
        return str(p)
    if dataset_root:
        root = Path(dataset_root)
        cand = root / image_path
        if cand.exists():
            return str(cand)
        cand2 = root / 'validation' / image_path
        if cand2.exists():
            return str(cand2)
        if p.name.startswith('episode'):
            cand3 = root / 'raw' / 'first_frame' / 'fixed_scene_task' / p.name
            if cand3.exists():
                return str(cand3)
    return image_path


def main():
    args = parse_args()
    device = set_multi_gpus_devices(1, 1)
    config = OmegaConf.load("/data/alice/cjtest/VideoX-Fun/config/wan2.1/wan_civitai.yaml")

    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(args.base_model, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.base_model, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(torch.bfloat16)
    tokenizer = T5Tokenizer.from_pretrained(
        os.path.join(args.base_model, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        local_files_only=True,
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.base_model, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.base_model, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(torch.bfloat16).eval()
    scheduler = FlowUniPCMultistepScheduler(
        **filter_kwargs(FlowUniPCMultistepScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    scheduler.config.shift = 1

    pipeline = WanI2VPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )

    if args.gpu_memory_mode != 'none':
        register_auto_device_hook(pipeline.transformer)
        safe_enable_group_offload(pipeline, onload_device=device, offload_device="cpu", offload_type="leaf_level", use_stream=True)
    else:
        pipeline.transformer.to(device)
        pipeline.vae.to(device)
        pipeline.text_encoder.to(device)
        pipeline.clip_image_encoder.to(device)
    pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=torch.bfloat16)
    if args.enable_teacache:
        coeffs = get_teacache_coefficients('wan2.1-i2v-14b-480p')
        if coeffs is not None and hasattr(pipeline.transformer, 'enable_teacache'):
            print(f'Enable TeaCache threshold={args.teacache_threshold}', flush=True)
            pipeline.transformer.enable_teacache(coeffs, args.steps, args.teacache_threshold, num_skip_start_steps=1, offload=False)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    if args.limit > 0:
        entries = entries[args.start:args.start + args.limit]
    else:
        entries = entries[args.start:]

    generator = torch.Generator(device=device).manual_seed(args.seed)

    results = []
    total = len(entries)
    for idx, item in enumerate(entries, start=1):
        prompt = item['prompt']
        image_path = resolve_image_path(item['image'], args.dataset_root)
        output_name = item['output_video']
        print(f"sample_start {idx}/{total} -> {output_name}", flush=True)
        input_video, input_video_mask, clip_image = get_image_to_video_latent(image_path, None, video_length=args.video_length, sample_size=[args.height, args.width])
        with torch.no_grad():
            sample = pipeline(
                prompt,
                num_frames=args.video_length,
                negative_prompt=NEGATIVE_PROMPT,
                height=args.height,
                width=args.width,
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                video=input_video,
                mask_video=input_video_mask,
                clip_image=clip_image,
                shift=args.shift,
            ).videos
        output_path = outdir / output_name
        save_videos_grid(sample, str(output_path), fps=args.fps)
        print(f"sample_done {idx}/{total} -> {output_path}", flush=True)
        results.append({'image': image_path, 'prompt': prompt, 'output_video': str(output_path), 'status': 'success'})

    with open(outdir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    pipeline = unmerge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=torch.bfloat16)


if __name__ == '__main__':
    main()
