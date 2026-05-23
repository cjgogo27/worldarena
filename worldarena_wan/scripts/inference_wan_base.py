#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ABOT_INFERENCE_ROOT = Path('/data/alice/cjtest/model_repros/ABot-PhysWorld/inference')
if str(ABOT_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(ABOT_INFERENCE_ROOT))

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig



DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

def extract_first_frame(media_path: str):
    from pathlib import Path
    import imageio

    media_path = Path(media_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    if media_path.suffix.lower() in image_extensions:
        image = Image.open(media_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    reader = imageio.get_reader(str(media_path))
    frame = reader.get_data(0)
    reader.close()
    image = Image.fromarray(frame)
    return image


def load_pipeline(device="cuda", enable_compile=False, enable_teacache=False, enable_quantization=False):
    """
    Load WAN pipeline with optional acceleration methods
    
    Args:
        device: Target device (cuda/cpu)
        enable_compile: Enable torch.compile() for DIT (~1.2-1.5x speedup)
        enable_teacache: Enable TeaCache attention caching (~2-3x speedup, requires 'pip install teacache')
        enable_quantization: Enable INT8 quantization (~1.5-2x speedup, quality trade-off)
    """
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                offload_device="cpu",
            ),
        ],
    )
    
    # === 加速方案集成 ===
    
    # 方案1: torch.compile() - DIT 编译加速 (1.2-1.5x)
    if enable_compile:
        print("[✓] 启用 torch.compile() 对 DIT...")
        pipe.dit = torch.compile(pipe.dit, mode='reduce-overhead')
    
    # 方案2: TeaCache - 注意力缓存加速 (2-3x) 
    if enable_teacache:
        try:
            from teacache import TeaCache as TeaCacheModule
            print("[✓] 启用 TeaCache 注意力缓存...")
            pipe = TeaCacheModule(pipe)
        except ImportError:
            print("[⚠] TeaCache 未安装，跳过。运行: pip install teacache")
    
    # 方案3: Flash Attention - 自动启用（torch.cuda.is_available() && flash_attn 已安装）
    # Flash Attention 会自动检测并启用，无需手动配置
    print("[✓] Flash Attention 已自动启用（如果已安装 flash-attn>=2.6.0）")
    
    # 方案4: 量化 - INT8 权重量化 (1.5-2x, 质量有损)
    if enable_quantization:
        try:
            from torch.quantization import quantize_dynamic
            print("[✓] 启用 INT8 动态量化...")
            pipe.dit = quantize_dynamic(
                pipe.dit,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        except Exception as e:
            print(f"[⚠] 量化失败: {e}")
    
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(description="Pure Wan2.1 I2V baseline batch inference")
    parser.add_argument("--jsonl_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--no_tiled", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    
    # === 加速选项 ===
    parser.add_argument("--enable-compile", action="store_true", help="启用 torch.compile()，加速 1.2-1.5x")
    parser.add_argument("--enable-teacache", action="store_true", help="启用 TeaCache，加速 2-3x（需要: pip install teacache）")
    parser.add_argument("--enable-quantization", action="store_true", help="启用 INT8 量化，加速 1.5-2x（质量有损）")
    
    return parser.parse_args()


def main():
    args = parse_args()
    args.tiled = not args.no_tiled

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cuda"
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f if line.strip()]
    if args.num_samples > 0:
        samples = samples[:args.num_samples]

    print(f"Loaded {len(samples)} samples from {args.jsonl_path}")
    pipe = load_pipeline(
        device=device,
        enable_compile=args.enable_compile,
        enable_teacache=args.enable_teacache,
        enable_quantization=args.enable_quantization
    )
    pipe.enable_vram_management()

    if args.lora_path:
        state_dict = torch.load(args.lora_path, map_location="cpu")
        converted_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("base_model.model."):
                key = key[len("base_model.model."):]
            converted_state_dict[key] = value
        pipe.load_lora(
            pipe.dit,
            state_dict=converted_state_dict,
            alpha=args.lora_alpha / args.lora_rank,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sample_idx, sample in enumerate(tqdm(samples, desc="Inference")):
        video_path = Path(sample["video"])
        prompt = sample["prompt"]
        if not video_path.exists():
            results.append({"index": sample_idx, "video": str(video_path), "prompt": prompt, "status": "error: file not found"})
            continue

        path_parts = video_path.parts
        if len(path_parts) >= 2:
            unique_id = f"{path_parts[-2]}_{video_path.stem}"
        else:
            unique_id = f"sample_{sample_idx:04d}_{video_path.stem}"

        try:
            input_image = extract_first_frame(str(video_path))
            if input_image.size != (args.width, args.height):
                input_image = input_image.resize((args.width, args.height), Image.Resampling.LANCZOS)

            video = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                input_image=input_image,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                tiled=args.tiled,
            )

            output_video_path = output_dir / f"{unique_id}_generated.mp4"
            save_video(video, str(output_video_path), fps=args.fps, quality=5)
            results.append({
                "index": sample_idx,
                "video": str(video_path),
                "prompt": prompt,
                "output_video": str(output_video_path),
                "status": "success",
            })
            del video
            torch.cuda.empty_cache()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results.append({"index": sample_idx, "video": str(video_path), "prompt": prompt, "status": f"error: {exc}"})

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\nDone! {success_count}/{len(results)} samples succeeded.")
    print(f"Results saved to: {results_path}")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
