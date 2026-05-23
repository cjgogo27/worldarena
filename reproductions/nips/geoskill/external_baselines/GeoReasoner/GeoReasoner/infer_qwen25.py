import argparse
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


DEFAULT_PROMPT = """You are GeoReasoner, a geolocation specialist.
Analyze the image carefully and reason step by step about where it was taken.
Focus on geographic evidence such as road markings, driving side, utility poles, signs, language, vegetation, architecture, terrain, vehicles, climate, and other location clues.
Return ONLY valid JSON with this schema:
{
  "country": "<full country name>",
  "country_code": "<iso-3166-1 alpha-2 lowercase code>",
  "city": "<city or empty string>",
  "reasoning": "<concise paragraph of reasoning>",
  "reasoning_chain": [
    "<bullet 1>",
    "<bullet 2>",
    "<bullet 3>"
  ]
}
Do not wrap the JSON in markdown."""


def _load_model(model_name: str, cache_dir: str | None, load_in_4bit: bool):
    kwargs = {
        "cache_dir": cache_dir,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, use_fast=True)
    return model, processor


def _decode_response(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return JSON. Raw output: {raw_text[:500]}")
    data = json.loads(match.group(0))
    chain = data.get("reasoning_chain", [])
    if isinstance(chain, str):
        chain = [line.strip("-* ").strip() for line in chain.splitlines() if line.strip()]
    data["reasoning_chain"] = [str(item).strip() for item in chain if str(item).strip()]
    return data


def predict_single(
    image_path: str,
    model,
    processor,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 512,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_tokens = generated[:, inputs["input_ids"].shape[-1] :]
    raw_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
    prediction = _decode_response(raw_text)
    prediction["image_path"] = str(Path(image_path).resolve())
    prediction["raw_output"] = raw_text
    return prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=os.environ.get("GEOREASONER_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    model, processor = _load_model(
        model_name=args.model,
        cache_dir=args.cache_dir,
        load_in_4bit=not args.no_4bit,
    )
    prediction = predict_single(
        image_path=args.image,
        model=model,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(prediction, ensure_ascii=False))


if __name__ == "__main__":
    main()
