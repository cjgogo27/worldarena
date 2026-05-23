import argparse
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


PROMPT = """You are a professional GeoGuessr player.
Analyze this image and produce a concise geographic reasoning chain.
Focus on concrete visual evidence such as road markings, utility poles, driving side, signs, language, vegetation, terrain, architecture, vehicles, and climate.
Return ONLY valid JSON with this schema:
{
  "country": "<full country name>",
  "country_code": "<iso-3166-1 alpha-2 lowercase code or empty string>",
  "city": "<city or empty string>",
  "reasoning_chain": [
    "<one concise evidence-based bullet>",
    "<one concise evidence-based bullet>",
    "<one concise evidence-based bullet>",
    "<one concise evidence-based bullet>"
  ]
}
Rules:
- Each bullet must mention a concrete visual clue and the geographic implication.
- No markdown.
- No extra explanation outside the JSON."""


def load_model(model_path: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    force_single = os.getenv("HF_FORCE_SINGLE_DEVICE", "0") == "1"
    if force_single and torch.cuda.is_available():
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        model = model.to("cuda")
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).eval()
    return model, processor, tokenizer


def predict_image(model, processor, tokenizer, image_path: Path, max_new_tokens: int = 256) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = outputs[:, inputs["input_ids"].shape[-1] :]
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        data = json.loads(text[start : end + 1])
    else:
        data = _recover_partial_json(text)
    chain = data.get("reasoning_chain", [])
    if isinstance(chain, str):
        chain = [line.strip("-* ").strip() for line in chain.splitlines() if line.strip()]
    data["reasoning_chain"] = [str(x).strip() for x in chain if str(x).strip()]
    data["raw_output"] = text
    return data


def _recover_partial_json(text: str) -> dict:
    country = ""
    country_code = ""
    city = ""
    country_match = re.search(r'"country"\s*:\s*"([^"]*)"', text)
    if country_match:
        country = country_match.group(1).strip()
    code_match = re.search(r'"country_code"\s*:\s*"([^"]*)"', text)
    if code_match:
        country_code = code_match.group(1).strip()
    city_match = re.search(r'"city"\s*:\s*"([^"]*)"', text)
    if city_match:
        city = city_match.group(1).strip()

    chain = []
    if '"reasoning_chain"' in text:
        tail = text.split('"reasoning_chain"', 1)[1]
        quoted = re.findall(r'"([^"\n]+)"', tail)
        for item in quoted:
            cleaned = item.strip().strip(",")
            if cleaned and cleaned not in {"country", "country_code", "city", "reasoning_chain"}:
                chain.append(cleaned)
    if not chain:
        lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
        chain = [line for line in lines if len(line) > 20][:4]
    return {
        "country": country,
        "country_code": country_code,
        "city": city,
        "reasoning_chain": chain,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--modulo", type=int, default=1)
    parser.add_argument("--remainder", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.challenge_path).resolve()
    challenge_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if args.modulo > 1:
        challenge_dirs = [p for idx, p in enumerate(challenge_dirs) if idx % args.modulo == args.remainder]
    if args.limit > 0:
        challenge_dirs = challenge_dirs[: args.limit]

    model, processor, tokenizer = load_model(args.model_path)
    records = []

    for challenge_dir in challenge_dirs:
        for round_idx in range(1, 6):
            image_path = challenge_dir / f"{challenge_dir.name}_{round_idx}.png"
            if not image_path.exists():
                continue
            txt_path = challenge_dir / f"candidate_reasoning_chain_georeasoner_{round_idx}.txt"
            json_path = challenge_dir / f"candidate_prediction_georeasoner_{round_idx}.json"
            if txt_path.exists() and not args.force:
                records.append({"challenge": challenge_dir.name, "round": round_idx, "status": "skipped_existing"})
                continue

            pred = predict_image(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image_path=image_path,
                max_new_tokens=args.max_new_tokens,
            )
            lines = pred.get("reasoning_chain", [])
            txt_path.write_text("\n".join([line for line in lines if line]).strip() + "\n", encoding="utf-8")
            json_path.write_text(json.dumps(pred, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            records.append({"challenge": challenge_dir.name, "round": round_idx, "status": "generated"})
            print(f"[{challenge_dir.name}] round {round_idx} generated")

    summary_path = root / "georeasoner_qwen35_generation_summary.json"
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
