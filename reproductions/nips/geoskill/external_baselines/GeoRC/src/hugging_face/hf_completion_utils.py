import os
import base64

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import Gemma3ForConditionalGeneration, MllamaForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, pipeline

model_references = {}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_pipeline(model: str, cache_dir: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model == "google":
        g_pipe = pipeline(
            task="image-text-to-text", # required for image input
            model="google/gemma-3-12b-it",
            device=device,
            torch_dtype=torch.bfloat16, 
            cache_dir=cache_dir
        )
        return g_pipe
def print_utilizations():
    # print(torch.cuda.memory_summary())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}, Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB, Memory used: {torch.cuda.memory_allocated(i) / (1024 ** 2):.2f} MB")
    print(f" is cuda available {torch.cuda.is_available()}")

def get_processor_and_model(model_name: str, cache_dir: str = None, device_id: int = 0):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("Before allocation:")
    print_utilizations()
    if model_name in model_references:
        return model_references[model_name]
    if os.path.isdir(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).eval()
        model_references[model_name] = (model, tokenizer)
        print("After allocation:")
        print_utilizations()
        return model, tokenizer
    if model_name == "google":
        model_id = "google/gemma-3-12b-it"
        try:
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, 
                cache_dir=cache_dir,
                device_map="auto",  # Let transformers handle device placement
                torch_dtype=torch.float16  # Use fp16 for better memory efficiency
            ).eval()
            return model, processor
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    elif model_name == "meta_llm":
        # meta-llama/Llama-2-7b-chat-hf
        #meta-llama/Meta-Llama-3-70B-Instruct
        # meta-llama/Llama-3.1-70B-Instruct
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, 
        cache_dir=cache_dir,
        trust_remote_code=True
        )

        return model, tokenizer
    elif model_name == "meta":
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
        )

        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            cache_dir=cache_dir
        )
        model = model.to(device)

        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        return model, processor
    elif model_name == "qwen":
        model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, 
                                                                   quantization_config=bnb_config,
                                                                   cache_dir=cache_dir).eval()
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
        print("After allocation:")
        print_utilizations()
        return model, processor
    elif model_name == "deepseek_llm":
        model_id = "deepseek-ai/DeepSeek-R1"
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, device_map="cuda")
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_quant_type="nf4",
        # )
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir).eval()
        model = model.to(device)
        return model, tokenizer
    elif model_name == "qwen_llm":
        model_id = "Qwen/Qwen3-4B-Instruct-2507"
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, device_map="cuda")
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir).eval()
        model = model.to(device)
        print("After allocation:")
        print_utilizations()
        return model, tokenizer
    elif model_name == "qwen_tiny_llm":
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir).eval()
        model = model.to(device)
        print("After allocation:")
        print_utilizations()
        model_references[model_name] = (model, tokenizer)
        return model, tokenizer
    elif model_name == "gpt_oss":
        model_id = "openai/gpt-oss-20b"
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, device_map="cuda")
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir).eval()
        model = model.to(device)
        model_references[model_name] = (model, tokenizer)
        return model, tokenizer

def image_and_text_to_text_using_pipeline(image_path: str, prompt: str, pipeline, max_new_tokens: int = 10, model_name: str = "default"):
    image = Image.open(image_path).convert("RGB")
    out = pipeline(
        image,
        text="<start_of_image> "+prompt,
        max_new_tokens=max_new_tokens
    )
    return out[0]['generated_text'][len(out[0]['input_text']):]
def image_and_text_to_text(image_path: str, prompt: str, model, processor, max_new_tokens: int = 10, model_name: str = "default"):
    # print(f"Model name: {model_name}")
    if model_name == "gpt_oss":
        image = encode_image(image_path)
        messages = [
            {"role": "user", "content": f"<image_url>:data:image/png;base64,{image}\n\n{prompt}"}
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        # print("Output tokens from GPT-OSS:", output)
    else:
        image = Image.open(image_path)
        if isinstance(model, Gemma3ForConditionalGeneration):
            image = image.convert('RGB')
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # do_sample=False,         # deterministic, faster than sampling
                # num_beams=1,             # greedy decoding, faster than beam search
            )
    # Only keep the newly generated tokens (ignore the input prompt)
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated_tokens, skip_special_tokens=True)

def text_to_text(prompt: str, model, tokenizer, max_new_tokens: int = 128):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_length = getattr(model.config, "max_position_embeddings", None)
    if max_length is None and hasattr(model.config, "text_config"):
        max_length = getattr(model.config.text_config, "max_position_embeddings", None)
    prompt_len = inputs["input_ids"].shape[-1]
    if max_length is not None and prompt_len >= max_length:
        print("Prompt too long; will be truncated or fail to generate.")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    if "</think>" in result:
        result = result.split("</think>", 1)[-1].lstrip()
    return result
