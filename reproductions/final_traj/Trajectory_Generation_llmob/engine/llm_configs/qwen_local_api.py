from __future__ import annotations

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.llm_configs.base_gpt_api import BaseGPTAPI


QWEN_DEFAULT_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/data/alice/cjtest/FinalTraj/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B",
)


class QwenLocalGPTAPI(BaseGPTAPI):
    """Local Qwen model via HuggingFace AutoModelForCausalLM (no API key needed)."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or QWEN_DEFAULT_PATH
        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _build_prompt(self, messages: list[dict]) -> str:
        """Convert OpenAI-style message list to a single prompt string.

        Qwen3-8B uses a chat template; apply it if available, otherwise
        fall back to concatenation.
        """
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return text

    def completion(self, messages: list[dict]) -> dict:
        prompt = self._build_prompt(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                top_p=1.0,
                top_k=50,
                temperature=0.8,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[0][prompt_len:]
        ans_output = self.tokenizer.decode(generated, skip_special_tokens=True)

        return {
            "choices": [
                {
                    "message": {"content": ans_output},
                }
            ]
        }

    def get_choice_text(self, rsp: dict) -> str:
        return rsp["choices"][0]["message"]["content"]
