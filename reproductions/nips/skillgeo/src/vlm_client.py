import base64

import io

import json

import re

import time

from dataclasses import dataclass

from pathlib import Path

from typing import Any



from openai import OpenAI

from PIL import Image





@dataclass

class VLMConfig:

    base_url: str

    api_key: str

    model: str

    max_tokens: int = 1500

    max_image_side: int = 1024

    retries: int = 3

    backoff_seconds: float = 1.0

    request_timeout_seconds: float = 45.0





class VLMClient:

    def __init__(self, config: VLMConfig) -> None:

        self.config = config

        self.client = OpenAI(base_url=self._normalize_base_url(config.base_url), api_key=config.api_key)



    @staticmethod

    def _normalize_base_url(base_url: str) -> str:

        normalized = base_url.strip().rstrip("/")

        if normalized.endswith("/chat"):

            normalized = normalized[: -len("/chat")]

        return normalized



    def _prepare_image_data_uri(self, image_path: str | Path) -> str:

        img = Image.open(image_path).convert("RGB")

        max_side = max(img.size)

        if max_side > self.config.max_image_side:

            scale = self.config.max_image_side / max_side

            img = img.resize(

                (int(img.width * scale), int(img.height * scale)),

                Image.Resampling.LANCZOS,

            )

        buffer = io.BytesIO()

        img.save(buffer, format="JPEG", quality=88, optimize=True)

        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/jpeg;base64,{b64}"



    def query(

        self,

        image_path: str | Path | None = None,

        system_prompt: str = "",

        user_prompt: str = "",

        temperature: float = 0.2,

    ) -> str:

        """Query VLM with optional image. If image_path is None, text-only query."""

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

        if image_path is not None:

            image_data_uri = self._prepare_image_data_uri(image_path)

            content.append({"type": "image_url", "image_url": {"url": image_data_uri}})



        last_error: Exception | None = None

        for attempt in range(self.config.retries):

            try:

                response = self.client.chat.completions.create(

                    model=self.config.model,

                    max_tokens=self.config.max_tokens,

                    temperature=temperature,

                    timeout=self.config.request_timeout_seconds,

                    messages=[

                        {"role": "system", "content": system_prompt},

                        {"role": "user", "content": content},

                    ],

                )

                text = response.choices[0].message.content

                return "" if text is None else str(text)

            except Exception as exc:

                last_error = exc

                if attempt < self.config.retries - 1:

                    time.sleep(self.config.backoff_seconds * (2**attempt))

                continue



        raise RuntimeError(f"VLM query failed after retries: {last_error}")



    @staticmethod

    def extract_json(text: str) -> dict[str, Any]:

        """Extract JSON object from model response, handling markdown code blocks."""

        text = text.strip()





        try:

            return json.loads(text)

        except Exception:

            pass





        md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)

        if md_match:

            try:

                return json.loads(md_match.group(1).strip())

            except Exception:

                pass





        start = text.find("{")

        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:

            try:

                return json.loads(text[start : end + 1])

            except Exception:

                pass



        raise ValueError(f"Could not parse JSON from model response: {text[:200]}")
