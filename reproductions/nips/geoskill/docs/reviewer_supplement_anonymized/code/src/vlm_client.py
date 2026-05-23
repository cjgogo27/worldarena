import base64
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import requests


@dataclass
class VLMConfig:
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 1500
    max_image_side: int = 1024
    retries: int = 1
    backoff_seconds: float = 0.0
    request_timeout_seconds: float = 60.0
    enable_thinking: bool | None = None


class VLMClient:
    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self.base_url = self._normalize_base_url(config.base_url)
        self._session = requests.Session()
        # Direct session bypasses env proxies, used as fallback when local proxy is unavailable.
        self._session_direct = requests.Session()
        self._session_direct.trust_env = False

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        normalized = base_url.strip().rstrip("/")
        if normalized.endswith("/chat/completions"):
            normalized = normalized[: -len("/chat/completions")]
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

    @staticmethod
    def _is_auth_like_error(msg: str) -> bool:
        return any(token in msg for token in ["unauthorized", "invalid token", "401"])

    @staticmethod
    def _is_proxy_like_error(msg: str) -> bool:
        return any(
            token in msg
            for token in [
                "proxyerror",
                "unable to connect to proxy",
                "connection refused",
                "failed to establish a new connection",
            ]
        )

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
        max_attempts = max(1, int(self.config.retries))
        use_direct_session = False

        for attempt in range(1, max_attempts + 1):
            try:
                payload = {
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                }
                if self.config.enable_thinking is not None:
                    payload["enable_thinking"] = bool(self.config.enable_thinking)
                session = self._session_direct if use_direct_session else self._session
                resp = session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.config.request_timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
                message = data.get("choices", [{}])[0].get("message", {})
                text = message.get("content")

                # Some providers return content blocks as list[{type,text}].
                if isinstance(text, list):
                    blocks: list[str] = []
                    for block in text:
                        if isinstance(block, dict):
                            block_text = block.get("text")
                            if block_text:
                                blocks.append(str(block_text))
                        elif block:
                            blocks.append(str(block))
                    text = "\n".join(blocks).strip()

                if text is None:
                    text = ""
                text = str(text)

                # Fallback for reasoning-enabled models that leave content empty.
                if not text.strip():
                    reasoning_text = message.get("reasoning_content")
                    if reasoning_text is not None:
                        text = str(reasoning_text)

                return text
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts and self.config.backoff_seconds > 0:
                    msg = str(exc).lower()
                    backoff = float(self.config.backoff_seconds) * attempt
                    # Some upstream gateways transiently return auth-like responses under load.
                    # Back off more aggressively so short-lived lockouts can clear.
                    if self._is_auth_like_error(msg):
                        backoff = max(backoff, 12.0 * attempt)
                    # If a local/system proxy is down, retry with direct connection and longer cool-down.
                    if self._is_proxy_like_error(msg):
                        use_direct_session = True
                        backoff = max(backoff, 8.0 * attempt)
                    time.sleep(backoff)
                else:
                    msg = str(exc).lower()
                    if self._is_proxy_like_error(msg):
                        use_direct_session = True
                continue

        raise RuntimeError(f"VLM query failed after retries: {last_error}")

    @staticmethod
    def extract_json(text: str) -> dict[str, Any]:
        """Extract JSON object from model response, handling markdown code blocks."""
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try extracting from markdown code block
        md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if md_match:
            try:
                return json.loads(md_match.group(1).strip())
            except Exception:
                pass

        # Try finding outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass

        raise ValueError(f"Could not parse JSON from model response: {text[:200]}")
