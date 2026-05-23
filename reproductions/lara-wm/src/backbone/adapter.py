"""Backbone adapter for LaRA-WM shared local feature extraction.

The fair default is a shared local CLIP image backbone with optional on-disk
feature caching. Heavyweight local VLMs remain available only as explicit
native loads; they are not used in the shared baseline loop because the repo
does not implement a stable, comparable native image-feature API for them.
"""

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportOptionalMemberAccess=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false, reportDeprecated=false, reportImplicitStringConcatenation=false, reportAttributeAccessIssue=false, reportUnknownParameterType=false, reportAny=false, reportImplicitOverride=false

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, CLIPModel

from .config import BackboneConfig

logger = logging.getLogger(__name__)


class BackboneAdapter:
    """Unified adapter for fair shared-backbone experiments."""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        runtime_strategy: str = "native",
        requested_model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        normalize_features: bool = True,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.runtime_strategy = runtime_strategy
        self.requested_model_name = requested_model_name or model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.normalize_features = normalize_features
        self.model: Optional[torch.nn.Module] = None
        self.processor: Optional[AutoProcessor] = None
        self._vision_dim: Optional[int] = None

    @classmethod
    def from_config(cls, config: Optional[BackboneConfig] = None) -> "BackboneAdapter":
        if config is None:
            config = BackboneConfig.from_defaults()

        working_path = config.get_working_path()
        if working_path is None:
            raise RuntimeError("No valid backbone model path found")

        logger.info(
            "Backbone request=%s runtime=%s path=%s strategy=%s",
            config.preferred_model_name,
            config.resolved_model_name,
            working_path,
            config.runtime_strategy,
        )
        return cls(
            model_path=working_path,
            model_name=config.resolved_model_name,
            requested_model_name=config.preferred_model_name,
            device=config.device,
            dtype=config.dtype,
            runtime_strategy=config.runtime_strategy,
            cache_dir=config.cache_dir,
            normalize_features=config.normalize_features,
        )

    @property
    def uses_shared_backbone(self) -> bool:
        return self.model_name == "clip-vit-large-patch14"

    def load(self) -> "BackboneAdapter":
        if self.model is not None:
            return self

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float16)

        if self.uses_shared_backbone:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            clip_model = CLIPModel.from_pretrained(self.model_path, torch_dtype=torch_dtype)
            self.model = clip_model.to(self.device)
            self.model.eval()
            self._vision_dim = int(clip_model.config.projection_dim)
            return self

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        self._vision_dim = self._get_vision_dim()
        logger.warning(
            "Loaded native backbone %s. Shared-baseline experiments should prefer runtime_strategy='cached-shared-clip'.",
            self.model_name,
        )
        return self

    def _get_vision_dim(self) -> int:
        if self.model is None:
            return 768
        config = getattr(self.model, "config", None)
        if config is None:
            return 768
        projection_dim = getattr(config, "projection_dim", None)
        if isinstance(projection_dim, int):
            return projection_dim
        hidden_size = getattr(config, "hidden_size", None)
        if isinstance(hidden_size, int):
            return hidden_size
        vision_config = getattr(config, "vision_config", None)
        if hasattr(vision_config, "hidden_size") and isinstance(vision_config.hidden_size, int):
            return int(vision_config.hidden_size)
        return 768

    def _cache_path(self, cache_key: str) -> Path:
        if self.cache_dir is None:
            raise RuntimeError("Backbone cache is not configured")
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.pt"

    def _load_cached(self, cache_key: Optional[str]) -> Optional[torch.Tensor]:
        if cache_key is None or self.cache_dir is None:
            return None
        cache_path = self._cache_path(cache_key)
        if not cache_path.exists():
            return None
        return torch.load(cache_path, map_location=self.device)

    def _save_cached(self, cache_key: Optional[str], features: torch.Tensor) -> None:
        if cache_key is None or self.cache_dir is None:
            return
        cache_path = self._cache_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features.detach().cpu(), cache_path)

    def _maybe_normalize(self, features: torch.Tensor) -> torch.Tensor:
        if not self.normalize_features:
            return features
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _coerce_feature_tensor(self, outputs: object) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs

        image_embeds = getattr(outputs, "image_embeds", None)
        if isinstance(image_embeds, torch.Tensor):
            return image_embeds

        pooler_output = getattr(outputs, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor):
            return pooler_output

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if isinstance(last_hidden_state, torch.Tensor):
            return last_hidden_state.mean(dim=1)

        raise TypeError(f"Could not coerce backbone outputs of type {type(outputs)!r} into a feature tensor")

    def _extract_shared_backbone_features(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Backbone model is not loaded")

        model = self.model
        if hasattr(model, "get_image_features"):
            outputs = model.get_image_features(**inputs)
            if isinstance(outputs, torch.Tensor):
                return outputs
            coerced = self._coerce_feature_tensor(outputs)
            if isinstance(coerced, torch.Tensor):
                return coerced

        outputs = model(**inputs)
        return self._coerce_feature_tensor(outputs)

    def _image_to_pil(self, image: torch.Tensor | np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            array = image.detach().cpu().float().numpy()
        else:
            array = np.asarray(image)

        if array.ndim != 3:
            raise ValueError(f"Expected 3D image, got shape {array.shape}")
        if array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] not in (1, 3):
            raise ValueError(f"Unsupported channel layout for shape {array.shape}")
        if array.dtype != np.uint8:
            scale = 255.0 if float(array.max(initial=0.0)) <= 1.5 else 1.0
            array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
        if array.shape[-1] == 1:
            array = array[..., 0]
        return Image.fromarray(array).convert("RGB")

    def _prepare_pil_images(
        self, images: torch.Tensor | Sequence[str | Path] | Sequence[Image.Image | np.ndarray | torch.Tensor]
    ) -> tuple[list[Image.Image], tuple[int, ...]]:
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                return [self._image_to_pil(images)], (1,)
            if images.dim() == 4:
                return [self._image_to_pil(img) for img in images], (images.shape[0],)
            if images.dim() == 5:
                batch, time = images.shape[:2]
                flat = images.reshape(batch * time, *images.shape[2:])
                return [self._image_to_pil(img) for img in flat], (batch, time)
            raise ValueError(f"Unsupported image tensor rank: {images.dim()}")

        pil_images: list[Image.Image] = []
        for image in images:
            if isinstance(image, (str, Path)):
                pil_images.append(Image.open(image).convert("RGB"))
            else:
                pil_images.append(self._image_to_pil(image))
        return pil_images, (len(pil_images),)

    def encode_image(
        self, images: torch.Tensor, cache_key: Optional[str] = None
    ) -> torch.Tensor:
        if self.model is None:
            self.load()
        if self.model is None or self.processor is None:
            raise RuntimeError("Backbone is not loaded")
        if not self.uses_shared_backbone:
            raise NotImplementedError(
                "Native Qwen/BAGEL image feature extraction is not part of the fair shared experiment loop. "
                "Use runtime_strategy='cached-shared-clip' or 'shared-clip' for comparable baselines."
            )

        cached = self._load_cached(cache_key)
        if cached is not None:
            return cached.to(self.device)

        pil_images, original_shape = self._prepare_pil_images(images)
        model = self.model
        processor = self.processor
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            features = self._extract_shared_backbone_features(inputs)
        features = self._maybe_normalize(features)

        if len(original_shape) == 2:
            batch, time = original_shape
            features = features.reshape(batch, time, -1)
        self._save_cached(cache_key, features)
        return features

    def encode_image_paths(
        self, image_paths: Sequence[str | Path], cache_key: Optional[str] = None
    ) -> torch.Tensor:
        if self.model is None:
            self.load()
        if self.model is None or self.processor is None:
            raise RuntimeError("Backbone is not loaded")
        if not self.uses_shared_backbone:
            raise NotImplementedError(
                "Path-based image caching is implemented only for the shared CLIP backbone."
            )

        cached = self._load_cached(cache_key)
        if cached is not None:
            return cached.to(self.device)

        pil_images, _ = self._prepare_pil_images(image_paths)
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            features = self._extract_shared_backbone_features(inputs)
        features = self._maybe_normalize(features)
        self._save_cached(cache_key, features)
        return features

    def encode_state(self, observations: torch.Tensor) -> torch.Tensor:
        return observations.to(self.device)

    def forward(
        self, images: torch.Tensor, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(images), self.encode_state(states)

    @property
    def vision_dim(self) -> int:
        if self._vision_dim is None:
            self.load()
        assert self._vision_dim is not None
        return self._vision_dim

    def __repr__(self) -> str:
        return (
            f"BackboneAdapter(requested={self.requested_model_name}, runtime={self.model_name}, "
            f"vision_dim={self._vision_dim}, strategy={self.runtime_strategy})"
        )
