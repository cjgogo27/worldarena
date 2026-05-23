"""Backbone configuration for LaRA-WM.

Fair experiment policy:
- Qwen3.5-9B remains the preferred requested local backbone.
- Full experiment loops default to one shared local CLIP encoder.
- Shared CLIP features can be cached once and reused across LaRA-WM and all baselines.
- Native Qwen/BAGEL loading is retained only as an explicit opt-in path.
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportAttributeAccessIssue=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, cast

import yaml

BackboneModelName = Literal[
    "qwen3.5-9b",
    "bagel-7b-mot",
    "qwen3-8b",
    "clip-vit-large-patch14",
]
RuntimeStrategy = Literal["auto", "native", "shared-clip", "cached-shared-clip"]

PROJECT_ROOT = Path("/data/alice/cjtest/lara-wm")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "backbone.yaml"
DEFAULT_ASSET_MANIFEST = PROJECT_ROOT / "configs" / "asset_manifest.yaml"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "processed" / "backbone_cache" / "clip-vit-large-patch14"

_DEFAULT_PATHS: dict[BackboneModelName, str] = {
    "qwen3.5-9b": "/data/alice/cjtest/styledpo/Style-DPO-v2.8-fullcode-20260328_extracted/Upload_Server_FullCode_20260328/models/Qwen3.5-9B",
    "bagel-7b-mot": "/data/alice/cjtest/styledpo/Style-DPO-v2.8-fullcode-20260328_extracted/Upload_Server_FullCode_20260328/models/BAGEL-7B-MoT",
    "qwen3-8b": "/data/alice/cjtest/FinalTraj/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B",
    "clip-vit-large-patch14": "/data/alice/cjtest/styledpo/Style-DPO-v2.8-fullcode-20260328_extracted/Upload_Server_FullCode_20260328/models/clip-vit-large-patch14",
}

_FEATURE_DIMS: dict[BackboneModelName, int] = {
    "qwen3.5-9b": 4096,
    "bagel-7b-mot": 3584,
    "qwen3-8b": 4096,
    "clip-vit-large-patch14": 768,
}

_PATH_TO_NAME = {
    "Qwen3.5-9B": "qwen3.5-9b",
    "BAGEL-7B-MoT": "bagel-7b-mot",
    "Qwen3-8B": "qwen3-8b",
    "clip-vit-large-patch14": "clip-vit-large-patch14",
}


def _normalize_name(name: str) -> BackboneModelName:
    lowered = name.strip().lower()
    aliases: dict[str, BackboneModelName] = {
        "qwen3.5-9b": "qwen3.5-9b",
        "qwen3-5-9b": "qwen3.5-9b",
        "bagel-7b-mot": "bagel-7b-mot",
        "qwen3-8b": "qwen3-8b",
        "clip-vit-large-patch14": "clip-vit-large-patch14",
        "clip": "clip-vit-large-patch14",
    }
    if lowered not in aliases:
        raise ValueError(f"Unsupported backbone name: {name}")
    return aliases[lowered]


def _existing_path(path: str | None) -> str | None:
    if path is None:
        return None
    return path if Path(path).exists() else None


def _extract_markdown_source(text: str, heading: str) -> str | None:
    pattern = rf"{re.escape(heading)}\n(?:.*\n)*?- \*\*Source\*\*: (.+)"
    match = re.search(pattern, text)
    if match is None:
        return None
    return match.group(1).strip()


def _as_str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_bool(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


@dataclass
class BackboneConfig:
    preferred_model_name: BackboneModelName = "qwen3.5-9b"
    runtime_strategy: RuntimeStrategy = "cached-shared-clip"
    primary_path: str | None = None
    alternate_path: str | None = None
    fallback_path: str | None = None
    shared_encoder_path: str | None = None
    shared_encoder_name: BackboneModelName = "clip-vit-large-patch14"
    cache_dir: str | None = None
    device: str = "cuda"
    dtype: str = "float16"
    normalize_features: bool = True

    def __post_init__(self) -> None:
        self.preferred_model_name = _normalize_name(self.preferred_model_name)
        self.shared_encoder_name = _normalize_name(self.shared_encoder_name)
        self.primary_path = self.primary_path or _DEFAULT_PATHS["qwen3.5-9b"]
        self.alternate_path = self.alternate_path or _DEFAULT_PATHS["bagel-7b-mot"]
        self.fallback_path = self.fallback_path or _DEFAULT_PATHS["qwen3-8b"]
        self.shared_encoder_path = self.shared_encoder_path or _DEFAULT_PATHS[self.shared_encoder_name]
        if self.cache_dir is None and self.runtime_strategy == "cached-shared-clip":
            self.cache_dir = str(DEFAULT_CACHE_DIR)

    @classmethod
    def from_yaml(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "BackboneConfig":
        data = cast(dict[str, object], yaml.safe_load(Path(config_path).read_text()) or {})
        backbone_cfg = cast(dict[str, object], data.get("backbone", {}))
        paths = cast(dict[str, object], data.get("backbone_models", {}))
        return cls(
            preferred_model_name=cast(
                BackboneModelName,
                _as_str(backbone_cfg.get("preferred_model_name"), "qwen3.5-9b"),
            ),
            runtime_strategy=cast(
                RuntimeStrategy,
                _as_str(backbone_cfg.get("runtime_strategy"), "cached-shared-clip"),
            ),
            shared_encoder_name=cast(
                BackboneModelName,
                _as_str(backbone_cfg.get("shared_encoder_name"), "clip-vit-large-patch14"),
            ),
            cache_dir=_as_optional_str(backbone_cfg.get("cache_dir")),
            device=_as_str(backbone_cfg.get("device"), "cuda"),
            dtype=_as_str(backbone_cfg.get("dtype"), "float16"),
            normalize_features=_as_bool(backbone_cfg.get("normalize_features"), True),
            primary_path=_as_optional_str(paths.get("primary_path")),
            alternate_path=_as_optional_str(paths.get("alternate_path")),
            fallback_path=_as_optional_str(paths.get("fallback_path")),
            shared_encoder_path=_as_optional_str(paths.get("shared_encoder_path")),
        )

    @classmethod
    def from_asset_manifest(cls, manifest_path: str | Path = DEFAULT_ASSET_MANIFEST) -> "BackboneConfig":
        text = Path(manifest_path).read_text()
        return cls(
            primary_path=_extract_markdown_source(text, "### 2.1 Qwen3.5-9B (Multimodal VLM)"),
            alternate_path=_extract_markdown_source(text, "### 2.2 BAGEL-7B-MoT (Multimodal / World Model)"),
            fallback_path=_extract_markdown_source(text, "### 2.3 Qwen3-8B (Text-only LLM)"),
            shared_encoder_path=_extract_markdown_source(text, "### 2.4 CLIP ViT-L/14 (Shared Vision Encoder)"),
        )

    @classmethod
    def from_defaults(cls) -> "BackboneConfig":
        if DEFAULT_CONFIG_PATH.exists():
            return cls.from_yaml(DEFAULT_CONFIG_PATH)
        if DEFAULT_ASSET_MANIFEST.exists():
            return cls.from_asset_manifest(DEFAULT_ASSET_MANIFEST)
        return cls()

    @property
    def model_paths(self) -> dict[BackboneModelName, str]:
        return {
            "qwen3.5-9b": self.primary_path or _DEFAULT_PATHS["qwen3.5-9b"],
            "bagel-7b-mot": self.alternate_path or _DEFAULT_PATHS["bagel-7b-mot"],
            "qwen3-8b": self.fallback_path or _DEFAULT_PATHS["qwen3-8b"],
            "clip-vit-large-patch14": self.shared_encoder_path
            or _DEFAULT_PATHS["clip-vit-large-patch14"],
        }

    @property
    def requested_path(self) -> str:
        return self.model_paths[self.preferred_model_name]

    @property
    def resolved_model_name(self) -> BackboneModelName:
        if self.runtime_strategy in {"shared-clip", "cached-shared-clip"}:
            return self.shared_encoder_name
        if self.runtime_strategy == "auto" and _existing_path(self.shared_encoder_path) is not None:
            return self.shared_encoder_name
        if _existing_path(self.requested_path) is not None:
            return self.preferred_model_name
        for model_name in ("qwen3.5-9b", "bagel-7b-mot", "qwen3-8b"):
            if _existing_path(self.model_paths[model_name]) is not None:
                return model_name
        return self.shared_encoder_name

    @property
    def resolved_path(self) -> str | None:
        if self.resolved_model_name == self.shared_encoder_name:
            shared = _existing_path(self.shared_encoder_path)
            if shared is not None:
                return shared
        requested = _existing_path(self.model_paths[self.resolved_model_name])
        if requested is not None:
            return requested
        for model_name in ("qwen3.5-9b", "bagel-7b-mot", "qwen3-8b", "clip-vit-large-patch14"):
            path = _existing_path(self.model_paths[model_name])
            if path is not None:
                return path
        return None

    @property
    def feature_dim(self) -> int:
        return _FEATURE_DIMS[self.resolved_model_name]

    @property
    def uses_shared_backbone(self) -> bool:
        return self.resolved_model_name == self.shared_encoder_name

    def get_primary(self) -> str | None:
        return self.primary_path

    def get_alternate(self) -> str | None:
        return self.alternate_path

    def get_fallback(self) -> str | None:
        return self.fallback_path

    def get_working_path(self) -> str | None:
        return self.resolved_path

    def infer_model_name_from_path(self, path: str) -> BackboneModelName:
        name = _PATH_TO_NAME.get(Path(path).name)
        if name is None:
            raise ValueError(f"Cannot infer backbone model name from path: {path}")
        return cast(BackboneModelName, name)
