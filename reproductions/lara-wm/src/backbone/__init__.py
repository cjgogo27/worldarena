"""Backbone adapter module for LaRA-WM.

Provides a fair shared-backbone interface supporting:
- Qwen3.5-9B as the preferred requested local backbone
- shared local CLIP ViT-L/14 for comparable experiment-time feature extraction
- optional native Qwen/BAGEL loading outside the shared baseline loop
"""

from .adapter import BackboneAdapter
from .config import BackboneConfig

__all__ = ["BackboneAdapter", "BackboneConfig"]
