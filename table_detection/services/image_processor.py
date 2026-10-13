"""Aspect-ratio preserving resize and optional image sharpening."""
from __future__ import annotations
from typing import Literal
from dataclasses import dataclass, asdict

import cv2
import numpy as np

type SharpenMethod = Literal["none", "unsharp", "clahe_unsharp"]

@dataclass(frozen=True)
class PreprocessConfig:
    enabled_resize: bool = False
    target_long_side: int = 2048
    allow_upscale: bool = False
    sharpen_method: SharpenMethod = "none"
    sharpen_amount: float = 1.0
    sharpen_sigma: float = 1.2
    clahe_clip_limit: float = 2.0

    def __post_init__(self):
        if self.target_long_side < 16:
            raise ValueError('target_long_side must be >= 16')
        if self.sharpen_method not in ('none', 'unsharp', 'clahe_unsharp'):
            raise ValueError('Invalid sharpen_method')
        if not 0 <= self.sharpen_amount <= 5:
            raise ValueError('sharpen_amount must be in [0, 5]')
        if self.sharpen_sigma <= 0 or self.clahe_clip_limit <= 0:
            raise ValueError('sharpen_sigma and clahe_clip_limit must be positive')

class ImagePreprocessor:
    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()


    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        cfg = self.config
        height, width = image.shape[:2]
        scale = cfg.target_long_side / max(height, width)

        if not cfg.allow_upscale:
            scale = min(1.0, scale)
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        if (new_w, new_h) != (width, height):
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        return image


    def _unsharp(self, image: np.ndarray) -> np.ndarray:
        cfg = self.config
        blurred = cv2.GaussianBlur(image, (0, 0), cfg.sharpen_sigma)
        image = cv2.addWeighted(image, 1 + cfg.sharpen_amount, blurred, -cfg.sharpen_amount, 0)
        return image


    def _clahe_unsharp(self, image: np.ndarray) -> np.ndarray:
        cfg = self.config
        if image.ndim == 2:
            image = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=(8, 8)).apply(image)
        else:
            conversion = cv2.COLOR_RGB2LAB if image.shape[2] == 3 else cv2.COLOR_RGBA2RGB
            rgb = image if image.shape[2] == 3 else cv2.cvtColor(image, conversion)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image


    def process(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3) or image.size == 0:
            raise ValueError('Expected non-empty uint8 image')
        height, width = image.shape[:2]
        cfg = self.config

        result = image.copy()
        if cfg.enabled_resize:
            result = self._resize_image(result)

        if cfg.sharpen_method != 'none':
            if cfg.sharpen_method == 'clahe_unsharp':
                result = self._clahe_unsharp(result)
            if cfg.sharpen_amount > 0:
                result = self._unsharp(result)
        out_h, out_w = result.shape[:2]
        metadata = {
            'original': {'width': width, 'height': height},
            'processed': {'width': out_w, 'height': out_h},
            'scale_x': out_w / width,
            'scale_y': out_h / height,
            'config': asdict(cfg),
        }
        return result, metadata