from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from table_detection.schemas.geometry import Line, line_set
from table_detection.services._adaptive_hough_common import (
    AdaptiveHoughConfig, ImageSource, adaptive_binary, detect_and_merge, read_gray,
)


@dataclass(frozen=True, slots=True)
class AdaptiveMorphHoughConfig(AdaptiveHoughConfig):
    horizontal_kernel_length: int = 25
    vertical_kernel_length: int = 25
    morphology_gap: int = 0

    def __post_init__(self) -> None:
        super(AdaptiveMorphHoughConfig, self).__post_init__()
        if self.horizontal_kernel_length < 2 or self.vertical_kernel_length < 2:
            raise ValueError('Morphology kernel lengths must be >= 2')
        if self.morphology_gap < 0:
            raise ValueError('morphology_gap must be non-negative')


class AdaptiveMorphHoughLineDetectionService:
    """Adaptive threshold -> directional morphology -> separate HoughLinesP -> merge."""

    def __init__(self, config: AdaptiveMorphHoughConfig | None = None):
        self.config = config or AdaptiveMorphHoughConfig()

    def process(self, source: ImageSource) -> tuple[dict, dict[str, np.ndarray]]:
        gray = read_gray(source)
        cfg = self.config
        binary = adaptive_binary(gray, cfg)
        horizontal = binary
        vertical = binary
        if cfg.morphology_gap:
            horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morphology_gap + 1, 1)))
            vertical = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, cfg.morphology_gap + 1)))
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.horizontal_kernel_length, 1)))
        vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, cfg.vertical_kernel_length)))
        h = detect_and_merge(horizontal, cfg, 'horizontal')
        v = detect_and_merge(vertical, cfg, 'vertical')
        raw_h = h['raw']['lines'] + v['raw']['lines']
        merged_h = h['merged']['lines'] + v['merged']['lines']
        raw = [Line(i, *(item[k] for k in ('x1', 'y1', 'x2', 'y2'))) for i, item in enumerate(raw_h)]
        merged = [Line(i, *(item[k] for k in ('x1', 'y1', 'x2', 'y2'))) for i, item in enumerate(merged_h)]
        result = {'method': 'adaptive_morph_hough', 'image_size': [gray.shape[1], gray.shape[0]],
                  'parameters': asdict(cfg), 'raw': line_set(raw, cfg.angle_tolerance),
                  'merged': line_set(merged, cfg.angle_tolerance)}
        return result, {'binary': binary, 'horizontal': horizontal, 'vertical': vertical,
                        'combined': cv2.bitwise_or(horizontal, vertical)}
