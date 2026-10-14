from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from table_detection.schemas.geometry import Line, line_set

type ImageSource = str | Path | np.ndarray


@dataclass(frozen=True, slots=True)
class LSDConfig:
    min_length: float = 30.0
    angle_tolerance: float = 5.0
    refine: int = cv2.LSD_REFINE_STD

    def __post_init__(self) -> None:
        if self.min_length <= 0 or not 0 <= self.angle_tolerance < 45:
            raise ValueError('Invalid LSD configuration')
        if self.refine not in (cv2.LSD_REFINE_NONE, cv2.LSD_REFINE_STD, cv2.LSD_REFINE_ADV):
            raise ValueError('Invalid LSD refinement mode')


class LSDLineDetectionService:
    def __init__(self, config: LSDConfig | None = None):
        self.config = config or LSDConfig()
        self.detector = cv2.createLineSegmentDetector(self.config.refine)

    @staticmethod
    def _read(source: ImageSource) -> np.ndarray:
        if isinstance(source, np.ndarray):
            image = source
            if image.dtype != np.uint8 or image.size == 0:
                raise ValueError('Expected non-empty uint8 image')
            if image.ndim == 2:
                return image
            if image.ndim == 3 and image.shape[2] in (3, 4):
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY if image.shape[2] == 4 else cv2.COLOR_RGB2GRAY)
            raise ValueError('Expected grayscale, RGB, or RGBA image')
        data = np.fromfile(str(source), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Cannot decode image: {source}')
        return image

    def process(self, source: ImageSource) -> dict:
        gray = self._read(source)
        detected, _, _, _ = self.detector.detect(gray)
        lines: list[Line] = []
        if detected is not None:
            for x1, y1, x2, y2 in detected.reshape(-1, 4):
                candidate = Line(len(lines), int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
                if candidate.length < self.config.min_length:
                    continue
                oriented = candidate.to_orientation_line(self.config.angle_tolerance)
                if oriented is None:
                    continue
                lines.append(candidate)
        return {'method': 'lsd', 'image_size': [gray.shape[1], gray.shape[0]],
                'parameters': asdict(self.config), **line_set(lines, self.config.angle_tolerance)}
