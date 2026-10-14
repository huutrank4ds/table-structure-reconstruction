
"""Detect table line segments using OpenCV LSD."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from document_processing.schemas.image import ImageData
from table_detection.schemas.geometry import Line
from table_detection.schemas.detection import LineDetectionResult


type ImageInput = ImageData | np.ndarray


@dataclass(frozen=True, slots=True)
class LSDConfig:
    min_length: float = 30.0
    angle_tolerance: float = 5.0
    refine: int = cv2.LSD_REFINE_STD

    def __post_init__(self) -> None:
        if self.min_length <= 0:
            raise ValueError("min_length must be positive")

        if not 0 <= self.angle_tolerance < 45:
            raise ValueError("angle_tolerance must be in [0, 45)")

        if self.refine not in (
            cv2.LSD_REFINE_NONE,
            cv2.LSD_REFINE_STD,
            cv2.LSD_REFINE_ADV,
        ):
            raise ValueError("Invalid LSD refinement mode")


class LSDLineDetectionService:
    def __init__(
        self,
        config: LSDConfig | None = None,
    ) -> None:
        self.config = config or LSDConfig()

        self.detector = cv2.createLineSegmentDetector(
            self.config.refine
        )

    @staticmethod
    def _get_image(source: ImageInput) -> np.ndarray:
        image = (
            source.image
            if isinstance(source, ImageData)
            else source
        )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Expected ImageData or numpy.ndarray"
            )

        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.size == 0
        ):
            raise ValueError(
                "Expected non-empty RGB uint8 image "
                "with shape (H, W, 3)"
            )

        return image

    def _detect_lines(self, gray: np.ndarray) -> list[Line]:
        detected, _, _, _ = self.detector.detect(gray)

        if detected is None:
            return []

        cfg = self.config
        lines: list[Line] = []

        for x1, y1, x2, y2 in detected.reshape(-1, 4):
            candidate = Line(
                id=len(lines),
                x1=round(float(x1)),
                y1=round(float(y1)),
                x2=round(float(x2)),
                y2=round(float(y2)),
            )

            if candidate.length < cfg.min_length:
                continue

            if candidate.to_orientation_line(
                cfg.angle_tolerance
            ) is None:
                continue

            lines.append(candidate)

        return lines

    def process(self, source: ImageInput) -> LineDetectionResult:
        start = time.perf_counter()

        image = self._get_image(source)

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

        lines = self._detect_lines(gray)

        return LineDetectionResult(
            method="lsd",
            lines=lines,
            runtime_ms=round(
                (time.perf_counter() - start) * 1000,
                3,
            ),
            parameters=asdict(self.config),
        )
