
"""Detect horizontal and vertical lines using Adaptive Threshold and Run-Length."""

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
class AdaptiveRunLengthConfig:
    block_size: int = 31
    threshold_c: float = 9.0
    min_run_length: int = 45
    max_run_gap: int = 3
    max_line_thickness: int = 8
    angle_tolerance: float = 5.0

    def __post_init__(self) -> None:
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError("block_size must be odd and >= 3")

        if self.min_run_length < 2:
            raise ValueError("min_run_length must be >= 2")

        if self.max_run_gap < 0:
            raise ValueError("max_run_gap must be non-negative")

        if self.max_line_thickness < 1:
            raise ValueError("max_line_thickness must be positive")

        if not 0 <= self.angle_tolerance < 45:
            raise ValueError("angle_tolerance must be in [0, 45)")


class AdaptiveRunLengthLineDetectionService:
    def __init__(
        self,
        config: AdaptiveRunLengthConfig | None = None,
    ) -> None:
        self.config = config or AdaptiveRunLengthConfig()

    @staticmethod
    def _get_image(source: ImageInput) -> np.ndarray:
        image = source.image if isinstance(source, ImageData) else source

        if not isinstance(image, np.ndarray):
            raise TypeError("Expected ImageData or numpy.ndarray")

        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.size == 0
        ):
            raise ValueError(
                "Expected non-empty RGB uint8 image with shape (H, W, 3)"
            )

        return image

    @staticmethod
    def _runs(
        row: np.ndarray,
        minimum: int,
        max_gap: int,
    ) -> list[tuple[int, int]]:
        """
        Extract runs along one scan row.

        Coordinates:
        - start: inclusive
        - end: exclusive
        """
        mask = row != 0
        transitions = np.diff(
            np.pad(mask.astype(np.int8), (1, 1))
        )

        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)

        merged: list[tuple[int, int]] = []

        for start, end in zip(starts.tolist(), ends.tolist()):
            if merged and start - merged[-1][1] <= max_gap:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        return [
            (start, end)
            for start, end in merged
            if end - start >= minimum
        ]

    def _extract(
        self,
        binary: np.ndarray,
        horizontal: bool,
    ) -> tuple[list[Line], np.ndarray]:
        cfg = self.config

        # Transpose for vertical scanning.
        scan = binary if horizontal else binary.T

        run_mask = np.zeros_like(scan)

        for axis, row in enumerate(scan):
            for start, end in self._runs(
                row,
                cfg.min_run_length,
                cfg.max_run_gap,
            ):
                run_mask[axis, start:end] = 255

        # Join adjacent scan rows (perpendicular to run direction).
        kernel = np.ones((3, 1), dtype=np.uint8)

        run_mask = cv2.morphologyEx(
            run_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # Connected component extraction.
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            run_mask,
            connectivity=8,
        )

        lines: list[Line] = []

        for index in range(1, count):
            x, y, width, thickness, _ = map(int, stats[index])

            if (
                width < cfg.min_run_length
                or thickness > cfg.max_line_thickness
            ):
                continue

            axis = round(y + (thickness - 1) / 2)
            end = x + width - 1

            if horizontal:
                line = Line(
                    id=len(lines),
                    x1=x,
                    y1=axis,
                    x2=end,
                    y2=axis,
                )
            else:
                line = Line(
                    id=len(lines),
                    x1=axis,
                    y1=x,
                    x2=axis,
                    y2=end,
                )

            lines.append(line)

        # Return masks in the original image orientation.
        mask = run_mask if horizontal else run_mask.T.copy()

        return lines, mask

    def process(self, source: ImageInput) -> LineDetectionResult:
        start = time.perf_counter()

        image = self._get_image(source)
        cfg = self.config

        # RGB -> Grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

        # Adaptive Gaussian Threshold
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            cfg.block_size,
            cfg.threshold_c,
        )

        # Horizontal / Vertical Run-Length extraction
        horizontal_lines, horizontal_mask = self._extract(
            binary,
            horizontal=True,
        )
        vertical_lines, vertical_mask = self._extract(
            binary,
            horizontal=False,
        )

        # Assign unique IDs across both orientations.
        lines = [
            Line(
                id=index,
                x1=line.x1,
                y1=line.y1,
                x2=line.x2,
                y2=line.y2,
            )
            for index, line in enumerate(
                horizontal_lines + vertical_lines
            )
        ]

        return LineDetectionResult(
            method="adaptive_run_length",
            lines=lines,
            masks={
                "binary": binary,
                "horizontal": horizontal_mask,
                "vertical": vertical_mask,
            },
            runtime_ms=round(
                (time.perf_counter() - start) * 1000,
                3,
            ),
            parameters=asdict(cfg),
        )
