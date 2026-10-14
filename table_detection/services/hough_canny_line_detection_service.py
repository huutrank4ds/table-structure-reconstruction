
"""Detect and merge table lines using Canny and HoughLinesP."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from document_processing.schemas.image import ImageData
from table_detection.schemas.geometry import Line, OrientationLine
from table_detection.schemas.detection import LineDetectionResult


type ImageInput = ImageData | np.ndarray


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    canny_low: int = 50
    canny_high: int = 150
    blur_kernel: int = 3

    min_length: int = 60
    max_gap: int = 10
    hough_votes: int = 35

    angle_tolerance: float = 7.0
    axis_tolerance: int = 3
    merge_gap: int = 10
    length_power: float = 2.0

    def __post_init__(self) -> None:
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError(
                "Expected 0 <= canny_low < canny_high <= 255"
            )

        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError(
                "blur_kernel must be a positive odd integer"
            )

        if self.min_length < 1 or self.hough_votes < 1:
            raise ValueError(
                "min_length and hough_votes must be positive"
            )

        if min(self.max_gap, self.merge_gap, self.axis_tolerance) < 0:
            raise ValueError(
                "Gap and axis tolerance must be non-negative"
            )

        if not 0 <= self.angle_tolerance < 45:
            raise ValueError(
                "angle_tolerance must be in [0, 45)"
            )

        if self.length_power < 0:
            raise ValueError(
                "length_power must be non-negative"
            )


class HoughCannyLineDetectionService:
    def __init__(
        self,
        config: DetectionConfig | None = None,
    ) -> None:
        self.config = config or DetectionConfig()

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

    def _detect_lines(self, edges: np.ndarray) -> list[Line]:
        cfg = self.config

        detected = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=cfg.hough_votes,
            minLineLength=cfg.min_length,
            maxLineGap=cfg.max_gap,
        )

        if detected is None:
            return []

        return [
            Line(
                id=index,
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
            )
            for index, (x1, y1, x2, y2)
            in enumerate(detected.reshape(-1, 4))
        ]

    def _merge_lines(self, lines: list[Line]) -> list[Line]:
        cfg = self.config

        oriented = [
            converted
            for line in lines
            if (
                converted := line.to_orientation_line(
                    cfg.angle_tolerance
                )
            ) is not None
        ]

        merged_lines: list[Line] = []

        for orientation in ("horizontal", "vertical"):
            segments = sorted(
                (
                    line
                    for line in oriented
                    if line.orientation == orientation
                ),
                key=lambda line: (line.axis, line.start),
            )

            groups: list[dict[str, Any]] = []

            for line in segments:
                weight = max(line.length, 1) ** cfg.length_power

                groups.append({
                    "axis_sum": line.axis * weight,
                    "weight": weight,
                    "start": line.start,
                    "end": line.end,
                })

            # Merge groups until no compatible pairs remain.
            changed = True

            while changed:
                changed = False

                for i in range(len(groups)):
                    if changed:
                        break

                    for j in range(i + 1, len(groups)):
                        a = groups[i]
                        b = groups[j]

                        axis_a = a["axis_sum"] / a["weight"]
                        axis_b = b["axis_sum"] / b["weight"]

                        if abs(axis_a - axis_b) > cfg.axis_tolerance:
                            continue

                        if (
                            max(a["start"], b["start"])
                            > min(a["end"], b["end"]) + cfg.merge_gap
                        ):
                            continue

                        a["axis_sum"] += b["axis_sum"]
                        a["weight"] += b["weight"]
                        a["start"] = min(a["start"], b["start"])
                        a["end"] = max(a["end"], b["end"])

                        groups.pop(j)
                        changed = True
                        break

            for group in groups:
                merged = OrientationLine(
                    id=len(merged_lines),
                    orientation=orientation,
                    axis=round(
                        group["axis_sum"] / group["weight"]
                    ),
                    start=group["start"],
                    end=group["end"],
                )

                # No additional length filtering.
                merged_lines.append(merged.to_line())

        merged_lines.sort(
            key=lambda line: (
                line.y1,
                line.x1,
                line.y2,
                line.x2,
            )
        )

        # Assign sequential IDs after sorting.
        return [
            Line(
                id=index,
                x1=line.x1,
                y1=line.y1,
                x2=line.x2,
                y2=line.y2,
            )
            for index, line in enumerate(merged_lines)
        ]

    def process(self, source: ImageInput) -> LineDetectionResult:
        start = time.perf_counter()

        image = self._get_image(source)
        cfg = self.config

        # RGB -> Grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

        # Optional Gaussian Blur
        if cfg.blur_kernel > 1:
            gray = cv2.GaussianBlur(
                gray,
                (cfg.blur_kernel, cfg.blur_kernel),
                0,
            )

        # Canny Edge Detection
        edges = cv2.Canny(
            gray,
            cfg.canny_low,
            cfg.canny_high,
            L2gradient=True,
        )

        # Hough Detection and Line Merging
        raw_lines = self._detect_lines(edges)
        merged_lines = self._merge_lines(raw_lines)

        return LineDetectionResult(
            method="hough_canny",
            lines=merged_lines,
            raw_lines=raw_lines,
            masks={"canny": edges},
            runtime_ms=round(
                (time.perf_counter() - start) * 1000,
                3,
            ),
            parameters=asdict(cfg),
        )
