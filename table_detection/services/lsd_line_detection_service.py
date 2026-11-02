"""Detect and merge table line segments using OpenCV LSD."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from document_processing.schemas.image import ImageData
from table_detection.schemas.geometry import Line, OrientationLine
from table_detection.schemas.detection import LineDetectionResult


type ImageInput = ImageData | np.ndarray


@dataclass(frozen=True, slots=True)
class LSDConfig:
    min_length: float = 30.0
    angle_tolerance: float = 5.0
    refine: int = cv2.LSD_REFINE_STD

    # Line merging
    axis_tolerance: int = 5
    merge_gap: int = 10
    length_power: float = 2.0

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

        if self.axis_tolerance < 0:
            raise ValueError("axis_tolerance must be non-negative")

        if self.merge_gap < 0:
            raise ValueError("merge_gap must be non-negative")

        if self.length_power < 0:
            raise ValueError("length_power must be non-negative")


@dataclass(slots=True)
class _LineGroup:
    orientation: str
    axis_sum: float
    weight: float
    axis_min: int
    axis_max: int
    start: int
    end: int

    @property
    def axis(self) -> float:
        return self.axis_sum / self.weight

    @classmethod
    def from_line(
        cls,
        line: OrientationLine,
        length_power: float,
    ) -> _LineGroup:
        weight = max(line.length, 1) ** length_power

        return cls(
            orientation=line.orientation,
            axis_sum=line.axis * weight,
            weight=weight,
            axis_min=line.axis,
            axis_max=line.axis,
            start=line.start,
            end=line.end,
        )

    def can_merge(
        self,
        other: _LineGroup,
        axis_tolerance: int,
        merge_gap: int,
    ) -> bool:
        if self.orientation != other.orientation:
            return False

        # Prevent gradual axis drift when merging many lines.
        if (
            max(self.axis_max, other.axis_max)
            - min(self.axis_min, other.axis_min)
            > axis_tolerance
        ):
            return False

        # Intervals must overlap or be separated by a small gap.
        if (
            max(self.start, other.start)
            > min(self.end, other.end) + merge_gap
        ):
            return False

        return True

    def merge(self, other: _LineGroup) -> None:
        self.axis_sum += other.axis_sum
        self.weight += other.weight

        self.axis_min = min(self.axis_min, other.axis_min)
        self.axis_max = max(self.axis_max, other.axis_max)

        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)


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

    def _merge_lines(self, lines: list[Line]) -> list[Line]:
        cfg = self.config

        if not lines:
            return []

        oriented = [
            converted
            for line in lines
            if (
                converted := line.to_orientation_line(
                    cfg.angle_tolerance
                )
            ) is not None
        ]

        groups = [
            _LineGroup.from_line(line, cfg.length_power)
            for line in oriented
        ]

        # Iteratively merge compatible groups.
        changed = True

        while changed:
            changed = False

            for i in range(len(groups)):
                if changed:
                    break

                for j in range(i + 1, len(groups)):
                    if not groups[i].can_merge(
                        groups[j],
                        cfg.axis_tolerance,
                        cfg.merge_gap,
                    ):
                        continue

                    groups[i].merge(groups[j])
                    groups.pop(j)

                    changed = True
                    break

        merged_lines: list[Line] = []

        for group in groups:
            merged = OrientationLine(
                id=len(merged_lines),
                orientation=group.orientation,
                axis=round(group.axis),
                start=group.start,
                end=group.end,
            )

            merged_lines.append(merged.to_line())

        merged_lines.sort(
            key=lambda line: (
                line.y1,
                line.x1,
                line.y2,
                line.x2,
            )
        )

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

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

        raw_lines = self._detect_lines(gray)
        merged_lines = self._merge_lines(raw_lines)

        return LineDetectionResult(
            method="lsd",
            lines=merged_lines,
            raw_lines=raw_lines,
            runtime_ms=round(
                (time.perf_counter() - start) * 1000,
                3,
            ),
            parameters=asdict(self.config),
        )