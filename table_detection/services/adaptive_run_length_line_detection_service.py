from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from table_detection.schemas.geometry import Line, line_set

type ImageSource = str | Path | np.ndarray


@dataclass(frozen=True, slots=True)
class AdaptiveRunLengthConfig:
    block_size: int = 31
    threshold_c: float = 9.0
    min_run_length: int = 45
    max_run_gap: int = 3
    max_line_thickness: int = 8
    angle_tolerance: float = 5.0

    def __post_init__(self) -> None:
        if self.block_size < 3 or self.block_size % 2 != 1:
            raise ValueError('block_size must be odd and >= 3')
        if self.min_run_length < 2 or self.max_run_gap < 0 or self.max_line_thickness < 1:
            raise ValueError('Invalid run-length configuration')
        if not 0 <= self.angle_tolerance < 45:
            raise ValueError('angle_tolerance must be in [0, 45)')


class AdaptiveRunLengthLineDetectionService:
    def __init__(self, config: AdaptiveRunLengthConfig | None = None):
        self.config = config or AdaptiveRunLengthConfig()

    @staticmethod
    def _read(source: ImageSource) -> np.ndarray:
        if isinstance(source, np.ndarray):
            image = source
        else:
            data = np.fromfile(str(source), np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f'Cannot decode image: {source}')
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8 or image.size == 0:
            raise ValueError('Expected non-empty uint8 image')
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] in (3, 4):
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY if image.shape[2] == 4 else cv2.COLOR_RGB2GRAY)
        raise ValueError('Expected grayscale, RGB, or RGBA image')

    @staticmethod
    def _runs(row: np.ndarray, minimum: int, max_gap: int) -> list[tuple[int, int]]:
        # Inclusive start, exclusive end; join short gaps before filtering by length.
        mask = row != 0
        transitions = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        merged: list[tuple[int, int]] = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            if merged and start - merged[-1][1] <= max_gap:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return [(start, end) for start, end in merged if end - start >= minimum]

    def _extract(self, binary: np.ndarray, horizontal: bool) -> tuple[list[Line], np.ndarray]:
        cfg = self.config
        scan = binary if horizontal else binary.T
        run_mask = np.zeros_like(scan)
        for axis, row in enumerate(scan):
            for start, end in self._runs(row, cfg.min_run_length, cfg.max_run_gap):
                run_mask[axis, start:end] = 255

        # Join adjacent scan rows, then use connected components to recover centerline.
        run_mask = cv2.morphologyEx(run_mask, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(run_mask, connectivity=8)
        lines: list[Line] = []
        for i in range(1, n):
            x, y, width, thickness, area = map(int, stats[i])
            if width < cfg.min_run_length or thickness > cfg.max_line_thickness:
                continue
            axis = int(round(y + (thickness - 1) / 2))
            end = x + width - 1
            if horizontal:
                lines.append(Line(len(lines), x, axis, end, axis))
            else:
                lines.append(Line(len(lines), axis, x, axis, end))
        return lines, run_mask if horizontal else run_mask.T.copy()

    def process(self, source: ImageSource) -> tuple[dict, dict[str, np.ndarray]]:
        gray = self._read(source)
        cfg = self.config
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, cfg.block_size, cfg.threshold_c)
        horizontal, horizontal_mask = self._extract(binary, horizontal=True)
        vertical, vertical_mask = self._extract(binary, horizontal=False)
        lines = [Line(i, ln.x1, ln.y1, ln.x2, ln.y2) for i, ln in enumerate(horizontal + vertical)]
        result = {'method': 'adaptive_run_length', 'image_size': [gray.shape[1], gray.shape[0]],
                  'parameters': asdict(cfg), **line_set(lines, cfg.angle_tolerance)}
        return result, {'binary': binary, 'horizontal': horizontal_mask, 'vertical': vertical_mask}
