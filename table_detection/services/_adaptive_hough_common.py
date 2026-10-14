from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from table_detection.schemas.geometry import Line, line_set

type ImageSource = str | Path | np.ndarray


@dataclass(frozen=True, slots=True)
class AdaptiveHoughConfig:
    block_size: int = 31
    threshold_c: float = 9.0
    blur_kernel: int = 3
    hough_votes: int = 25
    min_line_length: int = 35
    max_line_gap: int = 8
    angle_tolerance: float = 7.0
    axis_tolerance: int = 2
    merge_gap: int = 3

    def __post_init__(self) -> None:
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError('block_size must be an odd integer >= 3')
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError('blur_kernel must be an odd integer >= 1')
        if self.hough_votes < 1 or self.min_line_length < 1 or self.max_line_gap < 0:
            raise ValueError('Invalid Hough parameter')
        if not 0 <= self.angle_tolerance < 45:
            raise ValueError('angle_tolerance must be in [0, 45)')
        if self.axis_tolerance < 0 or self.merge_gap < 0:
            raise ValueError('axis_tolerance and merge_gap must be non-negative')


def read_gray(source: ImageSource) -> np.ndarray:
    if isinstance(source, np.ndarray):
        image = source
        if image.dtype != np.uint8 or not image.size:
            raise ValueError('Image must be non-empty uint8')
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] in (3, 4):
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY if image.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
        raise ValueError('Expected grayscale, RGB, or RGBA ndarray')
    data = np.fromfile(str(source), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Unable to read image: {source}')
    return image


def adaptive_binary(gray: np.ndarray, cfg: AdaptiveHoughConfig) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), 0) if cfg.blur_kernel > 1 else gray
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, cfg.block_size, cfg.threshold_c)


def detect_and_merge(mask: np.ndarray, cfg: AdaptiveHoughConfig, orientation: str | None = None) -> dict:
    raw = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=cfg.hough_votes,
                          minLineLength=cfg.min_line_length, maxLineGap=cfg.max_line_gap)
    lines = [] if raw is None else [Line(i, *map(int, item)) for i, item in enumerate(raw.reshape(-1, 4))]
    orient = []
    for line in lines:
        oriented = line.to_orientation_line(cfg.angle_tolerance)
        if oriented is not None and (orientation is None or oriented.orientation == orientation):
            orient.append(oriented)
    # Merge only collinear segments with nearby intervals. Retain real start/end boundaries.
    groups: list[dict] = []
    for item in sorted(orient, key=lambda r: (r.orientation, r.axis, r.start)):
        matching = [g for g in groups if g['orientation'] == item.orientation
                    and abs(g['axis'] - item.axis) <= cfg.axis_tolerance
                    and item.start <= g['end'] + cfg.merge_gap
                    and item.end >= g['start'] - cfg.merge_gap]
        if matching:
            group = matching[0]
            group['axis'] = round((group['axis'] * group['weight'] + item.axis * item.length) /
                                  (group['weight'] + item.length))
            group['weight'] += item.length
            group['start'] = min(group['start'], item.start)
            group['end'] = max(group['end'], item.end)
            for redundant in matching[1:]:
                group['start'] = min(group['start'], redundant['start'])
                group['end'] = max(group['end'], redundant['end'])
                groups.remove(redundant)
        else:
            groups.append({'orientation': item.orientation, 'axis': item.axis,
                           'start': item.start, 'end': item.end, 'weight': max(item.length, 1)})
    merged = [Line(i, g['start'], g['axis'], g['end'], g['axis']) if g['orientation'] == 'horizontal'
              else Line(i, g['axis'], g['start'], g['axis'], g['end']) for i, g in enumerate(groups)]
    return {'raw': line_set(lines, cfg.angle_tolerance), 'merged': line_set(merged, cfg.angle_tolerance)}
