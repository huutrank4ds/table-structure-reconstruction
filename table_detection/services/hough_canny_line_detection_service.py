"""Image-only Canny -> HoughLinesP -> merge -> replaceable filter."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import math
import time
import cv2
import numpy as np

from table_detection.schemas.geometry import Line, OrientationLine, line_set
from table_detection.services.image_processor import ImagePreprocessor, PreprocessConfig


@dataclass(frozen=True)
class DetectionConfig:
    canny_low: int = 50
    canny_high: int = 150
    blur_kernel: int = 3
    min_length: int = 60
    max_gap: int = 12
    hough_votes: int = 35
    angle_tolerance: float = 7.0
    merge_gap: int = 10
    axis_tolerance: int = 3
    support_tolerance: int = 2

    def __post_init__(self):
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError('Expected 0 <= canny_low < canny_high <= 255')
        if self.blur_kernel < 1 or self.blur_kernel % 2 != 1:
            raise ValueError('blur_kernel must be a positive odd integer')
        if min(self.min_length, self.hough_votes) < 1 or min(self.max_gap, self.merge_gap, self.axis_tolerance, self.support_tolerance) < 0:
            raise ValueError('Invalid line detection parameters')
        if not 0 <= self.angle_tolerance <= 45:
            raise ValueError('angle_tolerance must be in [0, 45]')

class HoughCannyLineDetectionService:
    def __init__(self, config: DetectionConfig | None = None,
                 preprocess_config: PreprocessConfig | None = None):
        self.config = config or DetectionConfig()
        self.preprocessor = ImagePreprocessor(preprocess_config)


    @staticmethod
    def _support_score(edges: np.ndarray, line: Line, tolerance: int) -> float:
        """Fraction of positions along segment with at least one Canny pixel +/- tolerance.

        For merged segments this measures actual edge evidence (gaps reduce score).
        It is not a calibrated probability or model-provided confidence.
        """
        h, w = edges.shape
        if line.orientation == 'horizontal':
            indices = np.arange(line.x1, line.x2 + 1)
            if indices.size == 0: return 0.0
            y0, y1 = max(0, line.y1 - tolerance), min(h, line.y1 + tolerance + 1)
            hits = np.any(edges[y0:y1, indices] > 0, axis=0)
        else:
            indices = np.arange(line.y1, line.y2 + 1)
            if indices.size == 0: return 0.0
            x0, x1 = max(0, line.x1 - tolerance), min(w, line.x1 + tolerance + 1)
            hits = np.any(edges[indices, x0:x1] > 0, axis=1)
        return round(float(hits.mean()), 4)


    def _detect_line(self, edges: np.ndarray) -> list[Line]:
        cfg = self.config
        raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=cfg.hough_votes,
                              minLineLength=cfg.min_length, maxLineGap=cfg.max_gap)
        if raw is None:
            return []
        
        candidates: list[tuple[str, int, int, int]] = []
        for x1, y1, x2, y2 in raw.reshape(-1, 4).tolist():
            line = Line(len(candidates), x1, y1, x2, y2)
            candidates.append(line)
        return candidates

 
    def _merge_line(self, lines: list[Line]) -> list[Line]:
        cfg = self.config

        normalized = [
            converted for line in lines
            if (converted := line.to_orientation_line(cfg.angle_tolerance)) is not None
        ]

        output: list[Line] = []

        for orientation in ("horizontal", "vertical"):
            segments = sorted(
                (line for line in normalized if line.orientation == orientation),
                key=lambda line: (line.axis, line.start)
            )
            groups: list[dict] = []

            for line in segments:
                weight = max(line.length, 1)
                compatible = [
                    g for g in groups
                    if abs(g["axis"] - line.axis) <= cfg.axis_tolerance
                    and line.start <= g["end"] + cfg.merge_gap
                    and line.end >= g["start"] - cfg.merge_gap
                ]

                if compatible:
                    g = min(compatible, key=lambda g: abs(g["axis"] - line.axis))
                    g["sum"] += line.axis * weight
                    g["weight"] += weight
                    g["axis"] = g["sum"] / g["weight"]
                    g["start"] = min(g["start"], line.start)
                    g["end"] = max(g["end"], line.end)
                else:
                    groups.append({
                        "id": line.id, "axis": float(line.axis),
                        "sum": line.axis * weight, "weight": weight,
                        "start": line.start, "end": line.end
                    })

            changed = True
            while changed:
                changed = False
                for i in range(len(groups)):
                    if changed:
                        break

                    for j in range(i + 1, len(groups)):
                        a, b = groups[i], groups[j]

                        if abs(a["axis"] - b["axis"]) > cfg.axis_tolerance:
                            continue
                        if a["start"] > b["end"] + cfg.merge_gap or b["start"] > a["end"] + cfg.merge_gap:
                            continue

                        a["sum"] += b["sum"]
                        a["weight"] += b["weight"]
                        a["axis"] = a["sum"] / a["weight"]
                        a["start"] = min(a["start"], b["start"])
                        a["end"] = max(a["end"], b["end"])
                        groups.pop(j)
                        changed = True
                        break

            for g in groups:
                merged = OrientationLine(
                    id=g["id"],
                    orientation=orientation,
                    axis=round(g["axis"]),
                    start=g["start"],
                    end=g["end"]
                )

                if merged.length >= cfg.min_length:
                    output.append(merged.to_line())

        return sorted(output, key=lambda line: (line.y1, line.x1))


    def _to_original_line(self, line: Line, preprocessing: dict) -> Line:
        original = preprocessing["original"]
        sx, sy = preprocessing["scale_x"], preprocessing["scale_y"]
        width, height = original["width"], original["height"]

        return replace(
            line,
            x1=int(np.clip(round(line.x1 / sx), 0, width - 1)),
            y1=int(np.clip(round(line.y1 / sy), 0, height - 1)),
            x2=int(np.clip(round(line.x2 / sx), 0, width - 1)),
            y2=int(np.clip(round(line.y2 / sy), 0, height - 1)),
        )


    def process(self, image_rgb: np.ndarray) -> tuple[dict, np.ndarray]:
        if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim not in (2, 3):
            raise ValueError("Expected grayscale or RGB numpy image")
        if image_rgb.ndim == 3 and image_rgb.shape[2] not in (3, 4):
            raise ValueError("Expected 3 or 4 channels")
        if image_rgb.size == 0 or image_rgb.dtype != np.uint8:
            raise ValueError("Expected non-empty uint8 image")

        start = time.perf_counter()
        normalized, preprocessing = self.preprocessor.process(image_rgb)
        cfg = self.config

        if normalized.ndim == 2:
            gray = normalized
        else:
            code = cv2.COLOR_RGB2GRAY if normalized.shape[2] == 3 else cv2.COLOR_RGBA2GRAY
            gray = cv2.cvtColor(normalized, code)

        processed = cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), 0) if cfg.blur_kernel > 1 else gray
        edges = cv2.Canny(processed, cfg.canny_low, cfg.canny_high, L2gradient=True)

        # Detect once, then merge
        raw_lines = self._detect_line(edges)
        merged_lines = self._merge_line(raw_lines)

        # Restore coordinates to original image resolution
        raw_lines = [self._to_original_line(line, preprocessing) for line in raw_lines]
        merged_lines = [self._to_original_line(line, preprocessing) for line in merged_lines]

        result = {
            "image": preprocessing["original"],
            "preprocessing": preprocessing,
            "raw_hough": line_set(raw_lines, cfg.angle_tolerance),
            "merged_hough": line_set(merged_lines, cfg.angle_tolerance),
            "parameters": asdict(cfg),
            "runtime_ms": round((time.perf_counter() - start) * 1000, 3),
        }

        return result, edges