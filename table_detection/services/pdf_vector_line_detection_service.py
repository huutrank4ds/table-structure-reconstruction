from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from table_detection.schemas.geometry import Line, line_set

type PDFSource = str | Path | bytes

type BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PDFVectorConfig:
    dpi: int = 300
    min_length_px: float = 15.0
    angle_tolerance: float = 5.0

    def __post_init__(self) -> None:
        if self.dpi <= 0 or self.min_length_px < 0 or not 0 <= self.angle_tolerance < 45:
            raise ValueError('Invalid PDF vector configuration')


class PDFVectorLineDetectionService:
    def __init__(self, config: PDFVectorConfig | None = None):
        self.config = config or PDFVectorConfig()

    @staticmethod
    def _segments(page: pymupdf.Page):
        for drawing in page.get_drawings():
            # Filled-only shapes are not necessarily cell borders. Ignore them.
            if drawing.get('type') == 'f':
                continue
            for item in drawing['items']:
                kind = item[0]
                if kind == 'l':
                    yield item[1], item[2]
                elif kind == 're':
                    rect = item[1]
                    p1, p2, p3, p4 = (
                        pymupdf.Point(rect.x0, rect.y0), pymupdf.Point(rect.x1, rect.y0),
                        pymupdf.Point(rect.x1, rect.y1), pymupdf.Point(rect.x0, rect.y1))
                    yield p1, p2
                    yield p2, p3
                    yield p3, p4
                    yield p4, p1

    @staticmethod
    def _open(source: PDFSource) -> pymupdf.Document:
        return pymupdf.open(stream=source, filetype='pdf') if isinstance(source, bytes) else pymupdf.open(str(source))

    def process(self, source: PDFSource, page_id: int = 0, clip: BBox | None = None) -> dict:
        cfg = self.config
        with self._open(source) as document:
            if page_id < 0 or page_id >= len(document):
                raise ValueError(f'Invalid page_id: {page_id}')
            page = document[page_id]
            width = round(page.rect.width * cfg.dpi / 72)
            height = round(page.rect.height * cfg.dpi / 72)
            sx, sy = width / page.rect.width, height / page.rect.height
            lines: list[Line] = []
            bounds = pymupdf.Rect(clip) if clip is not None else page.rect
            if bounds.is_empty or bounds.is_infinite:
                raise ValueError('Invalid clip rectangle')

            for p1, p2 in self._segments(page):
                # Drawing coordinates are unrotated; page raster is in rotated display coordinates.
                a = p1 * page.rotation_matrix
                b = p2 * page.rotation_matrix
                # Clip against displayed page/ROI to avoid lines outside the table region.
                if not pymupdf.Rect(a, b).intersects(bounds) and not (bounds.contains(a) or bounds.contains(b)):
                    continue
                # Axis-aligned clipping for horizontal / vertical segments.
                ax, ay, bx, by = a.x, a.y, b.x, b.y
                if abs(ay - by) <= 1e-3:
                    lo, hi = sorted((ax, bx))
                    lo, hi = max(lo, bounds.x0), min(hi, bounds.x1)
                    if lo >= hi or not bounds.y0 <= ay <= bounds.y1:
                        continue
                    ax, bx = lo, hi
                elif abs(ax - bx) <= 1e-3:
                    lo, hi = sorted((ay, by))
                    lo, hi = max(lo, bounds.y0), min(hi, bounds.y1)
                    if lo >= hi or not bounds.x0 <= ax <= bounds.x1:
                        continue
                    ay, by = lo, hi
                else:
                    continue  # Only axis-aligned borders for table reconstruction.
                candidate = Line(len(lines), round((ax-page.rect.x0)*sx), round((ay-page.rect.y0)*sy),
                                 round((bx-page.rect.x0)*sx), round((by-page.rect.y0)*sy))
                if candidate.length >= cfg.min_length_px and candidate.to_orientation_line(cfg.angle_tolerance) is not None:
                    lines.append(candidate)
        return {'method': 'pdf_vector', 'page_id': page_id, 'image_size': [width, height],
                'coordinate_system': f'rendered_page_pixels_{cfg.dpi}dpi', 'parameters': asdict(cfg),
                **line_set(lines, cfg.angle_tolerance)}
