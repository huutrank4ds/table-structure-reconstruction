
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from document_processing.schemas.image import ImageData

@dataclass(frozen=False, slots=True)
class RasterizedPage(ImageData):
    page_id: int
    dpi: int
    pdf_width: float
    pdf_height: float

    def __post_init__(self) -> None:
        super(RasterizedPage, self).__post_init__()

        if self.page_id < 0:
            raise ValueError("page_id must be >= 0")

        if self.dpi <= 0:
            raise ValueError("dpi must be positive")

        if self.pdf_width <= 0 or self.pdf_height <= 0:
            raise ValueError("PDF dimensions must be positive")

    @property
    def scale_x(self) -> float:
        return self.width / self.pdf_width

    @property
    def scale_y(self) -> float:
        return self.height / self.pdf_height

    def pixel_to_pdf(self, coord: list[float]) -> list[float]:
        x1, y1, x2, y2 = coord
        return [x1 / self.scale_x, y1 / self.scale_y, x2 / self.scale_x, y2 / self.scale_y]

    def pdf_to_pixel(self, coord: list[float]) -> list[float]:
        x1, y1, x2, y2 = coord
        return [x1 * self.scale_x, y1 * self.scale_y, x2 * self.scale_x, y2 * self.scale_y]

    def to_dict(self) -> dict:
        return {
            **super(RasterizedPage, self).to_dict(),
            "page_id": self.page_id,
            "dpi": self.dpi,
            "pdf_width": self.pdf_width,
            "pdf_height": self.pdf_height,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
        }


@dataclass(frozen=True, slots=True)
class RasterizedDocument:
    file_path: str
    page_count: int
    pages: list[RasterizedPage]
    save_path: str | None = None

    def get_page(self, page_id: int) -> RasterizedPage | None:
        return next((page for page in self.pages if page.page_id == page_id), None)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "page_count": self.page_count,
            "save_path": self.save_path,
            "pages": [page.to_dict() for page in self.pages],
        }
