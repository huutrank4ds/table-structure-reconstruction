from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Self
import numpy as np

from layout_ocr_detection.schemas.layout_config import LayoutComponentConfig


@dataclass(frozen=True, slots=True)
class CroppedImage:
    crop_id: int
    image: np.ndarray
    component: LayoutComponentConfig
    image_path: str | None = None

    @property
    def bbox(self) -> list[float]:
        return self.component.coord

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    def to_source_coord(self, coord: list[float]) -> list[float]:
        """Convert crop-local XYXY coordinates to source coordinates."""
        x1, y1 = self.component.coord[:2]
        return [
            coord[0] + x1, coord[1] + y1,
            coord[2] + x1, coord[3] + y1
        ]

    def to_local_coord(self, coord: list[float]) -> list[float]:
        """Convert source XYXY coordinates to crop-local coordinates."""
        x1, y1 = self.component.coord[:2]
        return [
            coord[0] - x1, coord[1] - y1,
            coord[2] - x1, coord[3] - y1
        ]

    def to_dict(self) -> dict:
        return {
            "crop_id": self.crop_id,
            "component": self.component.to_dict(),
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
        }