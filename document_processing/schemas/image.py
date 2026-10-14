from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Self
import numpy as np
from uuid import uuid4
from pathlib import Path
from PIL import Image

from layout_ocr_detection.schemas.layout_config import LayoutComponentConfig


@dataclass(frozen=False, slots=True, kw_only=True, eq=False)
class ImageData:
    image: np.ndarray = field(repr=False)
    id: str = field(default_factory=lambda: uuid4().hex, init=False)
    image_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")

        if self.image.dtype != np.uint8:
            raise ValueError("Expected uint8 image")

        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("Expected RGB image with shape (H, W, 3)")

        if self.image.size == 0:
            raise ValueError("Image cannot be empty")

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.image.shape

    def to_dict(self) -> dict:
        return {
            "image_id": self.id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height
        }

    def save(self, path: str | Path, overwrite: bool = False) -> str:
        path = Path(path).expanduser()

        if not path.suffix:
            path = path.with_suffix(".png")

        if path.exists() and not overwrite:
            raise FileExistsError(f"Image already exists: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(self.image, mode="RGB").save(path)

        self.image_path = str(path.resolve())
        return self.image_path


@dataclass(frozen=False, slots=True)
class CroppedImage(ImageData):
    crop_id: int
    component: LayoutComponentConfig

    @property
    def bbox(self) -> list[float]:
        return self.component.coord

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
            **super(CroppedImage, self).to_dict(),
            "crop_id": self.crop_id,
            "component": self.component.to_dict(),
        }