from __future__ import annotations
from typing import Literal
from dataclasses import dataclass, asdict, replace

type CoordFormat = Literal["xyxy", "xywh", "cxcywh"]


@dataclass(frozen=True, slots=True)
class LayoutComponentConfig:
    label: str
    coord: list[float]
    confident: float

    def __post_init__(self):
        if len(self.coord) != 4:
            raise ValueError("coord must contain [x1, y1, x2, y2]")

        x1, y1, x2, y2 = self.coord
        if x1 > x2 or y1 > y2:
            raise ValueError("Invalid bounding box coordinates")

    @property
    def width(self) -> float:
        return self.coord[2] - self.coord[0]

    @property
    def height(self) -> float:
        return self.coord[3] - self.coord[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.coord
        return (x1 + x2) / 2, (y1 + y2) / 2

    def to_xyxy(self) -> list[float]:
        return list(self.coord)

    def to_xywh(self) -> list[float]:
        x1, y1, _, _ = self.coord
        return [x1, y1, self.width, self.height]

    def to_cxcywh(self) -> list[float]:
        cx, cy = self.center
        return [cx, cy, self.width, self.height]

    def to_polygon(self) -> list[list[float]]:
        x1, y1, x2, y2 = self.coord
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def to_format(self, fmt: CoordFormat = "xyxy") -> list[float]:
        if fmt == "xyxy":
            return self.to_xyxy()
        if fmt == "xywh":
            return self.to_xywh()
        if fmt == "cxcywh":
            return self.to_cxcywh()
        raise ValueError(f"Unsupported coordinate format: {fmt}")

    def to_normalized(self, image_width: int, image_height: int, fmt: CoordFormat = "xyxy") -> list[float]:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")

        x1, y1, x2, y2 = self.coord
        normalized = replace(self, coord=[
            x1 / image_width, y1 / image_height,
            x2 / image_width, y2 / image_height
        ])
        return normalized.to_format(fmt)

    def scale(self, scale_x: float, scale_y: float) -> LayoutComponentConfig:
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("Scale factors must be positive")

        x1, y1, x2, y2 = self.coord
        return replace(self, coord=[
            x1 * scale_x, y1 * scale_y,
            x2 * scale_x, y2 * scale_y
        ])


    def translate(self, dx: float, dy: float) -> LayoutComponentConfig:
        x1, y1, x2, y2 = self.coord
        return replace(self, coord=[
            x1 + dx, y1 + dy,
            x2 + dx, y2 + dy
        ])

    def clip(self, image_width: int, image_height: int) -> LayoutComponentConfig:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")

        x1, y1, x2, y2 = self.coord
        return replace(self, coord=[
            max(0.0, min(x1, image_width)),
            max(0.0, min(y1, image_height)),
            max(0.0, min(x2, image_width)),
            max(0.0, min(y2, image_height))
        ])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LayoutPageConfig:
    page_id: int
    components: list[LayoutComponentConfig]

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass(frozen=True)
class LayoutFileConfig:
    file_id: int
    file_path: str
    page_layout: list[LayoutPageConfig]

    def to_dict(self) -> dict:
        return asdict(self)