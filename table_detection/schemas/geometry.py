"""Stable serialization types: raw Hough and filtered collections share LineSet."""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Literal

type Orientation = Literal['horizontal', 'vertical']


@dataclass(frozen=True, slots=True)
class Line:
    id: int
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def length(self) -> float:
        return math.hypot(
            self.x2 - self.x1,
            self.y2 - self.y1,
        )

    @property
    def angle(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1

        return math.degrees(
            math.atan2(dy, dx)
        ) % 180

    def to_orientation_line(
        self,
        angle_tolerance: float = 5.0,
    ) -> OrientationLine | None:

        if not 0 <= angle_tolerance < 45:
            raise ValueError(
                "angle_tolerance must be in [0, 45)"
            )

        angle = self.angle

        if min(angle, 180 - angle) <= angle_tolerance:
            orientation = "horizontal"
            axis = round((self.y1 + self.y2) / 2)
            start, end = sorted((self.x1, self.x2))

        elif abs(angle - 90) <= angle_tolerance:
            orientation = "vertical"
            axis = round((self.x1 + self.x2) / 2)
            start, end = sorted((self.y1, self.y2))

        else:
            return None

        return OrientationLine(
            id=self.id,
            orientation=orientation,
            axis=axis,
            start=start,
            end=end,
        )


@dataclass(frozen=True, slots=True)
class OrientationLine:
    id: int
    orientation: Orientation
    axis: int
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def length(self) -> int:
        return abs(self.end - self.start)

    def to_line(self) -> Line:
        if self.orientation == "horizontal":
            return Line(
                id=self.id,
                x1=self.start,
                y1=self.axis,
                x2=self.end,
                y2=self.axis,
            )

        return Line(
            id=self.id,
            x1=self.axis,
            y1=self.start,
            x2=self.axis,
            y2=self.end,
        )



def summary(lines: list[Line]) -> dict:
    result = {}
    for direction in ('horizontal', 'vertical'):
        subset = [line for line in lines if line.to_orientation_line().orientation == direction]
        result[direction] = {
            'count': len(subset),
            'max_length_px': round(max((line.length for line in subset), default=0.0), 3)
        }
    return result


def line_set(lines: list[Line]) -> dict:
    """Identical schema for Hough and filtered output."""
    return {
        'count': len(lines),
        'by_orientation': summary(lines),
        'lines': [line.to_dict() for line in lines],
    }