from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from table_detection.schemas.geometry import Line


@dataclass(frozen=True, slots=True)
class LineDetectionResult:
    method: str
    lines: list[Line]
    raw_lines: list[Line] = field(default_factory=list)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    runtime_ms: float = 0.0
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "count": len(self.lines),
            "lines": [line.to_dict() for line in self.lines],
            "raw_count": len(self.raw_lines),
            "raw_lines": [line.to_dict() for line in self.raw_lines],
            "runtime_ms": self.runtime_ms,
            "parameters": self.parameters,
            "masks": {
                name: list(mask.shape)
                for name, mask in self.masks.items()
            },
        }