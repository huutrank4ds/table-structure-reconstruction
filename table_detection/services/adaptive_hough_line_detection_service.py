from __future__ import annotations

from dataclasses import asdict

import numpy as np

from table_detection.services._adaptive_hough_common import (
    AdaptiveHoughConfig, ImageSource, adaptive_binary, detect_and_merge, read_gray,
)


class AdaptiveHoughLineDetectionService:
    """Adaptive Gaussian threshold -> HoughLinesP -> collinear merge."""

    def __init__(self, config: AdaptiveHoughConfig | None = None):
        self.config = config or AdaptiveHoughConfig()

    def process(self, source: ImageSource) -> tuple[dict, dict[str, np.ndarray]]:
        gray = read_gray(source)
        binary = adaptive_binary(gray, self.config)
        detected = detect_and_merge(binary, self.config)
        result = {'method': 'adaptive_hough', 'image_size': [gray.shape[1], gray.shape[0]],
                  'parameters': asdict(self.config), **detected}
        return result, {'binary': binary}
