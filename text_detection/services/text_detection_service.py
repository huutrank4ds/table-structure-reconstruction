from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import paddle

from PIL import Image
from paddleocr import TextDetection

from document_processing.schemas.image import ImageData


type ImageInput = ImageData | np.ndarray | str | Path


@dataclass(frozen=True, slots=True)
class TextBox:
    id: int
    bbox: tuple[int, int, int, int]
    polygon: list[list[int]]
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


class PaddleTextDetectionService:
    def __init__(
        self,
        model_name: str = "PP-OCRv5_server_det",
        device: str | None = None,
        confidence: float = 0.5,
    ) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")

        self.device = device or (
            "gpu:0"
            if paddle.is_compiled_with_cuda()
            and "gpu:0" in paddle.device.get_available_device()
            else "cpu"
        )
        self.confidence = confidence

        self.model = TextDetection(
            model_name=model_name,
            device=self.device,
        )

    @staticmethod
    def _load_image(source: ImageInput) -> np.ndarray:
        if isinstance(source, ImageData):
            image = source.image

        elif isinstance(source, np.ndarray):
            image = source

        else:
            with Image.open(source) as img:
                image = np.asarray(img.convert("RGB")).copy()

        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.size == 0
        ):
            raise ValueError("Expected non-empty RGB uint8 image")

        return image

    def process(
        self,
        source: ImageInput,
    ) -> tuple[list[TextBox], float]:
        image = self._load_image(source)

        start = time.perf_counter()

        # PaddleOCR accepts NumPy arrays representing images.
        # Convert RGB -> BGR for the OpenCV-style ndarray input.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        results = self.model.predict(
            input=bgr,
            batch_size=1,
        )

        boxes: list[TextBox] = []

        for result in results:
            data = result.json
            data = data.get("res", data)

            polygons = data.get("dt_polys", [])
            scores = data.get("dt_scores", [])

            for polygon, score in zip(polygons, scores):
                if float(score) < self.confidence:
                    continue

                points = np.asarray(polygon, dtype=np.int32)

                x1 = int(points[:, 0].min())
                y1 = int(points[:, 1].min())
                x2 = int(points[:, 0].max())
                y2 = int(points[:, 1].max())

                boxes.append(
                    TextBox(
                        id=len(boxes),
                        bbox=(x1, y1, x2, y2),
                        polygon=points.tolist(),
                        score=round(float(score), 4),
                    )
                )

        runtime_ms = (time.perf_counter() - start) * 1000

        return boxes, round(runtime_ms, 3)

    @staticmethod
    def visualize(
        source: ImageInput,
        boxes: list[TextBox],
    ) -> np.ndarray:
        image = PaddleTextDetectionService._load_image(source)
        canvas = image.copy()

        for box in boxes:
            points = np.asarray(box.polygon, dtype=np.int32)

            cv2.polylines(
                canvas,
                [points],
                isClosed=True,
                color=(255, 50, 50),
                thickness=2,
            )

            x1, y1, _, _ = box.bbox

            cv2.putText(
                canvas,
                f"{box.score:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 50, 50),
                1,
                cv2.LINE_AA,
            )

        return canvas