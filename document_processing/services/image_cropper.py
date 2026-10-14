
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Sequence, TypeAlias

import cv2
import numpy as np

from document_processing.schemas.crop import CroppedImage
from layout_ocr_detection.schemas.layout_config import LayoutComponentConfig


ImageSource: TypeAlias = str | Path | np.ndarray
BBox: TypeAlias = Sequence[float]


class ImageCropper:
    def __init__(self, padding: int = 0):
        if padding < 0:
            raise ValueError("padding must be non-negative")
        self.padding = padding

    @staticmethod
    def _load_image(source: ImageSource) -> np.ndarray:
        if isinstance(source, np.ndarray):
            image = source
        else:
            path = Path(source).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")

            data = np.fromfile(str(path), dtype=np.uint8)
            decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

            if decoded is None:
                raise ValueError(f"Cannot decode image: {path}")

            if decoded.ndim == 2:
                image = decoded
            elif decoded.shape[2] == 3:
                image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            elif decoded.shape[2] == 4:
                image = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
            else:
                raise ValueError("Unsupported image channels")

        if image.ndim not in (2, 3) or image.size == 0:
            raise ValueError("Expected a non-empty image")
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError("Expected RGB or RGBA image")
        if image.dtype != np.uint8:
            raise ValueError("Expected uint8 image")

        return image

    @staticmethod
    def _normalize_bbox(
        bbox: BBox,
        width: int,
        height: int,
        padding: int,
    ) -> tuple[int, int, int, int]:
        if len(bbox) != 4:
            raise ValueError("Bounding box must contain [x1, y1, x2, y2]")
        if not all(math.isfinite(float(value)) for value in bbox):
            raise ValueError("Bounding box contains non-finite coordinates")

        x1, y1, x2, y2 = map(float, bbox)
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid bounding box: {bbox}")

        x1 = max(0, math.floor(x1 - padding))
        y1 = max(0, math.floor(y1 - padding))
        x2 = min(width, math.ceil(x2 + padding))
        y2 = min(height, math.ceil(y2 + padding))

        if x1 >= x2 or y1 >= y2:
            raise ValueError("Bounding box is outside the image")

        return x1, y1, x2, y2

    @staticmethod
    def _save_image(image: np.ndarray, path: Path, overwrite: bool) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            raise FileExistsError(f"Crop already exists: {path}")

        if image.ndim == 2:
            encoded_image = image
        elif image.shape[2] == 4:
            encoded_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        else:
            encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        success, encoded = cv2.imencode(path.suffix.lower(), encoded_image)
        if not success:
            raise IOError(f"Cannot encode image: {path}")

        encoded.tofile(str(path))
        return str(path.resolve())

    def _crop(
        self,
        image: np.ndarray,
        component: LayoutComponentConfig,
        crop_id: int,
        padding: int,
        save_path: Path | None,
        overwrite: bool,
    ) -> CroppedImage:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self._normalize_bbox(component.coord, width, height, padding)
        cropped = image[y1:y2, x1:x2].copy()

        image_path = None
        if save_path is not None:
            image_path = self._save_image(
                cropped, save_path / f"crop_{crop_id + 1:04d}.png", overwrite
            )

        # Store the actual cropped coordinates, including padding and clipping.
        actual_component = replace(component, coord=[float(x1), float(y1), float(x2), float(y2)])

        return CroppedImage(
            crop_id=crop_id,
            image=cropped,
            component=actual_component,
            image_path=image_path,
        )

    def crop(
        self,
        source: ImageSource,
        bbox: BBox,
        crop_id: int = 0,
        padding: int | None = None,
        save_path: str | Path | None = None,
        overwrite: bool = False,
        label: str | None = None,
        confident: float | None = None,
    ) -> CroppedImage:
        image = self._load_image(source)
        pad = self.padding if padding is None else padding

        if pad < 0:
            raise ValueError("padding must be non-negative")

        output_dir = Path(save_path).expanduser() if save_path is not None else None

        return self._crop(
            image=image,
            bbox=bbox,
            crop_id=crop_id,
            padding=pad,
            save_path=output_dir,
            overwrite=overwrite,
            label=label,
            confident=confident,
        )

    def crop_many(
        self,
        source: ImageSource,
        boxes: Sequence[BBox],
        padding: int | None = None,
        save_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> list[CroppedImage]:
        image = self._load_image(source)
        pad = self.padding if padding is None else padding

        if pad < 0:
            raise ValueError("padding must be non-negative")

        output_dir = Path(save_path).expanduser() if save_path is not None else None

        return [
            self._crop(image, bbox, index, pad, output_dir, overwrite)
            for index, bbox in enumerate(boxes)
        ]

    def crop_layout(
        self,
        source: ImageSource,
        components: list[LayoutComponentConfig],
        labels: set[str] | None = None,
        min_confidence: float = 0.0,
        padding: int | None = None,
        save_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> list[CroppedImage]:
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")

        image = self._load_image(source)
        pad = self.padding if padding is None else padding
        if pad < 0:
            raise ValueError("padding must be non-negative")

        output_dir = Path(save_path).expanduser() if save_path is not None else None
        results: list[CroppedImage] = []

        for component in components:
            if labels is not None and component.label not in labels:
                continue
            if component.confident < min_confidence:
                continue

            results.append(self._crop(
                image=image,
                component=component,
                crop_id=len(results),
                padding=pad,
                save_path=output_dir,
                overwrite=overwrite,
            ))

        return results
