
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from document_processing.schemas.image import ImageData, CroppedImage
from layout_ocr_detection.schemas.layout_config import LayoutComponentConfig


type ImageSource = ImageData | str | Path | np.ndarray
type ImageInput = ImageData | np.ndarray
type BBox = Sequence[float]
type PixelBBox = tuple[int, int, int, int]


class ImageCropper:
    def __init__(self, padding: int = 0) -> None:
        if padding < 0:
            raise ValueError("padding must be non-negative")

        self.padding = padding

    @staticmethod
    def _load_image(source: ImageSource) -> ImageData:
        """Convert supported sources into RGB uint8 ImageData."""
        if isinstance(source, ImageData):
            return source

        if isinstance(source, (str, Path)):
            path = Path(source).expanduser().resolve()

            with Image.open(path) as img:
                image = np.asarray(
                    ImageOps.exif_transpose(img).convert("RGB")
                ).copy()

            return ImageData(
                image=image,
                image_path=str(path),
            )

        if isinstance(source, np.ndarray):
            if source.dtype != np.uint8:
                raise ValueError("Expected uint8 image")

            image = np.asarray(
                Image.fromarray(source).convert("RGB")
            ).copy()

            return ImageData(image=image)

        raise TypeError(
            f"Unsupported image source: {type(source).__name__}"
        )


    @staticmethod
    def _normalize_bbox(
        bbox: BBox,
        width: int,
        height: int,
        padding: int,
    ) -> PixelBBox:
        if len(bbox) != 4:
            raise ValueError(
                "Bounding box must contain [x1, y1, x2, y2]"
            )

        if not all(math.isfinite(float(v)) for v in bbox):
            raise ValueError(
                "Bounding box contains non-finite coordinates"
            )

        x1, y1, x2, y2 = map(float, bbox)

        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid bounding box: {bbox}")

        return_bbox = (
            max(0, min(width, math.floor(x1 - padding))),
            max(0, min(height, math.floor(y1 - padding))),
            max(0, min(width, math.ceil(x2 + padding))),
            max(0, min(height, math.ceil(y2 + padding))),
        )

        nx1, ny1, nx2, ny2 = return_bbox

        if nx1 >= nx2 or ny1 >= ny2:
            raise ValueError("Bounding box is outside the image")

        return return_bbox


    def _resolve_padding(self, padding: int | None) -> int:
        value = self.padding if padding is None else padding

        if value < 0:
            raise ValueError("padding must be non-negative")

        return value


    @classmethod
    def _crop(
        cls,
        source: ImageInput,
        bbox: BBox,
        padding: int = 0,
    ) -> tuple[np.ndarray, PixelBBox]:
        image = source.image if isinstance(source, ImageData) else source

        if not isinstance(image, np.ndarray):
            raise TypeError("Expected ImageData or np.ndarray")

        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected RGB uint8 image with shape (H, W, 3)")

        height, width = image.shape[:2]

        coords = cls._normalize_bbox(
            bbox=bbox,
            width=width,
            height=height,
            padding=padding,
        )

        x1, y1, x2, y2 = coords
        return image[y1:y2, x1:x2].copy(), coords


    def crop(
        self,
        source: ImageSource,
        bbox: BBox,
        padding: int | None = None,
    ) -> np.ndarray:
        pad = self._resolve_padding(padding)

        # Only file paths require loading.
        image = (
            self._load_image(source)
            if isinstance(source, (str, Path))
            else source
        )

        cropped, _ = self._crop(image, bbox, pad)
        return cropped


    def crop_many(
        self,
        source: ImageSource,
        boxes: Sequence[BBox],
        padding: int | None = None,
    ) -> list[np.ndarray]:
        """Crop multiple regions without reloading the input."""
        image = self._load_image(source)
        pad = self._resolve_padding(padding)

        return [
            self._crop(image, bbox, pad)[0]
            for bbox in boxes
        ]


    def crop_layout(
        self,
        source: ImageSource,
        components: Sequence[LayoutComponentConfig],
        labels: set[str] | None = None,
        min_confidence: float = 0.0,
        padding: int | None = None,
        save_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> list[CroppedImage]:
        """
        Crop layout components into CroppedImage instances.

        component.coord stores the actual bounding box after
        padding and clipping, in source-image coordinates.
        """
        if not 0 <= min_confidence <= 1:
            raise ValueError(
                "min_confidence must be between 0 and 1"
            )

        image = self._load_image(source)
        pad = self._resolve_padding(padding)

        results: list[CroppedImage] = []
        output_dir = (
            Path(save_path).expanduser()
            if save_path is not None
            else None
        )

        for component in components:
            if labels is not None and component.label not in labels:
                continue

            if component.confident < min_confidence:
                continue

            cropped, coords = self._crop(
                source=image,
                bbox=component.coord,
                padding=pad,
            )

            actual_component = replace(
                component,
                coord=[float(v) for v in coords],
            )

            result = CroppedImage(
                image=cropped,
                crop_id=len(results),
                component=actual_component,
            )

            if output_dir is not None:
                result.save(
                    output_dir / f"crop_{result.crop_id + 1:04d}.png",
                    overwrite=overwrite,
                )

            results.append(result)

        return results
