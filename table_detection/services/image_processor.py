
"""Shared RGB image preprocessing and image transformations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np

from document_processing.schemas.image import ImageData


type ImageInput = ImageData | np.ndarray
type SharpenMethod = Literal["none", "unsharp", "clahe_unsharp"]
type DenoiseMethod = Literal["none", "gaussian", "median", "bilateral"]
type MorphologyOperation = Literal[
    "erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"
]


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    enabled_resize: bool = False
    target_long_side: int = 2048
    allow_upscale: bool = False

    denoise_method: DenoiseMethod = "none"
    denoise_kernel: int = 3

    sharpen_method: SharpenMethod = "none"
    sharpen_amount: float = 1.0
    sharpen_sigma: float = 1.2
    clahe_clip_limit: float = 2.0

    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.target_long_side < 16:
            raise ValueError("target_long_side must be >= 16")

        if self.denoise_method not in (
            "none", "gaussian", "median", "bilateral"
        ):
            raise ValueError("Invalid denoise_method")

        if self.denoise_kernel < 1 or self.denoise_kernel % 2 == 0:
            raise ValueError("denoise_kernel must be positive and odd")

        if self.sharpen_method not in (
            "none", "unsharp", "clahe_unsharp"
        ):
            raise ValueError("Invalid sharpen_method")

        if not 0 <= self.sharpen_amount <= 5:
            raise ValueError("sharpen_amount must be in [0, 5]")

        if self.sharpen_sigma <= 0:
            raise ValueError("sharpen_sigma must be positive")

        if self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive")

        if self.gamma <= 0:
            raise ValueError("gamma must be positive")


class ImagePreprocessor:
    def __init__(
        self,
        config: PreprocessConfig | None = None,
    ) -> None:
        self.config = config or PreprocessConfig()


    @staticmethod
    def _get_image(source: ImageInput) -> np.ndarray:
        """Get RGB uint8 ndarray without changing the source."""
        image = source.image if isinstance(source, ImageData) else source

        if not isinstance(image, np.ndarray):
            raise TypeError("Expected ImageData or numpy.ndarray")

        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.size == 0
        ):
            raise ValueError(
                "Expected non-empty RGB uint8 image (H, W, 3)"
            )

        return image


    @classmethod
    def to_grayscale(cls, source: ImageInput) -> np.ndarray:
        """Convert RGB to grayscale uint8."""
        return cv2.cvtColor(
            cls._get_image(source),
            cv2.COLOR_RGB2GRAY,
        )


    @classmethod
    def resize(
        cls,
        source: ImageInput,
        target_long_side: int,
        allow_upscale: bool = False,
    ) -> np.ndarray:
        """Resize while preserving aspect ratio."""
        if target_long_side < 1:
            raise ValueError("target_long_side must be positive")

        image = cls._get_image(source)
        height, width = image.shape[:2]

        scale = target_long_side / max(height, width)

        if not allow_upscale:
            scale = min(scale, 1.0)

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        if (new_width, new_height) == (width, height):
            return image.copy()

        interpolation = (
            cv2.INTER_AREA
            if scale < 1
            else cv2.INTER_CUBIC
        )

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation,
        )


    @classmethod
    def denoise(
        cls,
        source: ImageInput,
        method: DenoiseMethod = "gaussian",
        kernel_size: int = 3,
    ) -> np.ndarray:
        """Remove noise using Gaussian, median or bilateral filtering."""
        image = cls._get_image(source)

        if method == "none":
            return image.copy()

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")

        if method == "gaussian":
            return cv2.GaussianBlur(
                image, (kernel_size, kernel_size), 0
            )

        if method == "median":
            return cv2.medianBlur(image, kernel_size)

        if method == "bilateral":
            return cv2.bilateralFilter(
                image,
                d=kernel_size,
                sigmaColor=50,
                sigmaSpace=50,
            )

        raise ValueError(f"Unsupported denoise method: {method}")


    @classmethod
    def clahe(
        cls,
        source: ImageInput,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Apply CLAHE to luminance while preserving RGB colors."""
        if clip_limit <= 0 or min(tile_grid_size) < 1:
            raise ValueError("Invalid CLAHE parameters")

        image = cls._get_image(source)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


    @classmethod
    def unsharp(
        cls,
        source: ImageInput,
        amount: float = 1.0,
        sigma: float = 1.2,
    ) -> np.ndarray:
        """Sharpen the image using an unsharp mask."""
        if amount < 0 or sigma <= 0:
            raise ValueError("Invalid unsharp parameters")

        image = cls._get_image(source)

        if amount == 0:
            return image.copy()

        blurred = cv2.GaussianBlur(
            image, (0, 0), sigmaX=sigma
        )

        return cv2.addWeighted(
            image,
            1.0 + amount,
            blurred,
            -amount,
            0,
        )


    @classmethod
    def adjust_gamma(
        cls,
        source: ImageInput,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Apply gamma correction (gamma < 1 brightens)."""
        if gamma <= 0:
            raise ValueError("gamma must be positive")

        image = cls._get_image(source)

        if gamma == 1:
            return image.copy()

        lut = np.array(
            [
                round(255 * ((value / 255.0) ** gamma))
                for value in range(256)
            ],
            dtype=np.uint8,
        )

        return cv2.LUT(image, lut)


    @classmethod
    def threshold(
        cls,
        source: ImageInput,
        value: int = 127,
        invert: bool = True,
        otsu: bool = False,
    ) -> np.ndarray:
        """Create a grayscale binary mask."""
        if not 0 <= value <= 255:
            raise ValueError("Threshold must be in [0, 255]")

        gray = cls.to_grayscale(source)

        mode = (
            cv2.THRESH_BINARY_INV
            if invert
            else cv2.THRESH_BINARY
        )

        if otsu:
            mode |= cv2.THRESH_OTSU

        _, binary = cv2.threshold(
            gray, value if not otsu else 0, 255, mode
        )

        return binary


    @classmethod
    def adaptive_threshold(
        cls,
        source: ImageInput,
        block_size: int = 31,
        c: float = 9.0,
        gaussian: bool = True,
        invert: bool = True,
    ) -> np.ndarray:
        """Create a binary mask using local adaptive thresholding."""
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("block_size must be odd and >= 3")

        gray = cls.to_grayscale(source)

        method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if gaussian
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )

        mode = (
            cv2.THRESH_BINARY_INV
            if invert
            else cv2.THRESH_BINARY
        )

        return cv2.adaptiveThreshold(
            gray, 255, method, mode, block_size, c
        )


    @staticmethod
    def morphology(
        mask: np.ndarray,
        operation: MorphologyOperation = "open",
        kernel_size: tuple[int, int] = (3, 3),
        iterations: int = 1,
    ) -> np.ndarray:
        """Apply morphology to a uint8 single-channel mask."""
        if (
            not isinstance(mask, np.ndarray)
            or mask.ndim != 2
            or mask.dtype != np.uint8
            or mask.size == 0
        ):
            raise ValueError("Expected non-empty uint8 2D mask")

        if min(kernel_size) < 1 or iterations < 1:
            raise ValueError("Invalid morphology parameters")

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            kernel_size,  # (width, height)
        )

        if operation == "erode":
            return cv2.erode(mask, kernel, iterations=iterations)

        if operation == "dilate":
            return cv2.dilate(mask, kernel, iterations=iterations)

        modes = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }

        if operation not in modes:
            raise ValueError(f"Unsupported morphology: {operation}")

        return cv2.morphologyEx(
            mask,
            modes[operation],
            kernel,
            iterations=iterations,
        )


    def process(
        self,
        source: ImageInput,
    ) -> tuple[ImageData, dict]:
        """
        Run the configured preprocessing pipeline.

        Returns:
            (processed ImageData, transformation metadata)

        Coordinates in processed image may differ if resize is enabled.
        """
        original = self._get_image(source)
        height, width = original.shape[:2]
        cfg = self.config

        result = original.copy()

        # 1. Resize
        if cfg.enabled_resize:
            result = self.resize(
                result,
                target_long_side=cfg.target_long_side,
                allow_upscale=cfg.allow_upscale,
            )

        # 2. Denoise
        if cfg.denoise_method != "none":
            result = self.denoise(
                result,
                method=cfg.denoise_method,
                kernel_size=cfg.denoise_kernel,
            )

        # 3. Contrast enhancement
        if cfg.sharpen_method == "clahe_unsharp":
            result = self.clahe(
                result,
                clip_limit=cfg.clahe_clip_limit,
            )

        # 4. Sharpen
        if (
            cfg.sharpen_method != "none"
            and cfg.sharpen_amount > 0
        ):
            result = self.unsharp(
                result,
                amount=cfg.sharpen_amount,
                sigma=cfg.sharpen_sigma,
            )

        # 5. Gamma correction
        if cfg.gamma != 1.0:
            result = self.adjust_gamma(result, cfg.gamma)

        out_height, out_width = result.shape[:2]

        metadata = {
            "original": {
                "width": width,
                "height": height,
            },
            "processed": {
                "width": out_width,
                "height": out_height,
            },
            "scale_x": out_width / width,
            "scale_y": out_height / height,
            "config": asdict(cfg),
        }

        return ImageData(image=result), metadata
