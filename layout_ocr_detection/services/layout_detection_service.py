from __future__ import annotations
from typing import Any, Literal
from paddleocr import LayoutDetection
import numpy as np

from layout_ocr_detection.schemas.layout_config import LayoutComponentConfig, LayoutPageConfig, LayoutFileConfig


class LayoutDetectionServices:
    def __init__(self, layout_model_name=None, device='cpu'):
        self.device = device
        self._init_pipeline(layout_model_name)


    def _init_pipeline(self, layout_model_name=None):
        if layout_model_name is None:
            layout_model_name = "PP-DocLayout_plus-L"
        if self.device == 'cpu':
            enable_mkldnn = False
        else:
            enable_mkldnn = True
        self.layout_detector = LayoutDetection(
            model_name=layout_model_name,
            device=self.device,
            enable_mkldnn=enable_mkldnn
        )       


    def get_predicted_layout(self, images: str | np.ndarray | list[str | np.ndarray]) -> list[LayoutFileConfig]:
        if isinstance(images, (str, np.ndarray)):
            images = [images]
        if not images:
            return []

        predicted_layouts = self.layout_detector.predict(input=images)
        layout_results: list[LayoutFileConfig] = []
        file_indices: dict[str, int] = {}

        for index, layout in enumerate(predicted_layouts):
            data = layout.json["res"]
            input_path = data.get("input_path")
            file_path = str(input_path) if input_path is not None else f"image_{index}"

            components = [
                LayoutComponentConfig(
                    label=str(box["label"]),
                    coord=[float(value) for value in box["coordinate"]],
                    confident=float(box["score"])
                )
                for box in data.get("boxes", [])
            ]

            if file_path not in file_indices:
                file_indices[file_path] = len(layout_results)
                layout_results.append(LayoutFileConfig(
                    file_id=file_indices[file_path],
                    file_path=file_path,
                    page_layout=[]
                ))

            file_config = layout_results[file_indices[file_path]]
            file_config.page_layout.append(LayoutPageConfig(
                page_id=len(file_config.page_layout),
                components=components
            ))

        return layout_results


    def get_table(self, layout: LayoutPageConfig) -> list[LayoutComponentConfig]:
        return [
            component
            for component in layout.components
            if component.label == "table"
        ]
