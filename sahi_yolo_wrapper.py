"""
sahi_yolo_wrapper.py
--------------------
Unified SAHI DetectionModel wrapper that works with YOLOX, YOLOv8, and YOLOv26
ONNX backends.

Usage:
    from sahi_yolo_wrapper import YOLOSAHIWrapper
    from sahi.predict import get_sliced_prediction

    model = YOLOSAHIWrapper(
        model_path="best_yolov8.onnx",
        confidence_threshold=0.25,
        category_mapping={0: "rbc", 1: "retic", ...},
        model_type="yolov8",   # "yolox" | "yolov8" | "yolov26"
        image_size=416,
    )

    result = get_sliced_prediction(image, model, slice_height=416, ...)
"""

from typing import List, Optional

import numpy as np
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction

from onnx_backends import MODEL_REGISTRY


class YOLOSAHIWrapper(DetectionModel):
    """
    A single SAHI wrapper that delegates inference to the correct
    ONNX backend based on ``model_type``.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        category_mapping: dict,
        model_type: str = "yolox",
        image_size: int = 416,
        load_at_init: bool = True,
        **kwargs,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.category_mapping = category_mapping
        self.image_size = image_size
        self.load_at_init = load_at_init
        self.model_type = model_type.lower()
        self.num_classes = None
        self.category_remapping = None

        if self.model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                f"Supported: {list(MODEL_REGISTRY.keys())}"
            )

        if self.load_at_init:
            self.load_model()

    # ─── DetectionModel interface ────────────────────────

    def load_model(self):
        backend_cls = MODEL_REGISTRY[self.model_type]
        model = backend_cls(
            model_path=self.model_path,
            img_size=self.image_size,
            conf_threshold=self.confidence_threshold,
        )
        self.set_model(model)

    def set_model(self, model):
        self.model = model

    @property
    def num_categories(self):
        return len(self.category_mapping)

    @property
    def has_mask(self):
        return False

    @property
    def category_names(self):
        return list(self.category_mapping.values())

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        Run the backend on a single image (BGR uint8 ndarray).
        SAHI will call this once per slice.
        """
        prediction_result = self.model.infer(image)
        self._original_predictions = prediction_result

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        Convert the (N, 6) array into SAHI ObjectPrediction objects.
        All backends share the same output contract:
            [x1, y1, x2, y2, score, class_id]
        """
        original_predictions = self._original_predictions

        # normalise list shapes
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        object_prediction_list = []

        if original_predictions is not None:
            boxes = original_predictions[:, :4]
            scores = original_predictions[:, 4]
            class_ids = original_predictions[:, 5].astype(int)


            for i, box in enumerate(boxes):
                object_prediction = ObjectPrediction(
                    bbox=box.tolist(),
                    category_id=int(class_ids[i]),
                    category_name=self.category_mapping[str(int(class_ids[i]))],
                    shift_amount=shift_amount,
                    score=float(scores[i]),
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)

        self._object_prediction_list_per_image = [object_prediction_list]
