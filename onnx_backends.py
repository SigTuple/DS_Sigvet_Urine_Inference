"""
onnx_backends.py
----------------
Unified ONNX inference backends for YOLOX, YOLOv8, and YOLOv26.

Every backend exposes the same contract:
    model.infer(image_rgb) -> np.ndarray | None
        Accepts an RGB image (HWC, uint8) — the format SAHI delivers.
        Returns an (N, 6) array: [x1, y1, x2, y2, score, class_id]
        or None when there are no detections.
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2
import onnxruntime as ort


# ──────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────
class BaseONNXModel(ABC):
    """Abstract base for all ONNX detection models."""

    def __init__(self, model_path: str, img_size: int = 416, conf_threshold: float = 0.25):
        self.model_path = model_path
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    @abstractmethod
    def infer(self, image: np.ndarray) -> np.ndarray | None:
        """
        Run inference on an RGB image (np.ndarray, HWC, uint8).
        Returns (N, 6) array [x1, y1, x2, y2, score, class_id] or None.
        """
        ...


# ──────────────────────────────────────────────
# YOLOX backend
# ──────────────────────────────────────────────
class YOLOXONNXModel(BaseONNXModel):
    """
    YOLOX ONNX inference.

    Preprocessing : pad to input_size, CHW float32 (no /255 normalization).
    Postprocessing: grid decode → cxcywh→xyxy → multiclass NMS.
    """

    def __init__(self, model_path: str, img_size: int = 416,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        super().__init__(model_path, img_size, conf_threshold)
        self.iou_threshold = iou_threshold
        self._ratio = 1.0

    # ---- preprocessing ----
    def _preprocess(self, img_rgb: np.ndarray) -> np.ndarray:
        """Pad + transpose. Expects RGB input; converts to BGR for YOLOX."""
        img = img_rgb[:, :, ::-1]  # RGB → BGR  (YOLOX models are trained on BGR)
        input_size = (self.img_size, self.img_size)
        padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized

        padded = padded.transpose(2, 0, 1)  # HWC → CHW
        padded = np.ascontiguousarray(padded, dtype=np.float32)
        self._ratio = r
        return padded

    # ---- NMS helpers ----
    @staticmethod
    def _nms(boxes, scores, nms_thr):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]
        return keep

    def _multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
        valid_mask = cls_scores > score_thr
        if valid_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_mask]
        valid_boxes = boxes[valid_mask]
        valid_cls_inds = cls_inds[valid_mask]
        keep = self._nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            return np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return None

    def _multiclass_nms_class_aware(self, boxes, scores, nms_thr, score_thr):
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_mask = cls_scores > score_thr
            if valid_mask.sum() == 0:
                continue
            valid_scores = cls_scores[valid_mask]
            valid_boxes = boxes[valid_mask]
            keep = self._nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def _multiclass_nms(self, boxes, scores, nms_thr, score_thr, class_agnostic=True):
        fn = self._multiclass_nms_class_agnostic if class_agnostic else self._multiclass_nms_class_aware
        return fn(boxes, scores, nms_thr, score_thr)

    # ---- postprocessing ----
    def _postprocess(self, outputs: np.ndarray) -> np.ndarray | None:
        """Decode grid predictions → xyxy + NMS."""
        img_size = (self.img_size, self.img_size)
        grids, expanded_strides = [], []
        strides = [8, 16, 32]

        hsizes = [img_size[0] // s for s in strides]
        wsizes = [img_size[1] // s for s in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((*grid.shape[:2], 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)

        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        predictions = outputs[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= self._ratio

        return self._multiclass_nms(boxes_xyxy, scores,
                                    nms_thr=self.iou_threshold,
                                    score_thr=self.conf_threshold)

    # ---- public API ----
    def infer(self, image: np.ndarray) -> np.ndarray | None:
        preprocessed = self._preprocess(image)
        raw = self.session.run(None, {self.input_name: np.expand_dims(preprocessed, 0)})
        return self._postprocess(raw[0])


# ──────────────────────────────────────────────
# YOLOv8 / YOLOv26 backend  (identical logic)
# ──────────────────────────────────────────────
class YOLOv8ONNXModel(BaseONNXModel):
    """
    YOLOv8 / YOLOv26 ONNX inference.

    Preprocessing : letterbox (centered padding) + /255 + CHW.
                    Expects RGB input (as provided by SAHI).
    Postprocessing: the ONNX model already outputs (N, 6) decoded detections
                    [x1, y1, x2, y2, score, class_id].
    """

    def __init__(self, model_path: str, img_size: int = 416,
                 conf_threshold: float = 0.25):
        super().__init__(model_path, img_size, conf_threshold)

    # ---- letterbox ----
    @staticmethod
    def _letterbox(img, new_shape, color=(114, 114, 114)):
        h, w = img.shape[:2]
        ratio = min(new_shape / h, new_shape / w)
        new_unpad = (int(round(w * ratio)), int(round(h * ratio)))

        dw = (new_shape - new_unpad[0]) / 2.0
        dh = (new_shape - new_unpad[1]) / 2.0

        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
        return padded, ratio, (dw, dh)

    # ---- preprocessing ----
    def _preprocess(self, img_rgb: np.ndarray):
        """Letterbox + normalize. Expects RGB input (SAHI convention)."""
        img_lb, ratio, dwdh = self._letterbox(img_rgb, self.img_size)
        img_norm = img_lb / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        tensor = np.expand_dims(img_chw, 0).astype(np.float32)
        return tensor, ratio, dwdh, img_rgb.shape[:2]

    # ---- scale boxes back to original image coordinates ----
    @staticmethod
    def _scale_boxes(boxes, ratio, dwdh, orig_shape):
        dw, dh = dwdh
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= ratio
        h, w = orig_shape
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
        return boxes

    # ---- public API ----
    def infer(self, image: np.ndarray) -> np.ndarray | None:
        tensor, ratio, dwdh, orig_shape = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        preds = outputs[0][0]  # (N, 6)

        if preds.shape[0] == 0:
            return None

        boxes = preds[:, :4]
        scores = preds[:, 4]
        class_ids = preds[:, 5]

        # confidence filter
        mask = scores > self.conf_threshold
        if mask.sum() == 0:
            return None

        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # rescale to original image space
        boxes = self._scale_boxes(boxes, ratio, dwdh, orig_shape)

        return np.column_stack([boxes, scores, class_ids])


# ──────────────────────────────────────────────
# Registry for easy lookup by name
# ──────────────────────────────────────────────
MODEL_REGISTRY = {
    "yolox": YOLOXONNXModel,
    "yolov8": YOLOv8ONNXModel,
    "yolov26": YOLOv8ONNXModel,  # identical pipeline
}
