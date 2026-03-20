"""BPU-accelerated YOLO inference for Horizon J6 (RDK X5 / S100P).

Replaces ultralytics CPU inference (~1000ms) with Nash BPU (~3ms).
Input: RGB numpy frame → NV12 conversion → BPU forward → NMS → detections.

Usage::

    detector = BPUYoloDetector("/path/to/model.hbm")
    detections = detector.detect(rgb_frame)
    # [{"class_id": "person", "confidence": 0.85, "bbox": [x1,y1,x2,y2]}, ...]
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# COCO 80 class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _rgb_to_nv12(rgb: np.ndarray, target_h: int = 640, target_w: int = 640) -> np.ndarray:
    """Convert RGB frame to NV12 format (Y + interleaved UV), resized to target.

    Uses integer approximation of BT.601 to avoid OpenCV dependency.
    """
    from PIL import Image as PILImage

    # Resize to target
    img = PILImage.fromarray(rgb)
    img = img.resize((target_w, target_h), PILImage.BILINEAR)
    rgb_resized = np.asarray(img, dtype=np.int32)

    r = rgb_resized[:, :, 0]
    g = rgb_resized[:, :, 1]
    b = rgb_resized[:, :, 2]

    # BT.601 RGB→YUV
    y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
    u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128
    v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128

    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)

    # Subsample UV (2x2 blocks)
    u_sub = u[0::2, 0::2]
    v_sub = v[0::2, 0::2]

    # Interleave UV
    uv = np.empty((target_h // 2, target_w), dtype=np.uint8)
    uv[:, 0::2] = u_sub
    uv[:, 1::2] = v_sub

    # Concatenate Y + UV as 1D
    nv12 = np.concatenate([y.ravel(), uv.ravel()])
    return nv12


def _decode_yolo_outputs(
    outputs: list[Any],
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    img_h: int = 640,
    img_w: int = 640,
) -> list[dict[str, Any]]:
    """Decode YOLO v11 seg BPU outputs into detection list.

    BPU outputs (10 tensors):
      [0,3,6] = class scores at 3 scales (80x80, 40x40, 20x20) × 80 classes
      [1,4,7] = bbox regression at 3 scales × 64 (4×16 DFL)
      [2,5,8] = mask coefficients at 3 scales × 32
      [9]     = proto masks (160x160x32)
    """
    all_boxes = []
    all_scores = []
    all_class_ids = []

    strides = [8, 16, 32]

    for scale_idx in range(3):
        cls_out = outputs[scale_idx * 3].buffer[0]      # (H, W, 80)
        bbox_out = outputs[scale_idx * 3 + 1].buffer[0]  # (H, W, 64)

        grid_h, grid_w, num_classes = cls_out.shape
        stride = strides[scale_idx]

        # Sigmoid on class scores
        cls_scores = 1.0 / (1.0 + np.exp(-cls_out.astype(np.float32)))

        # Find detections above threshold
        max_scores = cls_scores.max(axis=-1)
        mask = max_scores > conf_threshold
        if not mask.any():
            continue

        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            scores = cls_scores[y, x]
            cls_id = int(scores.argmax())
            score = float(scores[cls_id])

            if score < conf_threshold:
                continue

            # Decode DFL bbox (simplified: use argmax of 16-bin distribution)
            bbox_raw = bbox_out[y, x].astype(np.float32).reshape(4, 16)
            # Softmax per coordinate
            bbox_exp = np.exp(bbox_raw - bbox_raw.max(axis=1, keepdims=True))
            bbox_soft = bbox_exp / bbox_exp.sum(axis=1, keepdims=True)
            # Expected value
            indices = np.arange(16, dtype=np.float32)
            dfl = (bbox_soft * indices).sum(axis=1)

            # Convert to xyxy
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            x1 = cx - dfl[0] * stride
            y1 = cy - dfl[1] * stride
            x2 = cx + dfl[2] * stride
            y2 = cy + dfl[3] * stride

            # Clip to image
            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))

            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(score)
            all_class_ids.append(cls_id)

    if not all_boxes:
        return []

    # Simple NMS
    boxes = np.array(all_boxes)
    scores = np.array(all_scores)
    class_ids = np.array(all_class_ids)

    keep = _nms(boxes, scores, iou_threshold)

    detections = []
    for idx in keep:
        detections.append({
            "class_id": COCO_CLASSES[class_ids[idx]] if class_ids[idx] < len(COCO_CLASSES) else f"class_{class_ids[idx]}",
            "confidence": float(scores[idx]),
            "bbox": [float(v) for v in boxes[idx]],
            "center": [float((boxes[idx][0] + boxes[idx][2]) / 2),
                       float((boxes[idx][1] + boxes[idx][3]) / 2)],
        })

    return detections


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Simple greedy NMS."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


class BPUYoloDetector:
    """YOLO detector running on Horizon Nash BPU.

    Drop-in replacement for ultralytics CPU inference.
    ~3ms per frame vs ~1000ms on CPU.
    """

    def __init__(
        self,
        model_path: str = "/home/sunrise/data/models/yolo11s_seg_nashe_640x640_nv12.hbm",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> None:
        self._model_path = model_path
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._model: Any = None
        self._init_attempted = False

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._init_attempted:
            return False
        self._init_attempted = True
        try:
            from hobot_dnn import pyeasy_dnn
            models = pyeasy_dnn.load(self._model_path)
            self._model = models[0]
            logger.info("[BPU] YOLO loaded: %s", self._model_path.split("/")[-1])
            return True
        except ImportError:
            logger.info("[BPU] hobot_dnn not available (not on RDK platform)")
            return False
        except Exception as exc:
            logger.warning("[BPU] Model load failed: %s", exc)
            return False

    @property
    def available(self) -> bool:
        return self._ensure_model()

    def detect(self, rgb_frame: np.ndarray) -> list[dict[str, Any]]:
        """Run YOLO detection on an RGB frame. Returns list of detections."""
        if not self._ensure_model():
            return []

        t0 = time.monotonic()

        # RGB → NV12
        nv12 = _rgb_to_nv12(rgb_frame, 640, 640)
        t_preprocess = time.monotonic() - t0

        # BPU inference
        t1 = time.monotonic()
        outputs = self._model.forward(nv12)
        t_infer = time.monotonic() - t1

        # Decode
        t2 = time.monotonic()
        orig_h, orig_w = rgb_frame.shape[:2]
        detections = _decode_yolo_outputs(
            outputs, self._confidence, self._iou_threshold
        )

        # Scale bbox back to original image size
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0
        for det in detections:
            det["bbox"][0] *= scale_x
            det["bbox"][1] *= scale_y
            det["bbox"][2] *= scale_x
            det["bbox"][3] *= scale_y
            det["center"][0] *= scale_x
            det["center"][1] *= scale_y

        t_postprocess = time.monotonic() - t2
        total = (time.monotonic() - t0) * 1000

        logger.debug(
            "[BPU] YOLO: %.0fms total (pre=%.0f infer=%.0f post=%.0f) → %d objects",
            total, t_preprocess * 1000, t_infer * 1000, t_postprocess * 1000,
            len(detections),
        )

        return detections
