"""Object detection backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from askme.runtime.registry import BackendRegistry


class DetectorBackend(ABC):
    """Abstract object detector — frame to detections."""

    @abstractmethod
    def __init__(self, cfg: dict[str, Any]) -> None: ...

    @abstractmethod
    def detect(self, frame: Any) -> list[dict]:
        """Run detection on a frame. Returns list of detection dicts.

        Each dict: {"label": str, "confidence": float, "bbox": [x1,y1,x2,y2], "distance_m": float|None}
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model identifier."""

    def supports_segmentation(self) -> bool:
        """Whether this backend also produces segmentation masks."""
        return False


detector_registry = BackendRegistry("detector", DetectorBackend, default="bpu_yolo")
