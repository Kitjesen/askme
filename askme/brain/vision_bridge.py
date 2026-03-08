"""
Vision bridge to the qp-perception library.

Handles lazy initialisation of the YOLO detector and tracker.
Falls back gracefully when qp-perception is not installed or no camera is available.

Usage::
    from askme.brain.vision_bridge import VisionBridge

    vision = VisionBridge()
    description = await vision.describe_scene()
    result = await vision.find_object("cup")
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from askme.config import get_config

logger = logging.getLogger(__name__)


class VisionBridge:
    """Thin wrapper around ``qp_perception`` with lazy init and graceful fallback.

    Supports two vision backends:
      - **YOLO** (primary): Real-time object detection via qp-perception
      - **VLM** (fallback): Rich scene understanding via Claude Sonnet API

    If YOLO is unavailable (qp-perception not installed), falls back to VLM
    for ``describe_scene()``.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._vision_cfg: dict[str, Any] = cfg.get("vision", {})

        self._enabled: bool = self._vision_cfg.get("enabled", False)
        self._model_path: str = self._vision_cfg.get(
            "model_path", "models/perception/yolo11n-seg.pt"
        )
        self._confidence: float = self._vision_cfg.get("confidence_threshold", 0.40)
        self._device: str = self._vision_cfg.get("device", "")
        self._camera_index: int = self._vision_cfg.get("camera_index", 0)

        # VLM fallback config (uses same OpenAI-compatible relay as brain)
        self._vlm_enabled: bool = self._vision_cfg.get("vlm_enabled", False)
        self._vlm_api_key: str = self._vision_cfg.get("vlm_api_key", "")
        self._vlm_model: str = self._vision_cfg.get("vlm_model", "claude-haiku-4-5-20251001")
        brain_cfg = cfg.get("brain", {})
        self._vlm_base_url: str = brain_cfg.get("base_url", "https://cursor.scihub.edu.kg/api/v1")

        # Lazily initialised heavy objects
        self._tracker: Any | None = None
        self._selector: Any | None = None
        self._init_attempted: bool = False
        self._vlm_client: Any | None = None
        self._vlm_backend: str = "openai"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_detector(self) -> bool:
        """Attempt to create ``YoloSegTracker`` + ``WeightedTargetSelector`` (once).

        Returns ``True`` if the detector is ready.
        """
        if self._tracker is not None:
            return True
        if self._init_attempted:
            return False

        self._init_attempted = True

        if not self._enabled:
            logger.info("[Vision] Vision disabled in config.")
            return False

        try:
            from qp_perception.tracking.yolo_seg import YoloSegTracker  # type: ignore[import-untyped]
            from qp_perception.selection.weighted import WeightedTargetSelector  # type: ignore[import-untyped]

            self._tracker = YoloSegTracker(
                model_path=self._model_path,
                confidence_threshold=self._confidence,
                device=self._device,
            )
            self._selector = WeightedTargetSelector(
                frame_width=640,
                frame_height=480,
            )
            logger.info("[Vision] YoloSegTracker + WeightedTargetSelector initialised.")
            return True

        except ImportError:
            logger.warning("[Vision] qp-perception not installed -- vision disabled.")
            return False
        except Exception as exc:
            logger.warning("[Vision] Detector init failed: %s", exc)
            self._tracker = None
            self._selector = None
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether any vision backend (YOLO or VLM) is usable."""
        return self._tracker is not None or self._vlm_enabled

    async def describe_scene(self, frame: Any = None) -> str:
        """Detect objects in *frame* and return a natural language description.

        Tries YOLO first (fast, structured), then falls back to VLM (rich, slower).
        If *frame* is ``None``, attempts to capture from camera.
        Returns an empty string if all vision backends are unavailable.
        """
        # Try YOLO first
        if self._ensure_detector():
            try:
                if frame is None:
                    frame = await asyncio.to_thread(self._capture_frame)
                if frame is None:
                    return await self._describe_scene_vlm()

                tracks = await asyncio.to_thread(
                    self._tracker.detect_and_track, frame, time.monotonic()
                )
                if tracks:
                    return self._tracks_to_description(tracks)
            except Exception as exc:
                logger.warning("[Vision] YOLO describe_scene error: %s", exc)

        # Fallback to VLM
        return await self._describe_scene_vlm(frame)

    async def find_object(self, target: str, frame: Any = None) -> dict[str, Any] | None:
        """Find a specific object class in *frame*.

        Returns a dict with ``bbox``, ``confidence``, ``center``, ``track_id``,
        or ``None`` if not found.
        """
        if not self._ensure_detector():
            return None

        try:
            if frame is None:
                frame = await asyncio.to_thread(self._capture_frame)
            if frame is None:
                return None

            tracks = await asyncio.to_thread(
                self._tracker.detect_and_track, frame, time.monotonic()
            )

            target_lower = target.lower()
            for track in tracks:
                if track.class_id.lower() == target_lower:
                    center = track.mask_center or track.bbox.center
                    return {
                        "track_id": track.track_id,
                        "class": track.class_id,
                        "bbox": {
                            "x": track.bbox.x,
                            "y": track.bbox.y,
                            "w": track.bbox.w,
                            "h": track.bbox.h,
                        },
                        "confidence": track.confidence,
                        "center": {"x": center[0], "y": center[1]},
                    }
            return None

        except Exception as exc:
            logger.warning("[Vision] find_object error: %s", exc)
            return None

    async def get_tracks(self, frame: Any) -> list[Any]:
        """Lower-level: return raw Track objects for robot control use.

        Returns an empty list if vision is unavailable.
        """
        if not self._ensure_detector():
            return []

        try:
            tracks = await asyncio.to_thread(
                self._tracker.detect_and_track, frame, time.monotonic()
            )
            return tracks
        except Exception as exc:
            logger.warning("[Vision] get_tracks error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # VLM (Claude Sonnet) fallback
    # ------------------------------------------------------------------

    def _ensure_vlm_client(self) -> bool:
        """Lazily initialise the VLM client. Returns True if ready.

        Tries Anthropic native SDK first (better for relay), falls back to
        OpenAI-compatible client.
        """
        if self._vlm_client is not None:
            return True
        if not self._vlm_enabled or not self._vlm_api_key:
            return False

        # Try Anthropic native SDK (relay: /api endpoint, no dev-assistant injection)
        anthropic_url = self._vlm_base_url.rstrip("/").removesuffix("/v1")
        try:
            import anthropic
            self._vlm_client = anthropic.Anthropic(
                api_key=self._vlm_api_key,
                base_url=anthropic_url,
            )
            self._vlm_backend = "anthropic"
            logger.info("[Vision] VLM client: Anthropic SDK (model=%s).", self._vlm_model)
            return True
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("[Vision] Anthropic SDK init failed: %s — trying OpenAI", exc)

        # Fallback: OpenAI-compatible client
        try:
            from openai import OpenAI
            self._vlm_client = OpenAI(
                api_key=self._vlm_api_key,
                base_url=self._vlm_base_url,
            )
            self._vlm_backend = "openai"
            logger.info("[Vision] VLM client: OpenAI compat (model=%s).", self._vlm_model)
            return True
        except ImportError:
            logger.warning("[Vision] Neither anthropic nor openai SDK installed — VLM disabled.")
            self._vlm_enabled = False
            return False
        except Exception as exc:
            logger.warning("[Vision] VLM client init failed: %s", exc)
            self._vlm_enabled = False
            return False

    async def _describe_scene_vlm(self, frame: Any = None) -> str:
        """Use VLM (via OpenAI-compatible relay) to describe a camera frame.

        Captures a frame if none provided, base64-encodes it, and sends to
        the relay API with a vision prompt.
        """
        if not self._ensure_vlm_client():
            return ""

        try:
            if frame is None:
                frame = await asyncio.to_thread(self._capture_frame)
            if frame is None:
                return ""

            # Encode frame as base64 JPEG
            import base64
            import cv2  # type: ignore[import-untyped]

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_b64 = base64.b64encode(buf).decode("utf-8")

            _VLM_TEXT = (
                    "I'm building a YOLO object detection test dataset. "
                    "List all visible objects in this image for annotation. "
                    "Output format: Chinese comma-separated list, no explanation."
                )

            # Call VLM — use Anthropic or OpenAI backend depending on what initialised
            def _call_vlm() -> str:
                if getattr(self, "_vlm_backend", "openai") == "anthropic":
                    response = self._vlm_client.messages.create(
                        model=self._vlm_model,
                        max_tokens=150,
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_b64,
                                    },
                                },
                                {"type": "text", "text": _VLM_TEXT},
                            ],
                        }],
                    )
                    raw = response.content[0].text if response.content else ""
                else:
                    response = self._vlm_client.chat.completions.create(
                        model=self._vlm_model,
                        max_tokens=150,
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                                },
                                {"type": "text", "text": _VLM_TEXT},
                            ],
                        }],
                    )
                    raw = response.choices[0].message.content or ""
                return VisionBridge._clean_vlm_response(raw)

            return await asyncio.to_thread(_call_vlm)

        except Exception as exc:
            logger.warning("[Vision] VLM describe failed: %s", exc)
            return ""

    _VLM_REFUSAL_MARKERS = (
        # English relay identity markers
        "I can't help", "I cannot", "I won't", "I appreciate",
        "I'm Claude", "I need to clarify", "privacy", "consent",
        # Chinese relay dev-assistant refusals
        "无法", "不在我的", "核心能力", "无法帮助", "很乐意协助",
        "如果您需要帮助构建", "软件开发", "专注于软件",
        "图像分析和数据集", "我很乐意",
    )

    @staticmethod
    def _clean_vlm_response(text: str) -> str:
        """Extract only the Chinese scene description from VLM output.

        The relay injects its own system prompt, causing the VLM to prepend
        English preamble or a dev-assistant refusal.  We strip that and return
        only the Chinese object-list description.
        Returns empty string if the response is a refusal.
        """
        import re

        # Fast-path: if the whole response is a refusal, bail immediately
        is_refusal = any(m in text for m in VisionBridge._VLM_REFUSAL_MARKERS)
        if is_refusal:
            logger.info("[Vision] VLM response flagged as refusal, returning empty.")
            return ""

        # Try explicit markers first: "简洁描述：..." or "描述：..."
        for marker in ("简洁描述：", "简洁描述:", "描述：", "描述:"):
            idx = text.find(marker)
            if idx != -1:
                extracted = text[idx + len(marker) :].strip()
                if not any(m in extracted for m in VisionBridge._VLM_REFUSAL_MARKERS):
                    return extracted

        # Fallback: find the longest run of text that's mostly Chinese
        lines = text.strip().split("\n")
        best = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            cn_chars = len(re.findall(r"[\u4e00-\u9fff]", line))
            if cn_chars > 5 and cn_chars > len(best) // 2:
                # Skip lines that contain refusal markers
                if not any(m in line for m in VisionBridge._VLM_REFUSAL_MARKERS):
                    best = line

        return best

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_frame(self) -> Any:
        """Capture a single frame from the default camera (blocking)."""
        try:
            import cv2  # type: ignore[import-untyped]

            cap = cv2.VideoCapture(self._camera_index)
            if not cap.isOpened():
                logger.warning("[Vision] Cannot open camera %d.", self._camera_index)
                return None
            ret, frame = cap.read()
            cap.release()
            if not ret:
                logger.warning("[Vision] Failed to read frame from camera.")
                return None
            return frame
        except ImportError:
            logger.warning("[Vision] cv2 not installed -- cannot capture frame.")
            return None
        except Exception as exc:
            logger.warning("[Vision] Camera capture error: %s", exc)
            return None

    @staticmethod
    def _tracks_to_description(tracks: list[Any]) -> str:
        """Convert a list of Track objects to a Chinese natural language string."""
        from collections import Counter

        counts: Counter[str] = Counter()
        for track in tracks:
            counts[track.class_id] += 1

        if not counts:
            return ""

        items = ", ".join(f"{count}个{name}" for name, count in counts.items())
        return f"我看到了: {items}"
