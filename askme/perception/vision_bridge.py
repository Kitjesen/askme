"""
Vision bridge to the qp-perception library.

Handles lazy initialisation of the YOLO detector and tracker.
Falls back gracefully when qp-perception is not installed or no camera is available.

Usage::
    from askme.perception.vision_bridge import VisionBridge

    vision = VisionBridge()
    description = await vision.describe_scene()
    result = await vision.find_object("cup")
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from askme.config import get_config
from askme.constants import (
    DAEMON_COLOR_FRAME_PATH,
    DAEMON_DEPTH_FRAME_PATH,
    DAEMON_DETECTIONS_PATH,
    DAEMON_HEARTBEAT_PATH,
    DAEMON_ROS2_FRAME_PATH,
    DEFAULT_BPU_MODEL_PATH,
)

logger = logging.getLogger(__name__)


def _count_persons(objects: list[dict[str, Any]]) -> int:
    total = 0
    for item in objects:
        label = str(
            item.get("label")
            or item.get("class_id")
            or item.get("class")
            or item.get("name")
            or ""
        ).strip().lower()
        if label in {"person", "human", "visitor", "\u4eba", "\u6e38\u5ba2"}:
            total += 1
    return total


# ---------------------------------------------------------------------------
# ROS2 frame grabber — persistent subscriber, grabs latest frame on demand
# ---------------------------------------------------------------------------

class _ROS2FrameGrabber:
    """Grabs frames from a ROS2 Image topic via subprocess.

    Uses the system Python (with ROS2 sourced) to subscribe to the topic
    and write raw frame data to a temp file. This avoids rclpy/venv
    compatibility issues entirely.
    """

    # Small ROS2 script executed by system Python (outside venv)
    _GRAB_SCRIPT = '''\
import sys, time, struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

rclpy.init()
node = Node("askme_grab")
frame = [None]
def cb(msg):
    frame[0] = msg
node.create_subscription(Image, sys.argv[1], cb, 1)
deadline = time.monotonic() + float(sys.argv[3])
while frame[0] is None and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.5)
node.destroy_node()
rclpy.shutdown()
if frame[0] is None:
    sys.exit(1)
m = frame[0]
out = sys.argv[2]
# Write binary: 4 bytes width + 4 bytes height + raw RGB data
with open(out, "wb") as f:
    f.write(struct.pack("II", m.width, m.height))
    f.write(bytes(m.data))
'''

    def __init__(self, topic: str = "/camera/color/image_raw", timeout: float = 5.0) -> None:
        self._topic = topic
        self._timeout = timeout
        self._tmp_path = DAEMON_ROS2_FRAME_PATH

    def grab(self) -> Any:
        """Grab a single frame via subprocess. Returns numpy array (H, W, 3) or None."""
        import struct
        import subprocess

        import numpy as np

        # Use bash -c to source ROS2 setup, then run grab script with system python
        cmd = (
            f'source /opt/ros/humble/setup.bash && '
            f'python3 -c {_shell_quote(self._GRAB_SCRIPT)} '
            f'{_shell_quote(self._topic)} '
            f'{_shell_quote(self._tmp_path)} '
            f'{self._timeout}'
        )
        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True, timeout=self._timeout + 5,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace")[:200]
                logger.warning("[Vision] ROS2 grab subprocess failed: %s", stderr)
                return None

            with open(self._tmp_path, "rb") as f:
                header = f.read(8)
                if len(header) < 8:
                    return None
                w, h = struct.unpack("II", header)
                data = f.read(w * h * 3)
                if len(data) != w * h * 3:
                    return None
                return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)

        except subprocess.TimeoutExpired:
            logger.warning("[Vision] ROS2 grab timed out after %.1fs", self._timeout + 5)
            return None
        except Exception as exc:
            logger.warning("[Vision] ROS2 grab error: %s", exc)
            return None


def _shell_quote(s: str) -> str:
    """Shell-quote a string for bash -c."""
    import shlex
    return shlex.quote(s)


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
        # Capture backend: "auto" tries ros2 first, then cv2
        self._capture_backend: str = self._vision_cfg.get("capture_backend", "auto")
        self._ros2_topic: str = self._vision_cfg.get("ros2_topic", "/camera/color/image_raw")
        self._ros2_grabber: _ROS2FrameGrabber | None = None
        self._lingtu_repo: str = self._vision_cfg.get(
            "lingtu_repo", "/opt/lingtu/current"
        )
        self._lingtu_color_shm: str = self._vision_cfg.get(
            "lingtu_color_shm", "/dev/shm/lingtu_camera_color"
        )
        self._lingtu_max_age_s: float = max(
            0.05, float(self._vision_cfg.get("lingtu_max_age_s", 1.0))
        )
        self._lingtu_rotate_180: bool = bool(
            self._vision_cfg.get("lingtu_rotate_180", True)
        )

        # VLM fallback config
        self._vlm_enabled: bool = self._vision_cfg.get("vlm_enabled", False)
        self._vlm_api_key: str = self._vision_cfg.get("vlm_api_key", "")
        self._vlm_model: str = self._vision_cfg.get("vlm_model", "qwen-vl-max")
        # VLM base URL: vision-specific first, then brain relay fallback
        self._vlm_base_url: str = self._vision_cfg.get(
            "vlm_base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self._vlm_timeout: float = max(
            5.0, float(self._vision_cfg.get("vlm_timeout", 20.0))
        )
        self._vlm_image_max_width: int = max(
            160, int(self._vision_cfg.get("vlm_image_max_width", 320))
        )

        # BPU YOLO (fast path, ~3ms on Horizon J6)
        self._bpu_model_path: str = self._vision_cfg.get(
            "bpu_model_path", DEFAULT_BPU_MODEL_PATH
        )
        self._bpu_detector: Any | None = None
        self._bpu_init_attempted: bool = False

        # Lazily initialised heavy objects (ultralytics CPU fallback)
        self._tracker: Any | None = None
        self._selector: Any | None = None
        self._init_attempted: bool = False
        self._vlm_client: Any | None = None
        self._vlm_backend: str = self._vision_cfg.get("vlm_backend", "openai")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _read_daemon_detections(self, max_age: float = 2.0) -> list[dict[str, Any]] | None:
        """Read latest BPU detections from frame_daemon shared JSON.

        Enriches each detection with depth (meters) from the depth frame.
        Returns None if daemon is stale/missing.
        """
        import json as _json
        det_path = DAEMON_DETECTIONS_PATH
        try:
            with open(det_path) as f:
                data = _json.load(f)
            if time.time() - data.get("timestamp", 0) > max_age:
                return None
            dets = data.get("detections", [])
            # Enrich with depth
            for d in dets:
                bbox = d.get("bbox", [])
                if len(bbox) == 4:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    depth_m = self.read_depth_at(int(cx), int(cy))
                    if depth_m is not None:
                        d["distance_m"] = round(depth_m, 2)
            return dets
        except (FileNotFoundError, ValueError, KeyError):
            return None

    def interaction_snapshot(self, max_age: float = 2.0) -> dict[str, Any]:
        """Return camera evidence for the voice InteractionGate.

        The snapshot is intentionally conservative: if the daemon is missing or
        stale we still return the timestamp/reason, so the gate can avoid using
        old visual facts as proof that a nearby person is addressing the robot.
        """
        import json as _json

        try:
            with open(DAEMON_DETECTIONS_PATH) as f:
                data = _json.load(f)
        except (FileNotFoundError, ValueError, OSError):
            return {
                "source": "vision_daemon",
                "reason": "daemon_missing",
                "observed_at": None,
                "objects": [],
            }

        observed_at = float(data.get("timestamp", 0.0) or 0.0)
        frame_width = float(data.get("frame_width") or data.get("width") or 1280.0)
        stale = observed_at <= 0.0 or time.time() - observed_at > max_age
        detections = data.get("detections", [])
        objects = [dict(item) for item in detections if isinstance(item, dict)]
        for item in objects:
            bbox = item.get("bbox")
            if len(bbox or []) == 4 and item.get("distance_m") is None:
                cx = (float(bbox[0]) + float(bbox[2])) / 2.0
                cy = (float(bbox[1]) + float(bbox[3])) / 2.0
                depth_m = self.read_depth_at(int(cx), int(cy))
                if depth_m is not None:
                    item["distance_m"] = round(depth_m, 2)
            if len(bbox or []) == 4 and item.get("angle_deg") is None:
                center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
                item["angle_deg"] = round(((center_x / max(frame_width, 1.0)) - 0.5) * 70.0, 2)
                item["frame_width"] = frame_width

        return {
            "source": "vision_daemon",
            "snapshot_id": str(data.get("frame_id") or data.get("id") or ""),
            "observed_at": observed_at or None,
            "reason": "stale" if stale else "fresh",
            "objects": objects,
            "person_count": _count_persons(objects),
            "frame_width": frame_width,
        }

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
            from qp_perception.selection.weighted import (
                WeightedTargetSelector,  # type: ignore[import-untyped]
            )
            from qp_perception.tracking.yolo_seg import (
                YoloSegTracker,  # type: ignore[import-untyped]
            )

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
        """Whether any vision backend (YOLO, VLM, or ROS2 capture) is usable."""
        if self._tracker is not None:
            return True
        if self._vlm_enabled:
            return True
        # Even without YOLO/VLM, if we can capture frames we're "available"
        # (VLM fallback or frame-only usage)
        if self._enabled and self._capture_backend in ("lingtu_shm", "ros2", "auto"):
            return True
        return False

    def auto_capture_enabled(self) -> bool:
        """Whether conversation turns should capture a scene automatically."""
        return bool(self._vision_cfg.get("auto_capture", False))

    async def describe_scene(self, frame: Any = None) -> str:
        """Detect objects in *frame* and return a natural language description.

        Priority: BPU YOLO (~3ms) → CPU YOLO → VLM (rich, slower).
        If *frame* is ``None``, attempts to capture from camera.
        Returns an empty string if all vision backends are unavailable.
        """
        # Try daemon BPU detections first (0ms — pre-computed by frame_daemon)
        dets = self._read_daemon_detections()
        if dets is not None:
            # BPU ran: return results or fall to VLM (skip slow CPU YOLO)
            if dets:
                return self._detections_to_description(dets)
            # BPU found nothing — skip CPU YOLO, go straight to VLM
            return await self._describe_scene_vlm(frame)

        # Try CPU YOLO (qp-perception) — only when daemon is not running
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
        or ``None`` if not found. Tries BPU first, then CPU YOLO.
        """
        # Try daemon BPU detections first (0ms)
        dets = self._read_daemon_detections()
        if dets is not None:
            target_lower = target.lower()
            for d in dets:
                if d["class_id"].lower() == target_lower:
                    return d
            # BPU ran but target not found — don't fall through to slow CPU YOLO
            return None

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

    async def describe_scene_with_question(self, question: str, frame: Any = None) -> str:
        """Use VLM to answer a specific question about the current camera view.

        Unlike ``describe_scene()`` which lists all objects, this method asks
        the VLM a targeted question (e.g. "有没有方便面", "桌上有什么食物").
        Falls back to ``describe_scene()`` if VLM is unavailable.
        """
        if not self._ensure_vlm_client():
            # No VLM — fall back to generic describe
            return await self.describe_scene(frame)

        try:
            if frame is None:
                frame = await asyncio.to_thread(self._capture_frame)
            if frame is None:
                return ""

            media_type, image_b64 = self._encode_frame_for_vlm(
                frame, max_width=self._vlm_image_max_width
            )
            if not image_b64:
                return ""

            prompt = (
                f"观察这张图片，回答以下问题：{question}\n"
                "你是视觉问答助手，必须直接回答用户的问题，不要固定重复‘看到了’。"
                "用户问数量时认真数（例如几根手指），问有没有时直接回答有或没有，问是什么时回答物体类别，"
                "问位置或属性时回答对应位置或属性；手指、文字等问题也要按图像实际内容回答。"
                "如果目标被遮挡、太小或无法确认，明确说‘无法准确判断’，不要猜测。"
                "只有用户泛问‘看见了什么’时，才简短列出清晰可辨认的主要物体（如瓶子、杯子、手机、纸箱、椅子）。"
                "不要报告颜色、线条、线缆、墙面、地面、天花板、板面、阴影、纹理、网状结构或几何形状；"
                "这些不是用户要的物体答案。禁止根据模糊轮廓、局部遮挡或室内常识猜测类别；"
                "回答使用简短中文，不超过50字。"
            )
            if any(term in question for term in ("手指", "几根", "伸出")):
                prompt += (
                    "这是手指计数问题：只数同一只手从掌部清晰伸直的手指，逐根核对；"
                    "弯曲、遮挡、重叠或画面边缘不清楚的手指不要猜算。不要把手掌、手臂、衣物或其他物体算作手指。"
                    "如果不能清楚确认数量，回答‘无法准确判断手指数量’，不要默认回答五根。"
                )

            def _call_vlm() -> str:
                if self._vlm_backend == "minimax_vision":
                    response = self._vlm_client.post(
                        "/v1/coding_plan/vlm",
                        json={
                            "prompt": prompt,
                            "image_url": f"data:{media_type};base64,{image_b64}",
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    base_resp = payload.get("base_resp") or {}
                    if int(base_resp.get("status_code", 0)) != 0:
                        raise RuntimeError(
                            base_resp.get("status_msg") or "MiniMax vision request failed"
                        )
                    return str(payload.get("content") or "")
                if self._vlm_backend == "anthropic":
                    response = self._vlm_client.messages.create(
                        model=self._vlm_model, max_tokens=150,
                        messages=[{"role": "user", "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                            {"type": "text", "text": prompt},
                        ]}],
                    )
                    return response.content[0].text if response.content else ""
                else:
                    response = self._vlm_client.chat.completions.create(
                        model=self._vlm_model, max_tokens=150,
                        messages=[{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                            {"type": "text", "text": prompt},
                        ]}],
                    )
                    return response.choices[0].message.content or ""

            raw = await asyncio.to_thread(_call_vlm)
            # Targeted Q&A: keep the short Chinese answer, but remove a
            # sentence that positively names an uncertain fan/blade structure.
            # These are common hallucinations for partial ceiling/cable views.
            return self._ground_targeted_answer(raw)

        except Exception as exc:
            logger.warning("[Vision] VLM question failed: %s", exc)
            return "视觉识别服务暂时不可用，无法确认当前摄像头画面。"

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

    async def save_snapshot(
        self,
        frame: Any = None,
        *,
        label: str = "snapshot",
        output_dir: str = "data/captures",
    ) -> str | None:
        """Capture current frame and save to disk. Returns file path or None.

        Saved as: ``{output_dir}/{timestamp}_{label}.jpg``
        """
        try:
            if frame is None:
                frame = await asyncio.to_thread(self._capture_frame)
            if frame is None:
                return None
            return await asyncio.to_thread(self._write_frame, frame, label, output_dir)
        except Exception as exc:
            logger.warning("[Vision] save_snapshot error: %s", exc)
            return None

    @staticmethod
    def _write_frame(frame: Any, label: str, output_dir: str) -> str | None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        filename = f"{ts}_{safe_label}.jpg"
        filepath = os.path.join(output_dir, filename)
        try:
            import cv2  # type: ignore[import-untyped]
            cv2.imwrite(filepath, frame)
        except ImportError:
            try:
                import numpy as np
                from PIL import Image as PILImage
                img = PILImage.fromarray(np.asarray(frame))
                img.save(filepath, quality=85)
            except ImportError:
                # Last resort: save as PPM (no dependencies)
                import numpy as np
                arr = np.asarray(frame)
                filepath = filepath.replace(".jpg", ".ppm")
                with open(filepath, "wb") as f:
                    f.write(f"P6\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode())
                    f.write(arr.tobytes())
        logger.info("[Vision] Snapshot saved: %s", filepath)
        return filepath

    # ------------------------------------------------------------------
    # VLM (Claude Sonnet) fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_frame_for_vlm(
        frame: Any, *, max_width: int = 320
    ) -> tuple[str, str]:
        """Encode a compact OpenCV-style BGR frame without optional codecs."""
        import base64

        import numpy as np

        array = np.asarray(frame, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            return "", ""

        height, width = array.shape[:2]
        if width > max_width:
            target_height = max(1, round(height * max_width / width))
            y_indices = np.linspace(0, height - 1, target_height).astype(np.intp)
            x_indices = np.linspace(0, width - 1, max_width).astype(np.intp)
            array = array[y_indices][:, x_indices]
        try:
            import cv2  # type: ignore[import-untyped]

            ok, encoded = cv2.imencode(
                ".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if ok:
                return "image/jpeg", base64.b64encode(encoded).decode("ascii")
        except ImportError:
            pass

        import binascii
        import struct
        import zlib

        # Reducing each channel to 32 levels substantially improves PNG size
        # while retaining enough detail for scene-level visual questions.
        array = ((array.astype(np.uint16) // 8) * 8).astype(np.uint8)
        rgb = np.ascontiguousarray(array[:, :, ::-1])
        height, width = rgb.shape[:2]
        scanlines = b"".join(
            b"\x00" + rgb[row].tobytes() for row in range(height)
        )

        def _png_chunk(kind: bytes, payload: bytes) -> bytes:
            body = kind + payload
            return (
                struct.pack(">I", len(payload))
                + body
                + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        png += _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        png += _png_chunk(b"IEND", b"")
        return "image/png", base64.b64encode(png).decode("ascii")

    def _ensure_vlm_client(self) -> bool:
        """Lazily initialise the VLM client. Returns True if ready.

        Tries Anthropic native SDK first (better for relay), falls back to
        OpenAI-compatible client.
        """
        if self._vlm_client is not None:
            return True
        if not self._vlm_enabled or not self._vlm_api_key:
            return False

        if self._vlm_backend == "minimax_vision":
            try:
                import httpx

                self._vlm_client = httpx.Client(
                    base_url=self._vlm_base_url,
                    headers={"Authorization": f"Bearer {self._vlm_api_key}"},
                    timeout=self._vlm_timeout,
                )
                logger.info("[Vision] VLM client: MiniMax vision service.")
                return True
            except Exception as exc:
                logger.warning("[Vision] MiniMax VLM client init failed: %s", exc)
                return False

        # Try Anthropic native SDK (relay: /api endpoint, no dev-assistant injection)
        anthropic_url = self._vlm_base_url.rstrip("/").removesuffix("/v1")
        try:
            import anthropic
            self._vlm_client = anthropic.Anthropic(
                api_key=self._vlm_api_key,
                base_url=anthropic_url,
                timeout=self._vlm_timeout,
                max_retries=0,
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
                timeout=self._vlm_timeout,
                max_retries=0,
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

            # Encode frame as base64 JPEG (cv2 → PIL → raw PPM fallback)
            import base64

            import numpy as np

            image_b64 = ""
            try:
                import cv2  # type: ignore[import-untyped]
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_b64 = base64.b64encode(buf).decode("utf-8")
            except ImportError:
                try:
                    import io

                    from PIL import Image as PILImage
                    img = PILImage.fromarray(np.asarray(frame))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80)
                    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                except ImportError:
                    logger.warning("[Vision] Neither cv2 nor PIL available for JPEG encoding.")
                    return ""
            if not image_b64:
                return ""

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
    def _ground_targeted_answer(text: str) -> str:
        """Drop unsupported fan/blade assertions while retaining visible facts."""
        import re

        raw = str(text or "").strip()
        if not raw:
            return ""
        sentences = re.split(r"(?<=[。！？.!?])|[\r\n]+", raw)
        risky_terms = (
            "风扇", "电风扇", "叶片", "扇叶", "旋转", "线条", "线缆", "电线",
            "板面", "网状", "网格", "墙面", "地面", "天花板", "平面", "纹理",
            "阴影", "光影",
        )
        object_terms = (
            "瓶", "杯", "手机", "电话", "纸箱", "箱子", "椅", "桌", "沙发",
            "电脑", "键盘", "显示器", "门", "窗", "人", "手", "衣服", "包",
            "书", "袋", "玩具", "车", "猫", "狗", "植物", "垃圾桶", "灯", "设备",
        )
        negative_terms = ("未见", "没看到", "没有", "无法确认", "不确定", "未发现")
        kept = []
        for sentence in sentences:
            has_risky = any(term in sentence for term in risky_terms)
            has_object = any(term in sentence for term in object_terms)
            if has_risky and not has_object and not any(
                marker in sentence for marker in negative_terms
            ):
                continue
            if has_risky and has_object and not any(
                marker in sentence for marker in negative_terms
            ):
                clauses = re.split(r"[，,；;]", sentence)
                if len(clauses) > 1:
                    clauses = [
                        clause for clause in clauses
                        if not any(term in clause for term in risky_terms)
                    ]
                    sentence = "，".join(clauses)
                else:
                    for term in risky_terms:
                        sentence = sentence.replace(term, "")
                    sentence = re.sub(r"(旁边有|上方有|下方有|有)$", "", sentence)
            kept.append(sentence)
        result = "".join(kept).strip()
        return result or "图中有一些无法确认的结构，暂时不能可靠判断具体物体。"

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
        """Capture a single frame from the camera (blocking).

        Tries backends in order based on ``capture_backend`` config:
        - ``lingtu_shm``: read LingTu camera frames from its POSIX SHM data plane
        - ``ros2``: subscribe to ROS2 Image topic (for Orbbec / ROS cameras)
        - ``cv2``: OpenCV VideoCapture (for USB UVC cameras)
        - ``auto`` (default): try ros2 first, then cv2
        """
        backend = self._capture_backend

        if backend in ("lingtu_shm", "auto"):
            frame = self._capture_lingtu_shm()
            if frame is not None:
                return frame
            if backend == "lingtu_shm":
                return None

        if backend in ("ros2", "auto"):
            frame = self._capture_ros2()
            if frame is not None:
                return frame
            if backend == "ros2":
                return None  # don't fall through

        # cv2 fallback
        return self._capture_cv2()

    def _capture_lingtu_shm(self) -> Any:
        """Read the latest RGB frame from LingTu's native camera SHM ring."""
        try:
            import sys

            import numpy as np

            source_root = str(Path(self._lingtu_repo) / "src")
            if source_root not in sys.path:
                sys.path.insert(0, source_root)
            from drivers.real.camera.shm import ShmFrameReader

            reader = ShmFrameReader(
                self._lingtu_color_shm,
                max_age_s=self._lingtu_max_age_s,
            )
            try:
                frame = reader.read_latest()
            finally:
                reader.close()
            if frame is None or frame.width <= 0 or frame.height <= 0:
                return None

            encoding = str(frame.encoding or "").lower()
            if encoding not in {"rgb8", "bgr8"}:
                logger.warning("[Vision] Unsupported LingTu color encoding: %s", encoding)
                return None
            row_bytes = int(frame.stride or frame.width * 3)
            expected = int(frame.height) * row_bytes
            raw = np.frombuffer(frame.payload, dtype=np.uint8)
            if raw.size < expected:
                logger.warning(
                    "[Vision] LingTu color frame is short: %d < %d bytes",
                    raw.size,
                    expected,
                )
                return None
            rows = raw[:expected].reshape(int(frame.height), row_bytes)
            image = rows[:, : int(frame.width) * 3].reshape(
                int(frame.height), int(frame.width), 3
            )
            # The rest of VisionBridge follows OpenCV's BGR convention.
            if encoding == "rgb8":
                image = image[:, :, ::-1]
            if self._lingtu_rotate_180:
                image = np.rot90(image, 2)
            return np.ascontiguousarray(image)
        except Exception as exc:
            logger.warning("[Vision] LingTu SHM capture error: %s", exc)
            return None

    def _capture_ros2(self) -> Any:
        """Grab latest frame — daemon file first (0ms), subprocess fallback (3-5s)."""
        # Try daemon shared file first (instant)
        frame = self._read_daemon_frame()
        if frame is not None:
            return frame

        # Fallback to subprocess
        try:
            if self._ros2_grabber is None:
                self._ros2_grabber = _ROS2FrameGrabber(
                    topic=self._ros2_topic, timeout=5.0,
                )
            return self._ros2_grabber.grab()
        except Exception as exc:
            logger.warning("[Vision] ROS2 capture error: %s", exc)
            return None

    @staticmethod
    def _check_daemon_alive(max_age: float = 2.0) -> bool:
        try:
            with open(DAEMON_HEARTBEAT_PATH) as f:
                ts = float(f.read().strip())
            return time.time() - ts <= max_age
        except (FileNotFoundError, ValueError):
            return False

    @staticmethod
    def _read_daemon_frame(
        path: str = DAEMON_COLOR_FRAME_PATH,
        max_age: float = 2.0,
    ) -> Any:
        """Read latest frame from frame_daemon shared file. Returns None if stale/missing."""
        import struct

        import numpy as np

        if not VisionBridge._check_daemon_alive(max_age):
            return None

        try:
            with open(path, "rb") as f:
                header = f.read(8)
                if len(header) < 8:
                    return None
                w, h = struct.unpack("II", header)
                data = f.read(w * h * 3)
                if len(data) != w * h * 3:
                    return None
                return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).copy()
        except (FileNotFoundError, OSError):
            return None

    @staticmethod
    def read_depth_at(x: int, y: int) -> float | None:
        """Read depth (meters) at pixel (x, y) from daemon depth frame.

        Returns None if depth unavailable or pixel is invalid (0 = no reading).
        Depth frame is 848x480 uint16 mm from Orbbec Gemini 335.
        """
        import struct

        import numpy as np

        if not VisionBridge._check_daemon_alive():
            return None
        try:
            with open(DAEMON_DEPTH_FRAME_PATH, "rb") as f:
                header = f.read(8)
                if len(header) < 8:
                    return None
                w, h = struct.unpack("II", header)
                data = f.read(w * h * 2)
                if len(data) != w * h * 2:
                    return None
            depth = np.frombuffer(data, dtype=np.uint16).reshape(h, w)
            # Scale color pixel (1280x720) to depth pixel (848x480)
            dx = int(x * w / 1280)
            dy = int(y * h / 720)
            dx = max(0, min(w - 1, dx))
            dy = max(0, min(h - 1, dy))
            # Sample 5x5 area for robustness (avoid single noisy pixel)
            y1, y2 = max(0, dy - 2), min(h, dy + 3)
            x1, x2 = max(0, dx - 2), min(w, dx + 3)
            patch = depth[y1:y2, x1:x2]
            valid = patch[patch > 0]
            if valid.size == 0:
                return None
            return float(np.median(valid)) / 1000.0  # mm → meters
        except (FileNotFoundError, OSError):
            return None

    @staticmethod
    def read_depth_map() -> Any:
        """Read full depth map. Returns (848, 480) uint16 mm array or None."""
        import struct

        import numpy as np

        if not VisionBridge._check_daemon_alive():
            return None
        try:
            with open(DAEMON_DEPTH_FRAME_PATH, "rb") as f:
                header = f.read(8)
                if len(header) < 8:
                    return None
                w, h = struct.unpack("II", header)
                data = f.read(w * h * 2)
                if len(data) != w * h * 2:
                    return None
            return np.frombuffer(data, dtype=np.uint16).reshape(h, w).copy()
        except (FileNotFoundError, OSError):
            return None

    def _capture_cv2(self) -> Any:
        """Grab a frame via OpenCV VideoCapture."""
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
            logger.debug("[Vision] cv2 not installed — skipping cv2 capture.")
            return None
        except Exception as exc:
            logger.warning("[Vision] cv2 capture error: %s", exc)
            return None

    @staticmethod
    def _detections_to_description(detections: list[dict[str, Any]]) -> str:
        """Convert BPU detection list to Chinese description with distance."""
        if not detections:
            return ""
        parts = []
        for d in detections:
            name = d["class_id"]
            dist = d.get("distance_m")
            if dist and dist > 0:
                parts.append(f"{name}({dist:.1f}米)")
            else:
                parts.append(name)

        from collections import Counter
        counts: Counter[str] = Counter(parts)
        items = ", ".join(f"{c}个{n}" if c > 1 else n for n, c in counts.items())
        return f"我看到了: {items}"

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
