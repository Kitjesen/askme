"""Perception provider factories."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from askme.ports import (
    ChangeMonitorPort,
    InteractionPerceptionPort,
    SceneIntelligencePort,
    VisionPort,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerceptionStack:
    """Concrete perception adapters selected for one runtime instance."""

    vision: VisionPort
    interaction_provider: InteractionPerceptionPort
    change_monitor: ChangeMonitorPort | None


def build_perception(
    config: dict[str, Any] | None = None,
    *,
    pulse: Any = None,
) -> PerceptionStack:
    """Build vision, interaction perception, and optional change monitoring."""
    from askme.perception.interaction_provider import FileInteractionPerceptionProvider
    from askme.perception.vision_bridge import VisionBridge

    cfg = config if isinstance(config, dict) else {}
    perception_cfg = cfg.get("perception", {}) if isinstance(cfg.get("perception"), dict) else {}
    interaction_cfg = (
        perception_cfg.get("interaction_provider", {})
        if isinstance(perception_cfg.get("interaction_provider"), dict)
        else {}
    )

    vision = VisionBridge()
    interaction_provider = FileInteractionPerceptionProvider(interaction_cfg)
    change_monitor = _build_change_monitor(cfg, pulse=pulse)

    return PerceptionStack(
        vision=vision,
        interaction_provider=interaction_provider,
        change_monitor=change_monitor,
    )


def build_scene_intelligence(*, episodic: Any, session: Any = None) -> SceneIntelligencePort:
    """Build scene-intelligence service behind the perception provider boundary."""
    from askme.perception.scene_intelligence import SceneIntelligence

    return SceneIntelligence(episodic=episodic, session=session)


def _build_change_monitor(
    config: dict[str, Any],
    *,
    pulse: Any = None,
) -> ChangeMonitorPort | None:
    try:
        from askme.perception.change_detector import ChangeDetector
    except Exception as exc:
        logger.debug("ChangeDetector not available: %s", exc)
        return None

    try:
        return ChangeDetector(config=config, pulse=pulse)
    except Exception as exc:
        logger.debug("ChangeDetector could not be built: %s", exc)
        return None


async def capture_snapshot_payload(vision: VisionPort) -> dict[str, Any] | None:
    """Capture a frame through the concrete vision adapter and encode JPEG."""
    import asyncio
    import base64

    frame = await asyncio.to_thread(vision._capture_frame)  # type: ignore[attr-defined]
    if frame is None:
        return None
    try:
        import cv2  # type: ignore[import-untyped]

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(buf).decode()
        height, width = frame.shape[:2]
        return {
            "image_base64": image_b64,
            "width": width,
            "height": height,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        logger.warning("[Vision] Encode error: %s", exc)
        return None


async def analyze_image_base64(vision: VisionPort, image_b64: str) -> str:
    """Decode a base64 JPEG and analyze it through the concrete vision adapter."""
    try:
        import base64

        import cv2  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]

        image_bytes = base64.b64decode(image_b64)
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return await vision._describe_scene_vlm(frame)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("[Vision] Analyze error: %s", exc)
        return f"分析失败: {exc}"


def read_depth_info(
    *,
    heartbeat_path: str = "/tmp/askme_frame_daemon.heartbeat",
    depth_path: str = "/tmp/askme_frame_depth.bin",
) -> dict[str, Any]:
    """Read depth-daemon status and center depth from the provider edge."""
    import struct
    import time

    result: dict[str, Any] = {"daemon_alive": False, "center_depth_m": None}

    try:
        with open(heartbeat_path, encoding="utf-8") as f:
            ts = float(f.read().strip())
        result["daemon_alive"] = time.time() - ts < 3.0
    except (FileNotFoundError, ValueError):
        pass

    try:
        import numpy as np

        with open(depth_path, "rb") as f:
            header = f.read(8)
            if len(header) == 8:
                width, height = struct.unpack("II", header)
                data = f.read(width * height * 2)
                if len(data) == width * height * 2:
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                    center = depth[height // 2, width // 2]
                    result["center_depth_m"] = (
                        round(center / 1000.0, 2) if center > 0 else None
                    )
                    result["frame_size"] = f"{width}x{height}"
    except Exception:
        pass

    return result


__all__ = [
    "PerceptionStack",
    "analyze_image_base64",
    "build_perception",
    "build_scene_intelligence",
    "capture_snapshot_payload",
    "read_depth_info",
]
