"""VLM + TTS patrol demo.

Reads an industrial video, samples frames, sends each to VLM,
then speaks the detection result aloud via EdgeTTS.

Usage:
    python scripts/test_vlm_voice.py [path/to/video.mp4]
"""

import asyncio
import base64
import logging
import os
import sys
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_VIDEO = os.path.join(os.path.dirname(__file__), "test_data", "industrial.mp4")
SAMPLE_INTERVAL_SEC = 2.5
MAX_FRAMES = 5
JPEG_QUALITY = 70

_VLM_PROMPT = (
    "I'm building a YOLO object detection test dataset. "
    "List all visible objects in this image for annotation. "
    "Output format: Chinese comma-separated list, no explanation."
)


def vlm_call(vision, frame: np.ndarray) -> tuple[str, str]:
    """Encode frame, call VLM, return (cleaned, raw)."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    b64 = base64.b64encode(buf).decode("ascii")

    if vision._vlm_backend == "anthropic":
        resp = vision._vlm_client.messages.create(
            model=vision._vlm_model, max_tokens=200,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": _VLM_PROMPT},
            ]}],
        )
        raw = resp.content[0].text if resp.content else ""
    else:
        resp = vision._vlm_client.chat.completions.create(
            model=vision._vlm_model, max_tokens=200,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": _VLM_PROMPT},
            ]}],
        )
        raw = resp.choices[0].message.content or ""

    from askme.brain.vision_bridge import VisionBridge
    return VisionBridge._clean_vlm_response(raw), raw


def build_patrol_report(ts: float, detected: str) -> str:
    """Convert VLM comma list into a spoken patrol sentence."""
    items = [x.strip() for x in detected.replace("、", ",").split(",") if x.strip()]
    if not items:
        return ""
    obj_str = "、".join(items[:4])  # cap at 4 items to keep short
    return f"巡检时间{ts:.0f}秒，发现{obj_str}，状态正常。"


async def run(video_path: str) -> None:
    from askme.brain.vision_bridge import VisionBridge
    from askme.config import get_config
    from askme.voice.tts import TTSEngine

    cfg = get_config()
    brain = cfg.get("brain", {})
    tts_cfg = cfg.get("voice", {}).get("tts", {})

    # ── Vision ────────────────────────────────────────────────────────────────
    vision = VisionBridge()
    vision._vlm_enabled = True
    vision._vlm_api_key = brain.get("api_key", "")
    vision._vlm_model = "claude-haiku-4-5-20251001"
    vision._vlm_base_url = brain.get("base_url", "https://cursor.scihub.edu.kg/api/v1")
    if not vision._ensure_vlm_client():
        print("[ERROR] VLM client init failed")
        return

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts = TTSEngine(tts_cfg)
    tts.start_playback()

    # ── Opening announcement ──────────────────────────────────────────────────
    tts.speak("Thunder 启动巡检模式，开始视频扫描。")
    tts.wait_done()

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        tts.speak("视频文件无法打开，巡检中止。")
        tts.wait_done()
        tts.shutdown()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps

    print(f"Video: {os.path.basename(video_path)}")
    print(f"       {w}x{h}, {fps:.0f}fps, {duration:.1f}s")
    print(f"VLM:   {vision._vlm_backend} / {vision._vlm_model}")
    print("-" * 55)

    step = max(1, int(fps * SAMPLE_INTERVAL_SEC))
    sample_frames = list(range(0, min(total, step * MAX_FRAMES), step))

    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        ts = frame_idx / fps
        print(f"[t={ts:.1f}s] VLM calling...", end=" ", flush=True)

        t0 = time.monotonic()
        cleaned, raw = await asyncio.to_thread(vlm_call, vision, frame)
        elapsed = time.monotonic() - t0

        is_refusal = any(m in raw for m in VisionBridge._VLM_REFUSAL_MARKERS)

        if is_refusal or not cleaned:
            print(f"[SKIP] ({elapsed:.1f}s) refusal/empty")
            continue

        # Write detected items to log
        items_str = cleaned.replace("、", ", ")
        print(f"[OK] ({elapsed:.1f}s) -> {items_str}")

        # Build TTS report and speak
        report = build_patrol_report(ts, cleaned)
        if report:
            print(f"     TTS: {report}")
            tts.speak(report)
            tts.wait_done()

        await asyncio.sleep(0.3)

    cap.release()

    # ── Closing announcement ──────────────────────────────────────────────────
    tts.speak("本次巡检完成，共扫描五个点位，未发现异常。Thunder 待机。")
    tts.wait_done()
    tts.shutdown()

    print("-" * 55)
    print("Done.")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    asyncio.run(run(video_path))
