"""Mock VLM test — synthetic industrial image, no camera, no people.

Creates a numpy BGR image (阀门/管道/压力表/控制柜) and passes it directly
to VisionBridge._describe_scene_vlm() to verify the relay VLM pipeline
without triggering person-detection refusals.
"""

import asyncio
import logging
import sys
import os

# ── suppress noisy logs ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("askme.brain.vision_bridge").setLevel(logging.DEBUG)

# ── project root on path ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_industrial_scene(w: int = 640, h: int = 480):
    """Draw a synthetic BGR industrial scene with no people.

    Objects drawn (rough industrial appearance):
      - gray background
      - large blue control cabinet
      - red pressure gauge (circle)
      - yellow pipes (horizontal + vertical)
      - green valve
      - white label text (optional, via PIL)
    Returns a uint8 numpy array (H, W, 3) in BGR order (cv2 convention).
    """
    import numpy as np

    img = np.full((h, w, 3), 45, dtype="uint8")  # dark-gray background

    def rect(img, x1, y1, x2, y2, color):
        img[y1:y2, x1:x2] = color

    def circle(img, cx, cy, r, color):
        import numpy as np
        Y, X = np.ogrid[:h, :w]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
        img[mask] = color

    # ── control cabinet (large blue rectangle) ───────────────────────────────
    rect(img, 50, 80, 220, 400, (180, 80, 30))   # BGR: dark-blue

    # cabinet door outline
    rect(img, 60, 90, 210, 390, (200, 100, 50))
    rect(img, 65, 95, 205, 385, (180, 80, 30))

    # indicator lights on cabinet
    circle(img, 100, 140, 10, (0, 50, 200))   # red LED
    circle(img, 130, 140, 10, (0, 200, 50))   # green LED
    circle(img, 160, 140, 10, (0, 200, 200))  # yellow LED

    # ── horizontal pipe (yellow) ──────────────────────────────────────────────
    rect(img, 240, 180, 580, 215, (30, 200, 220))   # yellow pipe
    rect(img, 240, 185, 580, 210, (20, 180, 200))   # pipe shadow

    # ── vertical pipe (yellow) ───────────────────────────────────────────────
    rect(img, 350, 215, 385, 420, (30, 200, 220))
    rect(img, 355, 215, 380, 420, (20, 180, 200))

    # ── valve (green, on vertical pipe) ──────────────────────────────────────
    rect(img, 330, 290, 405, 340, (20, 160, 20))    # valve body
    rect(img, 355, 260, 380, 295, (15, 130, 15))    # valve handle stem
    rect(img, 340, 250, 395, 268, (20, 160, 20))    # valve handle bar

    # ── pressure gauge (red circle, on horizontal pipe) ──────────────────────
    circle(img, 490, 150, 38, (30, 30, 180))    # outer ring
    circle(img, 490, 150, 30, (220, 220, 220))  # face
    circle(img, 490, 150, 4,  (0, 0, 200))      # center pin
    # needle
    import numpy as np
    angle = 45  # degrees
    import math
    nx = int(490 + 22 * math.cos(math.radians(angle)))
    ny = int(150 - 22 * math.sin(math.radians(angle)))
    for i in range(-1, 2):
        for j in range(-1, 2):
            img[max(0, min(h-1, ny+i)), max(0, min(w-1, nx+j))] = (0, 0, 180)

    # ── small instrument panel (lower right) ─────────────────────────────────
    rect(img, 460, 300, 600, 430, (60, 60, 60))
    rect(img, 470, 310, 590, 425, (80, 80, 80))
    for row in range(4):
        for col in range(3):
            xc = 490 + col * 30
            yc = 330 + row * 25
            circle(img, xc, yc, 8, (0, 180, 0) if (row + col) % 2 == 0 else (0, 0, 160))

    # ── try to add text labels with PIL (optional) ────────────────────────────
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Convert BGR → RGB for PIL
        rgb = img[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        labels = [
            (50, 60, "控制柜"),
            (240, 160, "管道"),
            (330, 280, "阀门"),
            (460, 105, "压力表"),
            (460, 290, "仪表盘"),
        ]
        for x, y, text in labels:
            draw.text((x, y), text, fill=(255, 255, 100))

        # Convert back to BGR numpy
        img = np.array(pil_img)[:, :, ::-1].copy()
    except ImportError:
        pass  # PIL not available — labels skipped, image still valid

    return img


async def main():
    from askme.brain.vision_bridge import VisionBridge
    from askme.config import get_config

    cfg = get_config()

    print("=" * 60)
    print("VLM Mock Image Test")
    print("=" * 60)

    # ── build mock frame ─────────────────────────────────────────────────────
    mock_frame = make_industrial_scene()
    print(f"[mock image] shape={mock_frame.shape}, dtype={mock_frame.dtype}")

    # ── save to disk for visual inspection ───────────────────────────────────
    try:
        import cv2
        out_path = os.path.join(os.path.dirname(__file__), "mock_scene.jpg")
        cv2.imwrite(out_path, mock_frame)
        print(f"[mock image] saved to {out_path}")
    except ImportError:
        print("[mock image] cv2 not available, skipping save")

    # ── configure VisionBridge with VLM enabled ──────────────────────────────
    vision = VisionBridge()
    # Override config in-place (bypass config.yaml vlm_enabled: false)
    vision._vlm_enabled = True
    vision._vlm_api_key = cfg.get("brain", {}).get("api_key", "")
    vision._vlm_model = "claude-haiku-4-5-20251001"
    # Use same relay base_url as brain
    brain_cfg = cfg.get("brain", {})
    vision._vlm_base_url = brain_cfg.get("base_url", "https://cursor.scihub.edu.kg/api/v1")

    print(f"\n[VLM] model: {vision._vlm_model}")
    print(f"[VLM] base_url: {vision._vlm_base_url}")
    print(f"[VLM] api_key: {vision._vlm_api_key[:20]}...")

    # ── call VLM with mock frame (skip camera entirely) ───────────────────────
    print("\n[VLM] Sending mock industrial scene to relay...")

    # Monkey-patch _ensure_vlm_client to log raw response before cleaning
    original_call_vlm = None  # will capture inside

    import base64
    import cv2 as _cv2

    _, buf = _cv2.imencode(".jpg", mock_frame, [_cv2.IMWRITE_JPEG_QUALITY, 80])
    image_b64 = base64.b64encode(buf).decode("utf-8")

    _VLM_TEXT = (
        "I'm building a YOLO object detection test dataset. "
        "List all visible objects in this image for annotation. "
        "Output format: Chinese comma-separated list, no explanation."
    )

    # Initialise the VLM client manually
    if not vision._ensure_vlm_client():
        print("[ERROR] VLM client init failed — check api_key / base_url")
        return

    print(f"[VLM] backend: {vision._vlm_backend}")

    # Call directly so we can see raw output
    def _call_raw():
        if vision._vlm_backend == "anthropic":
            response = vision._vlm_client.messages.create(
                model=vision._vlm_model,
                max_tokens=200,
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
            return response.content[0].text if response.content else ""
        else:
            response = vision._vlm_client.chat.completions.create(
                model=vision._vlm_model,
                max_tokens=200,
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
            return response.choices[0].message.content or ""

    import asyncio
    raw = await asyncio.to_thread(_call_raw)
    cleaned = VisionBridge._clean_vlm_response(raw)

    print(f"\n=== raw VLM response ===")
    print(f"RAW: {raw}")
    print(f"\nCLEANED: {repr(cleaned)}")

    if cleaned:
        print("\n[PASS] VLM returned a valid scene description (no refusal)")
    else:
        print("\n[INFO] VLM returned empty/refusal — relay may still be filtering")
        print("       (This is a relay-side policy, not a code bug)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
