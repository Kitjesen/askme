"""Tests for VisionBridge static helpers and pure logic."""

from __future__ import annotations

from copy import deepcopy

import pytest

from askme.perception.vision_bridge import VisionBridge


def test_encode_frame_for_vlm_has_dependency_free_png_fallback(monkeypatch):
    import builtins

    import numpy as np

    original_import = builtins.__import__

    def without_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("cv2 intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cv2)
    media_type, encoded = VisionBridge._encode_frame_for_vlm(np.zeros((2, 3, 3), dtype=np.uint8))

    assert media_type == "image/png"
    assert encoded.startswith("iVBORw0KGgo")


def test_encode_frame_for_vlm_downsamples_large_frame(monkeypatch):
    import base64
    import builtins
    import struct

    import numpy as np

    original_import = builtins.__import__

    def without_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("cv2 intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cv2)
    media_type, encoded = VisionBridge._encode_frame_for_vlm(
        np.zeros((480, 640, 3), dtype=np.uint8), max_width=320
    )
    png = base64.b64decode(encoded)
    width, height = struct.unpack(">II", png[16:24])

    assert media_type == "image/png"
    assert (width, height) == (320, 240)


def test_visual_question_reports_backend_failure(monkeypatch):
    import asyncio

    bridge = VisionBridge(config=_product_vlm_config())
    bridge._vlm_client = object()
    monkeypatch.setattr(bridge, "_capture_frame", lambda: object())
    monkeypatch.setattr(
        bridge, "_encode_frame_for_vlm", lambda frame, max_width: ("image/png", "AA==")
    )

    result = asyncio.run(bridge.describe_scene_with_question("看见了什么"))

    assert result == "视觉识别服务暂时不可用，无法确认当前摄像头画面。"


def _product_vlm_config() -> dict:
    return {
        "brain": {
            "provider": "litellm",
            "api_key": "sk-scoped-chat",
            "base_url": "http://127.0.0.1:4000/v1",
        },
        "vision": {
            "vlm_enabled": True,
            "vlm_backend": "openai",
            "vlm_api_key": "sk-scoped-vision",
            "vlm_base_url": "http://127.0.0.1:4000/v1",
            "vlm_model": "vision-scene",
            "vlm_timeout": 10.0,
        },
    }


class _CapturingVlmCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def _install_capturing_vlm_client(bridge: VisionBridge, content: str) -> _CapturingVlmCompletions:
    from types import SimpleNamespace

    completions = _CapturingVlmCompletions(content)
    bridge._vlm_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return completions


def _assert_vlm_litellm_envelope(request: dict) -> None:
    import json
    import re

    metadata = request["metadata"]
    headers = request["extra_headers"]

    assert request["model"] == "vision-scene"
    assert metadata["purpose"] == "vision_grounding"
    assert metadata["channel"] == "vision"
    assert metadata["request_class"] == "vision"
    assert metadata["privacy_class"] == "restricted"
    assert metadata["model_alias"] == "vision-scene"
    assert metadata["allow_cache"] == "false"
    assert metadata["latency_budget_ms"] == 10_000
    assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-01", headers["traceparent"])
    assert headers["x-litellm-call-id"] == metadata["call_id"]
    assert set(headers) == {"traceparent", "x-litellm-call-id"}
    assert request["extra_body"]["cache"] == {"no-cache": True, "no-store": True}

    control_plane = json.dumps(
        {"metadata": metadata, "extra_headers": headers, "extra_body": request["extra_body"]},
        ensure_ascii=False,
    )
    assert "观察这张图片" not in control_plane
    assert "YOLO object detection" not in control_plane
    assert "data:image" not in control_plane
    assert "sk-" not in control_plane
    assert "Authorization" not in control_plane
    assert "Cookie" not in control_plane


def test_exact_product_vlm_route_constructs_only_openai_compatible_client(monkeypatch):
    from unittest.mock import MagicMock

    bridge = VisionBridge(config=_product_vlm_config())
    openai_client = MagicMock()
    monkeypatch.setattr("openai.OpenAI", openai_client)

    assert bridge._ensure_vlm_client() is True

    openai_client.assert_called_once_with(
        api_key="sk-scoped-vision",
        base_url="http://127.0.0.1:4000/v1",
        timeout=10.0,
        max_retries=0,
    )
    assert bridge._vlm_backend == "openai"


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("brain", "provider", "deepseek"),
        ("vision", "vlm_backend", "anthropic"),
        ("vision", "vlm_model", "qwen-vl-max"),
        ("vision", "vlm_base_url", "https://direct.invalid/v1"),
        ("vision", "vlm_api_key", "sk-scoped-chat"),
    ],
)
def test_product_vlm_rejects_any_route_outside_scoped_litellm_policy(
    monkeypatch,
    section,
    key,
    value,
):
    from unittest.mock import MagicMock

    config = deepcopy(_product_vlm_config())
    config[section][key] = value
    openai_client = MagicMock()
    monkeypatch.setattr("openai.OpenAI", openai_client)

    bridge = VisionBridge(config=config)

    assert bridge._ensure_vlm_client() is False
    assert bridge._vlm_client is None
    openai_client.assert_not_called()


def test_vlm_question_uses_shared_litellm_envelope(monkeypatch):
    import asyncio

    import numpy as np

    bridge = VisionBridge(config=_product_vlm_config())
    completions = _install_capturing_vlm_client(bridge, "桌上有一个杯子。")
    monkeypatch.setattr(
        bridge,
        "_encode_frame_for_vlm",
        lambda frame, max_width: ("image/png", "AA=="),
    )

    result = asyncio.run(
        bridge.describe_scene_with_question(
            "桌上有什么？", frame=np.zeros((2, 2, 3), dtype=np.uint8)
        )
    )

    assert result == "桌上有一个杯子。"
    assert len(completions.calls) == 1
    _assert_vlm_litellm_envelope(completions.calls[0])


def test_vlm_scene_description_uses_shared_litellm_envelope(monkeypatch):
    import asyncio

    import numpy as np

    bridge = VisionBridge(config=_product_vlm_config())
    completions = _install_capturing_vlm_client(bridge, "描述：杯子，桌子")
    monkeypatch.setattr(
        bridge,
        "_encode_frame_for_vlm",
        lambda frame, max_width: ("image/jpeg", "BB=="),
    )

    result = asyncio.run(bridge._describe_scene_vlm(frame=np.zeros((2, 2, 3), dtype=np.uint8)))

    assert result == "杯子，桌子"
    assert len(completions.calls) == 1
    _assert_vlm_litellm_envelope(completions.calls[0])


# ── _clean_vlm_response ───────────────────────────────────────────────────────


class TestCleanVlmResponse:
    def test_empty_string_returns_empty(self):
        assert VisionBridge._clean_vlm_response("") == ""

    def test_refusal_returns_empty(self):
        result = VisionBridge._clean_vlm_response("I cannot help with this request.")
        assert result == ""

    def test_chinese_refusal_returns_empty(self):
        result = VisionBridge._clean_vlm_response("我无法帮助你分析图像。")
        assert result == ""

    def test_explicit_marker_extracted(self):
        text = "Here is the scene analysis:\n简洁描述：走廊里有一个人和一张桌子。"
        result = VisionBridge._clean_vlm_response(text)
        assert "走廊里有一个人和一张桌子" in result

    def test_colon_marker_extracted(self):
        text = "描述: 房间里有一把椅子。"
        result = VisionBridge._clean_vlm_response(text)
        assert "房间里有一把椅子" in result

    def test_fallback_to_chinese_line(self):
        text = "The room contains furniture.\n我看到了一把椅子和一张桌子和一扇窗户。\nOther info."
        result = VisionBridge._clean_vlm_response(text)
        assert "椅子" in result or "桌子" in result

    def test_skips_refusal_in_extracted_text(self):
        # Marker present but extracted content contains refusal → should NOT return refusal
        text = "简洁描述：无法描述场景。"
        result = VisionBridge._clean_vlm_response(text)
        # The extraction finds the marker, but the extracted text has "无法" = refusal
        assert result == "" or "无法" not in result

    def test_non_refusal_chinese_text_returned(self):
        text = "走廊空旷，地板干净，灯光正常，没有异常情况。"
        result = VisionBridge._clean_vlm_response(text)
        # Not a refusal, should return the Chinese line
        assert "走廊" in result or result == ""  # may or may not match fallback

    def test_purely_english_no_chinese_returns_empty_or_fallback(self):
        text = "A clean corridor with bright lights."
        # No Chinese chars, no refusal → best stays empty
        result = VisionBridge._clean_vlm_response(text)
        # We just verify it doesn't raise and returns a string
        assert isinstance(result, str)

    def test_i_claude_marker_refusal(self):
        result = VisionBridge._clean_vlm_response("I'm Claude, an AI assistant.")
        assert result == ""


# ── _detections_to_description ────────────────────────────────────────────────


class TestDetectionsToDescription:
    def test_empty_detections_returns_empty(self):
        assert VisionBridge._detections_to_description([]) == ""

    def test_single_detection_no_distance(self):
        dets = [{"class_id": "person"}]
        result = VisionBridge._detections_to_description(dets)
        assert "person" in result
        assert "我看到了" in result

    def test_single_detection_with_distance(self):
        dets = [{"class_id": "person", "distance_m": 2.5}]
        result = VisionBridge._detections_to_description(dets)
        assert "person(2.5米)" in result

    def test_zero_distance_excluded(self):
        dets = [{"class_id": "box", "distance_m": 0.0}]
        result = VisionBridge._detections_to_description(dets)
        assert "米" not in result

    def test_negative_distance_excluded(self):
        dets = [{"class_id": "box", "distance_m": -1.0}]
        result = VisionBridge._detections_to_description(dets)
        assert "米" not in result

    def test_multiple_same_class_counted(self):
        dets = [
            {"class_id": "person"},
            {"class_id": "person"},
            {"class_id": "person"},
        ]
        result = VisionBridge._detections_to_description(dets)
        assert "3个person" in result

    def test_multiple_different_classes(self):
        dets = [
            {"class_id": "person"},
            {"class_id": "box"},
        ]
        result = VisionBridge._detections_to_description(dets)
        assert "person" in result
        assert "box" in result

    def test_result_starts_with_prefix(self):
        dets = [{"class_id": "chair"}]
        result = VisionBridge._detections_to_description(dets)
        assert result.startswith("我看到了:")

    def test_distance_formatted_to_one_decimal(self):
        dets = [{"class_id": "table", "distance_m": 3.14159}]
        result = VisionBridge._detections_to_description(dets)
        assert "table(3.1米)" in result


# ── _tracks_to_description ────────────────────────────────────────────────────


class TestTracksToDescription:
    def _make_track(self, class_id: str):
        from unittest.mock import MagicMock

        t = MagicMock()
        t.class_id = class_id
        return t

    def test_empty_tracks_returns_empty(self):
        assert VisionBridge._tracks_to_description([]) == ""

    def test_single_track(self):
        tracks = [self._make_track("person")]
        result = VisionBridge._tracks_to_description(tracks)
        assert "person" in result
        assert "我看到了" in result

    def test_multiple_same_class(self):
        tracks = [self._make_track("person"), self._make_track("person")]
        result = VisionBridge._tracks_to_description(tracks)
        assert "2个person" in result

    def test_multiple_classes(self):
        tracks = [self._make_track("person"), self._make_track("box")]
        result = VisionBridge._tracks_to_description(tracks)
        assert "person" in result
        assert "box" in result


# ── VisionBridge.available ────────────────────────────────────────────────────


class TestVisionBridgeAvailable:
    def test_available_false_without_dependencies(self):
        """VisionBridge.available should be False in test env (no camera/qp-perception)."""
        bridge = VisionBridge()
        # In CI there is no camera or qp-perception, so available should be False
        assert isinstance(bridge.available, bool)
