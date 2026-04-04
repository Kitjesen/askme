"""Tests for AlertDispatcher — multi-channel alert delivery."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from askme.pipeline.alert_dispatcher import AlertDispatcher


def _make_dispatcher(**kwargs) -> AlertDispatcher:
    return AlertDispatcher(**kwargs)


class TestInit:
    def test_defaults(self):
        d = _make_dispatcher()
        assert d._voice is None
        assert d._webhook_url is None
        assert d._robot_name == "Thunder"

    def test_custom_robot_name(self):
        d = _make_dispatcher(robot_name="Spot")
        assert d._robot_name == "Spot"

    def test_config_sets_webhook_url(self):
        d = _make_dispatcher(config={"webhook_url": "https://example.com/hook"})
        assert d._webhook_url == "https://example.com/hook"

    def test_config_sets_voice_cooldown(self):
        d = _make_dispatcher(config={"voice_cooldown": 5})
        assert d._voice_cooldown == 5.0

    def test_custom_severity_routes(self):
        routes = {"info": ["log"], "warning": ["log"], "error": ["log"]}
        d = _make_dispatcher(config={"severity_routes": routes})
        assert d._routes["info"] == ["log"]


class TestDispatchLog:
    def test_log_channel_always_added(self):
        d = _make_dispatcher(config={"severity_routes": {"info": ["log"]}})
        sent = d.dispatch("hello", severity="info")
        assert "log" in sent

    def test_returns_list(self):
        d = _make_dispatcher()
        result = d.dispatch("test message")
        assert isinstance(result, list)

    def test_unknown_severity_falls_back_to_info_route(self):
        d = _make_dispatcher(config={"severity_routes": {"info": ["log"]}})
        sent = d.dispatch("test", severity="critical")
        assert "log" in sent


class TestDispatchVoice:
    def _make_voice(self, *, is_busy: bool = False) -> MagicMock:
        voice = MagicMock()
        voice.is_busy = is_busy
        return voice

    def test_voice_sent_when_not_busy(self):
        voice = self._make_voice()
        d = _make_dispatcher(
            voice=voice,
            config={"severity_routes": {"info": ["voice", "log"]}},
        )
        sent = d.dispatch("test", severity="info")
        assert "voice" in sent
        voice.speak.assert_called_once_with("test")

    def test_voice_skipped_when_busy(self):
        voice = self._make_voice(is_busy=True)
        d = _make_dispatcher(
            voice=voice,
            config={"severity_routes": {"info": ["voice", "log"]}},
        )
        sent = d.dispatch("test", severity="info")
        assert "voice" not in sent
        voice.speak.assert_not_called()

    def test_voice_skipped_when_none(self):
        d = _make_dispatcher(config={"severity_routes": {"info": ["voice", "log"]}})
        sent = d.dispatch("test", severity="info")
        assert "voice" not in sent

    def test_voice_cooldown_suppresses_rapid_alerts(self):
        voice = self._make_voice()
        d = _make_dispatcher(
            voice=voice,
            config={"severity_routes": {"info": ["voice", "log"]}, "voice_cooldown": 999},
        )
        sent1 = d.dispatch("first", severity="info")
        sent2 = d.dispatch("second", severity="info")
        assert "voice" in sent1
        assert "voice" not in sent2  # suppressed by cooldown

    def test_voice_speak_called_with_full_pipeline(self):
        voice = self._make_voice()
        d = _make_dispatcher(
            voice=voice,
            config={"severity_routes": {"info": ["voice"]}},
        )
        d.dispatch("hello robot", severity="info")
        voice.start_playback.assert_called_once()
        voice.speak.assert_called_once_with("hello robot")
        voice.wait_speaking_done.assert_called_once()
        voice.stop_playback.assert_called_once()


class TestDispatchWebhook:
    def test_webhook_sent_when_url_configured(self):
        d = _make_dispatcher(
            config={
                "webhook_url": "http://fake.local/hook",
                "severity_routes": {"warning": ["webhook", "log"]},
            }
        )
        with patch.object(d, "_post_json", return_value=True) as mock_post:
            sent = d.dispatch("alert!", severity="warning")
        assert "webhook" in sent
        mock_post.assert_called_once()
        body = mock_post.call_args[0][1]
        assert body["message"] == "alert!"
        assert body["severity"] == "warning"

    def test_webhook_skipped_when_no_url(self):
        d = _make_dispatcher(
            config={"severity_routes": {"warning": ["webhook", "log"]}}
        )
        with patch.object(d, "_post_json") as mock_post:
            sent = d.dispatch("alert!", severity="warning")
        assert "webhook" not in sent
        mock_post.assert_not_called()

    def test_webhook_body_includes_robot_id(self):
        d = _make_dispatcher(
            robot_id="robot-42",
            config={
                "webhook_url": "http://fake.local/hook",
                "severity_routes": {"info": ["webhook"]},
            }
        )
        with patch.object(d, "_post_json", return_value=True) as mock_post:
            d.dispatch("msg", severity="info")
        body = mock_post.call_args[0][1]
        assert body["robot_id"] == "robot-42"

    def test_webhook_returns_false_does_not_add_to_sent(self):
        d = _make_dispatcher(
            config={
                "webhook_url": "http://fake.local/hook",
                "severity_routes": {"info": ["webhook"]},
            }
        )
        with patch.object(d, "_post_json", return_value=False):
            sent = d.dispatch("msg", severity="info")
        assert "webhook" not in sent


class TestDispatchWecom:
    def test_wecom_sent_when_url_configured(self):
        d = _make_dispatcher(
            config={
                "wecom_webhook": "http://wecom.local/send",
                "severity_routes": {"error": ["wecom"]},
            }
        )
        with patch.object(d, "_post_json", return_value=True):
            sent = d.dispatch("critical!", severity="error")
        assert "wecom" in sent

    def test_wecom_skipped_when_no_url(self):
        d = _make_dispatcher(config={"severity_routes": {"error": ["wecom"]}})
        with patch.object(d, "_post_json") as mock_post:
            sent = d.dispatch("msg", severity="error")
        assert "wecom" not in sent
        mock_post.assert_not_called()


class TestDispatchDingtalk:
    def test_dingtalk_sent(self):
        d = _make_dispatcher(
            config={
                "dingtalk_webhook": "http://ding.local/send",
                "severity_routes": {"error": ["dingtalk"]},
            }
        )
        with patch.object(d, "_post_json", return_value=True):
            sent = d.dispatch("msg", severity="error")
        assert "dingtalk" in sent

    def test_dingtalk_skipped_when_no_url(self):
        d = _make_dispatcher(config={"severity_routes": {"error": ["dingtalk"]}})
        with patch.object(d, "_post_json") as mock_post:
            sent = d.dispatch("msg", severity="error")
        assert "dingtalk" not in sent


class TestDispatchFeishu:
    def test_feishu_sent(self):
        d = _make_dispatcher(
            config={
                "feishu_webhook": "http://feishu.local/send",
                "severity_routes": {"error": ["feishu"]},
            }
        )
        with patch.object(d, "_post_json", return_value=True):
            sent = d.dispatch("msg", severity="error")
        assert "feishu" in sent

    def test_feishu_skipped_when_no_url(self):
        d = _make_dispatcher(config={"severity_routes": {"error": ["feishu"]}})
        with patch.object(d, "_post_json") as mock_post:
            sent = d.dispatch("msg", severity="error")
        assert "feishu" not in sent


class TestChannelFailureHandling:
    def test_channel_exception_does_not_crash_dispatch(self):
        voice = MagicMock()
        voice.is_busy = False
        voice.speak.side_effect = RuntimeError("TTS exploded")
        d = _make_dispatcher(
            voice=voice,
            config={"severity_routes": {"info": ["voice", "log"]}},
        )
        # Should not raise
        sent = d.dispatch("test", severity="info")
        assert "log" in sent  # log still processed


class TestImageHelpers:
    def test_read_image_base64_missing_file(self):
        result = AlertDispatcher._read_image_base64("/nonexistent/path.jpg")
        assert result is None

    def test_read_image_base64_valid_file(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff")  # fake JPEG header
        result = AlertDispatcher._read_image_base64(str(img))
        assert result is not None
        assert isinstance(result, str)

    def test_file_md5_missing(self):
        result = AlertDispatcher._file_md5("/nonexistent/file")
        assert result is None

    def test_file_md5_valid_file(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello")
        result = AlertDispatcher._file_md5(str(f))
        assert result is not None
        assert len(result) == 32  # hex MD5


class TestPostJson:
    def test_returns_false_on_url_error(self):
        from urllib.error import URLError
        with patch("askme.pipeline.alert_dispatcher.request.urlopen",
                   side_effect=URLError("no route")):
            result = AlertDispatcher._post_json("http://fake.local", {"key": "val"})
        assert result is False

    def test_payload_is_valid_json(self):
        """_post_json encodes body as UTF-8 JSON."""
        captured = {}

        def fake_urlopen(req, timeout):
            captured["data"] = req.data
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            resp.status = 200
            return resp

        with patch("askme.pipeline.alert_dispatcher.request.urlopen", fake_urlopen):
            AlertDispatcher._post_json("http://fake.local", {"msg": "hello 世界"})

        decoded = json.loads(captured["data"].decode("utf-8"))
        assert decoded["msg"] == "hello 世界"
