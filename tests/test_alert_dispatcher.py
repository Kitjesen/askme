"""Tests for AlertDispatcher — multi-channel alert delivery."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from askme.pipeline.alert_dispatcher import AlertDispatcher
from askme.pipeline.incident_alerts import INCIDENT_ALERTS, format_incident_alert


def _make_dispatcher(**kwargs) -> AlertDispatcher:
    return AlertDispatcher(**kwargs)


class TestInit:
    def test_defaults(self):
        d = _make_dispatcher()
        assert d._voice is None
        assert d._webhook_url is None
        assert d._robot_name == "现场机器人"

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
        assert d.last_delivery_report == [
            {"channel": "dingtalk", "status": "sent", "reason": ""}
        ]

    def test_dingtalk_skipped_when_no_url(self):
        d = _make_dispatcher(config={"severity_routes": {"error": ["dingtalk"]}})
        with patch.object(d, "_post_json") as mock_post:
            sent = d.dispatch("msg", severity="error")
        assert "dingtalk" not in sent
        assert d.last_delivery_report == [
            {"channel": "dingtalk", "status": "not_sent", "reason": "not_configured"}
        ]

    def test_dingtalk_uses_channel_specific_incident_text(self):
        d = _make_dispatcher(
            config={
                "dingtalk_webhook": "http://ding.local/send",
                "severity_routes": {"error": ["dingtalk"]},
            }
        )
        with patch.object(d, "_post_json", return_value=True) as mock_post:
            d.dispatch(
                "voice text",
                severity="error",
                payload={"dingtalk_message": "security team text"},
            )
        body = mock_post.call_args[0][1]
        assert "security team text" in body["markdown"]["text"]
        assert "voice text" not in body["markdown"]["text"]

    def test_dingtalk_signed_url_when_secret_configured(self):
        d = _make_dispatcher(
            config={
                "dingtalk_webhook": "http://ding.local/send?access_token=token",
                "dingtalk_secret": "SECabc123",
                "severity_routes": {"error": ["dingtalk"]},
            }
        )
        with (
            patch("askme.pipeline.alert_dispatcher.time.time", return_value=1710000000.123),
            patch.object(d, "_post_json", return_value=True) as mock_post,
        ):
            sent = d.dispatch("msg", severity="error")

        assert sent == ["dingtalk"]
        signed_url = mock_post.call_args[0][0]
        assert signed_url.startswith("http://ding.local/send?access_token=token&")
        assert "timestamp=1710000000123" in signed_url
        assert "sign=" in signed_url

    def test_dingtalk_unsigned_url_when_secret_missing(self):
        d = _make_dispatcher(
            config={
                "dingtalk_webhook": "http://ding.local/send?access_token=token",
                "severity_routes": {"error": ["dingtalk"]},
            }
        )
        with patch.object(d, "_post_json", return_value=True) as mock_post:
            d.dispatch("msg", severity="error")

        assert mock_post.call_args[0][0] == "http://ding.local/send?access_token=token"

    def test_delivery_report_is_archived(self, tmp_path: Path):
        archive_path = tmp_path / "incidents.jsonl"
        d = _make_dispatcher(
            config={
                "dingtalk_webhook": "http://ding.local/send",
                "incident_archive_path": str(archive_path),
                "severity_routes": {"error": ["dingtalk", "log"]},
            }
        )
        with patch.object(d, "_post_json", return_value=False):
            sent = d.dispatch("msg", severity="error", topic="robot.fall_unrecoverable")

        assert sent == ["log"]
        record = json.loads(archive_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["delivery_report"] == [
            {"channel": "dingtalk", "status": "not_sent", "reason": "not_configured_or_failed"},
            {"channel": "log", "status": "sent", "reason": ""},
        ]


class TestIncidentAlerts:
    def test_incident_templates_are_customer_readable_utf8(self):
        suspicious = ("�", "锟", "å", "ç", "é", "æ", "è", "Ð", "Ñ", "Â", "ã")
        sample = {
            "location": "A区东侧",
            "mission_id": "mission-1",
            "duration_s": 18,
            "distance_m": 0.8,
            "joint_id": "hip-left",
            "fault_code": "over_current",
            "zone_name": "窗户北侧",
            "image_path": "/evidence/evt-1.jpg",
            "plate_number": "沪A12345",
            "temperature_c": 68,
            "smoke_level": "high",
            "bin_id": "bin-a-01",
            "fill_ratio": "92%",
            "person_count": 8,
            "duration_min": 35,
            "target_location": "A区门厅",
            "operator_id": "admin-1",
            "interrupted_mission_id": "patrol-1",
        }
        for topic in INCIDENT_ALERTS:
            alert = format_incident_alert(topic, sample)
            assert alert is not None
            rendered = "\n".join(
                str(alert[key]) for key in ("voice", "dingtalk", "operator_action")
            )
            assert not any(char in rendered for char in suspicious)

    def test_required_robot_incidents_have_fixed_templates(self):
        required_topics = {
            "robot.fall_unrecoverable",
            "navigation.immobilized",
            "security.malicious_blocking",
            "actuator.joint_motor_fault",
            "security.night_stranger_photo",
            "traffic.illegal_parking",
            "safety.fire_or_smoke",
            "sanitation.trash_bin_full",
            "security.crowd_gathering",
            "patrol.urgent_dispatch",
        }
        assert required_topics <= set(INCIDENT_ALERTS)
        for topic in required_topics:
            alert = format_incident_alert(
                topic,
                {
                    "location": "A区东侧",
                    "mission_id": "mission-1",
                    "duration_s": 18,
                    "distance_m": 0.8,
                    "joint_id": "hip-left",
                    "fault_code": "over_current",
                    "zone_name": "窗户北侧",
                    "image_path": "/evidence/evt-1.jpg",
                    "plate_number": "沪A12345",
                    "temperature_c": 68,
                    "smoke_level": "high",
                    "bin_id": "bin-a-01",
                    "fill_ratio": "92%",
                    "person_count": 8,
                    "duration_min": 35,
                    "target_location": "A区北门",
                    "operator_id": "admin-1",
                    "interrupted_mission_id": "patrol-1",
                },
            )
            assert alert is not None
            assert alert["severity"] in {"error", "warning"}
            assert alert["voice"]
            assert alert["dingtalk"]
            assert alert["operator_action"]
            assert alert["archive_required"] is True
            assert alert["notification_group"] in {"security", "cleaning", "operations"}

    def test_missing_incident_placeholder_uses_dash(self):
        alert = format_incident_alert("robot.fall_unrecoverable", {})
        assert alert is not None
        assert "位置：-" in alert["dingtalk"]

    def test_trash_bin_alert_routes_to_cleaning_group(self):
        alert = format_incident_alert(
            "sanitation.trash_bin_full",
            {"location": "C区西门", "bin_id": "bin-c-02", "fill_ratio": "90%"},
        )
        assert alert is not None
        assert alert["notification_group"] == "cleaning"
        assert "保洁" in alert["dingtalk"]


class TestIncidentArchive:
    def test_dispatch_writes_jsonl_incident_archive(self, tmp_path: Path):
        archive = tmp_path / "incidents" / "incident-alerts.jsonl"
        d = _make_dispatcher(
            robot_id="robot-42",
            config={
                "incident_archive_path": str(archive),
                "severity_routes": {"error": ["log"]},
            },
        )
        sent = d.dispatch(
            "紧急情况：我被卡住，当前无法继续运动，请协助清理周围障碍物。",
            severity="error",
            topic="navigation.immobilized",
            payload={"event_id": "evt-1", "location": "A区"},
        )
        assert sent == ["log"]
        rows = archive.read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1
        record = json.loads(rows[0])
        assert record["event_id"] == "evt-1"
        assert record["robot_id"] == "robot-42"
        assert record["topic"] == "navigation.immobilized"
        assert record["channels"] == ["log"]


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

    def test_return_result_includes_http_status_and_response_excerpt(self):
        def fake_urlopen(req, timeout):
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            resp.status = 200
            resp.read.return_value = b'{"errcode":0,"errmsg":"ok"}'
            return resp

        with patch("askme.pipeline.alert_dispatcher.request.urlopen", fake_urlopen):
            result = AlertDispatcher._post_json(
                "http://fake.local",
                {"msg": "hello"},
                return_result=True,
            )

        assert result == {
            "ok": True,
            "http_status": 200,
            "reason": "",
            "response_excerpt": '{"errcode":0,"errmsg":"ok"}',
        }

    def test_return_result_includes_http_error_details(self):
        err = HTTPError(
            "http://fake.local",
            400,
            "Bad Request",
            hdrs=None,
            fp=BytesIO(b'{"errcode":310000,"errmsg":"keywords not in content"}'),
        )
        with patch("askme.pipeline.alert_dispatcher.request.urlopen", side_effect=err):
            result = AlertDispatcher._post_json(
                "http://fake.local",
                {"msg": "hello"},
                return_result=True,
            )

        assert result["ok"] is False
        assert result["http_status"] == 400
        assert result["reason"] == "http_400"
        assert "keywords not in content" in result["response_excerpt"]

    def test_delivery_report_includes_real_http_failure_details(self):
        d = _make_dispatcher(
            config={
                "dingtalk_webhook": "http://ding.local/send",
                "severity_routes": {"error": ["dingtalk"]},
            }
        )

        def fake_urlopen(req, timeout):
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            resp.status = 500
            resp.read.return_value = b'{"errcode":500,"errmsg":"internal"}'
            return resp

        with patch("askme.pipeline.alert_dispatcher.request.urlopen", fake_urlopen):
            sent = d.dispatch("msg", severity="error")

        assert sent == []
        assert d.last_delivery_report == [{
            "channel": "dingtalk",
            "status": "not_sent",
            "reason": "http_500",
            "http_status": 500,
            "response_excerpt": '{"errcode":500,"errmsg":"internal"}',
        }]
