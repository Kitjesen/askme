"""Protocol and configuration tests for Volcengine streaming ASR."""

from __future__ import annotations

import gzip
import json
import struct

from askme.voice.input.cloud_asr import CloudASR, cloud_asr_credentials_present
from askme.voice.input.volcengine_asr import VolcengineASR


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "enabled": True,
        "provider": "volcengine",
        "app_id": "app-test",
        "access_token": "access-test",
        "resource_id": "volc.seedasr.sauc.duration",
        "sample_rate": 16000,
    }
    config.update(overrides)
    return config


def _server_frame(
    payload: dict[str, object],
    *,
    flags: int = 1,
    sequence: int = 1,
) -> bytes:
    compressed = gzip.compress(json.dumps(payload).encode("utf-8"))
    header = bytes((0x11, 0x90 | flags, 0x11, 0x00))
    return header + struct.pack(">iI", sequence, len(compressed)) + compressed


class TestVolcengineASRConfig:
    def test_cloud_asr_dispatches_volcengine_provider(self) -> None:
        asr = CloudASR(_config())

        assert isinstance(asr, VolcengineASR)

    def test_old_console_credentials_are_available(self) -> None:
        asr = VolcengineASR(_config())

        assert asr.available is True

    def test_new_console_api_key_is_available(self) -> None:
        asr = VolcengineASR(
            _config(app_id="", access_token="", api_key="new-console-key")
        )

        assert asr.available is True

    def test_provider_aware_credential_check(self) -> None:
        assert cloud_asr_credentials_present(_config()) is True
        assert cloud_asr_credentials_present(_config(access_token="")) is False

    def test_old_console_headers_do_not_expose_secret_key(self) -> None:
        asr = VolcengineASR(_config(secret_key="must-not-be-used"))

        headers = asr._connection_headers()

        assert "X-Api-App-Key: app-test" in headers
        assert "X-Api-Access-Key: access-test" in headers
        assert "X-Api-Resource-Id: volc.seedasr.sauc.duration" in headers
        assert "must-not-be-used" not in repr(headers)

    def test_new_console_uses_x_api_key(self) -> None:
        asr = VolcengineASR(
            _config(app_id="", access_token="", api_key="new-console-key")
        )

        headers = asr._connection_headers()

        assert "X-Api-Key: new-console-key" in headers
        assert not any(header.startswith("X-Api-App-Key:") for header in headers)


class TestVolcengineBinaryProtocol:
    def test_full_request_is_json_gzip_binary_frame(self) -> None:
        asr = VolcengineASR(_config(hotwords=["小算", "聚龙科创e谷"]))

        frame = asr._build_full_request_frame()

        assert frame[:4] == bytes((0x11, 0x10, 0x11, 0x00))
        payload_size = struct.unpack(">I", frame[4:8])[0]
        payload = json.loads(gzip.decompress(frame[8 : 8 + payload_size]))
        assert payload["audio"] == {
            "format": "pcm",
            "codec": "raw",
            "rate": 16000,
            "bits": 16,
            "channel": 1,
        }
        assert payload["request"]["model_name"] == "bigmodel"
        assert payload["request"]["enable_nonstream"] is True
        context = json.loads(payload["request"]["corpus"]["context"])
        assert context["hotwords"] == [
            {"word": "小算"},
            {"word": "聚龙科创e谷"},
        ]

    def test_audio_frame_marks_last_packet(self) -> None:
        asr = VolcengineASR(_config())

        frame = asr._build_audio_frame(b"\x01\x02", final=True)

        assert frame[:4] == bytes((0x11, 0x22, 0x01, 0x00))
        payload_size = struct.unpack(">I", frame[4:8])[0]
        assert gzip.decompress(frame[8 : 8 + payload_size]) == b"\x01\x02"

    def test_server_result_frame_is_decoded(self) -> None:
        asr = VolcengineASR(_config())
        frame = _server_frame(
            {
                "result": {
                    "text": "小算，带我去服务中心。",
                    "utterances": [
                        {
                            "text": "小算，带我去服务中心。",
                            "definite": True,
                        }
                    ],
                }
            },
            flags=3,
            sequence=-1,
        )

        response = asr._parse_server_frame(frame)

        assert response.payload["result"]["text"] == "小算，带我去服务中心。"
        assert response.is_final is True
        assert response.error_code is None

    def test_error_frame_preserves_provider_error(self) -> None:
        asr = VolcengineASR(_config())
        payload = json.dumps({"message": "resource not granted"}).encode("utf-8")
        frame = (
            bytes((0x11, 0xF0, 0x10, 0x00))
            + struct.pack(">II", 45000001, len(payload))
            + payload
        )

        response = asr._parse_server_frame(frame)

        assert response.error_code == 45000001
        assert response.payload["message"] == "resource not granted"


class TestVolcengineStatus:
    def test_status_snapshot_is_non_secret(self) -> None:
        asr = VolcengineASR(_config())

        snapshot = asr.status_snapshot()

        assert snapshot["provider"] == "volcengine_seed_asr"
        assert snapshot["resource_id"] == "volc.seedasr.sauc.duration"
        assert snapshot["available"] is True
        assert "access-test" not in repr(snapshot)


def test_cancel_releases_concurrent_finish_wait() -> None:
    asr = VolcengineASR(_config())
    asr._result_ready.clear()

    asr.cancel_session()

    assert asr._result_ready.is_set()
