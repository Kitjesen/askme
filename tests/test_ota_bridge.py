from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
import requests

from askme.ota_bridge import OTABridge, OTABridgeMetrics


def _state_path(project_root: Path) -> Path:
    state_dir = project_root / "data" / "pytest-ota-bridge"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{uuid4().hex}.json"


class _Response:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, calls: list[dict]) -> None:
        self._calls = calls

    def post(self, url, json, headers, timeout):
        self._calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if url.endswith("/devices/register"):
            return _Response(
                {
                    "id": "INVX-THUNDER-001",
                    "device_token": "secret-device-token",
                    "registered_at": "2026-03-09T04:00:00Z",
                },
                status_code=201,
            )
        if url.endswith("/agent/heartbeat"):
            return _Response({"ok": True, "pending_configs": [], "diag_command": None})
        if url.endswith("/telemetry/report"):
            return _Response({"ok": True, "health_score": 97})
        raise AssertionError(f"unexpected OTA URL: {url}")

    def close(self) -> None:
        return


@pytest.mark.asyncio
async def test_ota_bridge_registers_and_reports_runtime_metrics(project_root, monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr("askme.ota_bridge.requests.Session", lambda: _Session(calls))

    metrics = OTABridgeMetrics()
    metrics.record_conversation_turn()
    metrics.record_conversation_turn()
    metrics.record_llm_call(0.245, success=True, mode="stream", model="claude-haiku")
    metrics.record_skill_execution(success=True)
    metrics.record_skill_execution(success=False)

    state_path = _state_path(project_root)
    bridge = OTABridge(
        {
            "enabled": True,
            "server_url": "https://ota.example.com/api",
            "channel": "stable",
            "package_name": "askme",
            "state_file": str(state_path),
            "timeout": 3.0,
            "device": {
                "product": "inovxio-dog",
                "tags": ["thunder", "askme"],
                "robot_id": "thunder-01",
                "site_id": "factory-a",
            },
        },
        metrics=metrics,
        voice_status_provider=lambda: {
            "mode": "voice",
            "pipeline_ok": True,
            "tts_backend": "edge",
            "tts_busy": False,
        },
        app_name="askme",
        app_version="4.0.0",
        voice_mode=True,
        robot_mode=False,
    )

    assert await bridge._ensure_registered() is True  # noqa: SLF001
    await bridge._send_heartbeat()  # noqa: SLF001
    await bridge._send_telemetry()  # noqa: SLF001

    assert len(calls) == 3
    assert calls[0]["url"] == "https://ota.example.com/api/devices/register"
    assert calls[0]["json"]["product"] == "inovxio-dog"
    assert calls[0]["json"]["channel"] == "stable"
    assert calls[1]["url"] == "https://ota.example.com/api/agent/heartbeat"
    assert calls[1]["headers"]["X-Device-Token"] == "secret-device-token"
    assert calls[1]["json"]["current_versions"] == {"askme": "4.0.0"}
    assert calls[2]["url"] == "https://ota.example.com/api/telemetry/report"
    custom_metrics = calls[2]["json"]["custom_metrics"]
    assert custom_metrics["conversation_count"] == 2
    assert custom_metrics["llm_latency_ms"]["last_latency_ms"] == 245.0
    assert custom_metrics["skill_success_rate"] == 0.5
    assert custom_metrics["voice_pipeline_status"]["pipeline_ok"] is True
    assert custom_metrics["robot_id"] == "thunder-01"
    assert custom_metrics["site_id"] == "factory-a"
    assert calls[0]["json"]["system_info"]["robot_id"] == "thunder-01"
    assert calls[1]["json"]["system_info"]["site_id"] == "factory-a"
    status = bridge.status_snapshot()
    assert status["state"] == "connected"
    assert status["registered"] is True
    assert status["device_id"] == "INVX-THUNDER-001"
    assert status["last_registration_attempt_at"] is not None
    assert status["last_heartbeat_at"] is not None
    assert status["last_telemetry_at"] is not None

    persisted = json.loads(Path(state_path).read_text(encoding="utf-8"))
    assert persisted["device_id"] == "INVX-THUNDER-001"
    assert persisted["device_token"] == "secret-device-token"


@pytest.mark.asyncio
async def test_ota_bridge_reloads_persisted_credentials(project_root, monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr("askme.ota_bridge.requests.Session", lambda: _Session(calls))

    state_path = _state_path(project_root)
    state_path.write_text(
        json.dumps(
            {
                "device_id": "INVX-STATE-001",
                "device_token": "persisted-token",
                "registered_at": "2026-03-09T04:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    bridge = OTABridge(
        {
            "enabled": True,
            "server_url": "https://ota.example.com/api",
            "state_file": str(state_path),
        },
        metrics=OTABridgeMetrics(),
    )

    await bridge._send_heartbeat()  # noqa: SLF001

    assert len(calls) == 1
    assert calls[0]["url"] == "https://ota.example.com/api/agent/heartbeat"
    assert calls[0]["headers"]["X-Device-Token"] == "persisted-token"


def test_ota_bridge_status_snapshot_reports_registration_state(project_root, monkeypatch) -> None:
    monkeypatch.setattr("askme.ota_bridge.requests.Session", lambda: _Session([]))

    bridge = OTABridge(
        {
            "enabled": True,
            "server_url": "https://ota.example.com/api",
            "channel": "stable",
            "state_file": str(_state_path(project_root)),
            "device": {
                "product": "inovxio-dog",
            },
        },
        metrics=OTABridgeMetrics(),
    )

    initial = bridge.status_snapshot()
    assert initial["enabled"] is True
    assert initial["registered"] is False
    assert initial["state"] == "stopped"
    assert initial["channel"] == "stable"

    bridge._set_registration(  # noqa: SLF001
        device_id="INVX-THUNDER-001",
        device_token="token",
        registered_at="2026-03-09T04:00:00Z",
    )
    bridge._set_connection_state("connected", clear_error=True)  # noqa: SLF001

    registered = bridge.status_snapshot()
    assert registered["registered"] is True
    assert registered["device_id"] == "INVX-THUNDER-001"
    assert registered["state"] == "connected"
