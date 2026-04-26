"""Tests for OTABridge core — config, state management, payload building."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from askme.robot.ota_bridge import (
    OTABridge,
    OTABridgeAuthError,
    _clean_optional,
    _iso_utc_now,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_bridge(
    tmp_path: Path,
    *,
    enabled: bool = True,
    server_url: str = "http://localhost:5090",
    product: str = "test-dog",
    channel: str = "stable",
    serial_number: str | None = None,
    robot_id: str | None = None,
    site_id: str | None = None,
    tags: list[str] | None = None,
    voice_mode: bool = False,
    robot_mode: bool = False,
    voice_status_provider=None,
    heartbeat_interval: float = 60.0,
    telemetry_interval: float = 60.0,
) -> OTABridge:
    state_file = tmp_path / "ota_state.json"
    cfg: dict = {
        "enabled": enabled,
        "server_url": server_url,
        "product": product,
        "channel": channel,
        "tags": tags or [],
        "state_file": str(state_file),
        "heartbeat_interval": heartbeat_interval,
        "telemetry_interval": telemetry_interval,
    }
    if serial_number:
        cfg["serial_number"] = serial_number
    if robot_id:
        cfg["robot_id"] = robot_id
    if site_id:
        cfg["site_id"] = site_id
    return OTABridge(
        cfg,
        voice_mode=voice_mode,
        robot_mode=robot_mode,
        voice_status_provider=voice_status_provider,
    )


# ── TestInit ─────────────────────────────────────────────────────────────────

class TestInit:
    def test_enabled_with_server_url(self, tmp_path):
        bridge = _make_bridge(tmp_path, enabled=True)
        assert bridge.enabled is True

    def test_disabled_when_no_server_url(self, tmp_path):
        bridge = _make_bridge(tmp_path, enabled=True, server_url="")
        assert bridge.enabled is False

    def test_disabled_flag_respected(self, tmp_path):
        bridge = _make_bridge(tmp_path, enabled=False)
        assert bridge.enabled is False

    def test_product_stored(self, tmp_path):
        bridge = _make_bridge(tmp_path, product="my-robot")
        assert bridge._product == "my-robot"

    def test_channel_stored(self, tmp_path):
        bridge = _make_bridge(tmp_path, channel="beta")
        assert bridge._channel == "beta"

    def test_tags_stored(self, tmp_path):
        bridge = _make_bridge(tmp_path, tags=["tag1", "tag2"])
        assert "tag1" in bridge._tags
        assert "tag2" in bridge._tags

    def test_serial_number_stored(self, tmp_path):
        bridge = _make_bridge(tmp_path, serial_number="SN123")
        assert bridge._serial_number == "SN123"

    def test_robot_id_stored(self, tmp_path):
        bridge = _make_bridge(tmp_path, robot_id="robot-007")
        assert bridge._robot_id == "robot-007"

    def test_site_id_stored(self, tmp_path):
        bridge = _make_bridge(tmp_path, site_id="warehouse-A")
        assert bridge._site_id == "warehouse-A"

    def test_voice_mode_flag(self, tmp_path):
        bridge = _make_bridge(tmp_path, voice_mode=True)
        assert bridge._voice_mode is True

    def test_robot_mode_flag(self, tmp_path):
        bridge = _make_bridge(tmp_path, robot_mode=True)
        assert bridge._robot_mode is True

    def test_connection_state_stopped_when_enabled(self, tmp_path):
        bridge = _make_bridge(tmp_path, enabled=True)
        assert bridge._connection_state == "stopped"

    def test_connection_state_disabled_when_not_enabled(self, tmp_path):
        bridge = _make_bridge(tmp_path, enabled=False)
        assert bridge._connection_state == "disabled"

    def test_heartbeat_interval_minimum_enforced(self, tmp_path):
        bridge = _make_bridge(tmp_path, heartbeat_interval=1.0)  # below minimum of 5.0
        assert bridge._heartbeat_interval_s >= 5.0

    def test_none_config_does_not_crash(self, tmp_path):
        # Minimal: just None config — should result in disabled bridge (no server_url)
        bridge = OTABridge(None)
        assert bridge.enabled is False


# ── TestStatusSnapshot ────────────────────────────────────────────────────────

class TestStatusSnapshot:
    def test_has_required_keys(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        snap = bridge.status_snapshot()
        for key in ("enabled", "state", "registered", "device_id", "server_url",
                    "product", "channel", "last_error", "task_running"):
            assert key in snap

    def test_not_registered_initially(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        snap = bridge.status_snapshot()
        assert snap["registered"] is False
        assert snap["device_id"] is None

    def test_task_running_false_before_start(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        assert bridge.status_snapshot()["task_running"] is False

    def test_state_reflects_connection_state(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        snap = bridge.status_snapshot()
        assert snap["state"] == "stopped"

    def test_product_and_channel_in_snapshot(self, tmp_path):
        bridge = _make_bridge(tmp_path, product="nova", channel="dev")
        snap = bridge.status_snapshot()
        assert snap["product"] == "nova"
        assert snap["channel"] == "dev"


# ── TestRegistrationState ─────────────────────────────────────────────────────

class TestRegistrationState:
    def test_not_registered_initially(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        assert bridge._is_registered() is False

    def test_set_registration_makes_registered(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._set_registration(
            device_id="dev-001",
            device_token="tok-xyz",
            registered_at="2026-01-01T00:00:00Z",
        )
        assert bridge._is_registered() is True

    def test_clear_registration_makes_unregistered(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._set_registration(
            device_id="dev-001",
            device_token="tok-xyz",
            registered_at="2026-01-01T00:00:00Z",
        )
        bridge._clear_registration()
        assert bridge._is_registered() is False

    def test_set_registration_persists_to_file(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._set_registration(
            device_id="dev-001",
            device_token="tok-xyz",
            registered_at="2026-01-01T00:00:00Z",
        )
        state_file = tmp_path / "ota_state.json"
        data = json.loads(state_file.read_text())
        assert data["device_id"] == "dev-001"
        assert data["device_token"] == "tok-xyz"

    def test_clear_registration_overwrites_file(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._set_registration(
            device_id="dev-001", device_token="tok-xyz",
            registered_at="2026-01-01T00:00:00Z",
        )
        bridge._clear_registration()
        state_file = tmp_path / "ota_state.json"
        data = json.loads(state_file.read_text())
        assert data["device_id"] is None
        assert data["device_token"] is None


# ── TestLoadState ─────────────────────────────────────────────────────────────

class TestLoadState:
    def test_loads_persisted_credentials(self, tmp_path):
        state_file = tmp_path / "ota_state.json"
        state_file.write_text(json.dumps({
            "device_id": "loaded-id",
            "device_token": "loaded-token",
            "registered_at": "2026-01-01T00:00:00Z",
        }))
        bridge = _make_bridge(tmp_path)
        # Constructor calls _load_state; check state_file path matches
        bridge._state_path = state_file
        bridge._load_state()
        assert bridge._device_id == "loaded-id"
        assert bridge._device_token == "loaded-token"

    def test_missing_state_file_no_crash(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        # No state file written — should not raise
        assert bridge._device_id is None

    def test_corrupt_state_file_no_crash(self, tmp_path):
        state_file = tmp_path / "ota_state.json"
        state_file.write_text("not json at all")
        bridge = _make_bridge(tmp_path)
        bridge._state_path = state_file
        bridge._load_state()  # should not raise
        assert bridge._device_id is None

    def test_incomplete_state_file_not_loaded(self, tmp_path):
        state_file = tmp_path / "ota_state.json"
        state_file.write_text(json.dumps({"device_id": "only-id"}))  # no token
        bridge = _make_bridge(tmp_path)
        bridge._state_path = state_file
        bridge._device_id = None
        bridge._device_token = None
        bridge._load_state()
        assert bridge._device_id is None  # incomplete → not loaded


# ── TestConnectionState ───────────────────────────────────────────────────────

class TestConnectionState:
    def test_set_state_updates(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._set_connection_state("connected")
        assert bridge._connection_state == "connected"

    def test_set_state_with_error(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._set_connection_state("degraded", error="network down")
        assert bridge._last_error == "network down"

    def test_clear_error_removes_last_error(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._set_connection_state("degraded", error="network down")
        bridge._set_connection_state("connected", clear_error=True)
        assert bridge._last_error is None

    def test_mark_registration_attempt(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._mark_registration_attempt()
        assert bridge._connection_state == "registering"
        assert bridge._last_registration_attempt_at is not None

    def test_mark_heartbeat(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._mark_heartbeat()
        assert bridge._connection_state == "connected"
        assert bridge._last_heartbeat_at is not None

    def test_mark_telemetry(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._mark_telemetry()
        assert bridge._connection_state == "connected"
        assert bridge._last_telemetry_at is not None


# ── TestBuildPayloads ─────────────────────────────────────────────────────────

class TestBuildRegistrationPayload:
    def test_has_required_keys(self, tmp_path):
        bridge = _make_bridge(tmp_path, tags=["tag1"])
        payload = bridge._build_registration_payload()
        for key in ("product", "tags", "channel", "system_info", "ip_address"):
            assert key in payload

    def test_product_in_payload(self, tmp_path):
        bridge = _make_bridge(tmp_path, product="nova-dog")
        payload = bridge._build_registration_payload()
        assert payload["product"] == "nova-dog"

    def test_tags_in_payload(self, tmp_path):
        bridge = _make_bridge(tmp_path, tags=["env:prod", "region:cn"])
        payload = bridge._build_registration_payload()
        assert "env:prod" in payload["tags"]

    def test_serial_number_included_when_set(self, tmp_path):
        bridge = _make_bridge(tmp_path, serial_number="SN-001")
        payload = bridge._build_registration_payload()
        assert payload.get("serial_number") == "SN-001"

    def test_serial_number_excluded_when_none(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        payload = bridge._build_registration_payload()
        assert "serial_number" not in payload


class TestBuildSystemInfo:
    def test_has_required_keys(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        info = bridge._build_system_info()
        for key in ("hostname", "platform", "python_version", "service_name",
                    "voice_mode", "robot_mode"):
            assert key in info

    def test_service_name_matches_app_name(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._app_name = "my-service"
        info = bridge._build_system_info()
        assert info["service_name"] == "my-service"

    def test_voice_mode_reflected(self, tmp_path):
        bridge = _make_bridge(tmp_path, voice_mode=True)
        assert bridge._build_system_info()["voice_mode"] is True

    def test_robot_id_included_when_set(self, tmp_path):
        bridge = _make_bridge(tmp_path, robot_id="bot-42")
        info = bridge._build_system_info()
        assert info.get("robot_id") == "bot-42"

    def test_robot_id_excluded_when_none(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        info = bridge._build_system_info()
        assert "robot_id" not in info


# ── TestStart ─────────────────────────────────────────────────────────────────

class TestStart:
    def test_disabled_bridge_start_returns_none(self, tmp_path):
        bridge = _make_bridge(tmp_path, enabled=False)
        result = bridge.start()
        assert result is None

    async def test_enabled_bridge_start_creates_task(self, tmp_path):
        bridge = _make_bridge(tmp_path, enabled=True)
        task = bridge.start()
        try:
            assert task is not None
        finally:
            task.cancel()
            try:
                await task
            except (Exception, BaseException):
                pass

    async def test_start_idempotent(self, tmp_path):
        bridge = _make_bridge(tmp_path, enabled=True)
        task1 = bridge.start()
        task2 = bridge.start()
        try:
            assert task1 is task2
        finally:
            task1.cancel()
            try:
                await task1
            except (Exception, BaseException):
                pass


# ── TestPostJsonSync ──────────────────────────────────────────────────────────

class TestPostJsonSync:
    def test_401_raises_auth_error(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._device_id = "dev-001"
        bridge._device_token = "token"
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.raise_for_status = MagicMock()
        bridge._session = MagicMock()
        bridge._session.post.return_value = mock_resp
        with pytest.raises(OTABridgeAuthError):
            bridge._post_json_sync("/heartbeat", {}, True)

    def test_403_raises_auth_error(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._device_id = "dev-001"
        bridge._device_token = "token"
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.raise_for_status = MagicMock()
        bridge._session = MagicMock()
        bridge._session.post.return_value = mock_resp
        with pytest.raises(OTABridgeAuthError):
            bridge._post_json_sync("/heartbeat", {}, True)

    def test_missing_token_raises_auth_error(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._device_id = None
        bridge._device_token = None
        with pytest.raises(OTABridgeAuthError):
            bridge._post_json_sync("/heartbeat", {}, True)

    def test_non_dict_response_raises_value_error(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        bridge._device_id = "dev-001"
        bridge._device_token = "token"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = ["a", "b"]  # list, not dict
        bridge._session = MagicMock()
        bridge._session.post.return_value = mock_resp
        with pytest.raises(ValueError, match="non-object"):
            bridge._post_json_sync("/register", {}, False)

    def test_successful_response_returned(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"device_id": "abc", "device_token": "tok"}
        bridge._session = MagicMock()
        bridge._session.post.return_value = mock_resp
        result = bridge._post_json_sync("/register", {"product": "test"}, False)
        assert result["device_id"] == "abc"


# ── TestHelperFunctions ───────────────────────────────────────────────────────

class TestCleanOptional:
    def test_none_returns_none(self):
        assert _clean_optional(None) is None

    def test_empty_string_returns_none(self):
        assert _clean_optional("") is None

    def test_whitespace_returns_none(self):
        assert _clean_optional("   ") is None

    def test_non_empty_returns_stripped(self):
        assert _clean_optional("  hello  ") == "hello"

    def test_integer_converted_to_string(self):
        assert _clean_optional(42) == "42"


class TestIsoUtcNow:
    def test_returns_string(self):
        result = _iso_utc_now()
        assert isinstance(result, str)

    def test_ends_with_z(self):
        result = _iso_utc_now()
        assert result.endswith("Z")

    def test_contains_date_separator(self):
        result = _iso_utc_now()
        assert "T" in result
