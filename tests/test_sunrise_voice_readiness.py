"""Tests for the aggregate Sunrise voice readiness gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from askme import cli
from askme.voice import sunrise_readiness


def test_readiness_is_ok_when_required_checks_pass() -> None:
    payload = sunrise_readiness.run_sunrise_voice_readiness(
        {},
        voice_health_runner=lambda _cfg: _ok("voice"),
        audio_doctor_runner=lambda _cfg, _guard: _ok("audio"),
    )

    assert payload["status"] == "ok"
    assert payload["summary"]["required_checks_ok"] is True
    assert payload["checks"]["room_loop"]["status"] == "skipped"
    assert any("room_loop" in warning for warning in payload["warnings"])


def test_readiness_degrades_when_audio_doctor_fails() -> None:
    payload = sunrise_readiness.run_sunrise_voice_readiness(
        {},
        voice_health_runner=lambda _cfg: _ok("voice"),
        audio_doctor_runner=lambda _cfg, _guard: _degraded("audio missing"),
    )

    assert payload["status"] == "degraded"
    assert "sunrise_audio_doctor: audio missing" in payload["errors"]


def test_readiness_requires_room_loop_when_requested() -> None:
    payload = sunrise_readiness.run_sunrise_voice_readiness(
        {},
        include_room_loop=True,
        room_loop_trials=2,
        live_tts_room_loop=True,
        voice_health_runner=lambda _cfg: _ok("voice"),
        audio_doctor_runner=lambda _cfg, _guard: _ok("audio"),
        room_loop_runner=lambda text, prefix, trials, live_tts, asr_backend: {
            "status": "ok",
            "required": True,
            "text": text,
            "prefix": prefix,
            "trials": trials,
            "live_tts": live_tts,
            "asr_backend": asr_backend,
            "errors": [],
            "warnings": [],
        },
    )

    assert payload["status"] == "ok"
    assert payload["summary"]["room_loop_required"] is True
    assert payload["checks"]["room_loop"]["trials"] == 2
    assert payload["checks"]["room_loop"]["live_tts"] is True
    assert payload["checks"]["room_loop"]["asr_backend"] == "local"


def test_readiness_can_require_cloud_asr(monkeypatch) -> None:
    monkeypatch.setattr(sunrise_readiness, "_dependency_available", lambda _module: True)
    monkeypatch.setattr(sunrise_readiness, "_websocket_client_available", lambda: True)

    payload = sunrise_readiness.run_sunrise_voice_readiness(
        {"voice": {"cloud_asr": {"enabled": True, "api_key": "sk-test"}}},
        require_cloud_asr=True,
        voice_health_runner=lambda _cfg: _ok("voice"),
        audio_doctor_runner=lambda _cfg, _guard: _ok("audio"),
    )

    assert payload["status"] == "ok"
    assert payload["summary"]["cloud_asr_required"] is True
    assert payload["summary"]["cloud_asr_ok"] is True
    assert payload["checks"]["cloud_asr"]["status"] == "ok"


def test_required_cloud_asr_makes_room_loop_use_cloud(monkeypatch) -> None:
    monkeypatch.setattr(sunrise_readiness, "_dependency_available", lambda _module: True)
    monkeypatch.setattr(sunrise_readiness, "_websocket_client_available", lambda: True)

    payload = sunrise_readiness.run_sunrise_voice_readiness(
        {"voice": {"cloud_asr": {"enabled": True, "api_key": "sk-test"}}},
        include_room_loop=True,
        require_cloud_asr=True,
        voice_health_runner=lambda _cfg: _ok("voice"),
        audio_doctor_runner=lambda _cfg, _guard: _ok("audio"),
        room_loop_runner=lambda _text, _prefix, _trials, _live_tts, asr_backend: {
            "status": "ok",
            "required": True,
            "asr_backend": asr_backend,
            "errors": [],
            "warnings": [],
        },
    )

    assert payload["summary"]["room_loop_asr"] == "cloud"
    assert payload["checks"]["room_loop"]["asr_backend"] == "cloud"


def test_readiness_degrades_when_required_cloud_asr_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(sunrise_readiness, "_dependency_available", lambda _module: True)
    monkeypatch.setattr(sunrise_readiness, "_websocket_client_available", lambda: False)

    payload = sunrise_readiness.run_sunrise_voice_readiness(
        {"voice": {"cloud_asr": {"enabled": False, "api_key": ""}}},
        require_cloud_asr=True,
        voice_health_runner=lambda _cfg: _ok("voice"),
        audio_doctor_runner=lambda _cfg, _guard: _ok("audio"),
    )

    assert payload["status"] == "degraded"
    assert payload["summary"]["cloud_asr_ok"] is False
    assert "cloud_asr: voice.cloud_asr.enabled is not true" in payload["errors"]
    assert "cloud_asr: voice.cloud_asr.api_key is empty" in payload["errors"]
    assert "cloud_asr: Cloud ASR dependency missing: websocket-client" in payload["errors"]


def test_cli_runtime_sunrise_voice_readiness_json(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_readiness(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "ok", "errors": [], "warnings": [], "checks": {}}

    monkeypatch.setattr(cli, "_run_sunrise_voice_readiness", fake_readiness)

    cli.main(
        [
            "runtime",
            "sunrise-voice-readiness",
            "--json",
            "--with-room-loop",
            "--room-loop-trials",
            "2",
            "--guard-min-seconds",
            "1.4",
            "--require-cloud-asr",
        ]
    )

    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "ok"
    assert seen["include_room_loop"] is True
    assert seen["room_loop_trials"] == 2
    assert seen["guard_min_seconds"] == 1.4
    assert seen["require_cloud_asr"] is True
    assert seen["room_loop_asr"] == "auto"


def test_cli_runtime_sunrise_voice_readiness_writes_json_out(monkeypatch, tmp_path: Path) -> None:
    out_path = tmp_path / "readiness" / "sunrise.json"
    monkeypatch.setattr(
        cli,
        "_run_sunrise_voice_readiness",
        lambda **_kwargs: {
            "status": "ok",
            "target": "sunrise-voice-readiness",
            "errors": [],
            "warnings": [],
            "checks": {},
        },
    )

    cli.main(["runtime", "sunrise-voice-readiness", "--json-out", str(out_path)])

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["target"] == "sunrise-voice-readiness"


def test_cli_runtime_sunrise_voice_readiness_exits_nonzero_when_degraded(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_run_sunrise_voice_readiness",
        lambda **_kwargs: {
            "status": "degraded",
            "errors": ["voice_health: missing"],
            "warnings": [],
            "checks": {},
        },
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "sunrise-voice-readiness"])

    assert exc.value.code == 1
    assert "sunrise-voice-readiness: degraded" in capsys.readouterr().out


def test_room_loop_sentinel_uses_fresh_artifact_dir(monkeypatch, tmp_path: Path) -> None:
    stale_json = tmp_path / "sunrise_voice_readiness_room_loop.json"
    stale_json.write_text(
        json.dumps({"summary": {"passed": True}}),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "fresh-run"

    monkeypatch.setattr(sunrise_readiness.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        sunrise_readiness.tempfile,
        "mkdtemp",
        lambda prefix: str(_mkdir(artifact_dir)),
    )

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(sunrise_readiness.subprocess, "run", lambda *_args, **_kwargs: Result())

    payload = sunrise_readiness._run_room_loop_sentinel(
        "一二三",
        "一",
        1,
        False,
        "cloud",
    )

    assert payload["status"] == "degraded"
    assert payload["artifact_dir"] == str(artifact_dir)
    assert "--asr-backend" in payload["command"]
    assert "cloud" in payload["command"]
    assert "room loop sentinel did not pass" in payload["errors"]


def _ok(name: str) -> dict[str, object]:
    return {"status": "ok", "name": name, "errors": [], "warnings": []}


def _degraded(error: str) -> dict[str, object]:
    return {"status": "degraded", "errors": [error], "warnings": []}


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
