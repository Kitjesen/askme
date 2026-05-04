"""Tests for offline voice health checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from askme import cli
from askme.voice import health_check


def test_voice_health_reports_ok_with_complete_local_models(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(health_check, "_dependency_available", lambda _name: True)
    config = _voice_config(tmp_path, kws_keywords=["thunder @thunder"])
    _write_voice_models(tmp_path, include_kws=True)

    payload = health_check.run_voice_health(config, root=tmp_path)

    assert payload["status"] == "ok"
    assert payload["config_ok"] is True
    assert payload["models_ok"] is True
    assert payload["asr_ok"] is True
    assert payload["vad_ok"] is True
    assert payload["kws_ok"] is True
    assert payload["tts_ok"] is True
    assert payload["hardware_required"] is False
    assert payload["checks"]["vad"]["configured_key"] == "model_path"
    assert payload["health_snapshot"]["pipeline_ok"] is True
    assert payload["health_snapshot"]["wake_word_enabled"] is True


def test_voice_health_flags_missing_vad_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(health_check, "_dependency_available", lambda _name: True)
    config = _voice_config(tmp_path)
    _write_voice_models(tmp_path, include_vad=False)

    payload = health_check.run_voice_health(config, root=tmp_path)

    assert payload["status"] == "degraded"
    assert payload["vad_ok"] is False
    assert any("VAD missing model" in error for error in payload["errors"])
    assert payload["health_snapshot"]["vad_available"] is False


def test_voice_health_treats_empty_kws_keywords_as_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(health_check, "_dependency_available", lambda _name: True)
    config = _voice_config(tmp_path, kws_keywords=[])
    _write_voice_models(tmp_path, include_kws=False)

    payload = health_check.run_voice_health(config, root=tmp_path)

    assert payload["status"] == "ok"
    assert payload["kws_ok"] is True
    assert payload["checks"]["kws"]["enabled"] is False
    assert payload["health_snapshot"]["wake_word_enabled"] is False
    assert payload["health_snapshot"]["woken_up"] is True


def test_cli_runtime_voice_health_json(monkeypatch, capsys) -> None:
    seen: dict[str, bool] = {}

    def _fake_voice_health(*, live: bool) -> dict[str, object]:
        seen["live"] = live
        return {
            "status": "ok",
            "config_ok": True,
            "models_ok": True,
            "asr_ok": True,
            "vad_ok": True,
            "kws_ok": True,
            "tts_ok": True,
            "runtime_bridge_ok": True,
            "health_snapshot_ok": True,
            "hardware_required": live,
            "live_requested": live,
            "errors": [],
            "warnings": [],
        }

    monkeypatch.setattr(cli, "_run_voice_health_check", _fake_voice_health)

    cli.main(["runtime", "voice-health", "--json", "--live"])

    data = json.loads(capsys.readouterr().out)
    assert seen == {"live": True}
    assert data["status"] == "ok"
    assert data["hardware_required"] is True


def test_cli_runtime_voice_health_exits_nonzero_when_degraded(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_run_voice_health_check",
        lambda *, live: {
            "status": "degraded",
            "config_ok": False,
            "models_ok": False,
            "asr_ok": False,
            "vad_ok": False,
            "kws_ok": True,
            "tts_ok": True,
            "runtime_bridge_ok": True,
            "health_snapshot_ok": True,
            "hardware_required": live,
            "live_requested": live,
            "errors": ["voice config section is missing or empty"],
            "warnings": [],
        },
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "voice-health"])

    assert exc.value.code == 1
    assert "voice-health: degraded" in capsys.readouterr().out


def _voice_config(tmp_path: Path, *, kws_keywords: list[str] | None = None) -> dict[str, object]:
    return {
        "_project_root": str(tmp_path),
        "voice": {
            "asr": {
                "model_dir": "models/asr/test-asr",
            },
            "vad": {
                "model_path": "models/vad/silero_vad.onnx",
            },
            "kws": {
                "model_dir": "models/kws/test-kws",
                "keywords": ["wake @wake"] if kws_keywords is None else kws_keywords,
            },
            "tts": {
                "backend": "local",
                "model_dir": "models/tts/test-tts",
            },
        },
        "runtime": {
            "voice_bridge": {
                "enabled": False,
                "text_enabled": False,
            },
        },
    }


def _write_voice_models(
    tmp_path: Path,
    *,
    include_vad: bool = True,
    include_kws: bool = True,
) -> None:
    asr_dir = tmp_path / "models/asr/test-asr"
    for filename in ("tokens.txt", "encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx"):
        _write(asr_dir / filename)

    if include_vad:
        _write(tmp_path / "models/vad/silero_vad.onnx")

    if include_kws:
        kws_dir = tmp_path / "models/kws/test-kws"
        for filename in (
            "tokens.txt",
            "encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            "joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
            "keywords.txt",
        ):
            _write(kws_dir / filename)

    tts_dir = tmp_path / "models/tts/test-tts"
    for filename in ("model.onnx", "tokens.txt"):
        _write(tts_dir / filename)


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
