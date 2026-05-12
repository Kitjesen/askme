"""Tests for Sunrise MCP01 audio diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from askme import cli
from askme.voice import sunrise_audio_doctor as doctor


def test_usb_and_asound_parsers_detect_mcp01_audio_tree() -> None:
    assert doctor.mcp01_visible("Bus 001 Device 004: ID 17ef:a03b Lenovo MCP01")
    assert doctor.lsusb_tree_has_audio_class("|__ Port 1: Dev 4, If 0, Class=Audio, Driver=, 480M")

    parsed = doctor.parse_asound_cards(
        " 0 [MCP01          ]: USB-Audio - MCP01 USB Audio\n"
        "                      Lenovo MCP01 USB Audio at usb-xhci, high speed\n"
    )

    assert parsed["cards_visible"] is True
    assert parsed["cards"][0]["id"] == "MCP01"


def test_parse_asound_cards_accepts_no_soundcards_state() -> None:
    parsed = doctor.parse_asound_cards("--- no soundcards ---\n")

    assert parsed == {"cards_visible": False, "cards": [], "detail": "no soundcards"}


def test_sunrise_audio_doctor_reports_ok_for_guarded_usb_direct_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(doctor.Path, "exists", lambda self: str(self) == "/proc/asound/cards")
    monkeypatch.setattr(
        doctor.Path,
        "read_text",
        lambda self, **_kwargs: "--- no soundcards ---\n",
    )

    payload = doctor.run_sunrise_audio_doctor(
        _config(
            tmp_path,
            usb_direct_speech_leadin_seconds=1.25,
            usb_direct_speech_wake_signal_seconds=1.05,
            usb_direct_speech_wake_signal_gain=0.28,
            usb_direct_speech_onset_cushion_seconds=0.35,
            usb_direct_speech_onset_cushion_gain=0.45,
        ),
        command_runner=_fake_runner,
    )

    assert payload["status"] == "ok"
    output = payload["checks"]["usb_output_shape"]
    assert output["final_shape_ok"] is True
    assert output["first_token_guard_ok"] is True
    assert output["speech_offset_seconds"] >= 1.5
    assert payload["checks"]["usb"]["mcp01_audio_tree_ok"] is True


def test_sunrise_audio_doctor_output_shape_accounts_for_speech_gain(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(doctor.Path, "exists", lambda self: str(self) == "/proc/asound/cards")
    monkeypatch.setattr(
        doctor.Path,
        "read_text",
        lambda self, **_kwargs: "--- no soundcards ---\n",
    )

    payload = doctor.run_sunrise_audio_doctor(
        _config(
            tmp_path,
            usb_direct_speech_gain=8.0,
            usb_direct_speech_leadin_seconds=1.25,
            usb_direct_speech_wake_signal_seconds=1.05,
            usb_direct_speech_wake_signal_gain=0.28,
            usb_direct_speech_onset_cushion_seconds=0.35,
            usb_direct_speech_onset_cushion_gain=0.45,
        ),
        command_runner=_fake_runner,
    )

    assert payload["status"] == "ok"
    output = payload["checks"]["usb_output_shape"]
    assert output["active_leadin_samples"] == output["cold_leadin_samples"]
    assert output["final_shape_ok"] is True
    assert output["first_token_guard_ok"] is True


def test_sunrise_audio_doctor_flags_short_first_token_guard(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(doctor.Path, "exists", lambda self: str(self) == "/proc/asound/cards")
    monkeypatch.setattr(
        doctor.Path,
        "read_text",
        lambda self, **_kwargs: "--- no soundcards ---\n",
    )

    payload = doctor.run_sunrise_audio_doctor(
        _config(
            tmp_path,
            usb_direct_speech_leadin_seconds=0.1,
            usb_direct_speech_wake_signal_seconds=0.0,
            usb_direct_speech_onset_cushion_seconds=0.0,
        ),
        command_runner=_fake_runner,
    )

    assert payload["status"] == "degraded"
    assert any("first real speech begins" in error for error in payload["errors"])


def test_sunrise_audio_doctor_allows_short_quiet_start_with_warning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(doctor.Path, "exists", lambda self: str(self) == "/proc/asound/cards")
    monkeypatch.setattr(
        doctor.Path,
        "read_text",
        lambda self, **_kwargs: "--- no soundcards ---\n",
    )

    payload = doctor.run_sunrise_audio_doctor(
        _config(
            tmp_path,
            usb_direct_quiet_start=True,
            usb_direct_speech_leadin_seconds=0.25,
            usb_direct_speech_warm_leadin_seconds=0.25,
            usb_direct_speech_wake_signal_seconds=1.05,
            usb_direct_speech_wake_signal_gain=0.0,
            usb_direct_speech_wake_noise_gain=0.0,
            usb_direct_speech_onset_cushion_seconds=0.0,
            usb_direct_speech_onset_cushion_gain=0.0,
        ),
        command_runner=_fake_runner,
    )

    assert payload["status"] == "ok"
    output = payload["checks"]["usb_output_shape"]
    assert output["quiet_start"] is True
    assert output["first_token_guard_ok"] is True
    assert output["speech_offset_seconds"] == pytest.approx(0.25)
    assert output["cold_leadin_peak"] == 0
    assert any("quiet start accepts" in warning for warning in payload["warnings"])


def test_sunrise_audio_doctor_rejects_trusting_persistent_stream_warmth(tmp_path: Path) -> None:
    payload = doctor.run_sunrise_audio_doctor(
        _config(
            tmp_path,
            usb_direct_persistent_stream=True,
            usb_direct_trust_persistent_warm_state=True,
            usb_direct_speech_leadin_seconds=1.25,
            usb_direct_speech_wake_signal_seconds=1.05,
            usb_direct_speech_onset_cushion_seconds=0.35,
        ),
        command_runner=_fake_runner,
        include_output_probe=False,
    )

    assert payload["status"] == "degraded"
    assert any("trust_persistent_warm_state" in error for error in payload["errors"])


def test_cli_runtime_sunrise_audio_doctor_json(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_doctor(
        *,
        include_command_probes: bool,
        include_output_probe: bool,
        guard_min_seconds: float,
    ) -> dict[str, object]:
        seen.update(
            include_command_probes=include_command_probes,
            include_output_probe=include_output_probe,
            guard_min_seconds=guard_min_seconds,
        )
        return {"status": "ok", "errors": [], "warnings": [], "checks": {}}

    monkeypatch.setattr(cli, "_run_sunrise_audio_doctor", fake_doctor)

    cli.main(
        [
            "runtime",
            "sunrise-audio-doctor",
            "--json",
            "--skip-command-probes",
            "--guard-min-seconds",
            "1.4",
        ]
    )

    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "ok"
    assert seen == {
        "include_command_probes": False,
        "include_output_probe": True,
        "guard_min_seconds": 1.4,
    }


def test_cli_runtime_sunrise_audio_doctor_exits_nonzero_when_degraded(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_run_sunrise_audio_doctor",
        lambda **_kwargs: {
            "status": "degraded",
            "errors": ["MCP01 missing"],
            "warnings": [],
            "checks": {},
        },
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "sunrise-audio-doctor"])

    assert exc.value.code == 1
    assert "sunrise-audio-doctor: degraded" in capsys.readouterr().out


def _fake_runner(command: list[str], _timeout: float) -> doctor.CommandResult:
    if command == ["lsusb"]:
        return doctor.CommandResult(
            command=command,
            returncode=0,
            stdout="Bus 001 Device 004: ID 17ef:a03b Lenovo MCP01 USB Audio\n",
        )
    if command == ["lsusb", "-t"]:
        return doctor.CommandResult(
            command=command,
            returncode=0,
            stdout="/:  Bus 01.Port 1: Dev 1, Class=root_hub\n"
            "    |__ Port 2: Dev 4, If 0, Class=Audio, Driver=, 480M\n",
        )
    raise AssertionError(f"unexpected command: {command}")


def _config(tmp_path: Path, **tts_overrides: object) -> dict[str, object]:
    return {
        "_project_root": str(tmp_path),
        "voice": {
            "tts": {
                "backend": "minimax",
                "sample_rate": 1000,
                "output_transport": "auto",
                "usb_direct_persistent_stream": True,
                "usb_direct_trust_persistent_warm_state": False,
                "usb_direct_background_prewarm": False,
                "usb_direct_speech_warm_leadin_seconds": 1.25,
                "usb_direct_speech_wake_signal_gain": 0.28,
                "usb_direct_speech_onset_gap_seconds": 0.0,
                **tts_overrides,
            }
        },
    }
