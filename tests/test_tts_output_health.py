"""Physical output readiness contracts for voice health reporting."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_missing_proc_asound_is_not_reported_as_available(monkeypatch) -> None:
    from askme.voice.tts import TTSEngine

    engine = object.__new__(TTSEngine)
    engine._output_device = "plughw:1,0"

    def missing_cards(_path: Path, **_kwargs: object) -> str:
        raise FileNotFoundError("/proc/asound/cards")

    monkeypatch.setattr(Path, "read_text", missing_cards)

    assert engine._alsa_output_available() is False


@pytest.mark.parametrize("cards_text", ["", "   \n", "--- no soundcards ---\n"])
def test_empty_or_cardless_proc_asound_is_unavailable(
    monkeypatch, cards_text: str
) -> None:
    from askme.voice.tts import TTSEngine

    engine = object.__new__(TTSEngine)
    engine._output_device = "default"
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _path, **_kwargs: cards_text,
    )

    assert engine._alsa_output_available() is False


@pytest.mark.parametrize("output_device", [None, "default"])
def test_default_alsa_output_is_available_when_a_card_exists(
    monkeypatch, output_device: str | None
) -> None:
    from askme.voice.tts import TTSEngine

    engine = object.__new__(TTSEngine)
    engine._output_device = output_device
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _path, **_kwargs: " 0 [PCH            ]: HDA-Intel\n",
    )

    assert engine._alsa_output_available() is True


@pytest.mark.parametrize("output_device", ["hw:2,0", "plughw:2,0"])
def test_numbered_alsa_output_requires_its_specific_card(
    monkeypatch, output_device: str
) -> None:
    from askme.voice.tts import TTSEngine

    engine = object.__new__(TTSEngine)
    engine._output_device = output_device
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _path, **_kwargs: " 0 [PCH            ]: HDA-Intel\n",
    )
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: str(path) == "/proc/asound/card0",
    )

    assert engine._alsa_output_available() is False


def test_resident_output_is_disabled_when_alsa_has_no_cards(monkeypatch) -> None:
    import askme.voice.output.tts as tts_module
    from askme.voice.tts import TTSEngine

    monkeypatch.setattr(
        tts_module.shutil,
        "which",
        lambda executable: (
            "/usr/bin/aplay" if executable == "aplay" else None
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda _path, **_kwargs: "--- no soundcards ---\n",
    )
    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "aplay",
            "output_device": "default",
            "resident_output_enabled": True,
        }
    )
    try:
        snapshot = engine.status_snapshot()
    finally:
        engine.shutdown()

    assert snapshot["resident_output"]["enabled"] is False
    assert snapshot["resident_output"]["disabled_reason"] == (
        "alsa_output_unavailable"
    )
    assert snapshot["output_ready"] is False


def test_tts_snapshot_reports_physical_output_unavailable(monkeypatch) -> None:
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "aplay",
            "output_device": "plughw:1,0",
        }
    )
    try:
        monkeypatch.setattr(engine, "_alsa_output_available", lambda: False)
        snapshot = engine.status_snapshot()
    finally:
        engine.shutdown()

    assert snapshot["output_ready"] is False
    assert snapshot["output_readiness_reason"] == "alsa_output_unavailable"


def test_audio_agent_uses_tts_physical_readiness() -> None:
    from askme.voice.audio_agent import AudioAgent

    class _TTS:
        backend = "edge"

        @staticmethod
        def status_snapshot() -> dict[str, object]:
            return {"output_ready": False}

    agent = object.__new__(AudioAgent)
    agent.tts = _TTS()

    assert agent._tts_output_ready() is False


def test_injected_resident_adapter_is_ready_without_host_alsa() -> None:
    from tests.test_tts_resident_output import _RecordingAdapter

    adapter = _RecordingAdapter()
    from askme.voice.tts import TTSEngine

    engine = TTSEngine(
        {
            "backend": "edge",
            "output_transport": "aplay",
            "resident_output_enabled": True,
            "resident_output_full_duplex_verified": True,
            "resident_output_cold_preroll_ms": 0,
            "resident_output_warm_leadin_ms": 0,
        },
        audio_output_adapter=adapter,
    )
    try:
        snapshot = engine.status_snapshot()
    finally:
        engine.shutdown()

    assert snapshot["output_ready"] is True
    assert snapshot["output_readiness_reason"] == "resident_adapter_injected"
