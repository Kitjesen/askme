"""TTSEngine applies pronunciation before synthesis and retries that call."""

from __future__ import annotations

import json
from typing import Any

import pytest

from askme.config import get_config
from askme.voice.output.tts import TTSEngine


@pytest.fixture(autouse=True)
def _clear_correct_words_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASKME_TTS_CORRECT_WORDS", raising=False)


def _engine(**config: Any) -> TTSEngine:
    base: dict[str, Any] = {
        "backend": "edge",
        "phrase_cache_enabled": False,
        "output_tail_silence_seconds": 0.0,
    }
    base.update(config)
    return TTSEngine(base)


class _Audio:
    def __init__(self, samples: list[float]) -> None:
        self.samples = samples
        self.sample_rate = 24000


class _ScriptedLocal:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.texts: list[str] = []

    def generate(self, text: str, sid: int = 0, speed: float = 1.0) -> _Audio:
        del sid, speed
        self.texts.append(text)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _Audio(item)


def _use_local(engine: TTSEngine, local: _ScriptedLocal) -> None:
    engine._backend = "local"
    engine._local_tts = local
    engine._local_sample_rate = 24000


def test_speak_keeps_display_text_and_pushes_emotion_before_cleanup() -> None:
    sent: list[str] = []
    engine = _engine(correct_words=["小算|小蒜"], device_session_id="board-1")
    try:
        engine.set_device_emotion_sender(sent.append)
        engine.speak("小算😂到了")

        _generation, queued = engine.tts_text_queue.get_nowait()
        payload = json.loads(sent[0])
        assert queued == "小算到了"
        assert "😂" not in queued
        assert payload == {
            "type": "llm",
            "text": "😂",
            "emotion": "funny",
            "session_id": "board-1",
        }
        assert engine._last_device_emotion == payload
    finally:
        engine.shutdown()


def test_generate_audio_corrects_pronunciation_and_can_restore_it() -> None:
    engine = _engine(correct_words=["小算|小蒜", "大算|大蒜"])
    local = _ScriptedLocal([[0.1, 0.2]])
    try:
        _use_local(engine, local)
        generated = engine._generate_audio("小算来自大算", engine._get_generation())

        assert generated == "local"
        assert local.texts == ["小蒜来自大蒜"]
        assert engine.display_text_for("小蒜来自大蒜") == "小算来自大算"
        assert engine._last_tts_display_text == "小算来自大算"
        assert engine._last_tts_spoken_text == "小蒜来自大蒜"
    finally:
        engine.shutdown()


def test_empty_word_list_disables_examples_and_missing_key_uses_them() -> None:
    disabled = _engine(correct_words=[])
    defaults = _engine()
    try:
        assert disabled.spoken_text_for("小算和大算") == "小算和大算"
        assert defaults.spoken_text_for("小算") == "小蒜"
        assert defaults.spoken_text_for("聚龙科创e谷") == "聚龙科创伊谷"
    finally:
        disabled.shutdown()
        defaults.shutdown()


def test_tts_retries_twice_then_succeeds_with_the_spoken_form(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = _engine(correct_words=["小算|小蒜"])
    local = _ScriptedLocal(
        [RuntimeError("net-1"), RuntimeError("net-2"), [0.25, 0.5]],
    )
    try:
        _use_local(engine, local)
        with caplog.at_level("WARNING"):
            generated = engine._generate_audio("请播报小算", engine._get_generation())

        assert generated == "local"
        assert local.texts == ["请播报小蒜", "请播报小蒜", "请播报小蒜"]
        assert "语音生成失败1次: 请播报小算" in caplog.text
        assert "语音生成失败2次: 请播报小算" in caplog.text
        assert engine._has_buffered_audio()
    finally:
        engine.shutdown()


def test_tts_retry_stops_after_five_failures_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = _engine(correct_words=[])
    local = _ScriptedLocal([OSError("down") for _ in range(5)])
    try:
        _use_local(engine, local)
        with caplog.at_level("ERROR"):
            generated = engine._generate_audio("请播报", engine._get_generation())

        assert generated is None
        assert len(local.texts) == 5
        assert local.script == []
        assert "语音生成失败: 请播报，请检查网络或服务是否正常" in caplog.text
        assert not engine._has_buffered_audio()
    finally:
        engine.shutdown()


def test_tts_does_not_retry_programmer_errors() -> None:
    engine = _engine(correct_words=[])
    local = _ScriptedLocal([TypeError("bug"), ValueError("bug")])
    try:
        _use_local(engine, local)
        with pytest.raises(TypeError):
            engine._generate_audio("请播报", engine._get_generation())
        assert len(local.texts) == 1

        local.script.append(ValueError("bug"))
        with pytest.raises(ValueError):
            engine._generate_audio("请播报", engine._get_generation())
        assert len(local.texts) == 2
    finally:
        engine.shutdown()


def test_shipped_config_loads_the_word_list() -> None:
    words = get_config(reload=True)["voice"]["tts"]["correct_words"]

    assert "小算|小蒜" in words
    assert "大算|大蒜" in words
    assert "聚龙科创e谷|聚龙科创伊谷" in words
    assert "巨龙科创e谷|巨龙科创伊谷" in words
