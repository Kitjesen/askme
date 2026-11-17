"""TTS retry helper: attempt counts, empty audio, and programmer errors."""

from __future__ import annotations

import pytest

from askme.voice.speech.tts_retry import (
    DEFAULT_TTS_MAX_ATTEMPTS,
    TTSSynthesisFailure,
    synthesize_with_retry,
)

pytestmark = pytest.mark.unit


def test_success_on_the_first_try_uses_one_attempt(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[str] = []

    def synthesize() -> bytes:
        calls.append("ok")
        return b"audio"

    with caplog.at_level("INFO"):
        result = synthesize_with_retry(synthesize, original_text="小算到了")

    assert result.ok is True
    assert result.attempts == 1
    assert result.retries == 0
    assert result.value == b"audio"
    assert calls == ["ok"]
    assert "语音生成成功: 小算到了，重试0次" in caplog.text


def test_two_failures_then_success_counts_three_attempts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = {"n": 0}

    def synthesize() -> bytes:
        state["n"] += 1
        if state["n"] <= 2:
            raise RuntimeError(f"down-{state['n']}")
        return b"pcm"

    with caplog.at_level("WARNING"):
        result = synthesize_with_retry(synthesize, original_text="原文")

    assert result.ok is True
    assert result.attempts == 3
    assert result.retries == 2
    assert result.value == b"pcm"
    assert state["n"] == 3
    assert "语音生成失败1次: 原文" in caplog.text
    assert "语音生成失败2次: 原文" in caplog.text
    assert "down-1" in caplog.text
    assert "down-2" in caplog.text


def test_empty_audio_is_retried_then_accepted() -> None:
    scripts = [b"", b"", b"wav"]

    def synthesize() -> bytes:
        return scripts.pop(0)

    result = synthesize_with_retry(synthesize, original_text="空")

    assert result.ok is True
    assert result.attempts == 3
    assert result.value == b"wav"
    assert scripts == []


def test_five_failures_return_a_clear_error_and_do_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = {"n": 0}

    def synthesize() -> bytes:
        state["n"] += 1
        raise OSError("network down")

    with caplog.at_level("ERROR"):
        result = synthesize_with_retry(synthesize, original_text="请播报")

    assert result.ok is False
    assert result.attempts == DEFAULT_TTS_MAX_ATTEMPTS == 5
    assert result.retries == 5
    assert state["n"] == 5
    assert isinstance(result.error, TTSSynthesisFailure)
    assert result.error is not None
    assert result.error.attempts == 5
    assert result.error.original_text == "请播报"
    assert "请检查网络或服务是否正常" in str(result.error)
    assert "network down" in str(result.error)
    assert "语音生成失败: 请播报，请检查网络或服务是否正常" in caplog.text


def test_all_empty_results_fail_without_raising() -> None:
    state = {"n": 0}

    def synthesize() -> bytes:
        state["n"] += 1
        return b""

    result = synthesize_with_retry(synthesize, original_text="静音", max_attempts=5)

    assert result.ok is False
    assert result.attempts == 5
    assert state["n"] == 5
    assert isinstance(result.error, TTSSynthesisFailure)
    assert result.error is not None
    assert result.error.last_error is None
    assert "空音频" in str(result.error)


def test_type_error_and_value_error_are_not_retried() -> None:
    type_calls = {"n": 0}
    value_calls = {"n": 0}

    def raise_type() -> bytes:
        type_calls["n"] += 1
        raise TypeError("bug")

    def raise_value() -> bytes:
        value_calls["n"] += 1
        raise ValueError("bug")

    with pytest.raises(TypeError):
        synthesize_with_retry(raise_type, original_text="原文")
    with pytest.raises(ValueError):
        synthesize_with_retry(raise_value, original_text="原文")

    assert type_calls["n"] == 1
    assert value_calls["n"] == 1


def test_false_is_empty_and_true_is_success() -> None:
    state = {"n": 0}

    def synthesize() -> bool:
        state["n"] += 1
        return state["n"] >= 2

    result = synthesize_with_retry(synthesize, original_text="布尔")

    assert result.ok is True
    assert result.attempts == 2
    assert result.value is True


def test_custom_empty_check_and_invalid_max_attempts() -> None:
    def synthesize() -> dict[str, int]:
        return {"samples": 0}

    result = synthesize_with_retry(
        synthesize,
        original_text="自定义",
        max_attempts=1,
        is_empty=lambda value: value["samples"] <= 0,
    )

    assert result.ok is False
    assert result.attempts == 1
    with pytest.raises(ValueError, match="max_attempts"):
        synthesize_with_retry(lambda: b"x", original_text="x", max_attempts=0)
