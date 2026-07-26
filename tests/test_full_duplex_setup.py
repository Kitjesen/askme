from __future__ import annotations

import numpy as np

from askme.voice.input.aec_processor import AecStats
from askme.voice.input.full_duplex_gate import FullDuplexDecision
from askme.voice.orchestration.full_duplex_setup import configure_full_duplex
from askme.voice.output.audio_router import AudioRouter


class _FakeProcessor:
    def process_render(self, frame: np.ndarray) -> None:
        return None

    def process_capture(self, frame: np.ndarray, *, delay_ms: int) -> np.ndarray:
        return frame

    def stats(self) -> AecStats:
        return AecStats(True, True, False, "fake")

    def reset(self) -> None:
        return None


class _AudioWithAecSeams:
    def __init__(self) -> None:
        self.tts = self
        self.render_callback = None
        self.render_failure_callback = None
        self.render_reset_callback = None
        self.transport_failure_callback = None
        self.capture_callback = None
        self.capture_failure_callback = None
        self.playback_events: list[str] = []
        self._post_tts_input_cooldown_s = 3.0
        self._echo_gate_peak = 30_000
        self._audio_proc = type("Proc", (), {"_echo_gate_peak": 30_000})()

    def set_render_reference_callback(
        self,
        callback,
        *,
        on_failure=None,
        on_reset=None,
        reset_existing=True,
    ) -> None:
        del reset_existing
        self.render_callback = callback
        self.render_failure_callback = on_failure
        self.render_reset_callback = on_reset

    def set_capture_processor(self, callback, *, on_failure=None) -> None:
        self.capture_callback = callback
        self.capture_failure_callback = on_failure

    def set_render_transport_failure_callback(self, callback) -> None:
        self.transport_failure_callback = callback

    def drain_buffers(self) -> None:
        self.playback_events.append("drain")

    def stop_immediately(self) -> None:
        self.playback_events.append("stop")


def _decision(echo_control: str = "native") -> FullDuplexDecision:
    return FullDuplexDecision(
        requested=True,
        enabled=True,
        echo_control=echo_control,
        reason="ready",
        aec_backend="fake",
    )


def test_native_aec_is_wired_before_router_enters_full_duplex() -> None:
    audio = _AudioWithAecSeams()
    router = AudioRouter()

    result = configure_full_duplex(
        audio=audio,
        audio_router=router,
        decision=_decision(),
        aec_processor=_FakeProcessor(),
        aec_sample_rate_hz=16_000,
        aec_delay_ms=40,
    )

    assert result.enabled is True
    assert router.mode == "full_duplex"
    assert callable(audio.render_callback)
    assert callable(audio.render_failure_callback)
    assert callable(audio.render_reset_callback)
    assert callable(audio.transport_failure_callback)
    assert callable(audio.capture_callback)
    assert audio._post_tts_input_cooldown_s == 0.0
    assert audio._audio_proc._echo_gate_peak == 0


def test_native_aec_fails_closed_when_media_seams_are_missing() -> None:
    audio = type("Audio", (), {"tts": object()})()
    router = AudioRouter()

    result = configure_full_duplex(
        audio=audio,
        audio_router=router,
        decision=_decision(),
        aec_processor=_FakeProcessor(),
        aec_sample_rate_hz=16_000,
    )

    assert result.enabled is False
    assert result.reason == "aec_media_seam_unavailable"
    assert router.mode == "exclusive"


def test_verified_hardware_aec_does_not_require_software_callbacks() -> None:
    audio = _AudioWithAecSeams()
    router = AudioRouter()

    result = configure_full_duplex(
        audio=audio,
        audio_router=router,
        decision=_decision("hardware"),
        aec_processor=None,
        aec_sample_rate_hz=16_000,
    )

    assert result.enabled is True
    assert router.mode == "full_duplex"
    assert callable(audio.transport_failure_callback)

    audio._full_duplex_fail_closed(
        "audio_device_runtime_failure",
        RuntimeError("device busy"),
    )

    assert router.mode == "exclusive"
    assert audio._post_tts_input_cooldown_s == 3.0
    assert audio.full_duplex_status["reason"] == "audio_device_runtime_failure"


def test_hardware_aec_still_fails_closed_on_output_transport_failure() -> None:
    audio = _AudioWithAecSeams()
    router = AudioRouter()
    configure_full_duplex(
        audio=audio,
        audio_router=router,
        decision=_decision("hardware"),
        aec_processor=None,
        aec_sample_rate_hz=16_000,
    )

    assert callable(audio.transport_failure_callback)
    audio.transport_failure_callback(RuntimeError("output stream busy"))

    assert router.mode == "exclusive"
    assert audio.full_duplex_enabled is False
    assert audio.full_duplex_status["reason"] == "render_transport_runtime_failure"


def test_native_aec_preserves_output_transport_failure_reason() -> None:
    import time

    from askme.voice.output.tts import TTSEngine

    audio = _AudioWithAecSeams()
    tts = TTSEngine({"backend": "edge", "phrase_cache_enabled": False})
    audio.tts = tts
    router = AudioRouter()
    configure_full_duplex(
        audio=audio,
        audio_router=router,
        decision=_decision(),
        aec_processor=_FakeProcessor(),
        aec_sample_rate_hz=16_000,
    )

    try:
        tts.report_render_transport_failure("aplay_exit_7")
        deadline = time.monotonic() + 1.0
        while audio.full_duplex_enabled and time.monotonic() < deadline:
            time.sleep(0.01)

        assert router.mode == "exclusive"
        assert audio.full_duplex_enabled is False
        assert audio.full_duplex_status["reason"] == "render_transport_runtime_failure"
    finally:
        tts.shutdown()


def test_native_aec_runtime_failure_restores_safe_half_duplex() -> None:
    audio = _AudioWithAecSeams()
    router = AudioRouter()
    configure_full_duplex(
        audio=audio,
        audio_router=router,
        decision=_decision(),
        aec_processor=_FakeProcessor(),
        aec_sample_rate_hz=16_000,
    )

    assert callable(audio.capture_failure_callback)
    with router.output_session():
        audio.capture_failure_callback(RuntimeError("native AEC failed"))

        assert router.mode == "exclusive"
        assert router.wait_for_input_ready(timeout=0) is False

    _wait_for_playback_cleanup(audio)
    assert audio.playback_events == ["stop", "drain"]
    assert audio.render_callback is None
    assert audio.capture_callback is None
    assert audio._post_tts_input_cooldown_s == 3.0
    assert audio._echo_gate_peak == 30_000
    assert audio._audio_proc._echo_gate_peak == 30_000
    assert audio.full_duplex_enabled is False
    assert audio.full_duplex_status["reason"] == "aec_runtime_failure"


def test_render_reference_failure_restores_safe_half_duplex() -> None:
    audio = _AudioWithAecSeams()
    router = AudioRouter()
    configure_full_duplex(
        audio=audio,
        audio_router=router,
        decision=_decision(),
        aec_processor=_FakeProcessor(),
        aec_sample_rate_hz=16_000,
    )

    assert callable(audio.render_failure_callback)
    audio.render_failure_callback(RuntimeError("render clock stalled"))

    assert router.mode == "exclusive"
    _wait_for_playback_cleanup(audio)
    assert audio.playback_events == ["stop", "drain"]
    assert audio.full_duplex_enabled is False
    assert audio.full_duplex_status["reason"] == "aec_runtime_failure"


def test_runtime_failure_publishes_safe_state_before_slow_cleanup() -> None:
    import threading
    import time

    class BlockingCleanupAudio(_AudioWithAecSeams):
        def __init__(self) -> None:
            super().__init__()
            self.cleanup_blocked = threading.Event()
            self.release_cleanup = threading.Event()

        def drain_buffers(self) -> None:
            super().drain_buffers()
            self.cleanup_blocked.set()
            self.release_cleanup.wait(timeout=1.0)

    audio = BlockingCleanupAudio()
    router = AudioRouter()
    configure_full_duplex(
        audio=audio,
        audio_router=router,
        decision=_decision(),
        aec_processor=_FakeProcessor(),
        aec_sample_rate_hz=16_000,
    )

    started = time.monotonic()
    audio.capture_failure_callback(RuntimeError("AEC stalled"))
    elapsed = time.monotonic() - started
    try:
        assert elapsed < 0.1
        assert router.mode == "exclusive"
        assert audio.full_duplex_enabled is False
        assert audio._echo_gate_peak == 30_000
        assert audio.cleanup_blocked.wait(timeout=1.0)
    finally:
        audio.release_cleanup.set()


def _wait_for_playback_cleanup(audio: _AudioWithAecSeams) -> None:
    import time

    deadline = time.monotonic() + 1.0
    while len(audio.playback_events) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
