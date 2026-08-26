from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import askme.voice.orchestration.audio_agent as audio_agent_module
from askme.voice.orchestration.audio_agent import AudioAgent


class _InlineThread:
    def __init__(self, *, target, daemon: bool, **_kwargs) -> None:
        assert daemon is True
        self._target = target

    def start(self) -> None:
        self._target()


class _ManualTimer:
    created: list[_ManualTimer] = []

    def __init__(self, interval: float, function) -> None:
        self.interval = interval
        self.function = function
        self.daemon = False
        self.started = False
        self.cancelled = False
        self.created.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.function()

def _bare_feedback_agent(events: list[str]) -> AudioAgent:
    agent = object.__new__(AudioAgent)
    agent._init_output_ownership_state()
    agent._realtime_output_lock = threading.Lock()
    agent._realtime_output_provider_generation = 0
    agent._chime_lock = threading.RLock()
    agent._last_chime_at = 0.0
    agent._last_thinking_chime_at = 0.0
    agent._feedback_generation = 0
    agent._feedback_active = False
    agent._feedback_process = None
    agent._feedback_sounddevice_active = False
    agent._feedback_sounddevice_cancel_event = None
    agent._spoken_wait_prompt_enabled = False
    agent._spoken_wait_prompt_text = ""
    agent._spoken_wait_prompt_cache_key = ""
    agent._spoken_wait_prompt_min_interval_s = 8.0
    agent._last_spoken_wait_prompt_at = 0.0
    agent._processing_feedback_delay_s = 1.5
    agent._processing_feedback_generation = 0
    agent._processing_feedback_armed = False
    agent._processing_feedback_timer = None
    agent._audio_router = None
    agent._refresh_voice_metrics = lambda: None  # type: ignore[method-assign]
    agent._chime_acknowledge = lambda: np.ones(8, dtype=np.float32)  # type: ignore[method-assign]
    agent._chime_wake = agent._chime_acknowledge  # type: ignore[method-assign]
    agent._chime_error = agent._chime_acknowledge  # type: ignore[method-assign]
    agent._chime_thinking = agent._chime_acknowledge  # type: ignore[method-assign]
    agent._chime_ready = agent._chime_acknowledge  # type: ignore[method-assign]
    agent.tts = SimpleNamespace(
        _aplay_bin=None,
        _output_device=None,
        play_feedback_audio=lambda _audio, _rate: events.append("feedback") or True,
        cancel_feedback_audio=lambda: events.append("cancel_feedback"),
        speak=lambda text: events.append(f"speak:{text}"),
    )
    return agent


def _lock_owned(lock: threading.RLock) -> bool:
    owned = getattr(lock, "_is_owned", None)
    return bool(callable(owned) and owned())


def test_ack_does_not_suppress_the_first_thinking_fuse(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)

    agent.acknowledge()
    agent.play_thinking()

    assert events == ["feedback", "feedback"]


def test_repeated_thinking_chime_is_rate_limited(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)

    agent.play_thinking()
    agent.play_thinking()

    assert events == ["feedback"]


def test_processing_feedback_is_armed_from_turn_acceptance(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    agent._processing_feedback_delay_s = 0.65
    _ManualTimer.created.clear()
    monkeypatch.setattr(audio_agent_module.threading, "Timer", _ManualTimer)
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)

    assert agent.arm_processing_feedback() is True
    assert agent.processing_feedback_armed is True
    assert len(_ManualTimer.created) == 1
    timer = _ManualTimer.created[0]
    assert timer.interval == 0.65
    assert timer.started is True
    assert events == []
    armed = agent.processing_feedback_status_snapshot()
    assert armed["armed"] is True
    assert armed["delay_ms"] == 650
    assert armed["armed_total"] == 1
    assert armed["last_transition"] == "armed"

    timer.fire()

    assert events == ["feedback"]
    assert agent.processing_feedback_armed is True
    played = agent.processing_feedback_status_snapshot()
    assert played["triggered_total"] == 1
    assert played["started_total"] == 1
    assert played["overlap_prevented_total"] == 0
    agent.cancel_processing_feedback()
    assert agent.processing_feedback_armed is False
    cancelled = agent.processing_feedback_status_snapshot()
    assert cancelled["cancelled_total"] == 1
    assert cancelled["last_transition"] == "cancelled"


def test_semantic_speech_cancels_pending_processing_feedback(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    _ManualTimer.created.clear()
    monkeypatch.setattr(audio_agent_module.threading, "Timer", _ManualTimer)

    assert agent.arm_processing_feedback() is True
    timer = _ManualTimer.created[0]
    agent.speak("真实回答")
    timer.fire()

    assert timer.cancelled is True
    assert agent.processing_feedback_armed is False
    assert events == ["speak:真实回答"]
    status = agent.processing_feedback_status_snapshot()
    assert status["cancelled_total"] == 1
    assert status["started_total"] == 0



class _CancelledToken:
    def __init__(self) -> None:
        self.ran = False

    def is_set(self) -> bool:
        return False

    def try_run(self, callback):
        self.ran = True
        return False, None


def test_processing_feedback_cancel_token_linearizes_before_playback(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    token = _CancelledToken()
    _ManualTimer.created.clear()
    monkeypatch.setattr(audio_agent_module.threading, "Timer", _ManualTimer)
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)

    assert agent.arm_processing_feedback(token) is True
    _ManualTimer.created[0].fire()

    assert token.ran is True
    assert events == []
    assert agent.processing_feedback_armed is False
    status = agent.processing_feedback_status_snapshot()
    assert status["cancelled_total"] == 1
    assert status["last_transition"] == "turn_cancelled"


def test_processing_feedback_generation_must_still_match_before_transport(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)

    agent._processing_feedback_generation = 2
    agent._processing_feedback_armed = False
    agent._play_chime(
        "waiting_prompt",
        audio=np.ones(4, dtype=np.float32),
        sample_rate=agent._SR,
        expected_processing_generation=1,
    )

    assert events == []
    assert agent._feedback_active is False
    assert agent.processing_feedback_status_snapshot()["suppressed_total"] == 0


def test_noncanonical_waiting_prompt_key_fails_closed(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    agent._spoken_wait_prompt_enabled = True
    agent._spoken_wait_prompt_text = "收到，我来看看。"
    agent._spoken_wait_prompt_cache_key = "other-key"
    lookups = 0

    def cached_phrase_pcm(_text, *, cache_key, target_sample_rate):
        nonlocal lookups
        lookups += 1
        return np.ones(4, dtype=np.float32), target_sample_rate

    agent.tts.cached_phrase_pcm = cached_phrase_pcm
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)

    agent.play_thinking()

    assert events == ["feedback"]
    assert lookups == 0
    assert agent._last_spoken_wait_prompt_at == 0.0


def test_feedback_config_parsing_tolerates_invalid_values(monkeypatch) -> None:
    agent = object.__new__(AudioAgent)

    parse = agent._finite_nonnegative_feedback_seconds

    assert parse("bad", default=1.5) == 1.5
    assert parse(True, default=1.5) == 1.5
    assert parse(float("nan"), default=1.5) == 1.5
    assert parse(float("inf"), default=1.5) == 1.5
    assert parse(-2, default=1.5) == 0.0
    assert parse("0.25", default=1.5) == 0.25



def test_processing_feedback_fails_closed_when_semantic_tts_is_busy(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    agent._processing_feedback_generation = 1
    agent._processing_feedback_armed = True
    agent.tts.is_active = lambda: True
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)

    agent._play_chime("thinking", expected_processing_generation=1)

    assert events == []
    assert agent._feedback_active is False
    status = agent.processing_feedback_status_snapshot()
    assert status["suppressed_total"] == 1
    assert status["overlap_prevented_total"] == 1
    assert status["last_transition"] == "semantic_overlap_prevented"


def test_feedback_transport_handoff_starts_under_chime_lock(monkeypatch) -> None:
    lock_owned: list[bool] = []
    agent = _bare_feedback_agent([])

    def play_feedback_audio(_audio, _rate):
        lock_owned.append(_lock_owned(agent._chime_lock))
        return True

    agent.tts.play_feedback_audio = play_feedback_audio
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)

    agent.play_thinking()

    assert lock_owned == [True]


def test_semantic_speech_queues_under_chime_lock() -> None:
    events: list[str] = []
    lock_owned: list[bool] = []
    agent = _bare_feedback_agent(events)
    agent._feedback_active = True

    def speak(text):
        lock_owned.append(_lock_owned(agent._chime_lock))
        events.append(f"speak:{text}")

    agent.tts.speak = speak

    agent.speak("真实回答")

    assert lock_owned == [True]
    assert events == ["cancel_feedback", "speak:真实回答"]


def test_cached_semantic_phrase_queues_under_chime_lock() -> None:
    events: list[str] = []
    lock_owned: list[bool] = []
    agent = _bare_feedback_agent(events)
    agent._feedback_active = True

    def queue_cached_phrase(_text, *, cache_key):
        lock_owned.append(_lock_owned(agent._chime_lock))
        events.append(f"queue:{cache_key}")
        return True

    agent.tts.queue_cached_phrase = queue_cached_phrase
    owner = object()
    agent.start_playback = (  # type: ignore[method-assign]
        lambda: events.append("start") or owner
    )
    agent.wait_speaking_done = lambda: events.append("wait")  # type: ignore[method-assign]
    agent.stop_playback = (  # type: ignore[method-assign]
        lambda token: events.append("stop") if token is owner else None
    )

    played = asyncio.run(agent.speak_cached_and_wait("好的。", cache_key="quick-okay"))

    assert played is True
    assert lock_owned == [True]
    assert events == ["start", "cancel_feedback", "queue:quick-okay", "wait", "stop"]


def test_cached_semantic_phrase_owner_conflict_never_queues_pcm() -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    agent.start_playback = lambda: None  # type: ignore[method-assign]
    agent.tts.queue_cached_phrase = (
        lambda _text, *, cache_key: events.append(f"queue:{cache_key}") or True
    )

    with pytest.raises(RuntimeError, match="playback owner conflict"):
        asyncio.run(
            agent.speak_cached_and_wait("不应入队。", cache_key="conflicting-owner")
        )

    assert events == []

def test_enabled_thinking_uses_cached_waiting_prompt(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    agent._spoken_wait_prompt_enabled = True
    agent._spoken_wait_prompt_text = "收到，我来看看。"
    agent._spoken_wait_prompt_cache_key = "feedback-waiting"
    lookups: list[tuple[str, str, int]] = []

    def cached_phrase_pcm(text, *, cache_key, target_sample_rate):
        lookups.append((text, cache_key, target_sample_rate))
        return np.ones(4, dtype=np.float32), target_sample_rate

    agent.tts.cached_phrase_pcm = cached_phrase_pcm
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)

    agent.play_thinking()

    assert events == ["feedback"]
    assert lookups == [("收到，我来看看。", "feedback-waiting", agent._SR)]
    assert agent._last_spoken_wait_prompt_at > 0


def test_waiting_prompt_cache_miss_falls_back_to_thinking_chime(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    agent._spoken_wait_prompt_enabled = True
    agent._spoken_wait_prompt_text = "收到，我来看看。"
    agent._spoken_wait_prompt_cache_key = "feedback-waiting"
    agent.tts.cached_phrase_pcm = lambda *_args, **_kwargs: None
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)

    agent.play_thinking()

    assert events == ["feedback"]
    assert agent._last_spoken_wait_prompt_at == 0.0


def test_waiting_prompt_is_not_spoken_twice_within_min_interval(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    agent._spoken_wait_prompt_enabled = True
    agent._spoken_wait_prompt_text = "收到，我来看看。"
    agent._spoken_wait_prompt_cache_key = "feedback-waiting"
    agent._spoken_wait_prompt_min_interval_s = 8.0
    lookups = 0

    def cached_phrase_pcm(_text, *, cache_key, target_sample_rate):
        nonlocal lookups
        lookups += 1
        return np.ones(4, dtype=np.float32), target_sample_rate

    agent.tts.cached_phrase_pcm = cached_phrase_pcm
    now = iter([100.0, 100.0, 105.0, 105.0])
    monkeypatch.setattr(audio_agent_module.time, "monotonic", lambda: next(now))
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)

    agent.play_thinking()
    agent.play_thinking()

    # This prompt is feedback-only; it is never queued as semantic reply text,
    # so it cannot enter the response ledger or memory path from this slice.
    assert events == ["feedback", "feedback"]
    assert lookups == 1
    assert agent._last_spoken_wait_prompt_at == 100.0



def test_cached_semantic_speech_cancels_feedback_before_queueing_pcm() -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    agent._feedback_active = True
    agent.tts.queue_cached_phrase = (
        lambda _text, *, cache_key: events.append(f"queue:{cache_key}") or True
    )
    owner = object()
    agent.start_playback = (  # type: ignore[method-assign]
        lambda: events.append("start") or owner
    )
    agent.wait_speaking_done = lambda: events.append("wait")  # type: ignore[method-assign]
    agent.stop_playback = (  # type: ignore[method-assign]
        lambda token: events.append("stop") if token is owner else None
    )

    played = asyncio.run(
        agent.speak_cached_and_wait("好的。", cache_key="quick-okay")
    )

    assert played is True
    assert events == [
        "start",
        "cancel_feedback",
        "queue:quick-okay",
        "wait",
        "stop",
    ]

def test_semantic_speech_cancels_feedback_once_before_queueing_text() -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    agent._feedback_active = True

    agent.speak("第一句")
    agent.speak("第二句")

    assert events == [
        "cancel_feedback",
        "speak:第一句",
        "speak:第二句",
    ]


def test_cancel_feedback_terminates_fallback_process_without_tts_generation_reset() -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: events.append("terminate"),
    )
    agent._feedback_active = True
    agent._feedback_process = process
    initial_generation = agent._feedback_generation

    agent.cancel_feedback()

    assert agent._feedback_generation == initial_generation + 1
    assert agent._feedback_active is False
    assert events == ["cancel_feedback", "terminate"]


def test_cancel_feedback_is_bounded_and_does_not_stop_global_sounddevice(
    monkeypatch,
) -> None:
    """Feedback cancellation must not enter PortAudio on the semantic path."""

    events: list[str] = []
    agent = _bare_feedback_agent(events)
    feedback_cancelled = threading.Event()
    global_stop_entered = threading.Event()
    release_global_stop = threading.Event()
    cancel_returned = threading.Event()
    agent._feedback_active = True
    agent._feedback_sounddevice_active = True
    agent._feedback_sounddevice_cancel_event = feedback_cancelled

    def blocking_global_stop() -> None:
        global_stop_entered.set()
        release_global_stop.wait(timeout=1.0)

    monkeypatch.setattr(audio_agent_module.sd, "stop", blocking_global_stop)

    worker = threading.Thread(
        target=lambda: (agent.cancel_feedback(), cancel_returned.set()),
        daemon=True,
    )
    worker.start()
    try:
        assert cancel_returned.wait(timeout=0.1)
        assert global_stop_entered.is_set() is False
        assert feedback_cancelled.is_set() is True
    finally:
        release_global_stop.set()
        worker.join(timeout=1.0)
def test_failed_chime_synthesis_does_not_cancel_the_next_semantic_transport() -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)

    def _fail() -> np.ndarray:
        raise RuntimeError("synthesis failed")

    agent._chime_thinking = _fail  # type: ignore[method-assign]

    agent.play_thinking()
    agent.speak("真实回答")

    assert agent._feedback_active is False
    assert events == ["speak:真实回答"]



def test_sounddevice_feedback_uses_a_private_cancellable_stream(monkeypatch) -> None:
    """A feedback cancel signal is consumed only by its own output stream."""

    events: list[str] = []
    agent = _bare_feedback_agent(events)
    agent.tts.play_feedback_audio = lambda _audio, _rate: False
    stream_started = threading.Event()
    stream_closed = threading.Event()
    instances: list[object] = []

    class _FeedbackOutputStream:
        def __init__(
            self,
            *,
            samplerate,
            channels,
            dtype,
            callback,
            finished_callback,
        ) -> None:
            assert samplerate == agent._SR
            assert channels == 1
            assert dtype == "float32"
            self.callback = callback
            self.finished_callback = finished_callback
            instances.append(self)

        def start(self) -> None:
            events.append("stream_start")
            stream_started.set()

        def render_after_cancel(self) -> None:
            outdata = np.full((8, 1), 1.0, dtype=np.float32)
            try:
                self.callback(outdata, len(outdata), None, None)
            except audio_agent_module.sd.CallbackAbort:
                events.append("callback_abort")
                assert np.count_nonzero(outdata) == 0
                self.finished_callback()

        def abort(self, ignore_errors=True) -> None:
            del ignore_errors
            events.append("stream_abort")

        def close(self, ignore_errors=True) -> None:
            del ignore_errors
            events.append("stream_close")
            stream_closed.set()

    monkeypatch.setattr(
        audio_agent_module.sd,
        "OutputStream",
        _FeedbackOutputStream,
    )
    monkeypatch.setattr(
        audio_agent_module.sd,
        "play",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("global sounddevice playback must not be used")
        ),
    )

    agent.play_thinking()
    assert stream_started.wait(timeout=1.0)

    agent.cancel_feedback()
    assert len(instances) == 1
    instances[0].render_after_cancel()  # type: ignore[attr-defined]

    assert stream_closed.wait(timeout=1.0)
    assert events == [
        "stream_start",
        "cancel_feedback",
        "callback_abort",
        "stream_close",
    ]
def test_background_feedback_device_failure_is_contained(monkeypatch) -> None:
    events: list[str] = []
    agent = _bare_feedback_agent(events)
    agent.tts.play_feedback_audio = lambda _audio, _rate: False
    monkeypatch.setattr(audio_agent_module.threading, "Thread", _InlineThread)
    monkeypatch.setattr(
        audio_agent_module.sd,
        "play",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("device lost")),
    )

    agent.play_thinking()

    assert agent._feedback_active is False
