from __future__ import annotations

import queue
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

import askme.voice.orchestration.audio_agent as audio_agent_module
from askme.voice.orchestration.audio_agent import AudioAgent


class _FakeTTSEngine:
    created: list[_FakeTTSEngine] = []

    def __init__(self, config: dict[str, Any], *, audio_router: Any = None) -> None:
        del audio_router
        self.backend = str(config.get("backend", "fake"))
        self.tts_text_queue: queue.Queue[str] = queue.Queue()
        self._is_playing = bool(config.get("active", False))
        self.prewarm_started = config.get("prewarm_started", threading.Event())
        self.prewarm_release = config.get("prewarm_release", threading.Event())
        self.prewarm_stopping = config.get("prewarm_stopping", threading.Event())
        self.prewarm_finish_release = config.get("prewarm_finish_release")
        self.prewarm_finished = config.get("prewarm_finished", threading.Event())
        self.prewarm_error = config.get("prewarm_error")
        self.cancel_releases_prewarm = bool(config.get("cancel_releases_prewarm", True))
        self.prewarm_calls = 0
        self.cancel_calls = 0
        self.shutdown_calls = 0
        construct_started = config.get("construct_started")
        construct_release = config.get("construct_release")
        if construct_started is not None:
            construct_started.set()
        if construct_release is not None:
            construct_release.wait(timeout=2.0)
        self.created.append(self)

    def is_active(self) -> bool:
        return self._is_playing

    def prewarm_provider_session(self) -> dict[str, object]:
        self.prewarm_calls += 1
        self.prewarm_started.set()
        if self.prewarm_error is not None:
            raise self.prewarm_error
        self.prewarm_release.wait(timeout=2.0)
        self.prewarm_stopping.set()
        if self.prewarm_finish_release is not None:
            self.prewarm_finish_release.wait(timeout=2.0)
        self.prewarm_finished.set()
        return {"ok": True, "status": "opened"}

    def cancel_provider_prewarm(self) -> None:
        self.cancel_calls += 1
        if self.cancel_releases_prewarm:
            self.prewarm_release.set()

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.cancel_provider_prewarm()
        self._is_playing = False

    def status_snapshot(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "is_playing": self._is_playing,
            "queue_size": self.tts_text_queue.qsize(),
        }


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> AudioAgent:
    _FakeTTSEngine.created.clear()
    monkeypatch.setattr(audio_agent_module, "TTSEngine", _FakeTTSEngine)
    metrics = MagicMock()
    instance = AudioAgent(
        {"voice": {"tts": {"backend": "initial"}}},
        voice_mode=False,
        metrics=metrics,
    )
    yield instance
    instance.shutdown()


def test_immediate_tts_switch_notifies_runtime_owner_without_local_prewarm(
    agent: AudioAgent,
) -> None:
    activated: list[_FakeTTSEngine] = []
    agent.set_tts_activation_callback(activated.append)

    result = agent.reconfigure_tts({"backend": "next"})
    next_tts = agent.tts

    assert result["state"] == "active"
    assert next_tts.backend == "next"
    assert next_tts.prewarm_calls == 0
    assert activated == [next_tts]
    assert not hasattr(agent, "_tts_provider_prewarm_threads")
    assert agent.status_snapshot()["output_ready"] is True


def test_pending_tts_switch_notifies_only_after_new_engine_becomes_active(
    agent: AudioAgent,
) -> None:
    current = _FakeTTSEngine.created[0]
    current._is_playing = True
    activated: list[_FakeTTSEngine] = []
    agent.set_tts_activation_callback(activated.append)

    pending = agent.reconfigure_tts({"backend": "pending"})

    assert pending["state"] == "pending"
    assert activated == []
    assert len(_FakeTTSEngine.created) == 1

    current._is_playing = False
    agent._apply_pending_runtime_updates()
    next_tts = agent.tts

    assert next_tts.backend == "pending"
    assert next_tts.prewarm_calls == 0
    assert activated == [next_tts]
    assert current.shutdown_calls == 1


def test_tts_activation_callback_failure_does_not_rollback_switch(
    agent: AudioAgent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger=audio_agent_module.__name__)

    def fail_callback(_tts: _FakeTTSEngine) -> None:
        raise RuntimeError("refresh callback failed")

    agent.set_tts_activation_callback(fail_callback)
    result = agent.reconfigure_tts({"backend": "next"})

    assert result["state"] == "active"
    assert agent.tts.backend == "next"
    assert agent.tts.prewarm_calls == 0
    assert "TTS activation callback failed: refresh callback failed" in caplog.text


def test_shutdown_racing_tts_construction_closes_and_rejects_new_engine(
    agent: AudioAgent,
) -> None:
    current = agent.tts
    construct_started = threading.Event()
    construct_release = threading.Event()
    errors: list[BaseException] = []

    def reconfigure() -> None:
        try:
            agent.reconfigure_tts(
                {
                    "backend": "constructed-after-shutdown",
                    "construct_started": construct_started,
                    "construct_release": construct_release,
                }
            )
        except BaseException as exc:
            errors.append(exc)

    caller = threading.Thread(target=reconfigure, daemon=True)
    caller.start()
    try:
        assert construct_started.wait(timeout=1.0)
        agent.shutdown()
        construct_release.set()
        caller.join(timeout=1.0)

        rejected = _FakeTTSEngine.created[-1]
        assert caller.is_alive() is False
        assert agent.tts is current
        assert rejected is not current
        assert rejected.shutdown_calls == 1
        assert rejected.prewarm_calls == 0
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
    finally:
        construct_release.set()
        caller.join(timeout=1.0)


def test_tts_reconfigure_after_shutdown_is_rejected_without_pending_state(
    agent: AudioAgent,
) -> None:
    agent.shutdown()
    agent.tts._is_playing = True

    with pytest.raises(RuntimeError, match="rejected after AudioAgent shutdown"):
        agent.reconfigure_tts({"backend": "too-late"})

    assert agent._pending_tts_config is None
    assert len(_FakeTTSEngine.created) == 1
