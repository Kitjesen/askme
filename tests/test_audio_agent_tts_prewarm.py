from __future__ import annotations

import queue
import threading
import time
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


def test_immediate_tts_switch_prewarm_does_not_block_reconfigure_or_readiness(
    agent: AudioAgent,
) -> None:
    prewarm_started = threading.Event()
    prewarm_release = threading.Event()
    reconfigure_done = threading.Event()
    result: dict[str, object] = {}

    def reconfigure() -> None:
        result.update(
            agent.reconfigure_tts(
                {
                    "backend": "next",
                    "prewarm_started": prewarm_started,
                    "prewarm_release": prewarm_release,
                }
            )
        )
        reconfigure_done.set()

    caller = threading.Thread(target=reconfigure, daemon=True)
    caller.start()
    try:
        assert prewarm_started.wait(timeout=1.0)
        assert reconfigure_done.wait(timeout=0.2)
        assert prewarm_release.is_set() is False
        assert result["state"] == "active"
        assert agent.status_snapshot()["output_ready"] is True
    finally:
        prewarm_release.set()
        caller.join(timeout=1.0)


def test_pending_tts_switch_starts_prewarm_only_after_it_becomes_active(
    agent: AudioAgent,
) -> None:
    current = _FakeTTSEngine.created[0]
    current._is_playing = True
    prewarm_started = threading.Event()
    prewarm_release = threading.Event()
    apply_done = threading.Event()

    pending = agent.reconfigure_tts(
        {
            "backend": "pending",
            "prewarm_started": prewarm_started,
            "prewarm_release": prewarm_release,
        }
    )

    assert pending["state"] == "pending"
    assert len(_FakeTTSEngine.created) == 1
    assert prewarm_started.is_set() is False

    current._is_playing = False
    caller = threading.Thread(
        target=lambda: (agent._apply_pending_runtime_updates(), apply_done.set()),
        daemon=True,
    )
    caller.start()
    try:
        assert prewarm_started.wait(timeout=1.0)
        assert apply_done.wait(timeout=0.2)
        assert prewarm_release.is_set() is False
        assert agent.tts.backend == "pending"
        assert current.shutdown_calls == 1
    finally:
        prewarm_release.set()
        caller.join(timeout=1.0)


def test_shutdown_cancels_and_harvests_active_tts_prewarm(agent: AudioAgent) -> None:
    prewarm_started = threading.Event()
    prewarm_release = threading.Event()
    prewarm_stopping = threading.Event()
    prewarm_finish_release = threading.Event()
    prewarm_finished = threading.Event()
    shutdown_done = threading.Event()

    agent.reconfigure_tts(
        {
            "backend": "next",
            "prewarm_started": prewarm_started,
            "prewarm_release": prewarm_release,
            "prewarm_stopping": prewarm_stopping,
            "prewarm_finish_release": prewarm_finish_release,
            "prewarm_finished": prewarm_finished,
        }
    )
    assert prewarm_started.wait(timeout=1.0)
    tracked_threads = tuple(agent._tts_provider_prewarm_threads)
    assert tracked_threads
    assert all(thread.daemon for thread in tracked_threads)

    caller = threading.Thread(
        target=lambda: (agent.shutdown(), shutdown_done.set()),
        daemon=True,
    )
    caller.start()
    try:
        assert prewarm_stopping.wait(timeout=1.0)
        assert shutdown_done.wait(timeout=0.1) is False
        prewarm_finish_release.set()
        assert shutdown_done.wait(timeout=1.0)
        assert prewarm_finished.is_set()
        assert all(not thread.is_alive() for thread in tracked_threads)
    finally:
        prewarm_finish_release.set()
        caller.join(timeout=1.0)


def test_rapid_tts_switch_skips_prewarm_for_engine_replaced_before_worker_runs(
    agent: AudioAgent,
) -> None:
    final_started = threading.Event()
    final_release = threading.Event()

    with agent._runtime_switch_lock:
        agent.reconfigure_tts({"backend": "superseded"})
        superseded = _FakeTTSEngine.created[-1]
        agent.reconfigure_tts(
            {
                "backend": "final",
                "prewarm_started": final_started,
                "prewarm_release": final_release,
            }
        )

    try:
        assert final_started.wait(timeout=1.0)
        assert superseded.prewarm_calls == 0
        assert superseded.cancel_calls >= 1
        assert superseded.shutdown_calls == 1
        assert agent.tts.backend == "final"
    finally:
        final_release.set()


def test_tts_switch_cancels_inflight_prewarm_on_replaced_engine(agent: AudioAgent) -> None:
    replaced_started = threading.Event()
    replaced_finished = threading.Event()
    final_started = threading.Event()
    final_release = threading.Event()

    agent.reconfigure_tts(
        {
            "backend": "inflight",
            "prewarm_started": replaced_started,
            "prewarm_finished": replaced_finished,
        }
    )
    replaced = agent.tts
    assert replaced_started.wait(timeout=1.0)

    agent.reconfigure_tts(
        {
            "backend": "final",
            "prewarm_started": final_started,
            "prewarm_release": final_release,
        }
    )
    try:
        assert replaced_finished.wait(timeout=1.0)
        assert final_started.wait(timeout=1.0)
        assert replaced.cancel_calls >= 1
        assert replaced.shutdown_calls == 1
        assert agent.tts.backend == "final"
    finally:
        final_release.set()


def test_tts_prewarm_exception_is_logged_without_affecting_active_engine(
    agent: AudioAgent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger=audio_agent_module.__name__)

    result = agent.reconfigure_tts(
        {
            "backend": "failing-prewarm",
            "prewarm_error": RuntimeError("prewarm boom"),
        }
    )

    deadline = time.monotonic() + 1.0
    while agent._tts_provider_prewarm_threads and time.monotonic() < deadline:
        time.sleep(0.01)

    assert result["state"] == "active"
    assert agent.tts.backend == "failing-prewarm"
    assert agent.status_snapshot()["output_ready"] is True
    assert not agent._tts_provider_prewarm_threads
    assert "TTS provider prewarm failed: prewarm boom" in caplog.text


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


def test_shutdown_timeout_detaches_uncooperative_prewarm_as_daemon(
    agent: AudioAgent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger=audio_agent_module.__name__)
    prewarm_started = threading.Event()
    prewarm_release = threading.Event()
    prewarm_finished = threading.Event()

    agent.reconfigure_tts(
        {
            "backend": "uncooperative",
            "prewarm_started": prewarm_started,
            "prewarm_release": prewarm_release,
            "prewarm_finished": prewarm_finished,
            "cancel_releases_prewarm": False,
        }
    )
    assert prewarm_started.wait(timeout=1.0)
    tracked_threads = tuple(agent._tts_provider_prewarm_threads)

    started_at = time.monotonic()
    agent.shutdown()
    elapsed = time.monotonic() - started_at
    try:
        assert elapsed < 1.25
        assert tracked_threads
        assert all(thread.daemon for thread in tracked_threads)
        assert any(thread.is_alive() for thread in tracked_threads)
        assert "daemon TTS provider prewarm thread(s) did not stop" in caplog.text
    finally:
        prewarm_release.set()
        assert prewarm_finished.wait(timeout=1.0)
        for thread in tracked_threads:
            thread.join(timeout=1.0)


def test_tts_reconfigure_after_shutdown_is_rejected_without_pending_state(
    agent: AudioAgent,
) -> None:
    agent.shutdown()
    agent.tts._is_playing = True

    with pytest.raises(RuntimeError, match="rejected after AudioAgent shutdown"):
        agent.reconfigure_tts({"backend": "too-late"})

    assert agent._pending_tts_config is None
    assert len(_FakeTTSEngine.created) == 1
