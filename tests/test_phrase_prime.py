from __future__ import annotations

import threading
from typing import Any

from askme.robot_interaction.routing_policy import DEFAULT_QUICK_REPLIES
from askme.voice.output.phrase_prime import (
    PhrasePrimeEntry,
    configured_feedback_phrases,
    prime_phrase_cache,
    resolve_phrase_prime_entries,
)


def test_phrase_prime_list_accepts_only_canonical_consumed_phrases() -> None:
    entries = resolve_phrase_prime_entries(
        [
            "好的。",
            "正在读取当前位置，请稍候。",
            "好的，请跟我来",
            {"cache_key": "invented", "text": "请稍等"},
        ],
        quick_replies=DEFAULT_QUICK_REPLIES,
    )

    assert [entry.text for entry in entries] == [
        "好的。",
        "正在读取当前位置，请稍候。",
    ]
    assert all(entry.cache_key.startswith(("quick-", "location-")) for entry in entries)


def test_phrase_prime_accepts_explicit_feedback_mapping_only() -> None:
    entries = resolve_phrase_prime_entries(
        [
            {"cache_key": "feedback-waiting", "text": "收到，我来看看。"},
            {"cache_key": "feedback-other", "text": "随便缓存一句。"},
            {"cache_key": "invented", "text": "收到，我来看看。"},
        ],
        quick_replies=DEFAULT_QUICK_REPLIES,
        feedback_phrases={"feedback-waiting": "收到，我来看看。"},
    )

    assert entries == [
        PhrasePrimeEntry(cache_key="feedback-waiting", text="收到，我来看看。")
    ]


def test_configured_feedback_phrases_accepts_only_enabled_waiting_key() -> None:
    enabled = {
        "feedback": {
            "spoken_wait_prompt_enabled": True,
            "cache_key": "feedback-waiting",
            "text": "收到，我来看看。",
        }
    }

    assert configured_feedback_phrases(enabled) == {
        "feedback-waiting": "收到，我来看看。"
    }
    assert configured_feedback_phrases(
        {"feedback": {**enabled["feedback"], "spoken_wait_prompt_enabled": False}}
    ) == {}
    assert configured_feedback_phrases(
        {"feedback": {**enabled["feedback"], "cache_key": "feedback-other"}}
    ) == {}

def test_phrase_prime_uses_an_isolated_engine_and_always_shuts_it_down() -> None:
    created: list[_FakeEngine] = []

    def factory(config: dict[str, Any]) -> _FakeEngine:
        engine = _FakeEngine(config)
        created.append(engine)
        return engine

    results = prime_phrase_cache(
        {"backend": "edge", "phrase_cache_enabled": True},
        [PhrasePrimeEntry(cache_key="quick-stable", text="好的。")],
        stop_event=threading.Event(),
        engine_factory=factory,
    )

    assert len(created) == 1
    assert created[0].calls == [("好的。", "quick-stable")]
    assert created[0].shutdown_called is True
    assert results == [
        {
            "cached": True,
            "created": True,
            "cache_key": "quick-stable",
        }
    ]


def test_phrase_prime_stop_event_skips_remaining_phrases() -> None:
    stop_event = threading.Event()

    class StopAfterFirst(_FakeEngine):
        def prime_cached_phrase(self, text: str, *, cache_key: str) -> dict[str, Any]:
            result = super().prime_cached_phrase(text, cache_key=cache_key)
            stop_event.set()
            return result

    engine = StopAfterFirst({})
    results = prime_phrase_cache(
        {},
        [
            PhrasePrimeEntry(cache_key="one", text="一"),
            PhrasePrimeEntry(cache_key="two", text="二"),
        ],
        stop_event=stop_event,
        engine_factory=lambda _config: engine,
    )

    assert len(results) == 1
    assert engine.calls == [("一", "one")]
    assert engine.shutdown_called is True


def test_phrase_prime_stop_interrupts_inflight_provider_work() -> None:
    stop_event = threading.Event()
    started = threading.Event()
    interrupted = threading.Event()

    class BlockingEngine(_FakeEngine):
        def prime_cached_phrase(self, text: str, *, cache_key: str) -> dict[str, Any]:
            started.set()
            assert interrupted.wait(timeout=1.0)
            return super().prime_cached_phrase(text, cache_key=cache_key)

        def drain_buffers(self) -> None:
            interrupted.set()

    engine = BlockingEngine({})
    result: list[list[dict[str, Any]]] = []
    thread = threading.Thread(
        target=lambda: result.append(
            prime_phrase_cache(
                {},
                [PhrasePrimeEntry(cache_key="one", text="一")],
                stop_event=stop_event,
                engine_factory=lambda _config: engine,
            )
        )
    )
    thread.start()
    assert started.wait(timeout=1.0)

    stop_event.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert interrupted.is_set()
    assert engine.shutdown_called is True
    assert len(result) == 1


class _FakeEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.calls: list[tuple[str, str]] = []
        self.shutdown_called = False

    def prime_cached_phrase(self, text: str, *, cache_key: str) -> dict[str, Any]:
        self.calls.append((text, cache_key))
        return {"cached": True, "created": True, "cache_key": cache_key}

    def drain_buffers(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_called = True
