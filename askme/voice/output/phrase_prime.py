"""Safe startup priming for deterministic, actually consumed voice phrases."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from askme.robot_interaction.routing.fast_voice_intents import default_cached_phrases
from askme.voice.output.tts import TTSEngine


@dataclass(frozen=True, slots=True)
class PhrasePrimeEntry:
    """A deterministic phrase and the cache key used by its runtime consumer."""

    cache_key: str
    text: str


class _PhrasePrimeEngine(Protocol):
    def prime_cached_phrase(self, text: str, *, cache_key: str) -> dict[str, Any]: ...

    def drain_buffers(self) -> None: ...

    def shutdown(self) -> None: ...


WAITING_FEEDBACK_CACHE_KEY = "feedback-waiting"


def configured_feedback_phrases(
    voice_config: Mapping[str, Any],
) -> dict[str, str]:
    """Return the one product-approved transient speech phrase, if enabled."""

    feedback = voice_config.get("feedback", {})
    if not isinstance(feedback, Mapping):
        return {}
    if not bool(feedback.get("spoken_wait_prompt_enabled", False)):
        return {}
    cache_key = str(feedback.get("cache_key") or "").strip()
    text = str(feedback.get("text") or "").strip()
    if cache_key != WAITING_FEEDBACK_CACHE_KEY or not text:
        return {}
    return {cache_key: text}

def resolve_phrase_prime_entries(
    configured: object,
    *,
    quick_replies: Mapping[str, str],
    feedback_phrases: Mapping[str, str] | None = None,
) -> list[PhrasePrimeEntry]:
    """Resolve configuration only against cache keys used by safe consumers.

    Arbitrary text is intentionally rejected: priming a phrase that has no
    stable runtime ``cache_key`` wastes startup work and can accidentally turn
    a future navigation promise into pre-generated product behaviour.
    """

    if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
        return []

    canonical = {
        key: text
        for key, text in default_cached_phrases(quick_replies).items()
        if key.startswith(("quick-", "location-"))
    }
    if feedback_phrases:
        canonical.update(
            {
                key: text
                for key, text in feedback_phrases.items()
                if key == WAITING_FEEDBACK_CACHE_KEY and text
            }
        )
    by_text: dict[str, list[str]] = {}
    for key, text in canonical.items():
        by_text.setdefault(text, []).append(key)

    resolved: list[PhrasePrimeEntry] = []
    seen: set[tuple[str, str]] = set()
    for item in configured:
        candidates: list[tuple[str, str]] = []
        if isinstance(item, str):
            value = item.strip()
            if value in canonical:
                candidates.append((value, canonical[value]))
            else:
                candidates.extend((key, value) for key in by_text.get(value, ()))
        elif isinstance(item, Mapping):
            key = str(item.get("cache_key") or "").strip()
            text = str(item.get("text") or "").strip()
            if key in canonical and (not text or canonical[key] == text):
                candidates.append((key, canonical[key]))
            elif text:
                candidates.extend((candidate, text) for candidate in by_text.get(text, ()))

        for key, text in candidates:
            identity = (key, text)
            if identity in seen:
                continue
            seen.add(identity)
            resolved.append(PhrasePrimeEntry(cache_key=key, text=text))
    return resolved


def prime_phrase_cache(
    tts_config: Mapping[str, Any],
    entries: Sequence[PhrasePrimeEntry],
    *,
    stop_event: threading.Event,
    engine_factory: Callable[[dict[str, Any]], _PhrasePrimeEngine] = TTSEngine,
) -> list[dict[str, Any]]:
    """Prime phrases with an isolated engine that never touches live playback."""

    if stop_event.is_set():
        return []
    isolated_config = deepcopy(dict(tts_config))
    isolated_config["usb_direct_background_prewarm"] = False
    engine = engine_factory(isolated_config)
    results: list[dict[str, Any]] = []
    finished = threading.Event()

    def _interrupt_on_stop() -> None:
        while not finished.wait(0.01):
            if not stop_event.is_set():
                continue
            drain_buffers = getattr(engine, "drain_buffers", None)
            if callable(drain_buffers):
                drain_buffers()
            return

    interrupt_watcher = threading.Thread(
        target=_interrupt_on_stop,
        name="voice-phrase-prime-cancel",
        daemon=True,
    )
    interrupt_watcher.start()
    try:
        for entry in entries:
            if stop_event.is_set():
                break
            results.append(
                engine.prime_cached_phrase(entry.text, cache_key=entry.cache_key)
            )
    finally:
        finished.set()
        interrupt_watcher.join(timeout=0.1)
        engine.shutdown()
    return results


__all__ = [
    "PhrasePrimeEntry",
    "WAITING_FEEDBACK_CACHE_KEY",
    "configured_feedback_phrases",
    "prime_phrase_cache",
    "resolve_phrase_prime_entries",
]
