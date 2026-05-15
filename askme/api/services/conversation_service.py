"""Conversation service helpers for chat and voice-turn APIs."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import deque
from collections.abc import Callable
from inspect import Parameter, isawaitable, signature
from typing import Any

from askme.pipeline.rag_policy import forced_rag_reply

ChatHandler = Callable[..., Any]
MemoryHandler = Any


class EmptyChatText(ValueError):
    """Raised when a chat request does not include text."""


class ChatUnavailable(RuntimeError):
    """Raised when the chat pipeline is not configured."""


class ChatTimeout(TimeoutError):
    """Raised when the chat handler exceeds the configured budget."""

    def __init__(self, timeout_s: float) -> None:
        super().__init__(f"chat handler timed out after {timeout_s:.3f}s")
        self.timeout_s = timeout_s


class ChatOverloaded(RuntimeError):
    """Raised when chat requests exceed the configured in-flight limit."""

    def __init__(self, max_concurrency: int) -> None:
        super().__init__(f"chat concurrency limit reached: {max_concurrency}")
        self.max_concurrency = max_concurrency


class ConversationService:
    """Build product-facing chat payloads around runtime handlers."""

    def __init__(
        self,
        *,
        chat_handler: ChatHandler | None,
        memory_handler: MemoryHandler | None = None,
        logger: logging.Logger | None = None,
        chat_timeout_s: float | None = 30.0,
        chat_max_concurrency: int = 8,
        chat_slow_threshold_ms: float | None = 2000.0,
        chat_diagnostics_history_limit: int = 20,
    ) -> None:
        self._chat_handler = chat_handler
        self._memory_handler = memory_handler
        self._logger = logger or logging.getLogger(__name__)
        self._chat_timeout_s = _optional_positive_float(chat_timeout_s)
        self._chat_max_concurrency = _positive_int(chat_max_concurrency, default=8)
        self._chat_slow_threshold_ms = _optional_positive_float(chat_slow_threshold_ms)
        self._chat_history_limit = _positive_int(chat_diagnostics_history_limit, default=20)
        self._state_lock = asyncio.Lock()
        self._chat_in_flight = 0
        self._chat_total_turns = 0
        self._chat_failures = 0
        self._chat_timeouts = 0
        self._chat_overloads = 0
        self._chat_slow_turns_total = 0
        self._last_chat_turn: dict[str, Any] | None = None
        self._recent_chat_turns: deque[dict[str, Any]] = deque(maxlen=self._chat_history_limit)
        self._slow_chat_turns: deque[dict[str, Any]] = deque(maxlen=self._chat_history_limit)

    @property
    def chat_available(self) -> bool:
        return self._chat_handler is not None

    async def chat_payload_from_body(
        self,
        body: dict[str, Any],
        *,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the chat handler and attach voice/RAG context for HTTP responses."""
        started = time.perf_counter()
        timings: dict[str, float] = {}
        turn_trace_id = _clean_trace_id(trace_id) or f"chat-{secrets.token_hex(8)}"
        raw_text = body.get("text") or body.get("message") or body.get("prompt") or ""
        text = str(raw_text).strip()
        if not text:
            raise EmptyChatText("empty text")
        speak = bool(body.get("speak") or body.get("voice") or body.get("play_audio"))
        voice_turn = self.voice_turn_payload_from_body(body, text=text)
        timings["parse_ms"] = _elapsed_ms(started)

        await self._enter_chat_turn()
        status = "ok"
        error_type = ""
        try:
            handler_started = time.perf_counter()
            result = await self._dispatch_chat_handler_with_timeout(text, speak=speak)
            timings["handler_ms"] = _elapsed_ms(handler_started)

            response_started = time.perf_counter()
            payload = self.chat_response_payload(
                result,
                text=text,
                speak=speak,
                voice_turn=voice_turn,
            )
            timings["response_build_ms"] = _elapsed_ms(response_started)

            memory_started = time.perf_counter()
            payload = await self.attach_memory_chat_context(payload)
            timings["memory_context_ms"] = _elapsed_ms(memory_started)
            return payload
        except TimeoutError as exc:
            status = "timeout"
            error_type = exc.__class__.__name__
            timeout_s = self._chat_timeout_s or 0.0
            raise ChatTimeout(timeout_s) from exc
        except Exception as exc:
            status = "error"
            error_type = exc.__class__.__name__
            raise
        finally:
            timings["total_ms"] = _elapsed_ms(started)
            await self._finish_chat_turn(
                status=status,
                timings=timings,
                text_chars=len(text),
                speak=speak,
                is_voice=voice_turn is not None,
                trace_id=turn_trace_id,
                error_type=error_type,
            )

    async def dispatch_chat_handler(self, text: str, *, speak: bool) -> Any:
        handler = self._chat_handler
        if handler is None:
            raise ChatUnavailable("chat not available")
        if _handler_accepts_speak(handler):
            return await _maybe_await(handler(text, speak=speak))
        return await _maybe_await(handler(text))

    async def _dispatch_chat_handler_with_timeout(self, text: str, *, speak: bool) -> Any:
        call = self.dispatch_chat_handler(text, speak=speak)
        if self._chat_timeout_s is None:
            return await call
        return await asyncio.wait_for(call, timeout=self._chat_timeout_s)

    async def _enter_chat_turn(self) -> None:
        async with self._state_lock:
            if self._chat_in_flight >= self._chat_max_concurrency:
                self._chat_overloads += 1
                raise ChatOverloaded(self._chat_max_concurrency)
            self._chat_in_flight += 1

    async def _finish_chat_turn(
        self,
        *,
        status: str,
        timings: dict[str, float],
        text_chars: int,
        speak: bool,
        is_voice: bool,
        trace_id: str,
        error_type: str = "",
    ) -> None:
        async with self._state_lock:
            self._chat_in_flight = max(0, self._chat_in_flight - 1)
            self._chat_total_turns += 1
            if status != "ok":
                self._chat_failures += 1
            if status == "timeout":
                self._chat_timeouts += 1
            total_ms = float(timings.get("total_ms") or 0.0)
            slow = (
                self._chat_slow_threshold_ms is not None
                and total_ms >= self._chat_slow_threshold_ms
            )
            if slow:
                self._chat_slow_turns_total += 1
            turn = {
                "trace_id": trace_id,
                "status": status,
                "timings_ms": dict(timings),
                "slow": slow,
                "text_chars": text_chars,
                "speak": speak,
                "voice": is_voice,
                "error_type": error_type,
                "recorded_at": time.time(),
            }
            self._last_chat_turn = turn
            self._recent_chat_turns.append(turn)
            if slow:
                self._slow_chat_turns.append(turn)

    def diagnostics_snapshot(self) -> dict[str, Any]:
        """Return non-sensitive execution diagnostics for operations dashboards."""
        last_turn = dict(self._last_chat_turn) if self._last_chat_turn is not None else None
        return {
            "chat": {
                "configured": self.chat_available,
                "timeout_s": self._chat_timeout_s,
                "max_concurrency": self._chat_max_concurrency,
                "slow_threshold_ms": self._chat_slow_threshold_ms,
                "diagnostics_history_limit": self._chat_history_limit,
                "in_flight": self._chat_in_flight,
                "total_turns": self._chat_total_turns,
                "failures": self._chat_failures,
                "timeouts": self._chat_timeouts,
                "overloads": self._chat_overloads,
                "slow_turns_total": self._chat_slow_turns_total,
                "last_turn": last_turn,
                "recent_turns": [dict(item) for item in self._recent_chat_turns],
                "slow_turns": [dict(item) for item in self._slow_chat_turns],
            }
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return chat counters/gauges suitable for health and Prometheus payloads."""
        last_turn = self._last_chat_turn or {}
        timings = last_turn.get("timings_ms") if isinstance(last_turn, dict) else {}
        if not isinstance(timings, dict):
            timings = {}
        return {
            "chat": {
                "configured": self.chat_available,
                "in_flight": self._chat_in_flight,
                "total_turns": self._chat_total_turns,
                "failures": self._chat_failures,
                "timeouts": self._chat_timeouts,
                "overloads": self._chat_overloads,
                "slow_turns_total": self._chat_slow_turns_total,
                "last_turn_latency_ms": timings.get("total_ms"),
                "last_handler_ms": timings.get("handler_ms"),
                "last_memory_context_ms": timings.get("memory_context_ms"),
            }
        }

    def voice_turn_payload_from_body(
        self,
        body: dict[str, Any],
        *,
        text: str,
        channel: str = "voice",
    ) -> dict[str, Any] | None:
        is_voice = bool(
            body.get("voice")
            or body.get("transcript_id")
            or body.get("asr_confidence") is not None
        )
        if not is_voice:
            return None
        payload: dict[str, Any] = {
            "transcript_id": str(body.get("transcript_id") or f"voice-turn-{secrets.token_hex(6)}"),
            "recognized_text": text,
            "is_final": bool(body.get("is_final", True)),
            "channel": str(body.get("channel") or channel or "voice"),
            "safety_bypass_allowed": False,
            "created_at": time.time(),
        }
        confidence = body.get("asr_confidence", body.get("confidence"))
        if confidence is not None:
            try:
                payload["confidence"] = min(max(float(confidence), 0.0), 1.0)
            except (TypeError, ValueError):
                payload["confidence"] = 0.0
        return payload

    def chat_response_payload(
        self,
        result: Any,
        *,
        text: str,
        speak: bool,
        voice_turn: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(result, dict):
            payload = dict(result)
            payload.setdefault("text", text)
            payload.setdefault("reply", "")
        else:
            payload = {"reply": result, "text": text}
        payload.setdefault("evidence", [])
        if speak:
            payload.setdefault("spoken", False)
        if voice_turn is not None:
            payload.setdefault("voice_turn", voice_turn)
        return payload

    async def attach_memory_chat_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Attach latest RAG evidence/policy when chat handler returned plain text."""
        if self._memory_handler is None:
            return payload
        health_method = getattr(self._memory_handler, "health", None)
        if not callable(health_method):
            return payload
        try:
            health = await _maybe_await(health_method())
        except Exception as exc:
            self._logger.debug("memory health unavailable for chat evidence: %s", exc)
            return payload
        if not isinstance(health, dict):
            return payload

        evidence = health.get("last_evidence")
        dropped = health.get("last_dropped_evidence")
        answer_policy = health.get("last_answer_policy")

        if not payload.get("evidence") and isinstance(evidence, list):
            payload["evidence"] = evidence

        rag_payload = payload.get("rag")
        if not isinstance(rag_payload, dict):
            rag_payload = {}
            payload["rag"] = rag_payload
        rag_payload.setdefault("enabled", health.get("enabled", False))
        rag_payload.setdefault("backend", health.get("backend", ""))
        rag_payload.setdefault("configured_backend", health.get("configured_backend", ""))
        rag_payload.setdefault("backend_selection", health.get("backend_selection", {}))
        rag_payload.setdefault("available", health.get("available", False))
        rag_payload.setdefault("last_backend", health.get("last_backend", ""))
        rag_payload.setdefault("last_retrieve_ms", health.get("last_retrieve_ms"))
        rag_payload.setdefault("last_retrieved_items", health.get("last_retrieved_items", 0))
        if isinstance(dropped, list):
            rag_payload.setdefault("dropped_evidence", dropped)
        if isinstance(answer_policy, dict):
            rag_payload.setdefault("answer_policy", answer_policy)
            forced_reply = forced_rag_reply(answer_policy)
            if forced_reply and not payload.get("evidence"):
                payload["reply"] = forced_reply
                payload["rag_blocked"] = True
                rag_payload["answer_blocked"] = True
                rag_payload["forced_reply"] = True
                rag_payload["block_reason"] = answer_policy.get("reason", "")
        return payload


def _handler_accepts_speak(handler: ChatHandler) -> bool:
    try:
        params = signature(handler).parameters
    except (TypeError, ValueError):
        return True
    return "speak" in params or any(
        param.kind == Parameter.VAR_KEYWORD for param in params.values()
    )


async def _maybe_await(value: Any) -> Any:
    if isawaitable(value):
        return await value
    return value


def _optional_positive_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _clean_trace_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return "".join(char for char in text[:128] if char.isalnum() or char in "-_.:")


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)
