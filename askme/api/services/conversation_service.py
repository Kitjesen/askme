"""Conversation service helpers for chat and voice-turn APIs."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from inspect import Parameter, isawaitable, signature
from typing import Any

from askme.api.services.scenario_intent_payloads import (
    requested_or_runtime_skills,
    scenario_intent_decision_payload,
)
from askme.api.services.space_preview import (
    SpaceDispatch,
    space_resolution_evidence_items,
    space_resolution_preview,
)
from askme.conversation import canonical_thread_id
from askme.pipeline.core.rag_policy import forced_rag_reply
from askme.robot_interaction.scenario_intents import classify_scenario_intent

ChatHandler = Callable[..., Any]
MemoryHandler = Any
CapabilitiesProvider = Callable[[], dict[str, Any]]


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
        capabilities_provider: CapabilitiesProvider | None = None,
        space_dispatch: SpaceDispatch | None = None,
    ) -> None:
        self._chat_handler = chat_handler
        self._memory_handler = memory_handler
        self._capabilities_provider = capabilities_provider
        self._space_dispatch = space_dispatch
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
        conversation_session_id = canonical_thread_id(
            thread_id=_clean_optional_text(body.get("thread_id")),
            conversation_thread_id=_clean_optional_text(
                body.get("conversation_thread_id")
            ),
            conversation_session_id=_clean_optional_text(
                body.get("conversation_session_id")
            ),
            conversation_id=_clean_optional_text(body.get("conversation_id")),
            chat_session_id=_clean_optional_text(body.get("chat_session_id")),
            session_id=_clean_optional_text(body.get("session_id")),
        ) or f"chat-thread-{secrets.token_hex(16)}"
        normalized_body = dict(body)
        normalized_body["conversation_thread_id"] = conversation_session_id
        voice_turn = self.voice_turn_payload_from_body(normalized_body, text=text)
        timings["parse_ms"] = _elapsed_ms(started)

        await self._enter_chat_turn()
        status = "ok"
        error_type = ""
        try:
            handler_started = time.perf_counter()
            planning_session_id = _clean_optional_text(body.get("planning_session_id"))
            runtime_policy = _clean_runtime_policy(
                body.get("runtime_policy") or body.get("runtime_bridge_mode")
            )

            if self._chat_handler is None:
                result = self._offline_chat_result(text=text, speak=speak)
                captured_turn_rag = None
            else:
                result, captured_turn_rag = await self._dispatch_chat_handler_with_timeout(
                    text,
                    speak=speak,
                    conversation_session_id=conversation_session_id,
                    planning_session_id=planning_session_id,
                    runtime_policy=runtime_policy,
                )
            timings["handler_ms"] = _elapsed_ms(handler_started)

            response_started = time.perf_counter()
            payload = self.chat_response_payload(
                result,
                text=text,
                speak=speak,
                voice_turn=voice_turn,
            )
            payload["conversation_thread_id"] = conversation_session_id
            payload["conversation_session_id"] = conversation_session_id
            payload = self._attach_handler_turn_rag(
                payload,
                turn_rag=captured_turn_rag,
            )
            timings["response_build_ms"] = _elapsed_ms(response_started)

            memory_started = time.perf_counter()
            payload = await self.attach_memory_chat_context(payload)
            timings["memory_context_ms"] = _elapsed_ms(memory_started)

            space_started = time.perf_counter()
            payload = await self.attach_space_chat_context(payload, body=body, text=text)
            timings["space_context_ms"] = _elapsed_ms(space_started)
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

    async def dispatch_chat_handler(
        self,
        text: str,
        *,
        speak: bool,
        conversation_session_id: str | None = None,
        planning_session_id: str | None = None,
        runtime_policy: str = "disabled",
    ) -> Any:
        handler = self._chat_handler
        if handler is None:
            raise ChatUnavailable("chat not available")
        kwargs: dict[str, Any] = {}
        if _handler_accepts_keyword(handler, "speak"):
            kwargs["speak"] = speak
        if conversation_session_id and _handler_accepts_keyword(
            handler,
            "conversation_session_id",
        ):
            kwargs["conversation_session_id"] = conversation_session_id
        if planning_session_id and _handler_accepts_keyword(
            handler,
            "planning_session_id",
        ):
            kwargs["planning_session_id"] = planning_session_id
        if _handler_accepts_keyword(handler, "runtime_policy"):
            kwargs["runtime_policy"] = runtime_policy
        elif _handler_accepts_keyword(handler, "runtime_bridge_mode"):
            kwargs["runtime_bridge_mode"] = runtime_policy
        return await _maybe_await(handler(text, **kwargs))

    async def _dispatch_chat_handler_with_timeout(
        self,
        text: str,
        *,
        speak: bool,
        conversation_session_id: str | None = None,
        planning_session_id: str | None = None,
        runtime_policy: str = "disabled",
    ) -> tuple[Any, dict[str, Any] | None]:
        async def dispatch_and_capture() -> tuple[Any, dict[str, Any] | None]:
            result = await self.dispatch_chat_handler(
                text,
                speak=speak,
                conversation_session_id=conversation_session_id,
                planning_session_id=planning_session_id,
                runtime_policy=runtime_policy,
            )
            # asyncio.wait_for runs this coroutine in a child Task. ContextVar
            # writes made by TurnExecutor do not flow back to the parent Task,
            # so capture the turn-scoped evidence before leaving this context.
            return result, self._handler_turn_rag_snapshot()

        call = dispatch_and_capture()
        if self._chat_timeout_s is None:
            return await call
        return await asyncio.wait_for(call, timeout=self._chat_timeout_s)

    def _offline_chat_result(self, *, text: str, speak: bool) -> dict[str, Any]:
        """Return a product-safe response when dashboard runs without a brain loop."""

        return {
            "reply": "对话大脑还没有接入当前服务。已接入的园区问路、知识证据和现场事件仍会继续检查；如果问题没有依据，系统不会编造答案。",
            "text": text,
            "spoken": False,
            "evidence": [],
            "degraded": True,
            "reply_source": "dashboard_offline_fallback",
            "chat_backend": {
                "configured": False,
                "mode": "dashboard_only",
                "speak_requested": bool(speak),
            },
            "answer_policy": {
                "state": "brain_not_connected",
                "action": "answer_only_with_deterministic_context",
                "reason": "chat_handler_not_configured",
            },
        }

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
        conversation_session_id = canonical_thread_id(
            thread_id=_clean_optional_text(body.get("thread_id")),
            conversation_thread_id=_clean_optional_text(
                body.get("conversation_thread_id")
            ),
            conversation_session_id=_clean_optional_text(
                body.get("conversation_session_id")
            ),
            conversation_id=_clean_optional_text(body.get("conversation_id")),
            chat_session_id=_clean_optional_text(body.get("chat_session_id")),
            session_id=_clean_optional_text(body.get("session_id")),
        )
        planning_session_id = _clean_optional_text(body.get("planning_session_id"))
        if conversation_session_id:
            payload["conversation_thread_id"] = conversation_session_id
            payload["conversation_session_id"] = conversation_session_id
        if planning_session_id:
            payload["planning_session_id"] = planning_session_id
        confidence = body.get("asr_confidence", body.get("confidence"))
        if confidence is not None:
            try:
                payload["confidence"] = min(max(float(confidence), 0.0), 1.0)
            except (TypeError, ValueError):
                payload["confidence"] = 0.0
        return payload

    async def attach_space_chat_context(
        self,
        payload: dict[str, Any],
        *,
        body: dict[str, Any],
        text: str,
    ) -> dict[str, Any]:
        """Attach read-only park-space resolution evidence for customer Q&A."""

        if self._space_dispatch is None:
            return payload
        try:
            available_skills = requested_or_runtime_skills(body, self._capabilities_provider)
            decision = classify_scenario_intent(text, available_skills=available_skills)
            space_resolution = await space_resolution_preview(
                text=text,
                body=body,
                decision=decision,
                space_dispatch=self._space_dispatch,
            )
        except Exception as exc:
            self._logger.debug("space preview unavailable for chat evidence: %s", exc)
            return payload

        if decision is None and space_resolution is None:
            return payload

        preview_payload = {
            "matched": decision is not None,
            "decision": scenario_intent_decision_payload(decision),
            "space_resolution": space_resolution,
            "policy": {
                "preview_only": True,
                "does_not_execute_skill": True,
                "does_not_start_guide": True,
                "safe_for_customer_acceptance_testing": True,
            },
        }
        payload.setdefault("scenario_preview", preview_payload)
        if space_resolution is not None:
            payload.setdefault("space_resolution", space_resolution)

        if payload.get("rag_blocked"):
            return payload

        evidence_items = space_resolution_evidence_items(space_resolution)
        if evidence_items:
            existing = payload.get("evidence")
            if not isinstance(existing, list):
                existing = []
                payload["evidence"] = existing
            _append_unique_evidence(existing, evidence_items)

            resolution = space_resolution.get("resolution") if isinstance(space_resolution, dict) else None
            reply = resolution.get("reply") if isinstance(resolution, dict) else ""
            if reply:
                payload["reply"] = reply
                payload["reply_source"] = "space_cognition"
                payload["space_answered"] = True
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
        existing_rag = payload.get("rag")
        if isinstance(existing_rag, dict) and existing_rag.get("turn_scoped") is True:
            return payload
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

    def _handler_turn_rag_snapshot(self) -> dict[str, Any] | None:
        handler_owner = getattr(self._chat_handler, "__self__", None)
        turn_rag = getattr(handler_owner, "current_turn_rag", None)
        if callable(turn_rag):
            turn_rag = turn_rag()
        if not isinstance(turn_rag, dict):
            return None
        return deepcopy(turn_rag)

    def _attach_handler_turn_rag(
        self,
        payload: dict[str, Any],
        *,
        turn_rag: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(turn_rag, dict):
            return payload
        rag = turn_rag.get("rag")
        if not isinstance(rag, dict) or rag.get("turn_scoped") is not True:
            return payload
        evidence = turn_rag.get("evidence")
        payload["evidence"] = (
            [dict(item) for item in evidence if isinstance(item, dict)]
            if isinstance(evidence, list)
            else []
        )
        payload["rag"] = dict(rag)
        return payload


def _handler_accepts_keyword(handler: ChatHandler, keyword: str) -> bool:
    try:
        params = signature(handler).parameters
    except (TypeError, ValueError):
        return True
    return keyword in params or any(
        param.kind == Parameter.VAR_KEYWORD for param in params.values()
    )


def _clean_optional_text(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _clean_runtime_policy(value: Any) -> str:
    policy = str(value or "disabled").strip().lower().replace("-", "_")
    if policy in {"runtime_first", "first", "bridge_first"}:
        return "runtime_first"
    if policy in {"control_only", "controls_only"}:
        return "control_only"
    return "disabled"


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


def _append_unique_evidence(
    existing: list[Any],
    new_items: list[dict[str, Any]],
) -> None:
    seen = {
        (
            str(item.get("source") or ""),
            str(item.get("record_id") or item.get("source_record_id") or ""),
        )
        for item in existing
        if isinstance(item, dict)
    }
    for item in new_items:
        key = (
            str(item.get("source") or ""),
            str(item.get("record_id") or item.get("source_record_id") or ""),
        )
        if key in seen:
            continue
        existing.append(item)
        seen.add(key)


def _clean_trace_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return "".join(char for char in text[:128] if char.isalnum() or char in "-_.:")


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)
