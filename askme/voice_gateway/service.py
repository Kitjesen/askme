"""Unified service facade for voice gateway turn handling."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from inspect import Parameter, signature
from typing import Any, TypeAlias

from askme.conversation import canonical_thread_id
from askme.ports import VoiceTurnBridgePort
from askme.voice_gateway.session import (
    ConversationSessionManager,
    SessionSnapshot,
)

VoiceTurnBridge: TypeAlias = VoiceTurnBridgePort


class VoiceGatewayService:
    """Stable voice-middle-layer API for channel loops.

    The current implementation delegates to ``VoiceRuntimeBridge``. Keeping the
    loop code pointed at this service lets provider selection and runtime
    switching evolve without leaking into voice/text channels.
    """

    def __init__(
        self,
        bridge: VoiceTurnBridge | None = None,
        *,
        session_manager: ConversationSessionManager | None = None,
    ) -> None:
        self._bridge = bridge
        self._session_manager = session_manager or ConversationSessionManager()

    @property
    def bridge(self) -> VoiceTurnBridge | None:
        """Underlying runtime bridge, kept for compatibility diagnostics."""
        return self._bridge

    @property
    def session_manager(self) -> ConversationSessionManager:
        """Process-local context projection/cache, not a turn lifecycle owner."""
        return self._session_manager

    def status_snapshot(self) -> dict[str, Any]:
        if self._bridge is None:
            return {
                "enabled": False,
                "circuit_open": False,
                "consecutive_failures": 0,
                "cooldown_remaining_s": 0.0,
            }
        return self._bridge.status_snapshot()

    def handle_voice_text(
        self,
        text: str,
        *,
        thread_id: str | None = None,
        conversation_thread_id: str | None = None,
        conversation_session_id: str | None = None,
        conversation_id: str | None = None,
        chat_session_id: str | None = None,
        session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: Any | None = None,
        person_id: str | None = None,
        operator_id: str | None = None,
        robot_id: str | None = None,
        site_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        include_session: bool = False,
        defer_recording: bool = False,
    ) -> dict[str, Any] | None:
        return self._handle_turn(
            "voice",
            "handle_voice_text",
            text,
            conversation_session_id=canonical_thread_id(
                thread_id=thread_id,
                conversation_thread_id=conversation_thread_id,
                conversation_session_id=conversation_session_id,
                conversation_id=conversation_id,
                chat_session_id=chat_session_id,
                session_id=session_id,
            ),
            voice_turn_id=voice_turn_id,
            turn_cancel_token=turn_cancel_token,
            person_id=person_id,
            operator_id=operator_id,
            robot_id=robot_id,
            site_id=site_id,
            metadata=metadata,
            include_session=include_session,
            defer_recording=defer_recording,
        )

    def handle_text_input(
        self,
        text: str,
        *,
        thread_id: str | None = None,
        conversation_thread_id: str | None = None,
        conversation_session_id: str | None = None,
        conversation_id: str | None = None,
        chat_session_id: str | None = None,
        session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: Any | None = None,
        person_id: str | None = None,
        operator_id: str | None = None,
        robot_id: str | None = None,
        site_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        include_session: bool = False,
        defer_recording: bool = False,
    ) -> dict[str, Any] | None:
        return self._handle_turn(
            "text",
            "handle_text_input",
            text,
            conversation_session_id=canonical_thread_id(
                thread_id=thread_id,
                conversation_thread_id=conversation_thread_id,
                conversation_session_id=conversation_session_id,
                conversation_id=conversation_id,
                chat_session_id=chat_session_id,
                session_id=session_id,
            ),
            voice_turn_id=voice_turn_id,
            turn_cancel_token=turn_cancel_token,
            person_id=person_id,
            operator_id=operator_id,
            robot_id=robot_id,
            site_id=site_id,
            metadata=metadata,
            include_session=include_session,
            defer_recording=defer_recording,
        )

    def conversation_snapshot(self, session_id: str) -> SessionSnapshot | None:
        return self._session_manager.snapshot(session_id)

    def conversation_context(
        self,
        session_id: str,
        *,
        recent_turn_limit: int = 6,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        """Return compact projected context for prompt/runtime assembly."""

        payload = self._session_manager.context_payload(
            session_id,
            recent_turn_limit=recent_turn_limit,
            max_chars=max_chars,
        )
        if payload:
            payload.setdefault("thread_id", session_id)
            payload.setdefault("conversation_thread_id", session_id)
        return payload

    def record_local_turn(
        self,
        session_id: str,
        *,
        user_text: str,
        assistant_text: str = "",
        channel: str = "voice",
        metadata: dict[str, Any] | None = None,
        intent: str | None = None,
        gate_decision: str | None = None,
        skill_name: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        handoff_id: str | None = None,
    ) -> bool:
        """Project an explicitly settled local turn into the gateway context cache."""

        clean_session_id = _clean_optional(session_id)
        if clean_session_id is None:
            return False
        if self._session_manager.snapshot(clean_session_id) is None:
            self._session_manager.get_or_create(
                channel=channel,
                session_id=clean_session_id,
            )
        turn_metadata = {
            "bridge_handled": False,
            "local_fallback": True,
            **deepcopy(metadata or {}),
        }
        try:
            self._session_manager.append_turn(
                clean_session_id,
                user_text=user_text,
                assistant_text=assistant_text,
                intent=_clean_optional(intent),
                gate_decision=_clean_optional(gate_decision),
                skill_name=_clean_optional(skill_name),
                tool_calls=[dict(item) for item in (tool_calls or []) if isinstance(item, dict)],
                handoff_id=_clean_optional(handoff_id),
                metadata=turn_metadata,
            )
        except ValueError:
            return False
        return True

    def _handle_turn(
        self,
        channel: str,
        method_name: str,
        text: str,
        *,
        conversation_session_id: str | None,
        voice_turn_id: str | None,
        turn_cancel_token: Any | None,
        person_id: str | None,
        operator_id: str | None,
        robot_id: str | None,
        site_id: str | None,
        metadata: dict[str, Any] | None,
        include_session: bool,
        defer_recording: bool,
    ) -> dict[str, Any] | None:
        if self._bridge is None:
            return {
                "handled": False,
                "disposition": "declined",
                "reason": "runtime_bridge_unavailable",
            }

        session = self._session_manager.get_or_create(
            channel=channel,
            session_id=_clean_optional(conversation_session_id),
            operator_id=_clean_optional(operator_id),
            robot_id=_clean_optional(robot_id),
            site_id=_clean_optional(site_id),
            metadata=metadata,
        )
        bridge_context = {
            "session_id": session.session_id,
            "conversation_thread_id": session.session_id,
            "conversation_session_id": session.session_id,
            "conversation_context": self._session_manager.context_payload(
                session.session_id,
                recent_turn_limit=6,
                max_chars=3000,
            ),
            "channel": channel,
            "voice_turn_id": _clean_optional(voice_turn_id),
            "turn_cancel_token": turn_cancel_token,
            "person_id": _clean_optional(person_id),
            "operator_id": operator_id,
            "robot_id": robot_id,
            "site_id": site_id,
            "metadata": metadata,
        }
        result = _call_with_supported_kwargs(
            getattr(self._bridge, method_name),
            text,
            bridge_context,
        )
        recording_deferred = bool(defer_recording)
        if isinstance(result, dict) and result.get("handled"):
            if not recording_deferred:
                self._record_handled_turn(
                    session.session_id,
                    text,
                    result,
                    metadata=metadata,
                )
        if (include_session or recording_deferred) and isinstance(result, dict):
            enriched = dict(result)
            enriched["conversation_thread_id"] = session.session_id
            enriched["conversation_session_id"] = session.session_id
            if recording_deferred:
                enriched["conversation_recording_deferred"] = True
            if include_session:
                enriched["conversation_session"] = _snapshot_payload(
                    self._session_manager.snapshot(session.session_id)
                )
            return enriched
        return result

    def _record_handled_turn(
        self,
        session_id: str,
        user_text: str,
        result: dict[str, Any],
        *,
        metadata: dict[str, Any] | None,
    ) -> None:
        turn = result.get("turn")
        if not isinstance(turn, dict):
            turn = {}
        assistant_text = (
            _clean_optional(
                turn.get("spoken_reply")
                or turn.get("assistant_text")
                or result.get("assistant_text")
                or result.get("reply")
            )
            or ""
        )
        active_planning_session_id = _clean_optional(
            turn.get("planning_session_id") or result.get("planning_session_id")
        )
        if active_planning_session_id:
            self._session_manager.update_session(
                session_id,
                active_planning_session_id=active_planning_session_id,
            )
        tool_calls = turn.get("tool_calls") or result.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = []
        turn_metadata = {
            "bridge_handled": True,
            **deepcopy(metadata or {}),
        }
        if result.get("status") is not None:
            turn_metadata["status"] = result.get("status")
        if result.get("reason") is not None:
            turn_metadata["reason"] = result.get("reason")
        intent = _clean_optional(
            turn.get("intent") or turn.get("action_type") or result.get("intent")
        )
        gate_decision = _clean_optional(turn.get("gate_decision") or result.get("gate_decision"))
        skill_name = _clean_optional(turn.get("skill_name") or result.get("skill_name"))
        handoff_id = _clean_optional(turn.get("handoff_id") or result.get("handoff_id"))
        self._session_manager.append_turn(
            session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            intent=intent,
            gate_decision=gate_decision,
            skill_name=skill_name,
            tool_calls=[dict(item) for item in tool_calls if isinstance(item, dict)],
            handoff_id=handoff_id,
            metadata=turn_metadata,
        )


def _call_with_supported_kwargs(
    method: Callable[..., dict[str, Any] | None],
    text: str,
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    cleaned = {key: value for key, value in kwargs.items() if value is not None}
    try:
        params = signature(method).parameters
    except (TypeError, ValueError):
        try:
            return method(text, **cleaned)
        except TypeError:
            return method(text)
    accepts_kwargs = any(param.kind == Parameter.VAR_KEYWORD for param in params.values())
    accepted = (
        cleaned
        if accepts_kwargs
        else {key: value for key, value in cleaned.items() if key in params}
    )
    return method(text, **accepted)


def _snapshot_payload(snapshot: SessionSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {}
    return {
        "thread_id": snapshot.session_id,
        "conversation_thread_id": snapshot.session_id,
        "session_id": snapshot.session_id,
        "channel": snapshot.channel,
        "operator_id": snapshot.operator_id,
        "robot_id": snapshot.robot_id,
        "site_id": snapshot.site_id,
        "status": snapshot.status,
        "active_planning_session_id": snapshot.active_planning_session_id,
        "summary": snapshot.summary,
        "turn_count": len(snapshot.turns),
    }


def _clean_optional(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None
