"""Helpers for calling voice-gateway/runtime bridge methods.

Channel loops should keep their code in conversation terms. This helper maps
that context to either the stable voice-gateway keyword
``conversation_session_id`` or the provider/runtime keyword ``session_id``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import Parameter, isawaitable, signature
from typing import Any

from askme.pipeline.channels.external_turns import record_external_turn

logger = logging.getLogger(__name__)

ReplyHandler = Callable[[str], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class RuntimeBridgeOutcome:
    handled: bool = False
    reply: str = ""


def call_bridge_turn(
    method: Callable[..., dict[str, Any] | None],
    text: str,
    *,
    conversation_session_id: str | None = None,
    operator_id: str | None = None,
    robot_id: str | None = None,
    site_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Call a bridge method with only the kwargs it declares."""
    context = {
        "conversation_session_id": conversation_session_id,
        "session_id": conversation_session_id,
        "operator_id": operator_id,
        "robot_id": robot_id,
        "site_id": site_id,
        "metadata": metadata,
    }
    return method(text, **_supported_kwargs(method, context))


async def try_handle_runtime_bridge_turn(
    method: Callable[..., dict[str, Any] | None],
    user_text: str,
    *,
    conversation_session_id: str | None = None,
    pipeline: Any,
    dispatcher: Any | None = None,
    on_spoken_reply: ReplyHandler | None = None,
    label: str,
) -> bool:
    """Call a bridge method and handle its stable turn result contract."""
    outcome = await try_runtime_bridge_turn(
        method,
        user_text,
        conversation_session_id=conversation_session_id,
        pipeline=pipeline,
        dispatcher=dispatcher,
        on_spoken_reply=on_spoken_reply,
        label=label,
    )
    return outcome.handled


async def try_runtime_bridge_turn(
    method: Callable[..., dict[str, Any] | None],
    user_text: str,
    *,
    conversation_session_id: str | None = None,
    pipeline: Any,
    dispatcher: Any | None = None,
    on_spoken_reply: ReplyHandler | None = None,
    label: str,
) -> RuntimeBridgeOutcome:
    """Call a bridge method and return the handled outcome."""
    bridge_result = await asyncio.to_thread(
        call_bridge_turn,
        method,
        user_text,
        conversation_session_id=conversation_session_id,
    )

    return await runtime_bridge_result_outcome(
        bridge_result,
        user_text=user_text,
        pipeline=pipeline,
        dispatcher=dispatcher,
        on_spoken_reply=on_spoken_reply,
        label=label,
    )


async def handle_runtime_bridge_result(
    bridge_result: Any,
    *,
    user_text: str,
    pipeline: Any,
    dispatcher: Any | None = None,
    on_spoken_reply: ReplyHandler | None = None,
    label: str,
) -> bool:
    """Dispatch a handled runtime bridge turn or request local fallback."""
    outcome = await runtime_bridge_result_outcome(
        bridge_result,
        user_text=user_text,
        pipeline=pipeline,
        dispatcher=dispatcher,
        on_spoken_reply=on_spoken_reply,
        label=label,
    )
    return outcome.handled


async def runtime_bridge_result_outcome(
    bridge_result: Any,
    *,
    user_text: str,
    pipeline: Any,
    dispatcher: Any | None = None,
    on_spoken_reply: ReplyHandler | None = None,
    label: str,
) -> RuntimeBridgeOutcome:
    """Dispatch a handled runtime bridge turn and return reply metadata."""
    if not isinstance(bridge_result, dict) or not bridge_result.get("handled"):
        return RuntimeBridgeOutcome()

    turn = bridge_result.get("turn")
    if not isinstance(turn, dict):
        logger.warning(
            "%s runtime bridge returned an invalid handled payload; "
            "falling back to local pipeline.",
            label,
        )
        return RuntimeBridgeOutcome()

    action_type = turn.get("action_type")
    skill_name = turn.get("skill_name")

    if isinstance(skill_name, str) and skill_name and (
        action_type == "skill" or action_type == "general"
    ):
        if dispatcher:
            result = await dispatcher.dispatch(skill_name, user_text, source="runtime")
        else:
            result = await pipeline.execute_skill(skill_name, user_text)
        return RuntimeBridgeOutcome(
            handled=True,
            reply=result if isinstance(result, str) else "",
        )

    spoken_reply = turn.get("spoken_reply")
    if isinstance(spoken_reply, str) and spoken_reply.strip():
        reply = spoken_reply.strip()
        record_external_turn(
            pipeline,
            user_text,
            reply,
            source="runtime",
        )
        if on_spoken_reply is not None:
            maybe_awaitable = on_spoken_reply(reply)
            if isawaitable(maybe_awaitable):
                await maybe_awaitable
        return RuntimeBridgeOutcome(handled=True, reply=reply)

    logger.warning(
        "%s runtime bridge marked the turn handled but returned no usable "
        "reply (action_type=%r skill_name=%r); falling back to local pipeline.",
        label,
        action_type,
        skill_name,
    )
    return RuntimeBridgeOutcome()


def _supported_kwargs(
    method: Callable[..., Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    try:
        params = signature(method).parameters
    except (TypeError, ValueError):
        return {}

    accepts_kwargs = any(
        param.kind is Parameter.VAR_KEYWORD for param in params.values()
    )
    if accepts_kwargs:
        return {key: value for key, value in context.items() if value is not None}
    return {
        key: value
        for key, value in context.items()
        if value is not None and key in params
    }
