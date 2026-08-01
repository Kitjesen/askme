from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any

from askme.conversation import ConversationLedgerError, GenerationStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExternalTurnHandle:
    """Correlation handle for a provider/runtime turn outside BrainPipeline."""

    thread_id: str
    turn_id: str
    generation_id: str | None = None


class ExternalGenerationBeginError(ConversationLedgerError):
    """Raised when a durable Turn exists but provider Generation creation failed."""


def begin_external_turn(
    pipeline: Any,
    user_text: str,
    *,
    source: str,
    channel: str = "voice",
    conversation_thread_id: str | None = None,
    conversation_session_id: str | None = None,
    turn_id: str | None = None,
    provider: str | None = None,
    provider_session_id: str | None = None,
    provider_generation_id: str | None = None,
    generation_id: str | None = None,
    response_text: str = "",
    metadata: dict[str, Any] | None = None,
) -> ExternalTurnHandle | None:
    """Start the canonical Turn/Generation before provider audio is settled."""

    ledger = _turn_ledger_for(pipeline)
    if ledger is None:
        return None
    try:
        thread = ledger.resolve_thread(
            conversation_thread_id=conversation_thread_id,
            conversation_session_id=conversation_session_id,
            # Provider/runtime path changes normally share the voice
            # Thread channel; text adapters can opt into their own channel.
            channel=channel,
            metadata=metadata,
        )
        turn = ledger.start_turn(
            thread.thread_id,
            turn_id=turn_id,
            source=source,
            user_text=user_text,
            metadata=metadata,
        )
    except ConversationLedgerError:
        # Closed/expired/erased threads and identity conflicts are product
        # decisions, not storage outages.  Never bypass them through the
        # plaintext compatibility projection.
        raise
    except Exception as exc:
        logger.exception("Conversation Core could not begin external turn")
        _report_ledger_failure(pipeline, "begin an external turn", exc)
        if _legacy_fallback_enabled(pipeline):
            return None
        raise _ledger_write_error("begin an external turn", exc) from exc

    resolved_generation_id: str | None = None
    start_generation = getattr(ledger, "start_generation", None)
    if provider and callable(start_generation):
        try:
            generation = start_generation(
                turn.turn_id,
                generation_id=generation_id,
                provider=provider,
                provider_session_id=provider_session_id,
                provider_generation_id=provider_generation_id,
                response_text=response_text,
                metadata=metadata,
            )
            resolved_generation_id = str(generation.generation_id)
        except ConversationLedgerError:
            try:
                ledger.fail_turn(
                    turn.turn_id,
                    reason="generation_start_rejected",
                )
            except ConversationLedgerError:
                pass
            raise
        except Exception as exc:
            logger.exception("Conversation Core could not start external generation")
            _report_ledger_failure(pipeline, "start an external generation", exc)
            if not _legacy_fallback_enabled(pipeline):
                try:
                    ledger.fail_turn(
                        turn.turn_id,
                        reason="generation_start_failed",
                    )
                except ConversationLedgerError:
                    pass
                except Exception as settlement_exc:
                    _report_ledger_failure(
                        pipeline,
                        "fail an external turn after generation start",
                        settlement_exc,
                    )
                raise ExternalGenerationBeginError(
                    f"Conversation Core could not start an external generation: "
                    f"{type(exc).__name__}"
                ) from exc
    return ExternalTurnHandle(
        thread_id=str(thread.thread_id),
        turn_id=str(turn.turn_id),
        generation_id=resolved_generation_id,
    )


def complete_external_turn(
    pipeline: Any,
    handle: ExternalTurnHandle | None,
    *,
    user_text: str,
    assistant_text: str,
    source: str,
    conversation_thread_id: str | None = None,
    conversation_session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Commit delivered provider output, then update compatibility projections."""

    ledger = _turn_ledger_for(pipeline)
    should_project = ledger is None
    if handle is None and ledger is not None:
        if _legacy_fallback_enabled(pipeline):
            should_project = True
        else:
            raise ConversationLedgerError(
                "Conversation Core requires a durable external turn handle"
            )
    if handle is not None and ledger is not None:
        before_status = _ledger_turn_status(ledger, handle.turn_id)
        try:
            settled = ledger.commit_turn(
                handle.turn_id,
                user_text=user_text,
                assistant_text=assistant_text,
                heard_text=assistant_text,
                metadata=metadata,
            )
            after_status = _status_value(getattr(settled, "status", None))
            should_project = before_status not in _TERMINAL_TURN_STATUSES and (
                after_status in {None, "committed"}
            )
        except ConversationLedgerError as exc:
            logger.info("Conversation Core rejected late external completion: %s", exc)
            should_project = False
        except Exception as exc:
            logger.exception("Conversation Core could not complete external turn")
            status = _ledger_turn_status(ledger, handle.turn_id)
            _report_ledger_failure(pipeline, "complete an external turn", exc)
            if status == "committed" and before_status not in _TERMINAL_TURN_STATUSES:
                should_project = True
            elif _legacy_fallback_enabled(pipeline):
                should_project = True
            else:
                raise _ledger_write_error("complete an external turn", exc) from exc
    if should_project:
        _record_legacy_projection(
            pipeline,
            user_text=user_text,
            assistant_text=assistant_text,
            source=source,
            thread_id=(
                handle.thread_id
                if handle is not None
                else conversation_session_id or conversation_thread_id
            ),
        )


def cancel_external_turn(
    pipeline: Any,
    handle: ExternalTurnHandle | None,
    *,
    user_text: str,
    source: str,
    reason: str,
    played_ms: int = 0,
    heard_text: str = "",
    conversation_thread_id: str | None = None,
    conversation_session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Cancel provider output and retain only an explicitly known heard prefix."""

    ledger = _turn_ledger_for(pipeline)
    should_project = ledger is None
    if handle is None and ledger is not None:
        if _legacy_fallback_enabled(pipeline):
            should_project = True
        else:
            raise ConversationLedgerError(
                "Conversation Core requires a durable external turn handle"
            )
    if handle is not None and ledger is not None:
        before_status = _ledger_turn_status(ledger, handle.turn_id)
        try:
            settled = ledger.cancel_turn(
                handle.turn_id,
                reason=reason,
                played_ms=played_ms,
                heard_text=heard_text,
                metadata=metadata,
            )
            after_status = _status_value(getattr(settled, "status", None))
            should_project = before_status not in _TERMINAL_TURN_STATUSES and (
                after_status in {None, "cancelled"}
            )
        except ConversationLedgerError as exc:
            logger.info("Conversation Core rejected late external cancellation: %s", exc)
            should_project = False
        except Exception as exc:
            logger.exception("Conversation Core could not cancel external turn")
            status = _ledger_turn_status(ledger, handle.turn_id)
            _report_ledger_failure(pipeline, "cancel an external turn", exc)
            if status == "cancelled" and before_status not in _TERMINAL_TURN_STATUSES:
                should_project = True
            elif _legacy_fallback_enabled(pipeline):
                should_project = True
            else:
                raise _ledger_write_error("cancel an external turn", exc) from exc
    if should_project:
        _record_legacy_projection(
            pipeline,
            user_text=user_text,
            assistant_text=heard_text,
            source=source,
            thread_id=(
                handle.thread_id
                if handle is not None
                else conversation_session_id or conversation_thread_id
            ),
            outcome_label=f"{source}打断({reason})",
        )


def discard_external_generation(
    pipeline: Any,
    handle: ExternalTurnHandle | None,
    *,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Discard a speculative provider attempt while keeping its Turn reusable."""

    if handle is None or not handle.generation_id:
        return
    ledger = _turn_ledger_for(pipeline)
    transition_generation = getattr(ledger, "transition_generation", None)
    if not callable(transition_generation):
        return
    try:
        transition_generation(
            handle.generation_id,
            GenerationStatus.DISCARDED,
            metadata={"discard_reason": reason, **(metadata or {})},
        )
    except Exception as exc:
        logger.exception("Conversation Core could not discard external generation")
        _report_ledger_failure(pipeline, "discard an external generation", exc)


def record_external_turn(
    pipeline: Any,
    user_text: str,
    assistant_text: str,
    *,
    source: str = "external",
    channel: str = "voice",
    conversation_thread_id: str | None = None,
    conversation_session_id: str | None = None,
    turn_id: str | None = None,
    provider: str | None = None,
    provider_session_id: str | None = None,
    provider_generation_id: str | None = None,
    generation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Settle a turn handled outside ``BrainPipeline.process``.

    Conversation Core is authoritative.  The legacy ConversationManager write
    remains a prompt-context projection until all readers consume committed
    ledger events directly.
    """
    if not assistant_text:
        return

    handle = begin_external_turn(
        pipeline,
        user_text,
        source=source,
        channel=channel,
        conversation_thread_id=conversation_thread_id,
        conversation_session_id=conversation_session_id,
        turn_id=turn_id,
        provider=provider,
        provider_session_id=provider_session_id,
        provider_generation_id=provider_generation_id,
        generation_id=generation_id,
        response_text=assistant_text,
        metadata=metadata,
    )
    complete_external_turn(
        pipeline,
        handle,
        user_text=user_text,
        assistant_text=assistant_text,
        source=source,
        conversation_thread_id=conversation_thread_id,
        conversation_session_id=conversation_session_id,
        metadata=metadata,
    )


def _record_legacy_projection(
    pipeline: Any,
    *,
    user_text: str,
    assistant_text: str,
    source: str,
    thread_id: str | None,
    outcome_label: str | None = None,
) -> None:
    conversation = getattr(pipeline, "_conversation", None)
    if conversation is not None:
        add_user_message = getattr(conversation, "add_user_message", None)
        add_assistant_message = getattr(conversation, "add_assistant_message", None)
        if callable(add_user_message):
            _call_with_thread_scope(
                add_user_message,
                user_text,
                thread_id,
            )
        if assistant_text and callable(add_assistant_message):
            _call_with_thread_scope(
                add_assistant_message,
                assistant_text,
                thread_id,
            )

    episodic = getattr(pipeline, "_episodic", None)
    if episodic is not None:
        log = getattr(episodic, "log", None)
        should_reflect = getattr(episodic, "should_reflect", None)
        reflect = getattr(episodic, "reflect", None)
        if callable(log):
            log("command", f"用户说: {user_text}")
            label = outcome_label or f"{source}回复"
            log("outcome", f"{label}: {assistant_text[:100]}")
        if callable(should_reflect) and callable(reflect) and should_reflect():
            try:
                task = asyncio.create_task(reflect())
                task.add_done_callback(
                    lambda t: (
                        logger.error("[Episodic] Reflection failed: %s", t.exception())
                        if not t.cancelled() and t.exception()
                        else None
                    )
                )
            except RuntimeError:
                logger.debug("No running loop for external-turn reflection")


def _call_with_thread_scope(
    method: Any,
    content: str,
    thread_id: str | None,
) -> None:
    """Pass the compatibility session alias only when a callable supports it."""

    if not thread_id:
        method(content)
        return
    try:
        parameters = signature(method).parameters.values()
    except (TypeError, ValueError):
        method(content)
        return
    supports_keyword = any(
        parameter.kind is Parameter.VAR_KEYWORD or parameter.name == "conversation_session_id"
        for parameter in parameters
    )
    if supports_keyword:
        method(content, conversation_session_id=thread_id)
    else:
        method(content)


_TERMINAL_TURN_STATUSES = {"committed", "cancelled", "failed", "suppressed"}


def _turn_ledger_for(pipeline: Any) -> Any | None:
    """Resolve an explicitly wired ledger without activating mock attributes."""

    namespace = getattr(pipeline, "__dict__", None)
    if isinstance(namespace, dict) and "_turn_ledger" in namespace:
        return namespace["_turn_ledger"]
    if getattr(type(pipeline), "turn_ledger", None) is not None:
        return getattr(pipeline, "turn_ledger", None)
    return None


def _legacy_fallback_enabled(pipeline: Any) -> bool:
    """Return an explicitly configured fallback without activating mocks."""

    namespace = getattr(pipeline, "__dict__", None)
    return bool(
        isinstance(namespace, dict) and namespace.get("_conversation_core_legacy_fallback", False)
    )


def _ledger_write_error(operation: str, exc: BaseException) -> ConversationLedgerError:
    return ConversationLedgerError(f"Conversation Core could not {operation}: {type(exc).__name__}")


def _report_ledger_failure(
    pipeline: Any,
    operation: str,
    exc: BaseException,
) -> None:
    reporter = getattr(pipeline, "_record_turn_ledger_failure", None)
    if callable(reporter):
        reporter(operation, exc)


def _ledger_turn_status(ledger: Any, turn_id: str) -> str | None:
    get_turn = getattr(ledger, "get_turn", None)
    if not callable(get_turn):
        return None
    try:
        turn = get_turn(turn_id)
    except Exception:
        return None
    return _status_value(getattr(turn, "status", None))


def _status_value(status: Any) -> str | None:
    if status is None:
        return None
    return str(getattr(status, "value", status) or "").strip().lower() or None
