"""Provider-neutral interaction lifecycle commands for Conversation Core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from askme.conversation.ledger import VoiceTurnLedger
from askme.conversation.models import InvalidTransition, TurnRecord, TurnStatus


class ConfirmationKind(StrEnum):
    """Reason a later input may be interpreted as confirmation."""

    QUESTION_FOLLOWUP = "question_followup"
    SKILL_SLOT = "skill_slot"
    SKILL_CONFIRMATION = "skill_confirmation"
    COGNITION_CONFIRMATION = "cognition_confirmation"
    TOOL_APPROVAL = "tool_approval"
    RUNTIME_HANDOFF = "runtime_handoff"


@runtime_checkable
class CancellationToken(Protocol):
    """Minimal cancellation contract shared across interaction adapters."""

    def is_set(self) -> bool: ...

    def set(self) -> None: ...


def _optional_identity(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


@dataclass(frozen=True, slots=True)
class InteractionInput:
    """One admitted input before a durable Turn is opened."""

    user_text: str
    source: str
    thread_id: str | None = None
    turn_id: str | None = None
    channel: str | None = None
    person_id: str | None = None
    operator_id: str | None = None
    robot_id: str | None = None
    site_id: str | None = None
    timezone: str = "UTC"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cancel_token: CancellationToken | None = None
    turn_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class InteractionTurnContext:
    """Canonical identity carried by every stage of one interaction."""

    thread_id: str
    turn_id: str
    channel: str
    source: str
    user_text: str
    person_id: str | None = None
    operator_id: str | None = None
    robot_id: str | None = None
    site_id: str | None = None
    generation_id: str | None = None
    generation_epoch: int | None = None
    provider: str | None = None
    provider_session_id: str | None = None
    provider_generation_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cancel_token: CancellationToken | None = None
    turn_epoch: int | None = None


def _matches_later_turn(
    *,
    thread_id: str,
    prompt_turn_id: str,
    person_id: str | None,
    operator_id: str | None,
    expires_at_monotonic: float,
    context: InteractionTurnContext,
    now_monotonic: float,
) -> bool:
    """Match one later Turn without weakening its Thread or identity boundary."""

    if float(now_monotonic) >= expires_at_monotonic:
        return False
    if str(context.thread_id).strip() != thread_id:
        return False
    response_turn_id = str(context.turn_id).strip()
    if not response_turn_id or response_turn_id == prompt_turn_id:
        return False
    if person_id is not None and _optional_identity(context.person_id) != person_id:
        return False
    if operator_id is not None and _optional_identity(context.operator_id) != operator_id:
        return False
    return True


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfirmationScope:
    """Identity and deadline constraints for interpreting a later confirmation."""

    kind: ConfirmationKind
    thread_id: str
    prompt_turn_id: str
    person_id: str | None
    operator_id: str | None
    expires_at_monotonic: float
    allows_short_reply: bool

    @classmethod
    def create(
        cls,
        context: InteractionTurnContext,
        *,
        kind: ConfirmationKind | str,
        expires_at_monotonic: float,
        allows_short_reply: bool = False,
    ) -> ConfirmationScope:
        """Bind a confirmation prompt to its canonical Thread and identities."""

        thread_id = str(context.thread_id).strip()
        prompt_turn_id = str(context.turn_id).strip()
        if not thread_id or not prompt_turn_id:
            raise ValueError("confirmation scope requires thread_id and prompt_turn_id")
        return cls(
            kind=ConfirmationKind(kind),
            thread_id=thread_id,
            prompt_turn_id=prompt_turn_id,
            person_id=_optional_identity(context.person_id),
            operator_id=_optional_identity(context.operator_id),
            expires_at_monotonic=float(expires_at_monotonic),
            allows_short_reply=bool(allows_short_reply),
        )

    def matches(
        self,
        context: InteractionTurnContext,
        *,
        now_monotonic: float,
    ) -> bool:
        """Return whether a later input is eligible to answer this prompt."""

        return _matches_later_turn(
            thread_id=self.thread_id,
            prompt_turn_id=self.prompt_turn_id,
            person_id=self.person_id,
            operator_id=self.operator_id,
            expires_at_monotonic=self.expires_at_monotonic,
            context=context,
            now_monotonic=now_monotonic,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalScope:
    """Confirmation scope additionally bound to one exact approval challenge."""

    kind: ConfirmationKind
    thread_id: str
    prompt_turn_id: str
    person_id: str | None
    operator_id: str | None
    expires_at_monotonic: float
    allows_short_reply: bool
    approval_id: str
    subject: str
    risk_level: str
    payload_digest: str

    @classmethod
    def create(
        cls,
        context: InteractionTurnContext,
        *,
        approval_id: str,
        subject: str,
        risk_level: str,
        payload_digest: str,
        expires_at_monotonic: float,
        allows_short_reply: bool = False,
    ) -> ApprovalScope:
        """Bind an approval challenge to its prompt, identities, and payload."""

        thread_id = str(context.thread_id).strip()
        prompt_turn_id = str(context.turn_id).strip()
        approval_id = str(approval_id).strip()
        subject = str(subject).strip()
        risk_level = str(risk_level).strip()
        payload_digest = str(payload_digest).strip()
        if not thread_id or not prompt_turn_id:
            raise ValueError("approval scope requires thread_id and prompt_turn_id")
        if not approval_id or not subject or not risk_level or not payload_digest:
            raise ValueError("approval scope requires complete challenge identity")
        return cls(
            kind=ConfirmationKind.TOOL_APPROVAL,
            thread_id=thread_id,
            prompt_turn_id=prompt_turn_id,
            person_id=_optional_identity(context.person_id),
            operator_id=_optional_identity(context.operator_id),
            expires_at_monotonic=float(expires_at_monotonic),
            allows_short_reply=bool(allows_short_reply),
            approval_id=approval_id,
            subject=subject,
            risk_level=risk_level,
            payload_digest=payload_digest,
        )

    def matches(
        self,
        context: InteractionTurnContext,
        *,
        approval_id: str,
        now_monotonic: float,
    ) -> bool:
        """Return whether a response is eligible for this exact approval."""

        if str(approval_id).strip() != self.approval_id:
            return False
        return _matches_later_turn(
            thread_id=self.thread_id,
            prompt_turn_id=self.prompt_turn_id,
            person_id=self.person_id,
            operator_id=self.operator_id,
            expires_at_monotonic=self.expires_at_monotonic,
            context=context,
            now_monotonic=now_monotonic,
        )


@dataclass(frozen=True, slots=True)
class GenerationStarted:
    """Provider-neutral description of a generation attempt."""

    provider: str
    generation_id: str | None = None
    provider_session_id: str | None = None
    provider_generation_id: str | None = None
    response_text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    """A terminal interaction result expressed without provider concepts."""

    status: TurnStatus
    user_text: str | None = None
    assistant_text: str = ""
    heard_text: str | None = None
    played_ms: int | None = None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def commit(
        cls,
        *,
        assistant_text: str,
        user_text: str | None = None,
        heard_text: str | None = None,
        played_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TurnOutcome:
        """Describe content that was delivered and may enter conversation history."""

        return cls(
            status=TurnStatus.COMMITTED,
            user_text=user_text,
            assistant_text=assistant_text,
            heard_text=heard_text,
            played_ms=played_ms,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def cancel(
        cls,
        *,
        reason: str = "cancelled",
        heard_text: str | None = None,
        played_ms: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> TurnOutcome:
        """Describe an interrupted Turn with only its known delivered prefix."""

        return cls(
            status=TurnStatus.CANCELLED,
            heard_text=heard_text,
            played_ms=played_ms,
            reason=reason,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def fail(
        cls,
        *,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TurnOutcome:
        """Describe a Turn that could not produce a deliverable result."""

        return cls(
            status=TurnStatus.FAILED,
            reason=reason,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def suppress(
        cls,
        *,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TurnOutcome:
        """Describe input intentionally excluded by an interaction policy."""

        return cls(
            status=TurnStatus.SUPPRESSED,
            reason=reason,
            metadata=dict(metadata or {}),
        )


class InteractionTurnManager:
    """Own Thread, Turn, Generation, and settlement orchestration."""

    def __init__(self, ledger: VoiceTurnLedger) -> None:
        self._ledger = ledger

    def open(self, interaction: InteractionInput) -> InteractionTurnContext:
        """Resolve the canonical Thread and open exactly one durable Turn."""

        source = str(interaction.source or "unknown")
        channel = str(interaction.channel or source)
        metadata = dict(interaction.metadata)
        thread = self._ledger.resolve_thread(
            thread_id=interaction.thread_id,
            channel=channel,
            person_id=interaction.person_id,
            operator_id=interaction.operator_id,
            robot_id=interaction.robot_id,
            site_id=interaction.site_id,
            timezone=interaction.timezone,
            metadata=metadata,
        )
        turn = self._ledger.start_turn(
            thread.thread_id,
            turn_id=interaction.turn_id,
            source=source,
            user_text=interaction.user_text,
            metadata=metadata,
        )
        return InteractionTurnContext(
            thread_id=thread.thread_id,
            turn_id=turn.turn_id,
            channel=thread.channel,
            source=turn.source,
            user_text=turn.user_text,
            person_id=thread.person_id,
            operator_id=thread.operator_id,
            robot_id=thread.robot_id,
            site_id=thread.site_id,
            cancel_token=interaction.cancel_token,
            turn_epoch=interaction.turn_epoch,
            metadata=dict(turn.metadata),
        )

    def advance(
        self,
        context: InteractionTurnContext,
        event: GenerationStarted,
    ) -> InteractionTurnContext:
        """Start or replace the active Generation without changing Turn identity."""

        generation = self._ledger.start_generation(
            context.turn_id,
            provider=event.provider,
            generation_id=event.generation_id,
            provider_session_id=event.provider_session_id,
            provider_generation_id=event.provider_generation_id,
            response_text=event.response_text,
            metadata=dict(event.metadata),
        )
        return replace(
            context,
            generation_id=generation.generation_id,
            generation_epoch=generation.epoch,
            provider=generation.provider,
            provider_session_id=generation.provider_session_id,
            provider_generation_id=generation.provider_generation_id,
        )

    def settle(
        self,
        context: InteractionTurnContext,
        outcome: TurnOutcome,
    ) -> TurnRecord:
        """Settle a Turn once, preserving only content represented by its outcome."""

        status = TurnStatus(outcome.status)
        if status is TurnStatus.COMMITTED:
            return self._ledger.commit_turn(
                context.turn_id,
                user_text=outcome.user_text,
                assistant_text=outcome.assistant_text,
                heard_text=outcome.heard_text,
                played_ms=outcome.played_ms,
                metadata=dict(outcome.metadata),
            )
        if status is TurnStatus.CANCELLED:
            return self._ledger.cancel_turn(
                context.turn_id,
                reason=outcome.reason or "cancelled",
                played_ms=outcome.played_ms or 0,
                heard_text=outcome.heard_text,
                metadata=dict(outcome.metadata),
            )
        if status is TurnStatus.FAILED:
            return self._ledger.fail_turn(
                context.turn_id,
                reason=outcome.reason or "failed",
                metadata=dict(outcome.metadata),
            )
        if status is TurnStatus.SUPPRESSED:
            return self._ledger.suppress_turn(
                context.turn_id,
                reason=outcome.reason or "suppressed",
                metadata=dict(outcome.metadata),
            )
        raise InvalidTransition(f"unsupported interaction outcome: {status.value}")


__all__ = [
    "ApprovalScope",
    "CancellationToken",
    "ConfirmationKind",
    "ConfirmationScope",
    "GenerationStarted",
    "InteractionInput",
    "InteractionTurnContext",
    "InteractionTurnManager",
    "TurnOutcome",
]
