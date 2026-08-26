from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from askme.conversation import (
    ApprovalScope,
    CancellationToken,
    ConfirmationKind,
    ConfirmationScope,
    GenerationStarted,
    GenerationStatus,
    InteractionInput,
    InteractionTurnContext,
    InteractionTurnManager,
    TurnOutcome,
    TurnStatus,
    VoiceTurnLedger,
)


def test_manager_owns_a_committed_interaction_from_input_to_delivery(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "interaction-ledger.jsonl")
    manager = InteractionTurnManager(ledger)

    opened = manager.open(
        InteractionInput(
            user_text="请介绍一下园区",
            source="voice",
            thread_id="thread-1",
            person_id="person-1",
            operator_id="operator-1",
            robot_id="robot-1",
            site_id="site-a",
            metadata={"request_id": "request-1"},
        )
    )
    generating = manager.advance(
        opened,
        GenerationStarted(
            provider="litellm",
            provider_session_id="connection-1",
            provider_generation_id="provider-generation-1",
            response_text="生成中的回答",
            metadata={"model": "voice-fast"},
        ),
    )
    settled = manager.settle(
        generating,
        TurnOutcome.commit(
            assistant_text="这里是智慧园区。",
            heard_text="这里是智慧园区。",
            played_ms=720,
            metadata={"delivery": "speaker"},
        ),
    )

    assert opened.thread_id == "thread-1"
    assert opened.person_id == "person-1"
    assert generating.turn_id == opened.turn_id
    assert generating.generation_id is not None
    assert generating.provider == "litellm"
    assert settled.status is TurnStatus.COMMITTED
    assert settled.thread_id == opened.thread_id
    assert settled.assistant_text == "这里是智慧园区。"
    assert settled.heard_text == "这里是智慧园区。"
    assert settled.played_ms == 720

    generation = ledger.get_generation(generating.generation_id)
    assert generation is not None
    assert generation.status is GenerationStatus.APPROVED
    assert generation.provider_session_id == "connection-1"
    assert generation.provider_generation_id == "provider-generation-1"


def test_cancelled_interaction_keeps_only_the_known_heard_prefix(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "interaction-ledger.jsonl")
    manager = InteractionTurnManager(ledger)

    opened = manager.open(
        InteractionInput(
            user_text="继续介绍",
            source="voice",
            thread_id="thread-cancel",
        )
    )
    generating = manager.advance(
        opened,
        GenerationStarted(
            provider="realtime-s2s",
            response_text="完整生成内容不等于用户已经听到的内容",
        ),
    )
    cancelled = manager.settle(
        generating,
        TurnOutcome.cancel(
            reason="barge_in",
            heard_text="完整生成内容",
            played_ms=300,
            metadata={"interrupted_by": "person-1"},
        ),
    )

    assert cancelled.status is TurnStatus.CANCELLED
    assert cancelled.cancel_reason == "barge_in"
    assert cancelled.assistant_text == "完整生成内容"
    assert cancelled.heard_text == "完整生成内容"
    assert cancelled.playback_disposition == "truncate_played"


def test_failed_interaction_records_a_reason_without_committing_content(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "interaction-ledger.jsonl")
    manager = InteractionTurnManager(ledger)

    opened = manager.open(
        InteractionInput(
            user_text="查询状态",
            source="mcp",
            thread_id="thread-failed",
        )
    )
    failed = manager.settle(
        opened,
        TurnOutcome.fail(reason="provider_timeout", metadata={"provider": "litellm"}),
    )

    assert failed.status is TurnStatus.FAILED
    assert failed.failure_reason == "provider_timeout"
    assert failed.assistant_text == ""
    assert ledger.list_committed_turn_events() == []


def test_suppressed_interaction_records_policy_reason_without_committing_content(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "interaction-ledger.jsonl")
    manager = InteractionTurnManager(ledger)

    opened = manager.open(
        InteractionInput(
            user_text="扬声器回声",
            source="voice",
            thread_id="thread-suppressed",
        )
    )
    suppressed = manager.settle(
        opened,
        TurnOutcome.suppress(reason="echo", metadata={"gate": "interaction"}),
    )

    assert suppressed.status is TurnStatus.SUPPRESSED
    assert suppressed.suppression_reason == "echo"
    assert suppressed.assistant_text == ""
    assert ledger.list_committed_turn_events() == []


def test_confirmation_scope_matches_later_turn_only_within_identity_and_deadline() -> None:
    prompt = InteractionTurnContext(
        thread_id="thread-confirm",
        turn_id="prompt-turn",
        channel="voice",
        source="voice",
        user_text="需要我继续吗？",
        person_id="person-1",
        operator_id="operator-1",
    )
    response = replace(
        prompt,
        turn_id="response-turn",
        user_text="继续",
    )
    scope = ConfirmationScope.create(
        prompt,
        kind=ConfirmationKind.QUESTION_FOLLOWUP,
        expires_at_monotonic=120.0,
        allows_short_reply=True,
    )

    assert scope.prompt_turn_id == "prompt-turn"
    assert scope.allows_short_reply is True
    assert response.turn_id != scope.prompt_turn_id
    assert scope.matches(response, now_monotonic=119.0)
    assert not scope.matches(prompt, now_monotonic=119.0)
    assert not scope.matches(
        replace(response, turn_id=""),
        now_monotonic=119.0,
    )
    assert not scope.matches(
        replace(response, thread_id="other-thread"),
        now_monotonic=119.0,
    )
    assert not scope.matches(
        replace(response, operator_id="operator-2"),
        now_monotonic=119.0,
    )
    assert not scope.matches(
        replace(response, operator_id=None),
        now_monotonic=119.0,
    )
    assert not scope.matches(
        replace(response, person_id="person-2"),
        now_monotonic=119.0,
    )
    assert not scope.matches(
        replace(response, person_id=None),
        now_monotonic=119.0,
    )
    assert not scope.matches(response, now_monotonic=120.0)


def test_approval_scope_requires_the_exact_approval_id() -> None:
    prompt = InteractionTurnContext(
        thread_id="thread-approval",
        turn_id="prompt-approval",
        channel="voice",
        source="voice",
        user_text="执行危险操作吗？",
        person_id="person-1",
        operator_id="operator-1",
    )
    response = replace(
        prompt,
        turn_id="response-approval",
        user_text="确认执行",
    )
    scope = ApprovalScope.create(
        prompt,
        approval_id="approval-1",
        subject="robot.motion.enable",
        risk_level="high",
        payload_digest="sha256:abc123",
        expires_at_monotonic=220.0,
        allows_short_reply=True,
    )

    assert scope.kind is ConfirmationKind.TOOL_APPROVAL
    assert scope.approval_id == "approval-1"
    assert scope.subject == "robot.motion.enable"
    assert scope.risk_level == "high"
    assert scope.payload_digest == "sha256:abc123"
    assert scope.matches(
        response,
        approval_id="approval-1",
        now_monotonic=219.0,
    )
    assert not scope.matches(
        prompt,
        approval_id="approval-1",
        now_monotonic=219.0,
    )
    assert not scope.matches(
        response,
        approval_id="approval-other",
        now_monotonic=219.0,
    )


class _Cancellation:
    def __init__(self) -> None:
        self.cancelled = False

    def is_set(self) -> bool:
        return self.cancelled

    def set(self) -> None:
        self.cancelled = True


def test_turn_context_preserves_cancel_token_and_epoch_across_generation(
    tmp_path: Path,
) -> None:
    token = _Cancellation()
    ledger = VoiceTurnLedger(tmp_path / "interaction-context-ledger.jsonl")
    manager = InteractionTurnManager(ledger)
    opened = manager.open(
        InteractionInput(
            user_text="开始",
            source="voice",
            thread_id="thread-context",
            cancel_token=token,
            turn_epoch=7,
        )
    )
    generating = manager.advance(opened, GenerationStarted(provider="litellm"))

    assert isinstance(token, CancellationToken)
    assert generating.cancel_token is token
    assert generating.turn_epoch == 7
    token.set()
    assert generating.cancel_token is not None
    assert generating.cancel_token.is_set()
