from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from askme.conversation import (
    ConflictingThreadAliases,
    DuplicateEntity,
    GenerationStatus,
    InvalidTransition,
    LedgerCorruptionError,
    ThreadStatus,
    TurnInProgress,
    TurnStatus,
    VoiceTurnLedger,
    canonical_thread_id,
    migrate_legacy_history,
)
from askme.conversation.models import CommittedTurnEvent


class AdvancingClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 7, 19, 1, 2, 3, tzinfo=UTC)

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(milliseconds=1)
        return value


def make_ledger(tmp_path: Path) -> VoiceTurnLedger:
    return VoiceTurnLedger(tmp_path / "turn-ledger.jsonl", clock=AdvancingClock())


def test_resolve_thread_reuses_the_same_open_identity(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)

    first = ledger.resolve_thread(
        channel="voice",
        person_id="person-7",
        robot_id="robot-1",
        site_id="site-a",
    )
    second = ledger.resolve_thread(
        channel="voice",
        person_id="person-7",
        robot_id="robot-1",
        site_id="site-a",
    )

    assert second.thread_id == first.thread_id
    assert len(ledger.list_threads()) == 1
    assert first.created_at.tzinfo is UTC


def test_anonymous_resolution_without_explicit_id_starts_a_new_thread(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)

    first = ledger.resolve_thread(channel="voice")
    second = ledger.resolve_thread(channel="voice")

    assert first.thread_id != second.thread_id
    assert len(ledger.list_threads()) == 2


def test_thread_local_day_uses_its_declared_timezone(tmp_path: Path) -> None:
    clock = AdvancingClock()
    clock.current = datetime(2026, 7, 18, 17, 2, 3, tzinfo=UTC)
    ledger = VoiceTurnLedger(tmp_path / "turn-ledger.jsonl", clock=clock)

    thread = ledger.resolve_thread(
        conversation_thread_id="thread-local-day",
        timezone="Asia/Shanghai",
    )

    assert thread.local_day == "2026-07-19"


def test_explicit_thread_aliases_are_canonical_and_conflicts_are_rejected(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)

    assert canonical_thread_id(conversation_session_id=" thread-1 ") == "thread-1"
    assert canonical_thread_id(
        conversation_thread_id="thread-1",
        conversation_id="thread-1",
        chat_session_id="thread-1",
    ) == "thread-1"
    with pytest.raises(ConflictingThreadAliases):
        canonical_thread_id(conversation_thread_id="thread-1", session_id="thread-2")

    first = ledger.resolve_thread(conversation_session_id="thread-1", channel="voice")
    second = ledger.resolve_thread(conversation_thread_id="thread-1", channel="voice")
    assert first.thread_id == second.thread_id == "thread-1"
    assert len(ledger.list_threads()) == 1
    with pytest.raises(ValueError):
        ledger.resolve_thread(conversation_thread_id="thread-1", channel="web")


def test_turn_state_machine_accepts_valid_path_and_rejects_terminal_reentry(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(channel="voice", person_id="person-1")
    turn = ledger.start_turn(thread.thread_id, user_text="你好")

    ledger.transition_turn(turn.turn_id, TurnStatus.LISTENING)
    ledger.transition_turn(turn.turn_id, TurnStatus.TRANSCRIBED, user_text="你好")
    ledger.transition_turn(turn.turn_id, TurnStatus.ROUTED)
    ledger.transition_turn(turn.turn_id, TurnStatus.GENERATING)
    ledger.transition_turn(turn.turn_id, TurnStatus.SPEAKING, assistant_text="你好呀")
    committed = ledger.commit_turn(turn.turn_id, assistant_text="你好呀")

    assert committed.status is TurnStatus.COMMITTED
    assert committed.committed_at is not None
    with pytest.raises(InvalidTransition):
        ledger.transition_turn(turn.turn_id, TurnStatus.GENERATING)


def test_committed_turn_reader_exposes_only_commits_in_global_ledger_order(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)
    first_thread = ledger.resolve_thread(conversation_thread_id="thread-events-a")
    first_turn = ledger.start_turn(
        first_thread.thread_id,
        turn_id="turn-committed-a",
        source="voice",
        user_text="first question",
        metadata={"started": True},
    )
    first_commit_at = datetime(2026, 7, 19, 2, 0, tzinfo=UTC)
    ledger.commit_turn(
        first_turn.turn_id,
        assistant_text="first answer",
        heard_text="heard first answer",
        played_ms=420,
        metadata={"classification": "public"},
        event_id="commit-event-a",
        at=first_commit_at,
    )
    cancelled_turn = ledger.start_turn(
        first_thread.thread_id,
        turn_id="turn-cancelled",
        user_text="do not publish",
    )
    ledger.cancel_turn(cancelled_turn.turn_id)
    second_thread = ledger.resolve_thread(conversation_thread_id="thread-events-b")
    second_turn = ledger.start_turn(
        second_thread.thread_id,
        turn_id="turn-committed-b",
        source="text",
        user_text="second question",
    )
    ledger.commit_turn(
        second_turn.turn_id,
        assistant_text="second answer",
        event_id="commit-event-b",
    )

    events = ledger.list_committed_turn_events()

    assert [event.event_id for event in events] == ["commit-event-a", "commit-event-b"]
    assert [event.sequence for event in events] == [3, 8]
    first_event = events[0]
    assert first_event.occurred_at == first_commit_at
    assert first_event.thread_id == first_thread.thread_id
    assert first_event.turn_id == first_turn.turn_id
    assert first_event.turn_sequence == 1
    assert first_event.source == "voice"
    assert first_event.user_text == "first question"
    assert first_event.assistant_text == "first answer"
    assert first_event.heard_text == "heard first answer"
    assert first_event.played_ms == 420
    assert first_event.playback_disposition == "delivered"
    assert first_event.metadata == {"started": True, "classification": "public"}


def test_committed_turn_reader_applies_cursor_limit_and_validates_window(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(conversation_thread_id="thread-window")
    for index in range(1, 4):
        turn = ledger.start_turn(
            thread.thread_id,
            turn_id=f"turn-window-{index}",
            user_text=f"question {index}",
        )
        ledger.commit_turn(
            turn.turn_id,
            assistant_text=f"answer {index}",
            event_id=f"commit-window-{index}",
        )

    page = ledger.list_committed_turn_events(after_sequence=3, limit=1)

    assert [event.event_id for event in page] == ["commit-window-2"]
    with pytest.raises(ValueError, match="after_sequence"):
        ledger.list_committed_turn_events(after_sequence=-1)
    for invalid_limit in (0, 1001):
        with pytest.raises(ValueError, match="limit"):
            ledger.list_committed_turn_events(limit=invalid_limit)


def test_committed_turn_reader_replays_migration_content_identically(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy-reader.json"
    legacy_path.write_text(
        json.dumps(
            {
                "sessions": {
                    "legacy-reader": [
                        {"role": "user", "content": "legacy question"},
                        {"role": "assistant", "content": "legacy answer"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "reader-ledger.jsonl"
    ledger = VoiceTurnLedger(path)
    ledger.migrate_legacy_history(legacy_path)

    before_restart = ledger.list_committed_turn_events()
    after_restart = VoiceTurnLedger(path).list_committed_turn_events()

    assert after_restart == before_restart
    assert len(after_restart) == 1
    event = after_restart[0]
    assert event.source == "legacy"
    assert event.user_text == "legacy question"
    assert event.assistant_text == event.heard_text == "legacy answer"
    assert event.metadata["legacy_import"] is True


def test_committed_turn_reader_excludes_erased_threads_after_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "erasure-reader-ledger.jsonl"
    ledger = VoiceTurnLedger(path)
    private_thread = ledger.resolve_thread(conversation_thread_id="thread-erased-reader")
    private_turn = ledger.start_turn(
        private_thread.thread_id,
        turn_id="turn-erased-reader",
        user_text="SECRET USER TEXT",
        metadata={"secret": "SECRET METADATA"},
    )
    ledger.commit_turn(
        private_turn.turn_id,
        assistant_text="SECRET ASSISTANT TEXT",
        event_id="commit-erased-reader",
    )
    visible_thread = ledger.resolve_thread(conversation_thread_id="thread-visible-reader")
    visible_turn = ledger.start_turn(
        visible_thread.thread_id,
        turn_id="turn-visible-reader",
        user_text="visible question",
    )
    ledger.commit_turn(
        visible_turn.turn_id,
        assistant_text="visible answer",
        event_id="commit-visible-reader",
    )

    assert {event.event_id for event in ledger.list_committed_turn_events()} == {
        "commit-erased-reader",
        "commit-visible-reader",
    }
    ledger.transition_thread(private_thread.thread_id, ThreadStatus.ERASED)

    assert [
        event.event_id for event in ledger.list_committed_turn_events()
    ] == ["commit-visible-reader"]
    assert [
        event.event_id for event in VoiceTurnLedger(path).list_committed_turn_events()
    ] == ["commit-visible-reader"]


def test_committed_turn_event_is_frozen_and_metadata_is_defensively_copied(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)
    started_metadata = {"policy": {"retention": "short"}}
    committed_metadata = {"delivery": {"mode": "voice"}}
    thread = ledger.resolve_thread(conversation_thread_id="thread-copy-reader")
    turn = ledger.start_turn(
        thread.thread_id,
        user_text="question",
        metadata=started_metadata,
    )
    ledger.commit_turn(
        turn.turn_id,
        assistant_text="answer",
        metadata=committed_metadata,
    )
    started_metadata["policy"]["retention"] = "changed at source"
    committed_metadata["delivery"]["mode"] = "changed at source"

    event = ledger.list_committed_turn_events()[0]

    assert isinstance(event, CommittedTurnEvent)
    with pytest.raises(FrozenInstanceError):
        setattr(event, "user_text", "changed by caller")
    event.metadata["policy"]["retention"] = "changed by caller"
    event.metadata["delivery"]["mode"] = "changed by caller"
    fresh_event = ledger.list_committed_turn_events()[0]
    materialized_turn = ledger.get_turn(turn.turn_id)
    assert fresh_event.metadata == {
        "policy": {"retention": "short"},
        "delivery": {"mode": "voice"},
    }
    assert materialized_turn is not None
    assert materialized_turn.metadata == fresh_event.metadata


@pytest.mark.parametrize(
    "status",
    [
        TurnStatus.COMMITTED,
        TurnStatus.CANCELLED,
        TurnStatus.FAILED,
        TurnStatus.SUPPRESSED,
    ],
)
def test_generic_turn_transition_cannot_bypass_terminal_settlement_rules(
    tmp_path: Path,
    status: TurnStatus,
) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(conversation_thread_id=f"thread-{status.value}")
    turn = ledger.start_turn(thread.thread_id)

    with pytest.raises(InvalidTransition, match="dedicated"):
        ledger.transition_turn(turn.turn_id, status)

    assert ledger.get_turn(turn.turn_id).status is TurnStatus.STARTED


def test_duplicate_event_id_is_idempotent(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(channel="voice")
    turn = ledger.start_turn(thread.thread_id)

    first = ledger.transition_turn(
        turn.turn_id,
        TurnStatus.TRANSCRIBED,
        user_text="第一版",
        event_id="asr-final-1",
    )
    line_count = len(ledger.path.read_text(encoding="utf-8").splitlines())
    duplicate = ledger.transition_turn(
        turn.turn_id,
        TurnStatus.GENERATING,
        user_text="重复投递不应覆盖",
        event_id="asr-final-1",
    )

    assert duplicate.status is first.status is TurnStatus.TRANSCRIBED
    assert duplicate.user_text == "第一版"
    assert len(ledger.path.read_text(encoding="utf-8").splitlines()) == line_count


def test_turn_id_retry_requires_the_same_user_payload(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(conversation_thread_id="thread-turn-idempotency")
    first = ledger.start_turn(
        thread.thread_id,
        turn_id="stable-turn-id",
        user_text="original request",
    )
    event_count = ledger.event_count

    retry = ledger.start_turn(
        thread.thread_id,
        turn_id="stable-turn-id",
        user_text="original request",
    )

    assert retry == first
    assert ledger.event_count == event_count
    with pytest.raises(DuplicateEntity, match="user_text"):
        ledger.start_turn(
            thread.thread_id,
            turn_id="stable-turn-id",
            user_text="different request",
        )
    assert ledger.event_count == event_count


def test_event_id_cannot_be_reused_for_another_entity(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(conversation_thread_id="thread-events")
    first_turn = ledger.start_turn(thread.thread_id, turn_id="turn-one")

    ledger.transition_turn(
        first_turn.turn_id,
        TurnStatus.TRANSCRIBED,
        event_id="shared-event-token",
    )
    ledger.commit_turn(first_turn.turn_id, assistant_text="first done")
    second_turn = ledger.start_turn(thread.thread_id, turn_id="turn-two")
    with pytest.raises(DuplicateEntity):
        ledger.transition_turn(
            second_turn.turn_id,
            TurnStatus.TRANSCRIBED,
            event_id="shared-event-token",
        )

    assert ledger.get_turn(second_turn.turn_id).status is TurnStatus.STARTED


def test_event_id_cannot_be_reused_for_another_operation_on_same_entity(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(conversation_thread_id="thread-operation-events")
    turn = ledger.start_turn(
        thread.thread_id,
        turn_id="turn-operation",
        event_id="same-token",
    )

    with pytest.raises(DuplicateEntity):
        ledger.commit_turn(
            turn.turn_id,
            assistant_text="不能被吞掉",
            event_id="same-token",
        )

    assert ledger.get_turn(turn.turn_id).status is TurnStatus.STARTED


def test_replay_restores_thread_turn_and_generation(tmp_path: Path) -> None:
    path = tmp_path / "turn-ledger.jsonl"
    ledger = VoiceTurnLedger(path, clock=AdvancingClock())
    thread = ledger.resolve_thread(
        conversation_thread_id="thread-replay",
        channel="voice",
        person_id="person-3",
    )
    turn = ledger.start_turn(thread.thread_id, turn_id="turn-replay", user_text="问题")
    generation = ledger.start_generation(
        turn.turn_id,
        generation_id="generation-replay",
        provider="volcengine",
        provider_session_id="provider-session-a",
    )
    ledger.transition_generation(
        generation.generation_id,
        GenerationStatus.PROVIDER_RESPONDING,
        response_text="回答",
    )
    ledger.commit_turn(turn.turn_id, assistant_text="回答")

    replayed = VoiceTurnLedger(path)

    assert replayed.get_thread(thread.thread_id) == ledger.get_thread(thread.thread_id)
    assert replayed.get_turn(turn.turn_id) == ledger.get_turn(turn.turn_id)
    assert replayed.get_generation(generation.generation_id) == ledger.get_generation(
        generation.generation_id
    )


def test_replay_durably_fails_an_orphaned_active_generation(tmp_path: Path) -> None:
    path = tmp_path / "turn-ledger.jsonl"
    ledger = VoiceTurnLedger(path)
    thread = ledger.resolve_thread(conversation_thread_id="thread-restart")
    turn = ledger.start_turn(thread.thread_id, turn_id="turn-restart")
    generation = ledger.start_generation(turn.turn_id, provider="volcengine")
    ledger.transition_turn(turn.turn_id, TurnStatus.GENERATING)

    replayed = VoiceTurnLedger(path)

    recovered_turn = replayed.get_turn(turn.turn_id)
    recovered_generation = replayed.get_generation(generation.generation_id)
    assert recovered_turn is not None
    assert recovered_generation is not None
    assert recovered_turn.status is TurnStatus.FAILED
    assert recovered_turn.failure_reason == "process_restart"
    assert recovered_generation.status is GenerationStatus.PROVIDER_FAILED
    event_count = replayed.event_count
    assert VoiceTurnLedger(path).event_count == event_count


def test_cancel_before_playback_deletes_unheard_assistant_content(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(channel="voice")
    turn = ledger.start_turn(thread.thread_id, user_text="停一下")
    generation = ledger.start_generation(
        turn.turn_id,
        provider="volcengine",
        response_text="这一整段都还没有播放",
    )

    cancelled = ledger.cancel_turn(turn.turn_id, reason="barge_in", played_ms=0)

    assert cancelled.status is TurnStatus.CANCELLED
    assert cancelled.assistant_text == ""
    assert cancelled.heard_text == ""
    assert cancelled.playback_disposition == "delete_unheard"
    discarded = ledger.get_generation(generation.generation_id)
    assert discarded is not None
    assert discarded.status is GenerationStatus.DISCARDED


def test_cancel_after_playback_commits_only_heard_prefix(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(channel="voice")
    turn = ledger.start_turn(thread.thread_id, user_text="介绍一下")
    generation = ledger.start_generation(
        turn.turn_id,
        provider="volcengine",
        response_text="你好，我是园区机器人，很高兴认识你。",
    )

    cancelled = ledger.cancel_turn(
        turn.turn_id,
        reason="barge_in",
        played_ms=820,
        heard_text="你好，我是园区机器人",
    )

    assert cancelled.status is TurnStatus.CANCELLED
    assert cancelled.played_ms == 820
    assert cancelled.assistant_text == cancelled.heard_text == "你好，我是园区机器人"
    assert cancelled.playback_disposition == "truncate_played"
    truncated = ledger.get_generation(generation.generation_id)
    assert truncated is not None
    assert truncated.status is GenerationStatus.TRUNCATED
    assert truncated.played_ms == 820
    assert truncated.heard_text == "你好，我是园区机器人"


def test_provider_session_replacement_creates_new_generation_not_thread(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(conversation_thread_id="business-thread", channel="voice")
    turn = ledger.start_turn(thread.thread_id)

    first = ledger.start_generation(
        turn.turn_id,
        provider="volcengine",
        provider_session_id="socket-a",
    )
    second = ledger.start_generation(
        turn.turn_id,
        provider="volcengine",
        provider_session_id="socket-b",
    )

    assert first.thread_id == second.thread_id == thread.thread_id
    assert first.epoch == 1
    assert second.epoch == 2
    rolled_back = ledger.get_generation(first.generation_id)
    assert rolled_back is not None
    assert rolled_back.status is GenerationStatus.ROLLED_BACK
    events = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["payload"]["replaced_generation_ids"] == [first.generation_id]
    assert len(ledger.list_threads()) == 1


def test_turn_commit_atomically_approves_active_generation_with_final_text(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(conversation_thread_id="thread-commit")
    turn = ledger.start_turn(thread.thread_id, user_text="问题")
    generation = ledger.start_generation(
        turn.turn_id,
        provider="volcengine",
        response_text="草稿",
    )

    ledger.commit_turn(
        turn.turn_id,
        assistant_text="最终播放文本",
        heard_text="最终播放文本",
    )

    committed_generation = ledger.get_generation(generation.generation_id)
    assert committed_generation is not None
    assert committed_generation.status is GenerationStatus.APPROVED
    assert committed_generation.response_text == "最终播放文本"
    assert committed_generation.heard_text == "最终播放文本"


def test_generation_cannot_be_approved_outside_turn_commit(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(conversation_thread_id="thread-approval-owner")
    turn = ledger.start_turn(thread.thread_id, user_text="问题")
    generation = ledger.start_generation(turn.turn_id, provider="volcengine")

    with pytest.raises(InvalidTransition, match="commit_turn"):
        ledger.transition_generation(generation.generation_id, GenerationStatus.APPROVED)

    unchanged = ledger.get_generation(generation.generation_id)
    assert unchanged is not None
    assert unchanged.status is GenerationStatus.STARTED


def test_turn_commit_leaves_at_most_one_approved_generation(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(conversation_thread_id="thread-one-winner")
    turn = ledger.start_turn(thread.thread_id, user_text="问题")
    first = ledger.start_generation(turn.turn_id, provider="volcengine")
    second = ledger.start_generation(turn.turn_id, provider="askme_pipeline")

    ledger.commit_turn(turn.turn_id, assistant_text="最终回答")

    generations = ledger.list_generations(turn_id=turn.turn_id)
    approved = [
        generation
        for generation in generations
        if generation.status is GenerationStatus.APPROVED
    ]
    assert len(approved) == 1
    assert approved[0].generation_id == second.generation_id
    rolled_back = ledger.get_generation(first.generation_id)
    assert rolled_back is not None
    assert rolled_back.status is GenerationStatus.ROLLED_BACK


def test_cancel_never_guesses_heard_text_from_a_full_generated_answer(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(channel="voice")
    turn = ledger.start_turn(thread.thread_id)
    ledger.transition_turn(
        turn.turn_id,
        TurnStatus.SPEAKING,
        assistant_text="完整生成内容并不等于用户已经听到",
    )

    cancelled = ledger.cancel_turn(turn.turn_id, played_ms=300)

    assert cancelled.heard_text == ""
    assert cancelled.assistant_text == ""


def test_fail_and_suppress_are_terminal_turn_settlements(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(channel="voice")
    failed_turn = ledger.start_turn(thread.thread_id)
    failed = ledger.fail_turn(failed_turn.turn_id, reason="provider_timeout")
    suppressed_turn = ledger.start_turn(thread.thread_id)
    suppressed = ledger.suppress_turn(suppressed_turn.turn_id, reason="echo")

    assert failed.status is TurnStatus.FAILED
    assert failed.failure_reason == "provider_timeout"
    assert suppressed.status is TurnStatus.SUPPRESSED
    assert suppressed.suppression_reason == "echo"


@pytest.mark.parametrize("settlement", ["commit", "cancel", "fail", "suppress"])
def test_repeating_same_terminal_settlement_is_a_noop_that_keeps_replay_valid(
    tmp_path: Path,
    settlement: str,
) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(channel="voice")
    turn = ledger.start_turn(thread.thread_id)

    def settle() -> object:
        if settlement == "commit":
            return ledger.commit_turn(turn.turn_id, assistant_text="完成")
        if settlement == "cancel":
            return ledger.cancel_turn(turn.turn_id, reason="barge_in")
        if settlement == "fail":
            return ledger.fail_turn(turn.turn_id, reason="timeout")
        return ledger.suppress_turn(turn.turn_id, reason="echo")

    first = settle()
    event_count = ledger.event_count
    second = settle()

    assert second == first
    assert ledger.event_count == event_count
    assert VoiceTurnLedger(ledger.path).get_turn(turn.turn_id) == first


def test_closed_thread_rejects_new_turns_and_erasure_redacts_read_models(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(channel="voice", person_id="private-person")
    turn = ledger.start_turn(thread.thread_id, user_text="敏感问题", metadata={"secret": 1})
    generation = ledger.start_generation(
        turn.turn_id,
        provider="volcengine",
        response_text="敏感回答",
        metadata={"token": "private"},
    )
    ledger.commit_turn(turn.turn_id, assistant_text="敏感回答")
    ledger.transition_thread(thread.thread_id, "closed")

    with pytest.raises(InvalidTransition):
        ledger.start_turn(thread.thread_id)

    ledger.transition_thread(thread.thread_id, "erased")
    erased_turn = ledger.get_turn(turn.turn_id)
    erased_generation = ledger.get_generation(generation.generation_id)
    assert erased_turn is not None
    assert erased_generation is not None
    assert erased_turn.user_text == erased_turn.assistant_text == erased_turn.heard_text == ""
    assert erased_turn.metadata == {}
    assert erased_generation.response_text == erased_generation.heard_text == ""
    assert erased_generation.metadata == {}


def test_erasure_settles_active_turn_and_rejects_late_mutations(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(conversation_thread_id="thread-private")
    turn = ledger.start_turn(thread.thread_id, user_text="SECRET USER")
    generation = ledger.start_generation(
        turn.turn_id,
        provider="volcengine",
        response_text="SECRET ASSISTANT",
    )

    ledger.transition_thread(thread.thread_id, ThreadStatus.ERASED)

    erased_turn = ledger.get_turn(turn.turn_id)
    erased_generation = ledger.get_generation(generation.generation_id)
    assert erased_turn is not None
    assert erased_generation is not None
    assert erased_turn.status is TurnStatus.CANCELLED
    assert erased_generation.status is GenerationStatus.DISCARDED
    assert erased_turn.user_text == erased_turn.assistant_text == erased_turn.heard_text == ""
    assert erased_generation.response_text == erased_generation.heard_text == ""
    event_count = ledger.event_count

    with pytest.raises(InvalidTransition, match="erased"):
        ledger.commit_turn(turn.turn_id, assistant_text="SECRET LATE")
    with pytest.raises(InvalidTransition, match="erased"):
        ledger.transition_generation(
            generation.generation_id,
            GenerationStatus.PROVIDER_RESPONDING,
        )

    assert ledger.event_count == event_count
    replayed = VoiceTurnLedger(ledger.path)
    replayed_turn = replayed.get_turn(turn.turn_id)
    replayed_generation = replayed.get_generation(generation.generation_id)
    assert replayed_turn is not None
    assert replayed_generation is not None
    assert replayed_turn.assistant_text == ""
    assert replayed_generation.response_text == ""


def test_replay_ignores_only_a_truncated_final_jsonl_record(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(channel="voice")
    turn = ledger.start_turn(thread.thread_id)
    with ledger.path.open("ab") as stream:
        stream.write(b'{"sequence":999,"event_id":')

    replayed = VoiceTurnLedger(ledger.path)

    assert replayed.get_thread(thread.thread_id) is not None
    assert replayed.get_turn(turn.turn_id) is not None

    replayed.start_turn(thread.thread_id, turn_id="turn-after-repair")
    repaired = VoiceTurnLedger(ledger.path)
    assert repaired.get_turn("turn-after-repair") is not None


def test_replay_rejects_corruption_in_a_completed_record(tmp_path: Path) -> None:
    path = tmp_path / "turn-ledger.jsonl"
    path.write_bytes(b'{"broken":true}\n')

    with pytest.raises(LedgerCorruptionError):
        VoiceTurnLedger(path)


def test_replay_rejects_unknown_schema_and_illegal_lifecycle_history(tmp_path: Path) -> None:
    schema_ledger = make_ledger(tmp_path / "schema")
    schema_ledger.resolve_thread(channel="voice")
    schema_events = schema_ledger.path.read_text(encoding="utf-8").splitlines()
    first_event = json.loads(schema_events[0])
    first_event["schema_version"] = 999
    schema_ledger.path.write_text(json.dumps(first_event) + "\n", encoding="utf-8")
    with pytest.raises(LedgerCorruptionError):
        VoiceTurnLedger(schema_ledger.path)

    lifecycle_ledger = make_ledger(tmp_path / "lifecycle")
    thread = lifecycle_ledger.resolve_thread(channel="voice")
    turn = lifecycle_ledger.start_turn(thread.thread_id)
    lifecycle_ledger.commit_turn(turn.turn_id, assistant_text="done")
    invalid_event = {
        "schema_version": 1,
        "sequence": lifecycle_ledger.event_count + 1,
        "event_id": "invalid-transition-event",
        "event_type": "turn.transitioned",
        "entity_type": "turn",
        "entity_id": turn.turn_id,
        "occurred_at": datetime.now(UTC).isoformat(),
        "payload": {"turn_id": turn.turn_id, "status": "generating", "metadata": {}},
    }
    with lifecycle_ledger.path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(invalid_event) + "\n")
    with pytest.raises(LedgerCorruptionError):
        VoiceTurnLedger(lifecycle_ledger.path)


def test_legacy_history_migration_is_read_only_stable_and_timestamped(tmp_path: Path) -> None:
    legacy_path = tmp_path / "conversation_history.json"
    legacy_payload = {
        "sessions": {
            "legacy-session": [
                {"role": "user", "content": "今天天气怎么样？"},
                {"role": "assistant", "content": "今天适合散步。"},
                {"role": "user", "content": "记住我喜欢安静。"},
                {"role": "assistant", "content": "好的。"},
            ]
        }
    }
    legacy_path.write_text(json.dumps(legacy_payload, ensure_ascii=False), encoding="utf-8")
    original_bytes = legacy_path.read_bytes()
    first_ledger = make_ledger(tmp_path / "first")
    second_ledger = make_ledger(tmp_path / "second")

    first_result = migrate_legacy_history(legacy_path, first_ledger)
    second_result = second_ledger.migrate_legacy_history(legacy_path)
    first_turns = first_ledger.list_turns(thread_id="legacy-session")
    second_turns = second_ledger.list_turns(thread_id="legacy-session")

    assert legacy_path.read_bytes() == original_bytes
    assert first_result.turn_count == second_result.turn_count == 2
    assert [turn.turn_id for turn in first_turns] == [turn.turn_id for turn in second_turns]
    assert [turn.sequence for turn in first_turns] == [1, 2]
    assert all(turn.created_at.tzinfo is UTC for turn in first_turns)
    assert all(turn.metadata["legacy_timestamp_inferred"] for turn in first_turns)

    # Rolling legacy history may evict the oldest exchange. IDs must be
    # message/content based rather than window-position based, otherwise every
    # remaining exchange would be imported a second time on restart.
    legacy_payload["sessions"]["legacy-session"] = legacy_payload["sessions"][
        "legacy-session"
    ][2:]
    legacy_path.write_text(
        json.dumps(legacy_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    first_ledger.migrate_legacy_history(legacy_path)
    assert len(first_ledger.list_turns(thread_id="legacy-session")) == 2


def test_legacy_migration_resumes_after_crash_between_start_and_commit(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy-crash.json"
    legacy_path.write_text(
        json.dumps(
            {
                "sessions": {
                    "legacy-crash": [
                        {"role": "user", "content": "问题"},
                        {"role": "assistant", "content": "回答"},
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    path = tmp_path / "legacy-crash-ledger.jsonl"
    interrupted = VoiceTurnLedger(path)
    original_commit = interrupted.commit_turn

    def _crash(*_args, **_kwargs):
        raise OSError("simulated crash")

    interrupted.commit_turn = _crash  # type: ignore[method-assign]
    with pytest.raises(OSError, match="simulated crash"):
        interrupted.migrate_legacy_history(legacy_path)
    interrupted.commit_turn = original_commit  # type: ignore[method-assign]

    restarted = VoiceTurnLedger(path)
    pending_turn = restarted.list_turns(thread_id="legacy-crash")[0]
    assert pending_turn.status is TurnStatus.STARTED

    restarted.migrate_legacy_history(legacy_path)
    completed_turn = restarted.get_turn(pending_turn.turn_id)
    assert completed_turn is not None
    assert completed_turn.status is TurnStatus.COMMITTED


def test_concurrent_appends_have_unique_ordered_sequences_and_valid_jsonl(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)

    def start_distinct_thread(index: int):
        thread = ledger.resolve_thread(
            conversation_thread_id=f"thread-{index}",
            channel="voice",
        )
        return ledger.start_turn(
            thread.thread_id,
            turn_id=f"turn-{index}",
            user_text=f"message-{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        turns = list(pool.map(start_distinct_thread, range(40)))

    events = [
        json.loads(line)
        for line in ledger.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_sequences = [event["sequence"] for event in events]

    assert {turn.sequence for turn in turns} == {1}
    assert event_sequences == list(range(1, len(events) + 1))
    assert len({event["event_id"] for event in events}) == len(events)


def test_thread_allows_only_one_non_terminal_turn_and_retry_after_settlement(
    tmp_path: Path,
) -> None:
    ledger = make_ledger(tmp_path)
    thread = ledger.resolve_thread(conversation_thread_id="thread-single-flight")
    first = ledger.start_turn(
        thread.thread_id,
        turn_id="turn-a",
        user_text="A",
    )
    event_count = ledger.event_count

    assert ledger.start_turn(
        thread.thread_id,
        turn_id="turn-a",
        user_text="A",
    ) == first
    assert ledger.event_count == event_count
    with pytest.raises(TurnInProgress) as caught:
        ledger.start_turn(
            thread.thread_id,
            turn_id="turn-b",
            user_text="B",
        )

    assert caught.value.thread_id == thread.thread_id
    assert caught.value.blocking_turn_id == "turn-a"
    assert ledger.event_count == event_count

    ledger.cancel_turn("turn-a", reason="barge_in")
    second = ledger.start_turn(
        thread.thread_id,
        turn_id="turn-b",
        user_text="B",
    )
    assert second.sequence == 2
