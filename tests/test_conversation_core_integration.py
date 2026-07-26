"""Integration contract for wiring Conversation Core into the runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from askme.pipeline.brain_pipeline import BrainPipeline
from askme.pipeline.external_turns import record_external_turn

from askme.api.services.conversation_service import ConversationService
from askme.conversation import (
    ConflictingThreadAliases,
    DuplicateEntity,
    InvalidTransition,
    TurnInProgress,
    TurnStatus,
    VoiceTurnLedger,
)
from askme.memory.core.conversation import ConversationManager
from askme.pipeline.channels.external_turns import (
    begin_external_turn,
    cancel_external_turn,
    complete_external_turn,
    discard_external_generation,
)
from askme.voice_gateway import VoiceGatewayService


class _RecordingLedger:
    """Small protocol fake that records lifecycle operations by business meaning."""

    def __init__(self) -> None:
        self.resolved: list[dict[str, Any]] = []
        self.started: list[dict[str, Any]] = []
        self.committed: list[dict[str, Any]] = []
        self.cancelled: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []
        self.suppressed: list[dict[str, Any]] = []

    def resolve_thread(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        self.resolved.append(dict(kwargs))
        thread_id = (
            kwargs.get("thread_id")
            or kwargs.get("conversation_thread_id")
            or kwargs.get("conversation_session_id")
            or (args[0] if args else None)
            or "thread-generated"
        )
        return SimpleNamespace(thread_id=thread_id)

    def start_turn(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        call = dict(kwargs)
        if args:
            call.setdefault("thread_id", args[0])
        self.started.append(call)
        return SimpleNamespace(
            turn_id=call.get("turn_id") or "turn-generated",
            thread_id=call.get("thread_id"),
            status=TurnStatus.STARTED,
            assistant_text="",
        )

    def commit_turn(self, *args: Any, **kwargs: Any) -> None:
        self.committed.append(self._terminal_call(args, kwargs))

    def cancel_turn(self, *args: Any, **kwargs: Any) -> None:
        self.cancelled.append(self._terminal_call(args, kwargs))

    def fail_turn(self, *args: Any, **kwargs: Any) -> None:
        self.failed.append(self._terminal_call(args, kwargs))

    def suppress_turn(self, *args: Any, **kwargs: Any) -> None:
        self.suppressed.append(self._terminal_call(args, kwargs))

    @staticmethod
    def _terminal_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        call = dict(kwargs)
        if args:
            call.setdefault("turn_id", args[0])
        return call


class _TurnExecutor:
    def __init__(
        self,
        *,
        result: str = "回答",
        error: Exception | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.cancel_event = cancel_event
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.last_spoken_text = ""
        self.current_turn_rag = None

    async def process(self, user_text: str, **kwargs: Any) -> str:
        self.calls.append((user_text, kwargs))
        if self.error is not None:
            raise self.error
        if self.cancel_event is not None:
            self.cancel_event.set()
        return self.result

    def clear_turn_context(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


def _pipeline(
    executor: _TurnExecutor,
    *,
    ledger: Any | None = None,
    conversation: Any | None = None,
    legacy_fallback: bool = False,
) -> BrainPipeline:
    stream_processor = MagicMock()
    skill_gate = MagicMock()
    skill_gate.last_spoken_text = ""
    return BrainPipeline(
        llm=MagicMock(),
        conversation=(
            conversation
            if conversation is not None
            else MagicMock(name="legacy_conversation_manager")
        ),
        memory=MagicMock(),
        tools=MagicMock(),
        skill_manager=MagicMock(),
        skill_executor=MagicMock(),
        audio=MagicMock(),
        splitter=MagicMock(),
        stream_processor=stream_processor,
        skill_gate=skill_gate,
        turn_executor=executor,
        turn_ledger=ledger,
        conversation_core_legacy_fallback=legacy_fallback,
    )


async def test_brain_pipeline_commits_successful_turn_and_preserves_executor_contract() -> None:
    ledger = _RecordingLedger()
    executor = _TurnExecutor(result="温度正常")
    pipeline = _pipeline(executor, ledger=ledger)

    reply = await pipeline.process(
        "检查温度",
        source="voice",
        conversation_session_id="thread-42",
        voice_turn_id="turn-42",
    )

    assert reply == "温度正常"
    assert executor.calls[0][1]["conversation_session_id"] == "thread-42"
    assert ledger.started[0]["thread_id"] == "thread-42"
    assert ledger.started[0]["turn_id"] == "turn-42"
    assert ledger.started[0]["user_text"] == "检查温度"
    assert ledger.committed[0]["turn_id"] == "turn-42"
    assert ledger.committed[0]["assistant_text"] == "温度正常"
    assert not ledger.cancelled
    assert not ledger.failed
    assert not ledger.suppressed


async def test_brain_pipeline_fails_closed_before_executor_for_erased_thread(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "erased-pipeline.jsonl")
    thread = ledger.resolve_thread(conversation_thread_id="thread-erased-pipeline")
    ledger.transition_thread(thread.thread_id, "erased")
    executor = _TurnExecutor(result="不得生成的新回答")
    pipeline = _pipeline(executor, ledger=ledger)

    with pytest.raises(InvalidTransition, match="erased"):
        await pipeline.process(
            "敏感新问题",
            source="voice",
            conversation_session_id=thread.thread_id,
            voice_turn_id="turn-after-erasure",
        )

    assert executor.calls == []
    pipeline._conversation.add_user_message.assert_not_called()
    pipeline._conversation.add_assistant_message.assert_not_called()


async def test_brain_pipeline_turn_id_retry_does_not_generate_a_second_answer(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "pipeline-idempotency.jsonl")
    executor = _TurnExecutor(result="first and only answer")
    pipeline = _pipeline(executor, ledger=ledger)

    first = await pipeline.process(
        "same request",
        source="voice",
        conversation_session_id="thread-idempotency",
        voice_turn_id="stable-voice-turn",
    )
    retry = await pipeline.process(
        "same request",
        source="voice",
        conversation_session_id="thread-idempotency",
        voice_turn_id="stable-voice-turn",
    )

    assert first == retry == "first and only answer"
    assert len(executor.calls) == 1
    turns = ledger.list_turns(thread_id="thread-idempotency")
    assert len(turns) == 1
    assert turns[0].assistant_text == "first and only answer"
    assert len(ledger.list_generations(turn_id=turns[0].turn_id)) == 1

    with pytest.raises(DuplicateEntity, match="user_text conflicts"):
        await pipeline.process(
            "different request",
            source="voice",
            conversation_session_id="thread-idempotency",
            voice_turn_id="stable-voice-turn",
        )
    assert len(executor.calls) == 1


async def test_in_flight_erasure_clears_late_legacy_projection(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "in-flight-erasure.jsonl")
    conversation = ConversationManager(
        history_file=tmp_path / "legacy-history.json",
        config={"conversation": {}},
    )
    started = asyncio.Event()
    release = asyncio.Event()

    class _BlockingLegacyExecutor(_TurnExecutor):
        async def process(self, user_text: str, **kwargs: Any) -> str:
            session_id = str(kwargs["conversation_session_id"])
            conversation.add_user_message(
                user_text,
                conversation_session_id=session_id,
            )
            started.set()
            await release.wait()
            conversation.add_assistant_message(
                "SECRET LATE ANSWER",
                conversation_session_id=session_id,
            )
            return "SECRET LATE ANSWER"

    executor = _BlockingLegacyExecutor()
    pipeline = _pipeline(executor, ledger=ledger, conversation=conversation)
    task = asyncio.create_task(
        pipeline.process(
            "SECRET USER",
            source="voice",
            conversation_session_id="thread-in-flight-erasure",
            voice_turn_id="turn-in-flight-erasure",
        )
    )

    await asyncio.wait_for(started.wait(), timeout=1.0)
    ledger.transition_thread("thread-in-flight-erasure", "erased")
    release.set()

    with pytest.raises(InvalidTransition, match="erased"):
        await task
    await asyncio.sleep(0.05)

    assert conversation.get_messages(
        "system",
        conversation_session_id="thread-in-flight-erasure",
    ) == [{"role": "system", "content": "system"}]
    persisted = (tmp_path / "legacy-history.json").read_text(encoding="utf-8")
    assert "SECRET USER" not in persisted
    assert "SECRET LATE ANSWER" not in persisted


async def test_same_thread_turns_are_serialized_without_blocking_other_threads(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "thread-serialization.jsonl")
    conversation = ConversationManager(
        history_file=tmp_path / "thread-history.json",
        config={"conversation": {}},
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    execution_order: list[str] = []
    prompt_roles: dict[str, list[tuple[str, str]]] = {}

    class _HistoryExecutor(_TurnExecutor):
        async def process(self, user_text: str, **kwargs: Any) -> str:
            session_id = str(kwargs["conversation_session_id"])
            execution_order.append(f"user:{user_text}")
            conversation.add_user_message(
                user_text,
                conversation_session_id=session_id,
            )
            prompt_roles[user_text] = [
                (str(item.get("role")), str(item.get("content")))
                for item in conversation.get_messages(
                    "system",
                    conversation_session_id=session_id,
                )[1:]
            ]
            if user_text == "A":
                first_started.set()
                await release_first.wait()
            reply = f"answer-{user_text}"
            conversation.add_assistant_message(
                reply,
                conversation_session_id=session_id,
            )
            execution_order.append(f"assistant:{user_text}")
            return reply

    executor = _HistoryExecutor()
    pipeline = _pipeline(executor, ledger=ledger, conversation=conversation)
    first = asyncio.create_task(
        pipeline.process(
            "A",
            source="text",
            conversation_session_id="shared-thread",
            voice_turn_id="turn-a",
        )
    )
    await asyncio.wait_for(first_started.wait(), timeout=1.0)
    second = asyncio.create_task(
        pipeline.process(
            "B",
            source="text",
            conversation_session_id="shared-thread",
            voice_turn_id="turn-b",
        )
    )
    await asyncio.sleep(0.02)

    assert execution_order == ["user:A"]
    release_first.set()
    assert await asyncio.gather(first, second) == ["answer-A", "answer-B"]
    assert execution_order == ["user:A", "assistant:A", "user:B", "assistant:B"]
    assert prompt_roles["B"] == [
        ("user", "A"),
        ("assistant", "answer-A"),
        ("user", "B"),
    ]

    final_history = conversation.get_messages(
        "system",
        conversation_session_id="shared-thread",
    )[1:]
    assert [(item["role"], item["content"]) for item in final_history] == [
        ("user", "A"),
        ("assistant", "answer-A"),
        ("user", "B"),
        ("assistant", "answer-B"),
    ]


async def test_active_local_turn_rejects_cross_path_external_interleaving(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "cross-path-single-flight.jsonl")
    conversation = ConversationManager(
        history_file=tmp_path / "cross-path-history.json",
        config={"conversation": {}},
    )
    started = asyncio.Event()
    release = asyncio.Event()

    class _BlockingExecutor(_TurnExecutor):
        async def process(self, user_text: str, **kwargs: Any) -> str:
            session_id = str(kwargs["conversation_session_id"])
            conversation.add_user_message(
                user_text,
                conversation_session_id=session_id,
            )
            started.set()
            await release.wait()
            conversation.add_assistant_message(
                "answer-A",
                conversation_session_id=session_id,
            )
            return "answer-A"

    pipeline = _pipeline(
        _BlockingExecutor(),
        ledger=ledger,
        conversation=conversation,
    )
    local_turn = asyncio.create_task(
        pipeline.process(
            "A",
            source="text",
            conversation_session_id="shared-cross-path-thread",
            voice_turn_id="turn-a",
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)

    with pytest.raises(TurnInProgress) as caught:
        record_external_turn(
            pipeline,
            "B",
            "answer-B",
            source="volcengine_realtime",
            conversation_session_id="shared-cross-path-thread",
            turn_id="turn-b",
        )

    assert caught.value.blocking_turn_id == "turn-a"
    assert conversation.get_messages(
        "system",
        conversation_session_id="shared-cross-path-thread",
    )[1:] == [{"role": "user", "content": "A"}]

    release.set()
    assert await local_turn == "answer-A"
    record_external_turn(
        pipeline,
        "B",
        "answer-B",
        source="volcengine_realtime",
        conversation_session_id="shared-cross-path-thread",
        turn_id="turn-b",
    )

    final_history = conversation.get_messages(
        "system",
        conversation_session_id="shared-cross-path-thread",
    )[1:]
    assert [(item["role"], item["content"]) for item in final_history] == [
        ("user", "A"),
        ("assistant", "answer-A"),
        ("user", "B"),
        ("assistant", "answer-B"),
    ]


async def test_brain_pipeline_records_cancelled_empty_turn() -> None:
    cancel_event = asyncio.Event()
    ledger = _RecordingLedger()
    pipeline = _pipeline(
        _TurnExecutor(result="", cancel_event=cancel_event),
        ledger=ledger,
    )

    reply = await pipeline.process(
        "继续说",
        source="voice",
        conversation_session_id="thread-cancel",
        voice_turn_id="turn-cancel",
        turn_cancel_token=cancel_event,
    )

    assert reply == ""
    assert ledger.cancelled[0]["turn_id"] == "turn-cancel"
    assert not ledger.committed
    assert not ledger.failed
    assert not ledger.suppressed


async def test_delivered_nonempty_result_wins_over_late_cancellation() -> None:
    cancel_event = asyncio.Event()
    ledger = _RecordingLedger()
    pipeline = _pipeline(
        _TurnExecutor(result="迟到回答", cancel_event=cancel_event),
        ledger=ledger,
    )

    reply = await pipeline.process(
        "停止",
        source="voice",
        conversation_session_id="thread-cancel-late",
        voice_turn_id="turn-cancel-late",
        turn_cancel_token=cancel_event,
    )

    assert reply == "迟到回答"
    assert ledger.committed[0]["turn_id"] == "turn-cancel-late"
    assert not ledger.cancelled


async def test_barge_in_after_delivery_commits_legacy_assistant_and_returns_it(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "barge-in-linearization.jsonl")
    conversation = ConversationManager(
        history_file=tmp_path / "barge-in-history.json",
        config={"conversation": {}},
    )
    assistant_written = asyncio.Event()
    release = asyncio.Event()

    class _LateExecutor(_TurnExecutor):
        async def process(self, user_text: str, **kwargs: Any) -> str:
            session_id = str(kwargs["conversation_session_id"])
            conversation.add_user_message(
                user_text,
                conversation_session_id=session_id,
            )
            conversation.add_assistant_message(
                "UNHEARD LATE ANSWER",
                conversation_session_id=session_id,
            )
            assistant_written.set()
            await release.wait()
            return "UNHEARD LATE ANSWER"

    pipeline = _pipeline(
        _LateExecutor(),
        ledger=ledger,
        conversation=conversation,
    )
    task = asyncio.create_task(
        pipeline.process(
            "please answer",
            source="voice",
            conversation_session_id="thread-barge-in",
            voice_turn_id="turn-barge-in",
        )
    )
    await asyncio.wait_for(assistant_written.wait(), timeout=1.0)
    assert pipeline.cancel_active_turn(reason="barge_in") is True
    release.set()

    assert await task == "UNHEARD LATE ANSWER"
    await asyncio.sleep(0.05)
    turn = ledger.get_turn("turn-barge-in")
    assert turn is not None
    assert turn.status.value == "committed"
    assert turn.assistant_text == turn.heard_text == "UNHEARD LATE ANSWER"
    assert conversation.get_messages(
        "system",
        conversation_session_id="thread-barge-in",
    )[1:] == [
        {"role": "user", "content": "please answer"},
        {"role": "assistant", "content": "UNHEARD LATE ANSWER"},
    ]
    persisted = (tmp_path / "barge-in-history.json").read_text(encoding="utf-8")
    assert "UNHEARD LATE ANSWER" in persisted


async def test_brain_pipeline_records_failure_and_reraises() -> None:
    ledger = _RecordingLedger()
    pipeline = _pipeline(
        _TurnExecutor(error=RuntimeError("provider unavailable")),
        ledger=ledger,
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await pipeline.process(
            "现在状态",
            source="voice",
            conversation_session_id="thread-fail",
            voice_turn_id="turn-fail",
        )

    assert ledger.failed[0]["turn_id"] == "turn-fail"
    assert not ledger.committed
    assert not ledger.cancelled


async def test_brain_pipeline_without_ledger_keeps_legacy_process_behavior() -> None:
    executor = _TurnExecutor(result="旧路径仍可用")
    pipeline = _pipeline(executor, ledger=None)

    reply = await pipeline.process(
        "兼容吗",
        source="text",
        conversation_session_id="legacy-session",
    )

    assert reply == "旧路径仍可用"
    assert executor.calls[0][1]["conversation_session_id"] == "legacy-session"


async def test_brain_pipeline_surfaces_degraded_conversation_core_health() -> None:
    class _FailingLedger:
        def resolve_thread(self, **_: Any) -> None:
            raise OSError("disk unavailable")

    pipeline = _pipeline(
        _TurnExecutor(result="仍然回答"),
        ledger=_FailingLedger(),
        legacy_fallback=True,
    )

    assert await pipeline.process("状态", source="text") == "仍然回答"
    assert pipeline.conversation_core_health() == {
        "enabled": True,
        "status": "degraded",
        "write_failures": 1,
        "last_error_type": "OSError",
    }


class _LegacyConversation:
    def __init__(self) -> None:
        self.user_messages: list[str] = []
        self.assistant_messages: list[str] = []

    def add_user_message(self, content: str) -> None:
        self.user_messages.append(content)

    def add_assistant_message(self, content: str) -> None:
        self.assistant_messages.append(content)


def test_external_turn_commits_to_ledger_and_keeps_legacy_history() -> None:
    ledger = _RecordingLedger()
    conversation = _LegacyConversation()
    pipeline = SimpleNamespace(
        _turn_ledger=ledger,
        _conversation=conversation,
        _episodic=None,
    )

    record_external_turn(
        pipeline,
        "门口有人吗",
        "门口检测到一人",
        source="realtime",
        conversation_session_id="thread-external",
    )

    assert conversation.user_messages == ["门口有人吗"]
    assert conversation.assistant_messages == ["门口检测到一人"]
    assert ledger.started[0]["thread_id"] == "thread-external"
    assert ledger.started[0]["source"] == "realtime"
    assert ledger.committed[0]["assistant_text"] == "门口检测到一人"


def test_external_turn_without_ledger_keeps_legacy_history() -> None:
    conversation = _LegacyConversation()
    pipeline = SimpleNamespace(_conversation=conversation, _episodic=None)

    record_external_turn(pipeline, "你好", "你好，我在")

    assert conversation.user_messages == ["你好"]
    assert conversation.assistant_messages == ["你好，我在"]


def test_pipeline_module_builds_default_persistent_ledger_and_injects_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from askme.runtime.modules import pipeline_module

    ledgers: list[Any] = []
    migrations: list[Path] = []

    class _PersistentLedger:
        def __init__(self, path: str | Path, **_: Any) -> None:
            self.path = Path(path)
            ledgers.append(self)

        def migrate_legacy_history(self, path: str | Path) -> SimpleNamespace:
            migrations.append(Path(path))
            return SimpleNamespace(turn_count=2)

    class _BrainPipeline:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(pipeline_module, "VoiceTurnLedger", _PersistentLedger, raising=False)
    monkeypatch.setattr(pipeline_module, "BrainPipeline", _BrainPipeline)
    monkeypatch.setattr(pipeline_module, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(pipeline_module, "_load_soul_seed", lambda _cfg: [])
    legacy_path = tmp_path / "data" / "conversation_history.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text("{}", encoding="utf-8")

    module = pipeline_module.PipelineModule()
    module.llm_in = SimpleNamespace(client=MagicMock())
    module.memory_context = SimpleNamespace(
        conversation=MagicMock(),
        memory_bridge=MagicMock(),
        session_memory=None,
        episodic=None,
        memory_system=None,
    )
    module.tool_registry_in = SimpleNamespace(registry=MagicMock())
    module.safety_client = SimpleNamespace(client=None)
    module.vision = None
    module.control_in = SimpleNamespace(client=None)

    module.build({"brain": {"system_prompt": "test"}}, MagicMock())

    assert len(ledgers) == 1
    assert ledgers[0].path == tmp_path / "data" / "conversation" / "turn_ledger.jsonl"
    assert migrations == [legacy_path]
    assert module.brain_pipeline.kwargs["turn_ledger"] is ledgers[0]
    consumer_status = module.memory_consumer.status()
    assert consumer_status.processing_allowed is False
    assert consumer_status.blocked_reason == "erasure_deletion_unsupported"


def test_pipeline_module_prefers_conversation_history_path(tmp_path: Path) -> None:
    from askme.runtime.modules.pipeline_module import _legacy_conversation_history_path

    configured = _legacy_conversation_history_path(
        {
            "conversation": {"history_file": str(tmp_path / "conversation.json")},
            "memory": {"history_file": str(tmp_path / "memory.json")},
        }
    )

    assert configured == tmp_path / "conversation.json"


def test_voice_gateway_exposes_canonical_thread_id_and_rejects_alias_conflicts() -> None:
    class _Bridge:
        def handle_text_input(self, text: str, **_: Any) -> dict[str, Any]:
            return {"handled": True, "turn": {"spoken_reply": f"reply:{text}"}}

        def status_snapshot(self) -> dict[str, Any]:
            return {"enabled": True}

    gateway = VoiceGatewayService(_Bridge())

    result = gateway.handle_text_input(
        "你好",
        conversation_thread_id="thread-canonical",
        include_session=True,
    )

    assert result is not None
    assert result["conversation_thread_id"] == "thread-canonical"
    assert result["conversation_session_id"] == "thread-canonical"
    with pytest.raises(ConflictingThreadAliases):
        gateway.handle_text_input(
            "不能串线",
            conversation_thread_id="thread-a",
            conversation_session_id="thread-b",
        )


async def test_chat_api_normalizes_new_thread_id_and_returns_both_contracts() -> None:
    calls: list[str | None] = []

    async def _handler(
        text: str,
        *,
        conversation_session_id: str | None = None,
    ) -> dict[str, str]:
        calls.append(conversation_session_id)
        return {"reply": f"reply:{text}"}

    service = ConversationService(chat_handler=_handler)
    result = await service.chat_payload_from_body(
        {"text": "状态", "conversation_thread_id": "thread-api"}
    )

    assert calls == ["thread-api"]
    assert result["conversation_thread_id"] == "thread-api"
    assert result["conversation_session_id"] == "thread-api"


async def test_chat_api_rejects_conflicting_canonical_thread_aliases() -> None:
    service = ConversationService(chat_handler=lambda _text: {"reply": "unused"})

    with pytest.raises(ConflictingThreadAliases):
        await service.chat_payload_from_body(
            {
                "text": "不能串线",
                "conversation_thread_id": "thread-a",
                "thread_id": "thread-b",
            }
        )


async def test_chat_api_accepts_legacy_session_id_alias() -> None:
    calls: list[str | None] = []

    async def _handler(
        _text: str,
        *,
        conversation_session_id: str | None = None,
    ) -> dict[str, str]:
        calls.append(conversation_session_id)
        return {"reply": "ok"}

    service = ConversationService(chat_handler=_handler)
    result = await service.chat_payload_from_body(
        {"text": "继续", "session_id": "thread-session-alias"}
    )
    voice_turn = service.voice_turn_payload_from_body(
        {"voice": True, "session_id": "thread-session-alias"},
        text="继续",
    )

    assert calls == ["thread-session-alias"]
    assert result["conversation_thread_id"] == "thread-session-alias"
    assert voice_turn is not None
    assert voice_turn["conversation_thread_id"] == "thread-session-alias"


def test_voice_gateway_accepts_all_legacy_thread_aliases() -> None:
    class _Bridge:
        def handle_text_input(self, text: str, **_: Any) -> dict[str, Any]:
            return {"handled": True, "turn": {"spoken_reply": text}}

        def status_snapshot(self) -> dict[str, Any]:
            return {"enabled": True}

    gateway = VoiceGatewayService(_Bridge())

    by_conversation = gateway.handle_text_input(
        "一",
        conversation_id="thread-legacy",
        include_session=True,
    )
    by_chat_session = gateway.handle_text_input(
        "二",
        chat_session_id="thread-legacy",
        include_session=True,
    )

    assert by_conversation is not None
    assert by_chat_session is not None
    assert by_conversation["conversation_thread_id"] == "thread-legacy"
    assert by_chat_session["conversation_thread_id"] == "thread-legacy"


async def test_thread_survives_switch_from_pipeline_to_realtime_provider(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "cross-path-ledger.jsonl")
    pipeline = _pipeline(_TurnExecutor(result="本地回答"), ledger=ledger)

    await pipeline.process(
        "先走本地",
        source="text",
        conversation_session_id="thread-cross-path",
    )
    record_external_turn(
        pipeline,
        "再走云端",
        "实时回答",
        source="volcengine_realtime",
        conversation_session_id="thread-cross-path",
        provider="volcengine",
        provider_generation_id="generation-2",
    )

    turns = ledger.list_turns(thread_id="thread-cross-path")
    assert len(ledger.list_threads()) == 1
    assert [turn.source for turn in turns] == ["text", "volcengine_realtime"]
    local_generations = ledger.list_generations(turn_id=turns[0].turn_id)
    assert len(local_generations) == 1
    assert local_generations[0].provider == "askme_pipeline"
    assert local_generations[0].status.value == "approved"
    assert len(ledger.list_generations(turn_id=turns[-1].turn_id)) == 1


def test_interrupted_realtime_turn_truncates_generation_and_projects_heard_prefix(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "interrupted-realtime.jsonl")
    conversation = _LegacyConversation()
    pipeline = SimpleNamespace(
        _turn_ledger=ledger,
        _conversation=conversation,
        _episodic=None,
    )
    handle = begin_external_turn(
        pipeline,
        "介绍园区",
        source="volcengine_realtime",
        conversation_session_id="thread-interrupted",
        turn_id="turn-interrupted",
        provider="volcengine",
        provider_session_id="provider-session-a",
        provider_generation_id="provider-generation-4",
        response_text="园区分为办公区和生产区。",
    )

    assert handle is not None
    cancel_external_turn(
        pipeline,
        handle,
        user_text="介绍园区",
        source="volcengine_realtime",
        reason="barge_in",
        played_ms=640,
        heard_text="园区分为办公区",
    )

    turn = ledger.get_turn("turn-interrupted")
    assert turn is not None
    assert turn.playback_disposition == "truncate_played"
    assert turn.assistant_text == "园区分为办公区"
    generation = ledger.get_generation(handle.generation_id or "")
    assert generation is not None
    assert generation.played_ms == 640
    assert conversation.user_messages == ["介绍园区"]
    assert conversation.assistant_messages == ["园区分为办公区"]


def test_late_provider_completion_cannot_overwrite_cancelled_legacy_projection(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "cancel-wins.jsonl")
    conversation = _LegacyConversation()
    pipeline = SimpleNamespace(
        _turn_ledger=ledger,
        _conversation=conversation,
        _episodic=None,
    )
    handle = begin_external_turn(
        pipeline,
        "介绍园区",
        source="volcengine_realtime",
        conversation_session_id="thread-race",
        turn_id="turn-race",
        provider="volcengine",
        response_text="完整但未全部听到的回答",
    )

    cancel_external_turn(
        pipeline,
        handle,
        user_text="介绍园区",
        source="volcengine_realtime",
        reason="barge_in",
        played_ms=240,
        heard_text="已听前缀",
    )
    complete_external_turn(
        pipeline,
        handle,
        user_text="介绍园区",
        assistant_text="完整但未全部听到的回答",
        source="volcengine_realtime",
    )

    assert conversation.user_messages == ["介绍园区"]
    assert conversation.assistant_messages == ["已听前缀"]
    turn = ledger.get_turn("turn-race")
    assert turn is not None
    assert turn.status.value == "cancelled"


def test_erasure_blocks_late_external_completion_and_legacy_projection(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "erase-vs-completion.jsonl")
    conversation = _LegacyConversation()
    pipeline = SimpleNamespace(
        _turn_ledger=ledger,
        _conversation=conversation,
        _episodic=None,
    )
    handle = begin_external_turn(
        pipeline,
        "SECRET USER",
        source="volcengine_realtime",
        conversation_session_id="thread-erasure-race",
        provider="volcengine",
        response_text="SECRET DRAFT",
    )
    assert handle is not None

    ledger.transition_thread(handle.thread_id, "erased")
    complete_external_turn(
        pipeline,
        handle,
        user_text="SECRET USER",
        assistant_text="SECRET LATE ANSWER",
        source="volcengine_realtime",
        conversation_session_id=handle.thread_id,
    )

    erased_turn = ledger.get_turn(handle.turn_id)
    erased_generation = ledger.get_generation(handle.generation_id or "")
    assert erased_turn is not None
    assert erased_generation is not None
    assert erased_turn.status.value == "cancelled"
    assert erased_turn.user_text == erased_turn.assistant_text == ""
    assert erased_generation.response_text == ""
    assert conversation.user_messages == []
    assert conversation.assistant_messages == []

    with pytest.raises(InvalidTransition, match="erased"):
        begin_external_turn(
            pipeline,
            "SECOND SECRET",
            source="runtime",
            conversation_session_id=handle.thread_id,
        )
    assert conversation.user_messages == []


def test_external_turn_write_outage_degrades_health_and_keeps_legacy_context() -> None:
    class _FailingLedger:
        def resolve_thread(self, **_: Any) -> None:
            raise OSError("ledger offline")

    conversation = _LegacyConversation()
    failures: list[tuple[str, str]] = []
    pipeline = SimpleNamespace(
        _turn_ledger=_FailingLedger(),
        _conversation=conversation,
        _episodic=None,
        _conversation_core_legacy_fallback=True,
        _record_turn_ledger_failure=lambda operation, exc: failures.append(
            (operation, type(exc).__name__)
        ),
    )

    handle = begin_external_turn(
        pipeline,
        "状态",
        source="runtime",
        conversation_session_id="thread-outage",
    )
    complete_external_turn(
        pipeline,
        handle,
        user_text="状态",
        assistant_text="兼容上下文仍可用",
        source="runtime",
    )

    assert handle is None
    assert failures == [("begin an external turn", "OSError")]
    assert conversation.user_messages == ["状态"]
    assert conversation.assistant_messages == ["兼容上下文仍可用"]


def test_external_generation_write_failure_still_settles_durable_turn(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "generation-outage.jsonl")
    conversation = _LegacyConversation()
    failures: list[tuple[str, str]] = []
    pipeline = SimpleNamespace(
        _turn_ledger=ledger,
        _conversation=conversation,
        _episodic=None,
        _conversation_core_legacy_fallback=True,
        _record_turn_ledger_failure=lambda operation, exc: failures.append(
            (operation, type(exc).__name__)
        ),
    )

    def _fail_generation(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("generation storage unavailable")

    ledger.start_generation = _fail_generation  # type: ignore[method-assign]
    handle = begin_external_turn(
        pipeline,
        "状态",
        source="volcengine_realtime",
        conversation_session_id="thread-generation-outage",
        provider="volcengine",
    )

    assert handle is not None
    assert handle.generation_id is None
    assert ledger.get_turn(handle.turn_id).status.value == "started"

    complete_external_turn(
        pipeline,
        handle,
        user_text="状态",
        assistant_text="已经播出的事实仍被提交",
        source="volcengine_realtime",
    )

    settled = ledger.get_turn(handle.turn_id)
    assert settled is not None
    assert settled.status.value == "committed"
    assert settled.assistant_text == "已经播出的事实仍被提交"
    assert failures == [("start an external generation", "OSError")]
    assert conversation.assistant_messages == ["已经播出的事实仍被提交"]


async def test_rejected_realtime_generation_falls_back_within_same_turn(
    tmp_path: Path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "realtime-fallback.jsonl")
    pipeline = _pipeline(_TurnExecutor(result="级联回答"), ledger=ledger)
    handle = begin_external_turn(
        pipeline,
        "继续",
        source="volcengine_realtime",
        conversation_session_id="thread-fallback",
        turn_id="turn-fallback",
        provider="volcengine",
        provider_generation_id="provider-generation-8",
    )

    assert handle is not None
    discard_external_generation(
        pipeline,
        handle,
        reason="response_without_audio",
    )
    reply = await pipeline.process(
        "继续",
        source="voice",
        conversation_session_id="thread-fallback",
        voice_turn_id="turn-fallback",
    )

    assert reply == "级联回答"
    turn = ledger.get_turn("turn-fallback")
    assert turn is not None
    assert turn.status.value == "committed"
    generations = ledger.list_generations(turn_id="turn-fallback")
    assert [generation.status.value for generation in generations] == [
        "discarded",
        "approved",
    ]
    assert generations[-1].provider == "askme_pipeline"
