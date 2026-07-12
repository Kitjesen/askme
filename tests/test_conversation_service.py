"""Tests for product-facing conversation service governance."""

from __future__ import annotations

import asyncio

import pytest

from askme.api.services.conversation_service import (
    ChatOverloaded,
    ChatTimeout,
    ConversationService,
)


async def test_chat_diagnostics_records_successful_turn() -> None:
    class MemoryHandler:
        def health(self):
            return {
                "enabled": True,
                "backend": "vector",
                "available": True,
                "last_evidence": [{"record_id": "rec-1", "text": "fact"}],
            }

    async def chat_handler(text: str, *, speak: bool = False):
        await asyncio.sleep(0)
        return {"reply": f"reply:{text}", "spoken": speak}

    service = ConversationService(
        chat_handler=chat_handler,
        memory_handler=MemoryHandler(),
        chat_timeout_s=1.0,
        chat_max_concurrency=2,
        chat_diagnostics_history_limit=2,
    )

    payload = await service.chat_payload_from_body(
        {"text": "hello", "speak": True},
        trace_id="trace-1",
    )
    diagnostics = service.diagnostics_snapshot()["chat"]
    metrics = service.metrics_snapshot()["chat"]

    assert payload["reply"] == "reply:hello"
    assert payload["evidence"][0]["record_id"] == "rec-1"
    assert diagnostics["configured"] is True
    assert diagnostics["timeout_s"] == 1.0
    assert diagnostics["max_concurrency"] == 2
    assert diagnostics["in_flight"] == 0
    assert diagnostics["total_turns"] == 1
    assert diagnostics["failures"] == 0
    assert diagnostics["last_turn"]["status"] == "ok"
    assert diagnostics["last_turn"]["trace_id"] == "trace-1"
    assert diagnostics["last_turn"]["text_chars"] == 5
    assert diagnostics["recent_turns"][0]["trace_id"] == "trace-1"
    assert set(diagnostics["last_turn"]["timings_ms"]) >= {
        "parse_ms",
        "handler_ms",
        "response_build_ms",
        "memory_context_ms",
        "total_ms",
    }
    assert metrics["total_turns"] == 1
    assert metrics["last_turn_latency_ms"] == diagnostics["last_turn"]["timings_ms"]["total_ms"]


async def test_chat_payload_passes_conversation_session_id_to_handler() -> None:
    calls: list[dict[str, object]] = []

    async def chat_handler(
        text: str,
        *,
        speak: bool = False,
        conversation_session_id: str | None = None,
        planning_session_id: str | None = None,
        runtime_policy: str = "disabled",
    ):
        calls.append({
            "text": text,
            "speak": speak,
            "conversation_session_id": conversation_session_id,
            "planning_session_id": planning_session_id,
            "runtime_policy": runtime_policy,
        })
        return {"reply": "ok"}

    service = ConversationService(chat_handler=chat_handler)

    payload = await service.chat_payload_from_body(
        {
            "text": "inspect area A",
            "speak": True,
            "conversation_session_id": "conv-1",
            "planning_session_id": "plan-1",
            "runtime_policy": "runtime_first",
        }
    )

    assert payload["reply"] == "ok"
    assert calls == [{
        "text": "inspect area A",
        "speak": True,
        "conversation_session_id": "conv-1",
        "planning_session_id": "plan-1",
        "runtime_policy": "runtime_first",
    }]


async def test_turn_scoped_rag_is_not_overwritten_by_global_memory_health() -> None:
    class MemoryHandler:
        def health(self):
            return {
                "enabled": True,
                "backend": "vector",
                "last_evidence": [{"record_id": "wrong", "text": "wrong fact"}],
                "last_answer_policy": {
                    "state": "stale",
                    "action": "refuse_and_request_update",
                },
            }

    async def chat_handler(text: str):
        return {
            "reply": f"reply:{text}",
            "evidence": [{"record_id": "right", "text": "right fact"}],
            "rag": {
                "turn_scoped": True,
                "answer_policy": {
                    "state": "grounded",
                    "action": "answer_with_evidence",
                },
            },
        }

    payload = await ConversationService(
        chat_handler=chat_handler,
        memory_handler=MemoryHandler(),
    ).chat_payload_from_body({"text": "hello"})

    assert payload["reply"] == "reply:hello"
    assert payload["evidence"][0]["record_id"] == "right"
    assert payload["rag"]["answer_policy"]["state"] == "grounded"


async def test_bound_text_handler_exposes_turn_scoped_rag_to_service() -> None:
    class TextHandler:
        current_turn_rag: dict[str, object] | None = None

        async def process_turn(self, text: str):
            self.current_turn_rag = {
                "evidence": [{"record_id": "bound-turn", "text": text}],
                "rag": {
                    "turn_scoped": True,
                    "answer_policy": {"state": "grounded"},
                },
            }
            return "bound reply"

    class MemoryHandler:
        def health(self):
            return {
                "last_evidence": [{"record_id": "global-wrong"}],
                "last_answer_policy": {"state": "stale", "action": "refuse"},
            }

    handler = TextHandler()
    payload = await ConversationService(
        chat_handler=handler.process_turn,
        memory_handler=MemoryHandler(),
    ).chat_payload_from_body({"text": "hello"})

    assert payload["reply"] == "bound reply"
    assert payload["evidence"][0]["record_id"] == "bound-turn"
    assert payload["rag"]["turn_scoped"] is True


async def test_chat_payload_accepts_conversation_id_aliases() -> None:
    calls: list[dict[str, object]] = []

    async def chat_handler(
        text: str,
        *,
        conversation_session_id: str | None = None,
        planning_session_id: str | None = None,
    ):
        calls.append({
            "text": text,
            "conversation_session_id": conversation_session_id,
            "planning_session_id": planning_session_id,
        })
        return {"reply": "ok"}

    service = ConversationService(chat_handler=chat_handler)

    await service.chat_payload_from_body(
        {"text": "inspect area A", "conversation_id": "conv-a"}
    )
    await service.chat_payload_from_body(
        {"text": "inspect area B", "chat_session_id": "conv-b"}
    )

    assert calls == [
        {
            "text": "inspect area A",
            "conversation_session_id": "conv-a",
            "planning_session_id": None,
        },
        {
            "text": "inspect area B",
            "conversation_session_id": "conv-b",
            "planning_session_id": None,
        },
    ]


async def test_chat_timeout_records_failure_diagnostics() -> None:
    async def chat_handler(text: str, *, speak: bool = False):
        await asyncio.sleep(0.05)
        return {"reply": text}

    service = ConversationService(
        chat_handler=chat_handler,
        chat_timeout_s=0.001,
        chat_max_concurrency=1,
    )

    with pytest.raises(ChatTimeout):
        await service.chat_payload_from_body({"text": "slow"})

    diagnostics = service.diagnostics_snapshot()["chat"]
    assert diagnostics["in_flight"] == 0
    assert diagnostics["total_turns"] == 1
    assert diagnostics["failures"] == 1
    assert diagnostics["timeouts"] == 1
    assert diagnostics["last_turn"]["status"] == "timeout"
    assert diagnostics["last_turn"]["error_type"] == "TimeoutError"


async def test_chat_concurrency_limit_rejects_extra_turns_without_queueing() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def chat_handler(text: str, *, speak: bool = False):
        started.set()
        await release.wait()
        return {"reply": text}

    service = ConversationService(
        chat_handler=chat_handler,
        chat_timeout_s=1.0,
        chat_max_concurrency=1,
    )

    first = asyncio.create_task(service.chat_payload_from_body({"text": "one"}))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    with pytest.raises(ChatOverloaded) as exc:
        await service.chat_payload_from_body({"text": "two"})

    release.set()
    await first

    diagnostics = service.diagnostics_snapshot()["chat"]
    assert exc.value.max_concurrency == 1
    assert diagnostics["overloads"] == 1
    assert diagnostics["total_turns"] == 1
    assert diagnostics["in_flight"] == 0


async def test_chat_diagnostics_tracks_slow_and_recent_turns() -> None:
    async def chat_handler(text: str, *, speak: bool = False):
        await asyncio.sleep(0.01)
        return {"reply": text}

    service = ConversationService(
        chat_handler=chat_handler,
        chat_timeout_s=1.0,
        chat_slow_threshold_ms=1.0,
        chat_diagnostics_history_limit=2,
    )

    await service.chat_payload_from_body({"text": "one"}, trace_id="trace-one")
    await service.chat_payload_from_body({"text": "two"}, trace_id="trace-two")
    await service.chat_payload_from_body({"text": "three"}, trace_id="trace-three")

    diagnostics = service.diagnostics_snapshot()["chat"]
    assert diagnostics["slow_turns_total"] == 3
    assert [item["trace_id"] for item in diagnostics["recent_turns"]] == [
        "trace-two",
        "trace-three",
    ]
    assert [item["trace_id"] for item in diagnostics["slow_turns"]] == [
        "trace-two",
        "trace-three",
    ]
    assert all(item["slow"] is True for item in diagnostics["slow_turns"])
