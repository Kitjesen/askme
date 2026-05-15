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
