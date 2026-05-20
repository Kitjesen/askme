from __future__ import annotations

import asyncio

from askme.runtime.module import Runtime
from fastapi.testclient import TestClient

from askme.health_server import create_health_app
from askme.runtime.modules import CognitionModule, MemoryModule, MissionModule
from tests.support.text_loop_harness import make_text_loop


def _runtime_snapshot() -> dict:
    return {
        "app": {"name": "askme", "version": "test"},
        "status": "ok",
        "uptime_seconds": 1.0,
    }


def test_api_chat_keeps_real_cognition_memory_isolated_by_conversation_session_id() -> None:
    runtime = Runtime.use(MemoryModule) + Runtime.use(MissionModule) + Runtime.use(CognitionModule)
    app = asyncio.run(
        runtime.build(
            {
                "cognition": {
                    "sync_enabled": False,
                    "working_memory_retention_seconds": 60,
                }
            }
        )
    )
    cognition = app.modules["cognition"]
    cognition.working_memory.record(
        "note",
        "session A secret",
        conversation_session_id="conv-a",
    )
    cognition.working_memory.record(
        "note",
        "session B route",
        conversation_session_id="conv-b",
    )
    loop, _pipeline = make_text_loop(cognition_handler=cognition)

    async def chat_handler(
        text: str,
        *,
        speak: bool = False,
        conversation_session_id: str | None = None,
        planning_session_id: str | None = None,
        runtime_policy: str = "disabled",
    ):
        reply = await loop.process_turn(
            text,
            speak=speak,
            conversation_session_id=conversation_session_id,
            planning_session_id=planning_session_id,
            runtime_policy=runtime_policy,
        )
        return {
            "reply": reply,
            "cognition": loop.last_cognition_result or {},
        }

    client = TestClient(create_health_app(_runtime_snapshot, chat_handler=chat_handler))

    conv_a = client.post(
        "/api/chat",
        json={"text": "inspect area-a", "conversation_session_id": "conv-a"},
    )
    conv_b = client.post(
        "/api/chat",
        json={"text": "inspect area-b", "conversation_session_id": "conv-b"},
    )

    assert conv_a.status_code == 200
    assert conv_b.status_code == 200
    plan_a = conv_a.json()["cognition"]["plan"]
    plan_b = conv_b.json()["cognition"]["plan"]
    assert plan_a["conversation_session_id"] == "conv-a"
    assert plan_b["conversation_session_id"] == "conv-b"
    assert plan_a["planning_session_id"] != "conv-a"
    assert plan_b["planning_session_id"] != "conv-b"
    assert "session A secret" in plan_a["context"]["working_memory"]
    assert "session B route" not in plan_a["context"]["working_memory"]
    assert "session B route" in plan_b["context"]["working_memory"]
    assert "session A secret" not in plan_b["context"]["working_memory"]

    snapshot = cognition.working_memory.snapshot()
    session_ids = {item["conversation_session_id"] for item in snapshot["items"]}
    assert {"conv-a", "conv-b"}.issubset(session_ids)

    context_a = cognition.working_memory.select_context(conversation_session_id="conv-a")
    context_b = cognition.working_memory.select_context(conversation_session_id="conv-b")
    assert "session A secret" in context_a["text"]
    assert "session B route" not in context_a["text"]
    assert "session B route" in context_b["text"]
    assert "session A secret" not in context_b["text"]
