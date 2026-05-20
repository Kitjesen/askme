from __future__ import annotations

from fastapi.testclient import TestClient

from askme.health_server import AskmeHealthServer, create_health_app
from tests.support.text_loop_harness import make_text_loop


def _runtime_snapshot() -> dict:
    return {
        "app": {"name": "askme", "version": "test"},
        "status": "ok",
        "uptime_seconds": 1.0,
    }


def test_askme_health_server_chat_wrapper_forwards_session_and_runtime_policy() -> None:
    seen: dict[str, object] = {}

    async def chat_handler(
        text: str,
        *,
        speak: bool = False,
        conversation_session_id: str | None = None,
        planning_session_id: str | None = None,
        runtime_policy: str = "disabled",
    ):
        seen.update({
            "text": text,
            "speak": speak,
            "conversation_session_id": conversation_session_id,
            "planning_session_id": planning_session_id,
            "runtime_policy": runtime_policy,
        })
        return {"reply": "ok", "spoken": speak}

    server = AskmeHealthServer({}, health_provider=lambda: _runtime_snapshot())
    server.set_chat_handler(chat_handler)
    client = TestClient(server._app)

    response = client.post(
        "/api/chat",
        json={
            "text": "inspect area A",
            "speak": True,
            "conversation_id": "conv-1",
            "planning_session_id": "plan-1",
            "runtime_policy": "runtime_first",
        },
    )

    assert response.status_code == 200
    assert seen == {
        "text": "inspect area A",
        "speak": True,
        "conversation_session_id": "conv-1",
        "planning_session_id": "plan-1",
        "runtime_policy": "runtime_first",
    }


def test_chat_endpoint_keeps_text_loop_plans_scoped_by_conversation_session_id() -> None:
    class Cognition:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def plan_from_payload(
            self,
            payload: dict[str, object],
        ) -> dict[str, object]:
            self.calls.append(dict(payload))
            conversation_session_id = str(payload.get("conversation_session_id") or "")
            if payload.get("operator_confirmation") is True:
                return {
                    "planned": True,
                    "plan": {
                        "planning_session_id": payload.get("planning_session_id"),
                        "interaction_state": "ready_for_arbiter",
                        "next_prompt": f"ready {conversation_session_id}",
                        "handoff_ready": True,
                    },
                }
            return {
                "planned": True,
                "plan": {
                    "planning_session_id": f"plan-{conversation_session_id}",
                    "interaction_state": "awaiting_confirmation",
                    "next_prompt": f"confirm {conversation_session_id}",
                    "handoff_ready": False,
                },
            }

    cognition = Cognition()
    loop, _ = make_text_loop(cognition_handler=cognition)

    async def chat_handler(
        text: str,
        *,
        speak: bool = False,
        conversation_session_id: str | None = None,
        planning_session_id: str | None = None,
        runtime_policy: str = "disabled",
    ):
        return await loop.process_turn(
            text,
            speak=speak,
            conversation_session_id=conversation_session_id,
            planning_session_id=planning_session_id,
            runtime_policy=runtime_policy,
        )

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            chat_handler=chat_handler,
        )
    )

    first = client.post(
        "/api/chat",
        json={"text": "inspect area a", "conversation_session_id": "conv-a"},
    )
    second = client.post(
        "/api/chat",
        json={"text": "inspect area b", "conversation_session_id": "conv-b"},
    )
    confirm_a = client.post(
        "/api/chat",
        json={"text": "confirm", "conversation_session_id": "conv-a"},
    )
    confirm_b = client.post(
        "/api/chat",
        json={"text": "confirm", "conversation_session_id": "conv-b"},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert confirm_a.status_code == 200
    assert confirm_b.status_code == 200
    assert first.json()["reply"] == "confirm conv-a"
    assert second.json()["reply"] == "confirm conv-b"
    assert confirm_a.json()["reply"] == "ready conv-a"
    assert confirm_b.json()["reply"] == "ready conv-b"
    assert "planning_session_id" not in cognition.calls[0]
    assert "planning_session_id" not in cognition.calls[1]
    assert cognition.calls[2]["planning_session_id"] == "plan-conv-a"
    assert cognition.calls[3]["planning_session_id"] == "plan-conv-b"


def test_runtime_voice_turn_forwards_conversation_and_planning_session_ids() -> None:
    class DummyRuntimeHandler:
        def __init__(self):
            self.seen = {}

        def voice_turn_payload(
            self,
            text,
            *,
            conversation_session_id=None,
            planning_session_id=None,
            **kwargs,
        ):
            self.seen = {
                "text": text,
                "conversation_session_id": conversation_session_id,
                "planning_session_id": planning_session_id,
                **kwargs,
            }
            return {
                "handled": True,
                "reply": "handled",
                "voice_turn": {
                    "recognized_text": text,
                    "conversation_session_id": conversation_session_id,
                    "planning_session_id": planning_session_id,
                    "safety_bypass_allowed": False,
                },
            }

    runtime = DummyRuntimeHandler()
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            runtime_handler=runtime,
        )
    )

    response = client.post(
        "/api/runtime/voice-turn",
        json={
            "text": "pause current task",
            "conversation_session_id": "conv-voice",
            "planning_session_id": "plan-voice",
            "transcript_id": "turn-1",
        },
    )

    assert response.status_code == 200
    assert runtime.seen["conversation_session_id"] == "conv-voice"
    assert runtime.seen["planning_session_id"] == "plan-voice"
    assert runtime.seen["conversation_session_id"] != runtime.seen["planning_session_id"]
    payload = response.json()["voice_turn"]
    assert payload["conversation_session_id"] == "conv-voice"
    assert payload["planning_session_id"] == "plan-voice"


def test_runtime_voice_turn_accepts_conversation_session_aliases() -> None:
    class DummyRuntimeHandler:
        def __init__(self):
            self.calls = []

        def voice_turn_payload(
            self,
            text,
            *,
            conversation_session_id=None,
            planning_session_id=None,
            **kwargs,
        ):
            self.calls.append({
                "text": text,
                "conversation_session_id": conversation_session_id,
                "planning_session_id": planning_session_id,
                **kwargs,
            })
            return {
                "handled": True,
                "reply": "handled",
                "voice_turn": {
                    "recognized_text": text,
                    "conversation_session_id": conversation_session_id,
                    "planning_session_id": planning_session_id,
                    "safety_bypass_allowed": False,
                },
            }

    runtime = DummyRuntimeHandler()
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            runtime_handler=runtime,
        )
    )

    by_conversation_id = client.post(
        "/api/runtime/voice-turn",
        json={"text": "pause current task", "conversation_id": "conv-alias"},
    )
    by_chat_session_id = client.post(
        "/api/runtime/voice-turn",
        json={"text": "pause current task", "chat_session_id": "conv-chat"},
    )

    assert by_conversation_id.status_code == 200
    assert by_chat_session_id.status_code == 200
    assert runtime.calls[0]["conversation_session_id"] == "conv-alias"
    assert runtime.calls[1]["conversation_session_id"] == "conv-chat"
    assert by_conversation_id.json()["voice_turn"]["conversation_session_id"] == "conv-alias"
    assert by_chat_session_id.json()["voice_turn"]["conversation_session_id"] == "conv-chat"
