from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from askme.api.services.conversation_service import (
    ChatTimeout,
    ConversationService,
    current_chat_runtime_context,
)
from askme.conversation import VoiceTurnLedger
from askme.health_server import AskmeHealthServer, create_health_app
from askme.runtime.modules.health_module import HealthModule
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
        seen.update(
            {
                "text": text,
                "speak": speak,
                "conversation_session_id": conversation_session_id,
                "planning_session_id": planning_session_id,
                "runtime_policy": runtime_policy,
            }
        )
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


def test_chat_wrapper_carries_authorized_operator_context_and_session() -> None:
    seen: dict[str, object] = {}

    async def chat_handler(
        text: str,
        *,
        conversation_session_id: str | None = None,
    ) -> dict[str, str]:
        context = current_chat_runtime_context()
        seen.update(
            {
                "text": text,
                "conversation_session_id": conversation_session_id,
                "context": context,
            }
        )
        return {"reply": "authorized"}

    server = AskmeHealthServer({}, health_provider=lambda: _runtime_snapshot())
    server.set_runtime_handler(
        SimpleNamespace(context_payload=lambda: {"active_run": {"run_id": "run-authorized"}})
    )
    server.set_chat_handler(chat_handler)

    response = TestClient(server._app).post(
        "/api/chat",
        headers={"X-Askme-Operator-Id": "security-1"},
        json={
            "text": "pause current task",
            "conversation_session_id": "thread-authorized",
        },
    )

    assert response.status_code == 200
    context = seen["context"]
    assert context is not None
    assert context.conversation_session_id == "thread-authorized"
    assert context.operator_id == "security-1"
    assert context.operator_roles == ("operator",)
    assert context.operator_authenticated is False
    assert context.permission == "runtime:pause"


def test_explicit_control_text_is_ordinary_chat_without_runtime_target() -> None:
    seen_contexts: list[object] = []

    async def chat_handler(text: str) -> dict[str, str]:
        seen_contexts.append(current_chat_runtime_context())
        return {"reply": f"ordinary:{text}"}

    server = AskmeHealthServer({}, health_provider=lambda: _runtime_snapshot())
    server.set_chat_handler(chat_handler)
    response = TestClient(server._app).post(
        "/api/chat",
        headers={"X-Askme-Operator-Id": "product.owner"},
        json={
            "text": "continue",
            "conversation_session_id": "thread-no-runtime-target",
        },
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "ordinary:continue"
    assert response.json()["conversation_session_id"] == "thread-no-runtime-target"
    assert seen_contexts == [None]


def test_runtime_context_rejects_untrusted_operator_decisions() -> None:
    seen_contexts: list[object] = []

    async def chat_handler(_text: str) -> dict[str, str]:
        seen_contexts.append(current_chat_runtime_context())
        return {"reply": "not controlled"}

    service = ConversationService(chat_handler=chat_handler)
    decisions = (
        {
            "allowed": True,
            "permission": "runtime:pause",
            "operator": {
                "operator_id": "untrusted.oidc",
                "roles": ["operator"],
                "source": "oidc",
                "authenticated": False,
                "known": True,
            },
            "audit": {"mode": "enterprise"},
        },
        {
            "allowed": True,
            "permission": "runtime:pause",
            "operator": {
                "operator_id": "wrong.mode",
                "roles": ["operator"],
                "source": "local_config",
                "authenticated": False,
                "known": True,
            },
            "audit": {"mode": "enterprise"},
        },
    )

    async def dispatch_untrusted_decisions() -> None:
        for index, decision in enumerate(decisions):
            operator_id = str(decision["operator"]["operator_id"])
            await service.chat_payload_from_body(
                {
                    "text": "pause current task",
                    "conversation_session_id": f"untrusted-{index}",
                    "operator_id": operator_id,
                    "operator_auth": decision,
                }
            )

    asyncio.run(dispatch_untrusted_decisions())

    assert seen_contexts == [None, None]


def test_unprivileged_operator_cannot_control_runtime_through_chat() -> None:
    chat_calls: list[str] = []

    async def chat_handler(text: str) -> dict[str, str]:
        chat_calls.append(text)
        return {"reply": "should not run"}

    server = AskmeHealthServer({}, health_provider=lambda: _runtime_snapshot())
    server.set_runtime_handler(
        SimpleNamespace(context_payload=lambda: {"active_run": {"run_id": "run-denied"}})
    )
    server.set_chat_handler(chat_handler)
    response = TestClient(server._app).post(
        "/api/chat",
        headers={"X-Askme-Operator-Id": "product.owner"},
        json={
            "text": "cancel task",
            "conversation_session_id": "thread-denied",
        },
    )

    assert response.status_code == 403
    assert response.json()["operator_auth"]["permission"] == "runtime:cancel"
    assert response.json()["operator_auth"]["operator"]["operator_id"] == "product.owner"
    assert response.json()["operator_auth"]["operator"]["roles"] == ["product_owner"]
    assert chat_calls == []


def test_ambiguous_runtime_words_remain_ordinary_chat() -> None:
    class TextLoop:
        def __init__(self) -> None:
            self.texts: list[str] = []

        async def process_turn(
            self,
            text: str,
            *,
            speak: bool = False,
            conversation_session_id: str | None = None,
        ) -> str:
            del speak, conversation_session_id
            self.texts.append(text)
            return "ordinary chat"

    class RuntimeHandler:
        def __init__(self) -> None:
            self.control_calls: list[str] = []

        def handle_chat_control(self, text: str) -> dict[str, object]:
            self.control_calls.append(text)
            return {
                "handled": True,
                "reply": "unsafe legacy interception",
                "runtime": {"active_run": {"run_id": "run-1"}},
            }

    text_loop = TextLoop()
    runtime = RuntimeHandler()
    chat_handler = HealthModule()._chat_handler(
        {
            "text": SimpleNamespace(text_loop=text_loop),
            "runtime_handoff": runtime,
        }
    )
    assert chat_handler is not None

    server = AskmeHealthServer({}, health_provider=lambda: _runtime_snapshot())
    server.set_chat_handler(chat_handler)
    client = TestClient(server._app)
    texts = (
        "continue explaining the design",
        "Who are the stakeholders?",
        "continue brunch planning",
        "progressive disclosure is useful",
        "continue writing the task description",
        "hold a discussion about the robot design",
        "status in runtime APIs is a field",
        "暂停是什么？",
        "系统文档只是提到取消任务，并没有要求执行。",
    )
    responses = [
        client.post(
            "/api/chat",
            json={"text": text, "conversation_session_id": f"ordinary-{index}"},
        )
        for index, text in enumerate(texts)
    ]

    assert [response.status_code for response in responses] == [200] * len(texts)
    assert [response.json()["reply"] for response in responses] == ["ordinary chat"] * len(texts)
    assert text_loop.texts == list(texts)
    assert runtime.control_calls == []


def test_authorized_chat_runtime_control_records_operator_and_canonical_turn(
    tmp_path,
) -> None:
    ledger = VoiceTurnLedger(tmp_path / "chat-runtime-turns.jsonl")
    pipeline = SimpleNamespace(_turn_ledger=ledger)

    class TextLoop:
        def __init__(self) -> None:
            self._pipeline = pipeline

        async def process_turn(self, *_args, **_kwargs):
            raise AssertionError("authorized runtime control must be intercepted")

    class RuntimeHandler:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def context_payload(self) -> dict[str, object]:
            return {"active_run": {"run_id": "run-1", "current_state": "executing"}}

        def pause_payload(
            self,
            run_id: str,
            *,
            operator_id: str,
            operator_roles: list[str],
            operator_authenticated: bool,
            operator_source: str,
            operator_auth: dict[str, object],
            conversation_session_id: str,
        ) -> dict[str, object]:
            admitted_turns = ledger.list_turns(thread_id=conversation_session_id)
            assert len(admitted_turns) == 1
            assert admitted_turns[0].status.value == "started"
            self.calls.append(
                {
                    "run_id": run_id,
                    "operator_id": operator_id,
                    "operator_roles": operator_roles,
                    "operator_authenticated": operator_authenticated,
                    "operator_source": operator_source,
                    "operator_auth": operator_auth,
                    "conversation_session_id": conversation_session_id,
                }
            )
            return {
                "handled": True,
                "reply": "TaskRun paused.",
                "run": {"run_id": run_id, "current_state": "paused"},
            }

        def handle_chat_control(self, _text: str):
            raise AssertionError("privileged mutation must not use the unauthenticated shortcut")

    runtime = RuntimeHandler()
    chat_handler = HealthModule()._chat_handler(
        {
            "text": SimpleNamespace(text_loop=TextLoop()),
            "runtime_handoff": runtime,
        }
    )
    assert chat_handler is not None

    server = AskmeHealthServer({}, health_provider=lambda: _runtime_snapshot())
    server.set_runtime_handler(runtime)
    server.set_chat_handler(chat_handler)
    response = TestClient(server._app).post(
        "/api/chat",
        headers={"X-Askme-Operator-Id": "security-1"},
        json={
            "text": "pause current task",
            "conversation_session_id": "thread-runtime-control",
        },
    )

    assert response.status_code == 200
    assert response.json()["conversation_session_id"] == "thread-runtime-control"
    assert runtime.calls[0]["run_id"] == "run-1"
    assert runtime.calls[0]["operator_id"] == "security-1"
    assert runtime.calls[0]["operator_roles"] == ["operator"]
    assert runtime.calls[0]["operator_authenticated"] is False
    assert runtime.calls[0]["operator_source"] == "local_config"
    assert runtime.calls[0]["conversation_session_id"] == "thread-runtime-control"
    assert runtime.calls[0]["operator_auth"]["permission"] == "runtime:pause"

    turns = ledger.list_turns(thread_id="thread-runtime-control")
    assert len(turns) == 1
    assert turns[0].status.value == "committed"
    assert turns[0].user_text == "pause current task"
    assert turns[0].assistant_text == "TaskRun paused."
    assert turns[0].metadata["operator_id"] == "security-1"
    assert turns[0].metadata["runtime_control_intent"] == "pause"


def test_pipeline_fallback_controls_runtime_without_llm_fallthrough(tmp_path) -> None:
    ledger = VoiceTurnLedger(tmp_path / "pipeline-fallback-runtime-turns.jsonl")

    class BrainPipeline:
        def __init__(self) -> None:
            self._turn_ledger = ledger
            self.process_calls: list[dict[str, object]] = []

        async def process(
            self,
            text: str,
            *,
            source: str,
            conversation_session_id: str | None,
        ) -> str:
            self.process_calls.append(
                {
                    "text": text,
                    "source": source,
                    "conversation_session_id": conversation_session_id,
                }
            )
            return "ordinary fallback"

    class RuntimeHandler:
        def __init__(self) -> None:
            self.active_run: dict[str, str] | None = {"run_id": "run-fallback"}
            self.action_calls: list[tuple[str, str]] = []

        def context_payload(self) -> dict[str, object]:
            return {"active_run": self.active_run}

        def pause_payload(
            self,
            run_id: str,
            *,
            operator_id: str,
        ) -> dict[str, object]:
            self.action_calls.append((run_id, operator_id))
            return {"handled": True, "reply": "TaskRun paused."}

    pipeline = BrainPipeline()
    runtime = RuntimeHandler()
    chat_handler = HealthModule()._chat_handler(
        {
            "pipeline": SimpleNamespace(brain_pipeline=pipeline),
            "runtime_handoff": runtime,
        }
    )
    assert chat_handler is not None
    service = ConversationService(chat_handler=chat_handler)

    def body(session_id: str) -> dict[str, object]:
        return {
            "text": "pause current task",
            "conversation_session_id": session_id,
            "operator_id": "enterprise.operator",
            "operator_auth": {
                "allowed": True,
                "permission": "runtime:pause",
                "operator": {
                    "operator_id": "enterprise.operator",
                    "roles": ["operator"],
                    "source": "oidc",
                    "authenticated": True,
                    "known": True,
                },
                "audit": {"mode": "enterprise"},
            },
        }

    async def dispatch_controls() -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]:
        applied = await service.chat_payload_from_body(body("fallback-applied"))
        runtime.active_run = None
        not_applied = await service.chat_payload_from_body(body("fallback-no-active-run"))
        ordinary = await service.chat_payload_from_body(
            {
                "text": "explain the design",
                "conversation_session_id": "fallback-ordinary",
            }
        )
        return applied, not_applied, ordinary

    applied, not_applied, ordinary = asyncio.run(dispatch_controls())

    assert applied["reply"] == "TaskRun paused."
    assert not_applied["runtime"]["handled"] is False
    assert not_applied["runtime"]["reason"] == "runtime_control_no_active_run"
    assert ordinary["reply"] == "ordinary fallback"
    assert pipeline.process_calls == [
        {
            "text": "explain the design",
            "source": "text",
            "conversation_session_id": "fallback-ordinary",
        }
    ]
    assert runtime.action_calls == [("run-fallback", "enterprise.operator")]

    applied_turns = ledger.list_turns(thread_id="fallback-applied")
    no_active_turns = ledger.list_turns(thread_id="fallback-no-active-run")
    assert len(applied_turns) == 1
    assert applied_turns[0].status.value == "committed"
    assert applied_turns[0].metadata["operator_id"] == "enterprise.operator"
    assert len(no_active_turns) == 1
    assert no_active_turns[0].status.value == "committed"
    assert no_active_turns[0].metadata["operator_id"] == "enterprise.operator"


def test_runtime_control_requires_canonical_turn_admission() -> None:
    pipeline = SimpleNamespace()

    class TextLoop:
        _pipeline = pipeline

        async def process_turn(self, *_args, **_kwargs):
            raise AssertionError("an explicit control must not fall through")

    class RuntimeHandler:
        def __init__(self) -> None:
            self.action_calls = 0

        def context_payload(self) -> dict[str, object]:
            return {"active_run": {"run_id": "run-no-ledger"}}

        def pause_payload(self, _run_id: str, *, operator_id: str) -> dict[str, object]:
            del operator_id
            self.action_calls += 1
            return {"handled": True, "reply": "unsafe mutation"}

    runtime = RuntimeHandler()
    chat_handler = HealthModule()._chat_handler(
        {
            "text": SimpleNamespace(text_loop=TextLoop()),
            "runtime_handoff": runtime,
        }
    )
    assert chat_handler is not None
    service = ConversationService(chat_handler=chat_handler)

    payload = asyncio.run(
        service.chat_payload_from_body(
            {
                "text": "pause current task",
                "conversation_session_id": "thread-no-ledger",
                "operator_id": "enterprise.operator",
                "operator_auth": {
                    "allowed": True,
                    "permission": "runtime:pause",
                    "operator": {
                        "operator_id": "enterprise.operator",
                        "roles": ["operator"],
                        "source": "oidc",
                        "authenticated": True,
                        "known": True,
                    },
                    "audit": {"mode": "enterprise"},
                },
            }
        )
    )

    assert runtime.action_calls == 0
    assert payload["runtime"]["handled"] is False
    assert payload["runtime"]["reason"] == "conversation_turn_admission_unavailable"


def test_cancelled_runtime_control_settles_started_turn(tmp_path) -> None:
    ledger = VoiceTurnLedger(tmp_path / "cancelled-chat-runtime-turns.jsonl")
    pipeline = SimpleNamespace(_turn_ledger=ledger)

    class TextLoop:
        _pipeline = pipeline

        async def process_turn(self, *_args, **_kwargs):
            raise AssertionError("an explicit control must not fall through")

    class RuntimeHandler:
        def __init__(self) -> None:
            self.action_started = False

        def context_payload(self) -> dict[str, object]:
            return {"active_run": {"run_id": "run-cancelled"}}

        async def pause_payload(self, _run_id: str, *, operator_id: str) -> dict[str, object]:
            del operator_id
            self.action_started = True
            await asyncio.Event().wait()
            return {"handled": True, "reply": "unreachable"}

    runtime = RuntimeHandler()
    chat_handler = HealthModule()._chat_handler(
        {
            "text": SimpleNamespace(text_loop=TextLoop()),
            "runtime_handoff": runtime,
        }
    )
    assert chat_handler is not None
    service = ConversationService(chat_handler=chat_handler, chat_timeout_s=0.01)

    async def run_timed_control() -> None:
        try:
            await service.chat_payload_from_body(
                {
                    "text": "pause current task",
                    "conversation_session_id": "thread-cancelled-control",
                    "operator_id": "enterprise.operator",
                    "operator_auth": {
                        "allowed": True,
                        "permission": "runtime:pause",
                        "operator": {
                            "operator_id": "enterprise.operator",
                            "roles": ["operator"],
                            "source": "oidc",
                            "authenticated": True,
                            "known": True,
                        },
                        "audit": {"mode": "enterprise"},
                    },
                }
            )
        except ChatTimeout:
            return
        raise AssertionError("runtime control should have timed out")

    asyncio.run(run_timed_control())

    assert runtime.action_started is True
    turns = ledger.list_turns(thread_id="thread-cancelled-control")
    assert len(turns) == 1
    assert turns[0].status.value == "cancelled"


def test_anonymous_chat_requests_receive_distinct_thread_ids() -> None:
    seen_thread_ids: list[str | None] = []

    async def chat_handler(
        text: str,
        *,
        conversation_session_id: str | None = None,
    ) -> dict[str, str]:
        seen_thread_ids.append(conversation_session_id)
        return {"reply": f"reply:{text}"}

    server = AskmeHealthServer({}, health_provider=lambda: _runtime_snapshot())
    server.set_chat_handler(chat_handler)
    client = TestClient(server._app)

    first = client.post("/api/chat", json={"text": "first anonymous request"})
    second = client.post("/api/chat", json={"text": "second anonymous request"})

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    first_thread_id = first_payload["conversation_thread_id"]
    second_thread_id = second_payload["conversation_thread_id"]
    assert first_thread_id
    assert second_thread_id
    assert first_thread_id != second_thread_id
    assert first_payload["conversation_session_id"] == first_thread_id
    assert second_payload["conversation_session_id"] == second_thread_id
    assert seen_thread_ids == [first_thread_id, second_thread_id]


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
        headers={"X-Askme-Operator-Id": "security-1"},
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
            self.calls.append(
                {
                    "text": text,
                    "conversation_session_id": conversation_session_id,
                    "planning_session_id": planning_session_id,
                    **kwargs,
                }
            )
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
        headers={"X-Askme-Operator-Id": "security-1"},
        json={"text": "pause current task", "conversation_id": "conv-alias"},
    )
    by_chat_session_id = client.post(
        "/api/runtime/voice-turn",
        headers={"X-Askme-Operator-Id": "security-1"},
        json={"text": "pause current task", "chat_session_id": "conv-chat"},
    )

    assert by_conversation_id.status_code == 200
    assert by_chat_session_id.status_code == 200
    assert runtime.calls[0]["conversation_session_id"] == "conv-alias"
    assert runtime.calls[1]["conversation_session_id"] == "conv-chat"
    assert by_conversation_id.json()["voice_turn"]["conversation_session_id"] == "conv-alias"
    assert by_chat_session_id.json()["voice_turn"]["conversation_session_id"] == "conv-chat"


def test_runtime_voice_turn_accepts_canonical_thread_id_and_rejects_conflicts() -> None:
    class DummyRuntimeHandler:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def voice_turn_payload(
            self,
            text,
            *,
            conversation_session_id=None,
            **kwargs,
        ):
            self.calls.append(
                {
                    "text": text,
                    "conversation_session_id": conversation_session_id,
                    **kwargs,
                }
            )
            return {
                "handled": True,
                "voice_turn": {
                    "recognized_text": text,
                    "conversation_session_id": conversation_session_id,
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

    accepted = client.post(
        "/api/runtime/voice-turn",
        headers={"X-Askme-Operator-Id": "security-1"},
        json={"text": "暂停", "conversation_thread_id": "thread-new"},
    )
    rejected = client.post(
        "/api/runtime/voice-turn",
        headers={"X-Askme-Operator-Id": "security-1"},
        json={
            "text": "暂停",
            "conversation_thread_id": "thread-a",
            "thread_id": "thread-b",
        },
    )

    assert accepted.status_code == 200
    assert runtime.calls[0]["conversation_session_id"] == "thread-new"
    assert rejected.status_code == 400
    assert len(runtime.calls) == 1


class _CapturingVoiceRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def voice_turn_payload(self, text: str, **kwargs: object) -> dict[str, object]:
        self.calls.append({"text": text, **kwargs})
        return {
            "handled": True,
            "reply": "handled",
            "voice_turn": {
                "recognized_text": text,
                "safety_bypass_allowed": False,
            },
        }


def test_runtime_voice_turn_forwards_action_scoped_trusted_operator() -> None:
    runtime = _CapturingVoiceRuntime()
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            runtime_handler=runtime,
        )
    )

    response = client.post(
        "/api/runtime/voice-turn",
        headers={"X-Askme-Operator-Id": "security-1"},
        json={
            "text": "pause current task",
            "conversation_session_id": "voice-control-thread",
            "operator_id": "admin-1",
            "operator_auth": {
                "allowed": True,
                "permission": "runtime:cancel",
                "operator": {"operator_id": "admin-1", "roles": ["admin"]},
            },
            "reason": "visitor entered the path",
            "risk_acknowledgement": True,
        },
    )

    assert response.status_code == 200
    assert len(runtime.calls) == 1
    call = runtime.calls[0]
    assert call["operator_id"] == "security-1"
    assert call["operator_roles"] == ["operator"]
    assert call["operator_authenticated"] is False
    assert call["operator_source"] == "local_config"
    assert call["runtime_permission"] == "runtime:pause"
    assert call["conversation_session_id"] == "voice-control-thread"
    assert call["reason"] == "visitor entered the path"
    assert call["risk_acknowledgement"] is True


def test_runtime_voice_cancel_requires_supervisor_permission() -> None:
    runtime = _CapturingVoiceRuntime()
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            runtime_handler=runtime,
        )
    )

    denied = client.post(
        "/api/runtime/voice-turn",
        headers={"X-Askme-Operator-Id": "security-1"},
        json={"text": "cancel current task"},
    )
    allowed = client.post(
        "/api/runtime/voice-turn",
        headers={"X-Askme-Operator-Id": "supervisor-1"},
        json={"text": "cancel current task"},
    )

    assert denied.status_code == 403
    assert denied.json()["operator_auth"]["permission"] == "runtime:cancel"
    assert allowed.status_code == 200
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["operator_id"] == "supervisor-1"
    assert runtime.calls[0]["operator_roles"] == ["supervisor"]
    assert runtime.calls[0]["runtime_permission"] == "runtime:cancel"


def test_runtime_voice_turn_maps_status_and_requires_operator_for_submit() -> None:
    runtime = _CapturingVoiceRuntime()
    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            runtime_handler=runtime,
        )
    )

    status = client.post(
        "/api/runtime/voice-turn",
        json={"text": "runtime status"},
    )
    denied_non_control = client.post(
        "/api/runtime/voice-turn",
        json={"text": "explain the runtime status field"},
    )
    authorized_non_control = client.post(
        "/api/runtime/voice-turn",
        headers={"X-Askme-Operator-Id": "security-1"},
        json={"text": "explain the runtime status field"},
    )

    assert status.status_code == 200
    assert denied_non_control.status_code == 403
    assert denied_non_control.json()["reason"] == "runtime_operator_context_required"
    assert authorized_non_control.status_code == 200
    assert [call["runtime_permission"] for call in runtime.calls] == [
        "runtime:read",
        "runtime:submit",
    ]
