import asyncio
import threading
from unittest.mock import patch

import askme.pipeline.text_loop as text_loop_module
import pytest
from askme.pipeline.text_loop import TextLoop
from askme.pipeline.trace import PipelineTracer

from askme.robot_interaction import Intent, IntentType


class _Router:
    def route(self, text: str) -> Intent:
        if text == "/quit":
            return Intent(type=IntentType.COMMAND, command="/quit", raw_text=text)
        return Intent(type=IntentType.GENERAL, raw_text=text)


class _QuickReplyRouter:
    def route(self, text: str) -> Intent:
        return Intent(type=IntentType.QUICK_REPLY, skill_name="quick reply", raw_text=text)


class _TraceRouter:
    def route(self, text: str) -> Intent:
        return Intent(
            type=IntentType.VOICE_TRIGGER,
            skill_name="get_time",
            raw_text=text,
            trigger_phrase="几点了",
            reason="voice_trigger",
        )


class _Pipeline:
    def __init__(self) -> None:
        self.process_calls: list[str] = []
        self.process_source_calls: list[str] = []
        self.process_conversation_session_ids: list[str | None] = []
        self.skill_calls: list[tuple[str, str]] = []
        self.pending_calls: list[str] = []
        self.pending_reply_map: dict[str, str] = {}

    def start_idle_reflection(self):
        return None

    def start_memory_prefetch(self, user_text: str):
        return asyncio.create_task(asyncio.sleep(0, result=""))

    async def handle_pending_tool_response(self, user_text: str):
        self.pending_calls.append(user_text)
        return self.pending_reply_map.get(user_text)

    async def process(
        self,
        user_text: str,
        *,
        memory_task=None,
        source: str = "voice",
        conversation_session_id: str | None = None,
    ):
        self.process_calls.append(user_text)
        self.process_source_calls.append(source)
        self.process_conversation_session_ids.append(conversation_session_id)
        return "fallback"

    async def execute_skill(self, skill_name: str, user_text: str, source: str = "voice"):
        self.skill_calls.append((skill_name, user_text))
        return "skill"


class _Commands:
    def handle(self, command: str) -> bool:
        return command in {"quit", "/quit"}


class _Conversation:
    history: list[str] = []


class _Skills:
    def get_skill_catalog(self):
        return []


class _Audio:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.started = 0
        self.stopped = 0
        self.waited = 0

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def start_playback(self) -> None:
        self.started += 1

    def wait_speaking_done(self) -> None:
        self.waited += 1
        return

    def stop_playback(self) -> None:
        self.stopped += 1
        return


class _Dispatcher:
    def __init__(self) -> None:
        self.general_calls: list[dict[str, object]] = []

    async def handle_general(
        self,
        user_text: str,
        *,
        source: str = "voice",
        memory_task=None,
        conversation_session_id: str | None = None,
    ) -> str:
        self.general_calls.append(
            {
                "user_text": user_text,
                "source": source,
                "memory_task": memory_task,
                "conversation_session_id": conversation_session_id,
            }
        )
        return "dispatcher fallback"


class _Bridge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def handle_text_input(self, text: str, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return {
            "handled": True,
            "turn": {
                "action_type": "runtime_query",
                "spoken_reply": "runtime handled",
            },
        }


class _ExplodingBridge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def handle_text_input(self, text: str, **kwargs):
        self.calls.append({"text": text, **kwargs})
        raise RuntimeError("runtime bridge offline")


class _Cognition:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def plan_from_payload(self, payload: dict):
        self.calls.append(dict(payload))
        if payload.get("action") == "cancel":
            return {
                "planned": True,
                "plan": {
                    "planning_session_id": payload.get("planning_session_id"),
                    "interaction_state": "cancelled",
                    "next_prompt": "已取消当前规划。",
                    "handoff_ready": False,
                },
            }
        if payload.get("operator_confirmation") is True:
            return {
                "planned": True,
                "plan": {
                    "planning_session_id": payload.get("planning_session_id"),
                    "interaction_state": "ready_for_arbiter",
                    "next_prompt": "计划已确认，可以交给运行时仲裁器继续处理。",
                    "handoff_ready": True,
                },
            }
        return {
            "planned": True,
            "plan": {
                "planning_session_id": "session-1",
                "interaction_state": "awaiting_confirmation",
                "next_prompt": "已生成巡检任务草案，请确认后再交给运行时仲裁器。",
                "handoff_ready": False,
            },
            "sync": {"fresh_object_count": 0},
        }


@pytest.mark.asyncio
async def test_text_loop_prefers_runtime_bridge_before_llm() -> None:
    bridge = _Bridge()
    pipeline = _Pipeline()
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=bridge,
    )

    with patch("builtins.input", side_effect=["status?", "/quit"]):
        await loop.run()

    assert [call["text"] for call in bridge.calls] == ["status?"]
    assert pipeline.process_calls == []


@pytest.mark.asyncio
async def test_text_loop_passes_conversation_session_to_runtime_bridge() -> None:
    from askme.voice_gateway import VoiceGatewayService

    class Bridge:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def handle_text_input(self, text: str, **kwargs):
            self.calls.append({"text": text, **kwargs})
            return {"handled": True, "turn": {"spoken_reply": "runtime handled"}}

    bridge = Bridge()
    gateway = VoiceGatewayService(bridge)
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=gateway,
    )

    with patch("builtins.input", side_effect=["status?", "/quit"]):
        await loop.run()

    assert bridge.calls[0]["session_id"]
    assert bridge.calls[0]["session_id"] == bridge.calls[0]["conversation_session_id"]
    assert bridge.calls[0]["channel"] == "text"


@pytest.mark.asyncio
async def test_text_loop_replaces_closed_cached_runtime_session() -> None:
    from askme.voice_gateway import VoiceGatewayService

    class Bridge:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def handle_text_input(self, text: str, **kwargs):
            self.calls.append({"text": text, **kwargs})
            return {"handled": True, "turn": {"spoken_reply": "runtime handled"}}

    bridge = Bridge()
    gateway = VoiceGatewayService(bridge)
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=gateway,
    )

    assert await loop._maybe_handle_runtime_bridge("status one") is True
    first_session_id = str(bridge.calls[0]["session_id"])
    gateway.session_manager.close_session(first_session_id)

    assert await loop._maybe_handle_runtime_bridge("status two") is True

    assert bridge.calls[1]["session_id"] != first_session_id
    assert bridge.calls[1]["channel"] == "text"


@pytest.mark.asyncio
async def test_text_loop_replaces_missing_cached_runtime_session() -> None:
    from askme.voice_gateway import VoiceGatewayService

    class Bridge:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def handle_text_input(self, text: str, **kwargs):
            self.calls.append({"text": text, **kwargs})
            return {"handled": True, "turn": {"spoken_reply": "runtime handled"}}

    bridge = Bridge()
    gateway = VoiceGatewayService(bridge)
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=gateway,
    )

    assert await loop._maybe_handle_runtime_bridge("status one") is True
    first_session_id = str(bridge.calls[0]["session_id"])
    assert gateway.session_manager.store.delete(first_session_id) is True

    assert await loop._maybe_handle_runtime_bridge("status two") is True

    assert bridge.calls[1]["session_id"] != first_session_id
    assert bridge.calls[1]["session_id"] == bridge.calls[1]["conversation_session_id"]
    assert bridge.calls[1]["channel"] == "text"


@pytest.mark.asyncio
async def test_text_loop_calls_runtime_bridge_without_session_when_manager_unavailable() -> None:
    class BrokenManager:
        def get_or_create(self, **kwargs):
            raise RuntimeError("session store offline")

    class Bridge:
        session_manager = BrokenManager()

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def handle_text_input(self, text: str, **kwargs):
            self.calls.append({"text": text, **kwargs})
            return {"handled": True, "turn": {"spoken_reply": "runtime handled"}}

    bridge = Bridge()
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=bridge,
    )

    assert await loop._maybe_handle_runtime_bridge("status") is True

    assert bridge.calls[0]["text"] == "status"
    degraded_session_id = bridge.calls[0]["conversation_session_id"]
    assert str(degraded_session_id).startswith("text-degraded-")
    assert bridge.calls[0]["session_id"] == degraded_session_id


@pytest.mark.asyncio
async def test_text_loop_handles_pending_tool_confirmation_before_llm() -> None:
    pipeline = _Pipeline()
    pipeline.pending_reply_map["approve"] = "approved"
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=_Bridge(),
    )

    with patch("builtins.input", side_effect=["approve", "/quit"]):
        await loop.run()

    assert pipeline.pending_calls == ["approve", "/quit"]
    assert pipeline.process_calls == []


@pytest.mark.asyncio
async def test_text_loop_falls_back_to_local_pipeline_when_runtime_bridge_fails() -> None:
    bridge = _ExplodingBridge()
    pipeline = _Pipeline()
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=bridge,
    )

    with patch("builtins.input", side_effect=["status?", "/quit"]):
        await loop.run()

    assert [call["text"] for call in bridge.calls] == ["status?"]
    assert pipeline.process_calls == ["status?"]
    assert len(pipeline.process_conversation_session_ids) == 1
    assert pipeline.process_conversation_session_ids[0]


@pytest.mark.asyncio
async def test_process_turn_runtime_fallback_keeps_conversation_session() -> None:
    from askme.voice_gateway import VoiceGatewayService

    bridge = _ExplodingBridge()
    gateway = VoiceGatewayService(bridge)
    pipeline = _Pipeline()
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=gateway,
    )

    reply = await loop.process_turn("status?", runtime_policy="runtime_first")

    assert reply == "fallback"
    assert len(bridge.calls) == 1
    bridge_session_id = bridge.calls[0]["session_id"]
    assert bridge_session_id
    assert pipeline.process_conversation_session_ids == [bridge_session_id]


@pytest.mark.asyncio
async def test_process_turn_speak_uses_voice_source_and_waits_in_pipeline() -> None:
    pipeline = _Pipeline()
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
    )

    reply = await loop.process_turn("status?", speak=True)

    assert reply == "fallback"
    assert pipeline.process_calls == ["status?"]
    assert pipeline.process_source_calls == ["voice"]


@pytest.mark.asyncio
async def test_process_turn_text_source_when_speak_false() -> None:
    pipeline = _Pipeline()
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
    )

    reply = await loop.process_turn("status?")

    assert reply == "fallback"
    assert pipeline.process_source_calls == ["text"]


@pytest.mark.asyncio
async def test_process_turn_does_not_use_runtime_bridge_by_default() -> None:
    bridge = _Bridge()
    pipeline = _Pipeline()
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=bridge,
    )

    reply = await loop.process_turn("status?")

    assert reply == "fallback"
    assert bridge.calls == []
    assert pipeline.process_calls == ["status?"]


@pytest.mark.asyncio
async def test_process_turn_can_use_runtime_bridge_when_explicitly_requested() -> None:
    bridge = _Bridge()
    pipeline = _Pipeline()
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=bridge,
    )

    reply = await loop.process_turn(
        "status?",
        runtime_policy="runtime_first",
        conversation_session_id="conv-explicit",
    )

    assert reply == "runtime handled"
    assert bridge.calls[0]["text"] == "status?"
    assert bridge.calls[0]["conversation_session_id"] == "conv-explicit"
    assert bridge.calls[0]["session_id"] == "conv-explicit"
    assert pipeline.process_calls == []


@pytest.mark.asyncio
async def test_concurrent_runtime_first_turns_keep_their_explicit_sessions() -> None:
    bridge = _Bridge()
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=bridge,
    )

    replies = await asyncio.gather(
        loop.process_turn(
            "request-a",
            runtime_policy="runtime_first",
            conversation_session_id="conv-a",
        ),
        loop.process_turn(
            "request-b",
            runtime_policy="runtime_first",
            conversation_session_id="conv-b",
        ),
    )

    assert replies == ["runtime handled", "runtime handled"]
    sessions_by_text = {
        str(call["text"]): call["conversation_session_id"]
        for call in bridge.calls
    }
    assert sessions_by_text == {"request-a": "conv-a", "request-b": "conv-b"}


@pytest.mark.asyncio
async def test_runtime_fallback_cognition_keeps_each_explicit_session() -> None:
    class _BarrierUnhandledBridge:
        def __init__(self) -> None:
            self.first_started = threading.Event()
            self.release_first = threading.Event()

        def handle_text_input(self, text: str, **_kwargs):
            if text == "巡检 A 区":
                self.first_started.set()
                assert self.release_first.wait(timeout=2.0)
            return {"handled": False}

    class _CapturingCognition:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def plan_from_payload(self, payload: dict) -> dict:
            self.calls.append(dict(payload))
            return {
                "planned": True,
                "plan": {
                    "planning_session_id": f"plan-{payload['conversation_session_id']}",
                    "interaction_state": "awaiting_confirmation",
                    "next_prompt": f"reply-{payload['conversation_session_id']}",
                    "handoff_ready": False,
                },
            }

    bridge = _BarrierUnhandledBridge()
    cognition = _CapturingCognition()
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=bridge,
        cognition_handler=cognition,
    )
    first = asyncio.create_task(
        loop.process_turn(
            "巡检 A 区",
            runtime_policy="runtime_first",
            conversation_session_id="conv-a",
        )
    )
    assert await asyncio.to_thread(bridge.first_started.wait, 1.0)
    second = asyncio.create_task(
        loop.process_turn(
            "巡检 B 区",
            runtime_policy="runtime_first",
            conversation_session_id="conv-b",
        )
    )
    assert await second == "reply-conv-b"
    bridge.release_first.set()
    assert await first == "reply-conv-a"

    session_by_text = {
        str(call["text"]): call["conversation_session_id"]
        for call in cognition.calls
    }
    assert session_by_text == {"巡检 A 区": "conv-a", "巡检 B 区": "conv-b"}


@pytest.mark.asyncio
async def test_process_turn_records_intent_route_trace(monkeypatch) -> None:
    tracer = PipelineTracer()
    monkeypatch.setattr(text_loop_module, "get_tracer", lambda: tracer)
    loop = TextLoop(
        router=_TraceRouter(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
    )

    reply = await loop.process_turn("现在几点了")

    history = tracer.get_history(1)
    route = history[0]["metadata"]["intent_route"]
    route_span = next(span for span in history[0]["spans"] if span["name"] == "intent_route")
    assert reply == "skill"
    assert route["type"] == "voice_trigger"
    assert route["source"] == "text"
    assert route["skill_name"] == "get_time"
    assert route["trigger_phrase"] == "几点了"
    assert route_span["metadata"]["reason"] == "voice_trigger"


@pytest.mark.asyncio
async def test_process_turn_quick_reply_is_silent_when_speak_false() -> None:
    pipeline = _Pipeline()
    audio = _Audio()
    loop = TextLoop(
        router=_QuickReplyRouter(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=audio,
    )

    reply = await loop.process_turn("hi", speak=False)

    assert reply == "quick reply"
    assert audio.spoken == []
    assert audio.started == 0
    assert audio.waited == 0
    assert audio.stopped == 0


@pytest.mark.asyncio
async def test_process_turn_quick_reply_speaks_when_speak_true() -> None:
    pipeline = _Pipeline()
    audio = _Audio()
    loop = TextLoop(
        router=_QuickReplyRouter(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=audio,
    )

    reply = await loop.process_turn("hi", speak=True)

    assert reply == "quick reply"
    assert audio.spoken == ["quick reply"]
    assert audio.started == 1
    assert audio.waited == 1
    assert audio.stopped == 1


@pytest.mark.asyncio
async def test_process_turn_routes_robot_task_to_cognition_before_llm() -> None:
    pipeline = _Pipeline()
    cognition = _Cognition()
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        cognition_handler=cognition,
    )

    reply = await loop.process_turn("巡检 A 区")

    assert "请确认" in reply
    assert pipeline.process_calls == []
    assert cognition.calls[0]["text"] == "巡检 A 区"
    assert loop.last_cognition_result["plan"]["planning_session_id"] == "session-1"


@pytest.mark.asyncio
async def test_concurrent_cognition_results_are_task_local() -> None:
    class _PerRequestCognition:
        async def plan_from_payload(self, payload: dict) -> dict:
            text = str(payload["text"])
            suffix = "a" if "A" in text else "b"
            return {
                "planned": True,
                "plan": {
                    "planning_session_id": f"plan-{suffix}",
                    "conversation_session_id": payload["conversation_session_id"],
                    "interaction_state": "awaiting_confirmation",
                    "next_prompt": f"reply-{suffix}",
                    "handoff_ready": False,
                },
            }

    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        cognition_handler=_PerRequestCognition(),
    )
    first_speaking = asyncio.Event()
    release_first = asyncio.Event()

    async def _controlled_speak(reply: str) -> None:
        if reply == "reply-a":
            first_speaking.set()
            await release_first.wait()

    loop._speak_reply = _controlled_speak  # type: ignore[method-assign]

    async def _invoke(text: str, session_id: str) -> tuple[str, dict]:
        reply = await loop.process_turn(
            text,
            speak=True,
            conversation_session_id=session_id,
        )
        return reply, dict(loop.last_cognition_result or {})

    first = asyncio.create_task(_invoke("巡检 A 区", "conv-a"))
    await asyncio.wait_for(first_speaking.wait(), timeout=1.0)
    second_result = await _invoke("巡检 B 区", "conv-b")
    release_first.set()
    first_result = await first

    assert first_result[0] == "reply-a"
    assert second_result[0] == "reply-b"
    assert first_result[1]["plan"]["planning_session_id"] == "plan-a"
    assert second_result[1]["plan"]["planning_session_id"] == "plan-b"
    assert first_result[1]["plan"]["conversation_session_id"] == "conv-a"
    assert second_result[1]["plan"]["conversation_session_id"] == "conv-b"


@pytest.mark.asyncio
async def test_same_thread_cognition_turns_wait_for_prior_plan_state() -> None:
    class _BlockingCognition:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def plan_from_payload(self, payload: dict) -> dict:
            self.calls.append(dict(payload))
            if len(self.calls) == 1:
                self.first_started.set()
                await self.release_first.wait()
                return {
                    "planned": True,
                    "plan": {
                        "planning_session_id": "plan-shared",
                        "interaction_state": "awaiting_confirmation",
                        "next_prompt": "confirm shared plan",
                        "handoff_ready": False,
                    },
                }
            return {
                "planned": True,
                "plan": {
                    "planning_session_id": payload.get("planning_session_id"),
                    "interaction_state": "ready_for_arbiter",
                    "next_prompt": "shared plan confirmed",
                    "handoff_ready": True,
                },
            }

    cognition = _BlockingCognition()
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        cognition_handler=cognition,
    )
    first = asyncio.create_task(
        loop.process_turn(
            "巡检 A 区",
            conversation_session_id="shared-cognition-thread",
        )
    )
    await asyncio.wait_for(cognition.first_started.wait(), timeout=1.0)
    confirmation = asyncio.create_task(
        loop.process_turn(
            "确认",
            conversation_session_id="shared-cognition-thread",
        )
    )
    await asyncio.sleep(0.02)
    assert len(cognition.calls) == 1

    cognition.release_first.set()
    replies = await asyncio.gather(first, confirmation)

    assert replies == ["confirm shared plan", "shared plan confirmed"]
    assert cognition.calls[1]["planning_session_id"] == "plan-shared"
    assert cognition.calls[1]["operator_confirmation"] is True


@pytest.mark.asyncio
async def test_process_turn_continues_cognition_session_on_confirmation() -> None:
    cognition = _Cognition()
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        cognition_handler=cognition,
    )

    await loop.process_turn("巡检 A 区")
    reply = await loop.process_turn("确认")

    assert "计划已确认" in reply
    assert cognition.calls[1]["planning_session_id"] == "session-1"
    assert cognition.calls[1]["operator_confirmation"] is True
    assert loop._active_planning_session_id is None


@pytest.mark.asyncio
async def test_process_turn_passes_conversation_and_planning_sessions_separately() -> None:
    from askme.voice_gateway import VoiceGatewayService

    cognition = _Cognition()
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=VoiceGatewayService(),
        cognition_handler=cognition,
    )

    await loop.process_turn("巡检 A 区")
    await loop.process_turn("确认")

    assert cognition.calls[0]["conversation_session_id"]
    assert "planning_session_id" not in cognition.calls[0]
    assert cognition.calls[1]["conversation_session_id"] == cognition.calls[0]["conversation_session_id"]
    assert cognition.calls[1]["planning_session_id"] == "session-1"
    assert cognition.calls[1]["conversation_session_id"] != cognition.calls[1]["planning_session_id"]


@pytest.mark.asyncio
async def test_process_turn_uses_explicit_conversation_session_id() -> None:
    cognition = _Cognition()
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        cognition_handler=cognition,
    )

    await loop.process_turn("巡检 A 区", conversation_session_id="conv-api")

    assert cognition.calls[0]["conversation_session_id"] == "conv-api"


@pytest.mark.asyncio
async def test_process_turn_passes_explicit_conversation_session_to_pipeline() -> None:
    pipeline = _Pipeline()
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
    )

    reply = await loop.process_turn("hello", conversation_session_id="conv-api")

    assert reply == "fallback"
    assert pipeline.process_calls == ["hello"]
    assert pipeline.process_conversation_session_ids == ["conv-api"]


@pytest.mark.asyncio
async def test_process_turn_dispatcher_general_passes_explicit_conversation_session() -> None:
    dispatcher = _Dispatcher()
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        dispatcher=dispatcher,
    )

    reply = await loop.process_turn("hello", conversation_session_id="conv-api")

    assert reply == "dispatcher fallback"
    assert dispatcher.general_calls[0]["conversation_session_id"] == "conv-api"


@pytest.mark.asyncio
async def test_process_turn_tracks_planning_sessions_per_conversation() -> None:
    class Cognition:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def plan_from_payload(self, payload: dict[str, object]) -> dict[str, object]:
            self.calls.append(dict(payload))
            conversation_session_id = str(payload.get("conversation_session_id") or "")
            if payload.get("operator_confirmation") is True:
                return {
                    "planned": True,
                    "plan": {
                        "planning_session_id": payload.get("planning_session_id"),
                        "interaction_state": "ready_for_arbiter",
                        "next_prompt": "ready",
                        "handoff_ready": True,
                    },
                }
            return {
                "planned": True,
                "plan": {
                    "planning_session_id": f"plan-{conversation_session_id}",
                    "conversation_session_id": conversation_session_id,
                    "interaction_state": "awaiting_confirmation",
                    "next_prompt": "confirm",
                    "handoff_ready": False,
                },
            }

    cognition = Cognition()
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        cognition_handler=cognition,
    )

    await loop.process_turn("inspect area a", conversation_session_id="conv-a")
    await loop.process_turn("inspect area b", conversation_session_id="conv-b")
    await loop.process_turn("confirm", conversation_session_id="conv-a")

    assert "planning_session_id" not in cognition.calls[0]
    assert "planning_session_id" not in cognition.calls[1]
    assert cognition.calls[2]["conversation_session_id"] == "conv-a"
    assert cognition.calls[2]["planning_session_id"] == "plan-conv-a"
    assert loop._active_planning_session_ids["conv-b"] == "plan-conv-b"
    assert "conv-a" not in loop._active_planning_session_ids


@pytest.mark.asyncio
async def test_process_turn_does_not_route_other_conversation_by_global_plan() -> None:
    pipeline = _Pipeline()
    cognition = _Cognition()
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        cognition_handler=cognition,
    )

    await loop.process_turn("inspect area a", conversation_session_id="conv-a")
    reply = await loop.process_turn("hello there", conversation_session_id="conv-b")

    assert reply == "fallback"
    assert len(cognition.calls) == 1
    assert pipeline.process_calls == ["hello there"]


@pytest.mark.asyncio
async def test_process_turn_cancels_active_cognition_session() -> None:
    cognition = _Cognition()
    loop = TextLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        cognition_handler=cognition,
    )

    await loop.process_turn("巡检 A 区")
    reply = await loop.process_turn("取消")

    assert "已取消" in reply
    assert cognition.calls[1]["planning_session_id"] == "session-1"
    assert cognition.calls[1]["action"] == "cancel"
    assert loop._active_planning_session_id is None


@pytest.mark.asyncio
async def test_process_turn_keeps_general_chat_out_of_cognition() -> None:
    pipeline = _Pipeline()
    cognition = _Cognition()
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        cognition_handler=cognition,
    )

    reply = await loop.process_turn("hello")

    assert reply == "fallback"
    assert cognition.calls == []
    assert pipeline.process_calls == ["hello"]
