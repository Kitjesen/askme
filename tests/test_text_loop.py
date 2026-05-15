import asyncio
from unittest.mock import patch

import pytest

import askme.pipeline.text_loop as text_loop_module
from askme.llm.intent_router import Intent, IntentType
from askme.pipeline.text_loop import TextLoop
from askme.pipeline.trace import PipelineTracer


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

    async def process(self, user_text: str, *, memory_task=None, source: str = "voice"):
        self.process_calls.append(user_text)
        self.process_source_calls.append(source)
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


class _Bridge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def handle_text_input(self, text: str):
        self.calls.append(text)
        return {
            "handled": True,
            "turn": {
                "action_type": "runtime_query",
                "spoken_reply": "runtime handled",
            },
        }


class _ExplodingBridge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def handle_text_input(self, text: str):
        self.calls.append(text)
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

    assert bridge.calls == ["status?"]
    assert pipeline.process_calls == []


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

    assert bridge.calls == ["status?"]
    assert pipeline.process_calls == ["status?"]


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
