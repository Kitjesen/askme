import asyncio
from unittest.mock import patch

import pytest

from askme.brain.intent_router import Intent, IntentType
from askme.pipeline.text_loop import TextLoop


class _Router:
    def route(self, text: str) -> Intent:
        if text == "/quit":
            return Intent(type=IntentType.COMMAND, command="/quit", raw_text=text)
        return Intent(type=IntentType.GENERAL, raw_text=text)


class _Pipeline:
    def __init__(self) -> None:
        self.process_calls: list[str] = []
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

    async def process(self, user_text: str, *, memory_task=None):
        self.process_calls.append(user_text)
        return "fallback"

    async def execute_skill(self, skill_name: str, user_text: str):
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
    def wait_speaking_done(self) -> None:
        return

    def stop_playback(self) -> None:
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
                "spoken_reply": "当前没有进行中的任务。",
            },
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

    with patch("builtins.input", side_effect=["当前状态", "/quit"]):
        await loop.run()

    assert bridge.calls == ["当前状态"]
    assert pipeline.process_calls == []


@pytest.mark.asyncio
async def test_text_loop_handles_pending_tool_confirmation_before_llm() -> None:
    pipeline = _Pipeline()
    pipeline.pending_reply_map["确认执行"] = "已确认执行"
    loop = TextLoop(
        router=_Router(),
        pipeline=pipeline,
        commands=_Commands(),
        conversation=_Conversation(),
        skill_manager=_Skills(),
        audio=_Audio(),
        voice_runtime_bridge=_Bridge(),
    )

    with patch("builtins.input", side_effect=["确认执行", "/quit"]):
        await loop.run()

    assert pipeline.pending_calls == ["确认执行", "/quit"]
    assert pipeline.process_calls == []
