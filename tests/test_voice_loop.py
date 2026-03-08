import asyncio

import pytest

from askme.brain.intent_router import Intent, IntentType
from askme.pipeline.voice_loop import VoiceLoop


class _Router:
    def route(self, text: str) -> Intent:
        if text == "exit":
            return Intent(type=IntentType.COMMAND, command="exit", raw_text=text)
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


class _Audio:
    def __init__(self) -> None:
        self._calls = 0
        self.spoken: list[str] = []

    def listen_loop(self):
        self._calls += 1
        if self._calls == 1:
            return "inspect zone"
        return "exit"

    def acknowledge(self) -> None:
        return

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def start_playback(self) -> None:
        return

    def wait_speaking_done(self) -> None:
        return

    def stop_playback(self) -> None:
        return


class _Bridge:
    def handle_voice_text(self, text: str):
        return {
            "handled": True,
            "turn": {
                "action_type": "mission",
                "spoken_reply": "runtime handled",
            },
        }


class _ExplodingBridge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def handle_voice_text(self, text: str):
        self.calls.append(text)
        raise RuntimeError("runtime bridge offline")


@pytest.mark.asyncio
async def test_voice_loop_prefers_runtime_bridge_before_llm() -> None:
    pipeline = _Pipeline()
    audio = _Audio()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
        voice_runtime_bridge=_Bridge(),
    )

    await loop.run()

    assert pipeline.process_calls == []
    assert audio.spoken[-1] == "runtime handled"


@pytest.mark.asyncio
async def test_voice_loop_handles_pending_tool_confirmation_before_llm() -> None:
    pipeline = _Pipeline()
    pipeline.pending_reply_map["inspect zone"] = "approved"
    audio = _Audio()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
        voice_runtime_bridge=_Bridge(),
    )

    await loop.run()

    assert pipeline.pending_calls == ["inspect zone", "exit"]
    assert pipeline.process_calls == []


@pytest.mark.asyncio
async def test_voice_loop_falls_back_to_local_pipeline_when_runtime_bridge_fails() -> None:
    pipeline = _Pipeline()
    audio = _Audio()
    bridge = _ExplodingBridge()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
        voice_runtime_bridge=bridge,
    )

    await loop.run()

    assert bridge.calls == ["inspect zone"]
    assert pipeline.process_calls == ["inspect zone"]
