from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from askme.pipeline.text_loop import TextLoop

from askme.robot_interaction import Intent, IntentType


class Router:
    def route(self, text: str) -> Intent:
        return Intent(type=IntentType.GENERAL, raw_text=text)


class Pipeline:
    def __init__(self) -> None:
        self.process_calls: list[str] = []
        self.process_source_calls: list[str] = []
        self.process_conversation_session_ids: list[str | None] = []

    def start_idle_reflection(self):
        return None

    def start_memory_prefetch(self, user_text: str):
        return asyncio.create_task(asyncio.sleep(0, result=""))

    async def process(
        self,
        user_text: str,
        *,
        memory_task=None,
        source: str = "text",
        conversation_session_id: str | None = None,
    ):
        self.process_calls.append(user_text)
        self.process_source_calls.append(source)
        self.process_conversation_session_ids.append(conversation_session_id)
        return "fallback"


class Commands:
    def handle(self, command: str) -> bool:
        return False


@dataclass
class Conversation:
    history: list[str]


class Skills:
    def get_skill_catalog(self):
        return []


class Audio:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def start_playback(self) -> None:
        return

    def wait_speaking_done(self) -> None:
        return

    def stop_playback(self) -> None:
        return


def make_text_loop(
    *,
    cognition_handler: Any | None = None,
    voice_runtime_bridge: Any | None = None,
) -> tuple[TextLoop, Pipeline]:
    pipeline = Pipeline()
    loop = TextLoop(
        router=Router(),
        pipeline=pipeline,
        commands=Commands(),
        conversation=Conversation(history=[]),
        skill_manager=Skills(),
        audio=Audio(),
        voice_runtime_bridge=voice_runtime_bridge,
        cognition_handler=cognition_handler,
    )
    return loop, pipeline
