from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from askme.pipeline.channels.voice_loop import VoiceLoop
from askme.pipeline.skills.outcome import (
    NAV_LOCATION_UNAVAILABLE_MESSAGE,
    SkillOutcome,
)
from askme.pipeline.skills.skill_gate import SkillGate
from askme.robot_interaction import Intent, IntentRouter, IntentType
from askme.skills.core.skill_executor import SkillExecutor


class _Audio:
    awaiting_confirmation = False

    def __init__(self, texts: list[tuple[str, str]]) -> None:
        self._texts = list(texts)
        self.last_turn_wake_authorized = False
        self.last_turn_wake_source = "none"
        self.spoken: list[str] = []
        self.cached_spoken: list[tuple[str, str]] = []
        self.ack_count = 0
        self._muted = False
        self._processing_feedback_armed = False

    def listen_loop(self) -> str:
        text, wake_source = self._texts.pop(0)
        self.last_turn_wake_source = wake_source
        return text

    async def speak_and_wait(self, text: str) -> None:
        self.spoken.append(text)

    async def speak_cached_and_wait(self, text: str, *, cache_key: str) -> bool:
        self.cached_spoken.append((text, cache_key))
        return True

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def acknowledge(self) -> None:
        self.ack_count += 1

    def drain_buffers(self) -> None:
        return None

    def start_playback(self) -> None:
        return None

    def stop_playback(self) -> None:
        return None

    def wait_speaking_done(self) -> None:
        return None

    def play_thinking(self) -> None:
        return None

    def mark_interaction_turn(self) -> None:
        return None

    def arm_processing_feedback(self, _cancel_token=None) -> bool:
        self._processing_feedback_armed = True
        return True

    def cancel_processing_feedback(self) -> None:
        self._processing_feedback_armed = False

    @property
    def processing_feedback_armed(self) -> bool:
        return self._processing_feedback_armed

    @property
    def is_muted(self) -> bool:
        return self._muted

    def mute(self) -> None:
        self._muted = True

    def unmute(self) -> None:
        self._muted = False


class _Pipeline:
    last_spoken_text = ""

    def __init__(self) -> None:
        self.process_calls: list[str] = []
        self.skill_calls: list[tuple[str, str]] = []
        self.memory_calls: list[str] = []

    def start_idle_reflection(self):
        return None

    def has_pending_tool_approval(self) -> bool:
        return False

    async def handle_pending_tool_response(self, _text: str):
        return None

    def start_memory_prefetch(self, text: str):
        self.memory_calls.append(text)
        return asyncio.create_task(asyncio.sleep(0, result=""))

    async def process(self, text: str, **_kwargs):
        self.process_calls.append(text)
        return "reply"

    async def execute_skill(self, skill_name: str, text: str):
        self.skill_calls.append((skill_name, text))
        return "result"


class _BlockedDispatcher:
    def __init__(self, outcome: SkillOutcome) -> None:
        self.outcome = outcome
        self.dispatch_calls: list[tuple[str, str]] = []

    @property
    def has_active_agent_task(self) -> bool:
        return False

    async def can_execute(self, _skill_name: str, _user_text: str = "", **_kwargs):
        return self.outcome

    async def dispatch(self, skill_name: str, user_text: str, **_kwargs):
        self.dispatch_calls.append((skill_name, user_text))
        return "unexpected"


def _skill(name: str, *, enabled: bool = True):
    return SimpleNamespace(
        name=name,
        enabled=enabled,
        safety_level="normal",
        execution="skill_executor",
        depends=[],
    )


def _gate(*, skill, executor) -> tuple[SkillGate, _Audio]:
    manager = MagicMock()
    manager.get.return_value = skill
    audio = _Audio([])
    gate = SkillGate(
        skill_manager=manager,
        skill_executor=executor,
        audio=audio,
        conversation=MagicMock(),
    )
    return gate, audio


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    ["nav_gateway_unconfigured", "nav_pose_stale"],
)
async def test_nav_preflight_failures_speak_one_fixed_customer_message(reason: str) -> None:
    class Executor:
        async def preflight_skill(self, _skill):
            return False, reason

        execute = AsyncMock(side_effect=AssertionError("executor must not run"))

    gate, audio = _gate(skill=_skill("nav_query"), executor=Executor())

    result = await gate.execute_skill("nav_query", "当前位置", source="voice")

    assert result.startswith("[Skill]")
    assert audio.spoken == [NAV_LOCATION_UNAVAILABLE_MESSAGE]


@pytest.mark.asyncio
async def test_disabled_nav_speaks_one_fixed_customer_message() -> None:
    executor = SimpleNamespace(execute=AsyncMock())
    gate, audio = _gate(skill=_skill("nav_query", enabled=False), executor=executor)

    result = await gate.execute_skill("nav_query", "当前位置", source="voice")

    assert "Disabled" in result
    assert audio.spoken == [NAV_LOCATION_UNAVAILABLE_MESSAGE]
    executor.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_executor_marker_is_never_sent_to_tts() -> None:
    executor = SimpleNamespace(execute=AsyncMock(return_value="[Error] private backend detail"))
    gate, audio = _gate(skill=_skill("patrol"), executor=executor)

    result = await gate.execute_skill("patrol", "开始巡逻", source="voice")

    assert result == "[Error] private backend detail"
    assert len(audio.spoken) == 1
    assert not audio.spoken[0].startswith("[Error]")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "status", "expected_ready", "expected_reason"),
    [
        (False, {}, False, "nav_gateway_unconfigured"),
        (
            True,
            {"ready": False, "pose_fresh": False, "reason": "odometry_missing"},
            False,
            "nav_pose_stale",
        ),
        (
            True,
            {"ready": True, "pose_fresh": True, "has_odometry": True},
            True,
            "ready",
        ),
    ],
)
async def test_real_nav_preflight_interprets_gateway_readiness(
    configured: bool,
    status: dict,
    expected_ready: bool,
    expected_reason: str,
) -> None:
    client = SimpleNamespace(
        is_configured=lambda: configured,
        status=lambda: status,
    )
    nav_tool = SimpleNamespace(_navigation_client=client)
    registry = SimpleNamespace(get=lambda name: nav_tool if name == "nav_status" else None)
    executor = SkillExecutor(None, registry)  # type: ignore[arg-type]

    ready, reason = await executor.preflight_skill(_skill("nav_query"))

    assert ready is expected_ready
    assert reason == expected_reason


@pytest.mark.asyncio
async def test_location_preflight_runs_before_preface_and_stops_dispatch() -> None:
    audio = _Audio([("当前位置", "keyword"), ("exit", "keyword")])
    pipeline = _Pipeline()
    dispatcher = _BlockedDispatcher(
        SkillOutcome.blocked(
            code="nav_disabled",
            result="[Skill] Disabled: nav_query",
            user_message=NAV_LOCATION_UNAVAILABLE_MESSAGE,
        )
    )
    loop = VoiceLoop(
        router=IntentRouter(),
        pipeline=pipeline,
        audio=audio,
        dispatcher=dispatcher,  # type: ignore[arg-type]
    )

    await loop.run()

    assert dispatcher.dispatch_calls == []
    assert pipeline.skill_calls == []
    assert "当前位置" not in pipeline.memory_calls
    assert audio.cached_spoken == []
    assert audio.spoken == [NAV_LOCATION_UNAVAILABLE_MESSAGE]


@pytest.mark.asyncio
async def test_location_without_readiness_probe_fails_closed_before_preface() -> None:
    audio = _Audio([("当前位置", "keyword"), ("exit", "keyword")])
    pipeline = _Pipeline()
    loop = VoiceLoop(router=IntentRouter(), pipeline=pipeline, audio=audio)

    await loop.run()

    assert pipeline.skill_calls == []
    assert "当前位置" not in pipeline.memory_calls
    assert audio.cached_spoken == []
    assert audio.spoken == [NAV_LOCATION_UNAVAILABLE_MESSAGE]


@pytest.mark.asyncio
async def test_kws_unavailable_blocks_general_turn_before_ack_memory_or_llm() -> None:
    audio = _Audio(
        [("给我讲个故事", "kws_unavailable_safety_only"), ("exit", "keyword")]
    )
    pipeline = _Pipeline()
    loop = VoiceLoop(router=IntentRouter(), pipeline=pipeline, audio=audio)

    await loop.run()

    assert pipeline.process_calls == []
    assert pipeline.skill_calls == []
    assert "给我讲个故事" not in pipeline.memory_calls
    assert audio.spoken == []
    assert audio.ack_count == 1  # only the explicit, wake-authorized exit turn


@pytest.mark.asyncio
async def test_kws_unavailable_allows_local_stop_without_ack_or_dispatch() -> None:
    class StopRouter:
        def route(self, text: str) -> Intent:
            if text == "stop":
                return Intent(
                    type=IntentType.VOICE_TRIGGER,
                    skill_name="stop_speaking",
                    raw_text=text,
                )
            return Intent(type=IntentType.COMMAND, command="exit", raw_text=text)

    audio = _Audio(
        [("stop", "kws_unavailable_safety_only"), ("exit", "keyword")]
    )
    pipeline = _Pipeline()
    loop = VoiceLoop(router=StopRouter(), pipeline=pipeline, audio=audio)  # type: ignore[arg-type]

    await loop.run()

    assert pipeline.process_calls == []
    assert pipeline.skill_calls == []
    assert "stop" not in pipeline.memory_calls
    assert audio.ack_count == 1  # exit only
