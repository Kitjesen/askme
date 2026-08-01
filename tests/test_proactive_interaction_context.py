"""Conversation-context regressions for autonomous proactive work."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from askme.runtime.core.module import ModuleRegistry
from askme.runtime.modules.proactive_module import ProactiveModule


class _RecordingLedger:
    def __init__(self) -> None:
        self.resolve_calls: list[dict[str, Any]] = []

    def resolve_thread(self, **kwargs: Any) -> SimpleNamespace:
        self.resolve_calls.append(kwargs)
        return SimpleNamespace(thread_id=kwargs["thread_id"])


class _ContextAwarePipeline:
    def __init__(self) -> None:
        self.turn_ledger = _RecordingLedger()
        self.skill_calls: list[dict[str, Any]] = []

    async def execute_skill(
        self,
        skill_name: str,
        user_text: str,
        *,
        source: str,
        conversation_session_id: str,
        voice_turn_id: str,
        turn_cancel_token: Any,
    ) -> str:
        self.skill_calls.append(
            {
                "skill_name": skill_name,
                "user_text": user_text,
                "source": source,
                "conversation_session_id": conversation_session_id,
                "voice_turn_id": voice_turn_id,
                "turn_cancel_token": turn_cancel_token,
            }
        )
        return "handled"


@pytest.mark.asyncio
async def test_solve_jobs_share_scoped_thread_but_get_unique_turns_and_shutdown_token() -> None:
    pipeline = _ContextAwarePipeline()
    module = ProactiveModule()
    module.pipeline_in = SimpleNamespace(brain_pipeline=pipeline)
    module.build(
        {
            "robot": {
                "robot_id": "robot-private-7",
                "site_id": "site-private-3",
            },
            "proactive": {
                "enabled": True,
                "patrol_interval": 600,
            },
        },
        ModuleRegistry(),
    )

    await module.start()
    stop_token = module._stop_event
    try:
        assert module.agent._solve_callback is not None
        await module.agent._solve_callback("first anomaly")
        await module.agent._solve_callback("second anomaly")
    finally:
        await module.stop()

    [thread_call] = pipeline.turn_ledger.resolve_calls
    thread_id = thread_call["thread_id"]
    assert thread_call["channel"] == "proactive"
    assert thread_call["robot_id"] == "robot-private-7"
    assert thread_call["site_id"] == "site-private-3"
    assert "robot-private-7" not in thread_id
    assert "site-private-3" not in thread_id

    assert [call["source"] for call in pipeline.skill_calls] == [
        "proactive",
        "proactive",
    ]
    assert {call["conversation_session_id"] for call in pipeline.skill_calls} == {thread_id}
    assert len({call["voice_turn_id"] for call in pipeline.skill_calls}) == 2
    assert all(call["turn_cancel_token"] is stop_token for call in pipeline.skill_calls)
    assert stop_token.is_set()


class _ContextAwareLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        context: Any,
        cancel_token: Any,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "context": context,
                "cancel_token": cancel_token,
            }
        )
        return "NORMAL"


@pytest.mark.asyncio
async def test_direct_llm_jobs_share_session_and_token_but_get_unique_turns() -> None:
    from askme.pipeline.reactions.proactive_agent import ProactiveAgent

    llm = _ContextAwareLLM()
    cancel_token = asyncio.Event()
    agent = ProactiveAgent(
        vision=None,
        audio=None,
        episodic=None,
        llm=llm,
        config={
            "proactive": {
                "enabled": True,
            },
        },
    )
    agent.set_interaction_context(
        session_id="opaque-proactive-thread",
        cancel_token=cancel_token,
    )
    agent._scene_history.append("baseline")

    await agent._detect_anomaly("first scene")
    await agent._detect_anomaly("second scene")

    contexts = [call["context"] for call in llm.calls]
    assert {context.session_id for context in contexts} == {"opaque-proactive-thread"}
    assert {context.channel for context in contexts} == {"proactive"}
    assert len({context.turn_id for context in contexts}) == 2
    assert all(call["cancel_token"] is cancel_token for call in llm.calls)


class _LegacyPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def execute_skill(self, skill_name: str, user_text: str) -> str:
        self.calls.append((skill_name, user_text))
        return "legacy handled"


@pytest.mark.asyncio
async def test_solve_callback_preserves_two_argument_legacy_pipeline_signature() -> None:
    pipeline = _LegacyPipeline()
    module = ProactiveModule()
    module.pipeline_in = SimpleNamespace(brain_pipeline=pipeline)
    module.build(
        {
            "proactive": {
                "enabled": False,
            },
        },
        ModuleRegistry(),
    )

    assert module.agent._solve_callback is not None
    result = await module.agent._solve_callback("legacy anomaly")

    assert result == "legacy handled"
    assert pipeline.calls == [("solve_problem", "legacy anomaly")]


class _LegacyLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
            }
        )
        return "NORMAL"


@pytest.mark.asyncio
async def test_direct_llm_preserves_legacy_chat_signature() -> None:
    from askme.pipeline.reactions.proactive_agent import ProactiveAgent

    llm = _LegacyLLM()
    agent = ProactiveAgent(
        vision=None,
        audio=None,
        episodic=None,
        llm=llm,
        config={
            "proactive": {
                "enabled": True,
            },
        },
    )
    agent.set_interaction_context(
        session_id="opaque-proactive-thread",
        cancel_token=asyncio.Event(),
    )
    agent._scene_history.append("baseline")

    result = await agent._detect_anomaly("legacy scene")

    assert result is None
    assert len(llm.calls) == 1
    assert set(llm.calls[0]) == {"messages", "model", "temperature"}


@pytest.mark.asyncio
async def test_restart_reuses_thread_with_fresh_token_and_turn() -> None:
    pipeline = _ContextAwarePipeline()
    module = ProactiveModule()
    module.pipeline_in = SimpleNamespace(brain_pipeline=pipeline)
    module.build(
        {
            "robot": {
                "robot_id": "robot-7",
                "site_id": "site-3",
            },
            "proactive": {
                "enabled": True,
                "patrol_interval": 600,
            },
        },
        ModuleRegistry(),
    )
    thread_id = module._proactive_thread_id

    await module.start()
    first_token = module._stop_event
    try:
        assert module.agent._solve_callback is not None
        await module.agent._solve_callback("before restart")
    finally:
        await module.stop()

    await module.start()
    second_token = module._stop_event
    try:
        assert module.agent._solve_callback is not None
        await module.agent._solve_callback("after restart")
    finally:
        await module.stop()

    assert len(pipeline.turn_ledger.resolve_calls) == 1
    assert module._proactive_thread_id == thread_id
    assert [call["conversation_session_id"] for call in pipeline.skill_calls] == [
        thread_id,
        thread_id,
    ]
    assert first_token is not second_token
    assert first_token.is_set()
    assert second_token.is_set()
    assert [call["turn_cancel_token"] for call in pipeline.skill_calls] == [
        first_token,
        second_token,
    ]
    assert len({call["voice_turn_id"] for call in pipeline.skill_calls}) == 2


class _AutoTaskLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        context: Any,
        cancel_token: Any,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "context": context,
                "cancel_token": cancel_token,
            }
        )
        return "task complete"


@pytest.mark.asyncio
async def test_auto_tasks_use_same_session_with_a_new_turn_per_job() -> None:
    from askme.pipeline.reactions.proactive_agent import ProactiveAgent

    llm = _AutoTaskLLM()
    cancel_token = asyncio.Event()
    agent = ProactiveAgent(
        vision=None,
        audio=None,
        episodic=None,
        llm=llm,
        config={
            "proactive": {
                "enabled": True,
                "auto_tasks": [
                    {
                        "name": "inspect",
                        "interval": 0,
                        "prompt": "inspect the scene",
                    }
                ],
            },
        },
    )
    agent.set_interaction_context(
        session_id="opaque-proactive-thread",
        cancel_token=cancel_token,
    )
    spoken: list[str] = []

    async def record_alert(message: str, **kwargs: Any) -> None:
        _ = kwargs
        spoken.append(message)

    agent._speak_alert = record_alert

    await agent._process_auto_tasks()
    await agent._process_auto_tasks()

    contexts = [call["context"] for call in llm.calls]
    assert spoken == ["task complete", "task complete"]
    assert {context.session_id for context in contexts} == {"opaque-proactive-thread"}
    assert {context.channel for context in contexts} == {"proactive"}
    assert len({context.turn_id for context in contexts}) == 2
    assert all(call["cancel_token"] is cancel_token for call in llm.calls)


class _CancellationAwareLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.context: Any = None
        self.cancel_token: Any = None

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        context: Any,
        cancel_token: Any,
    ) -> str:
        _ = messages, model, temperature
        self.context = context
        self.cancel_token = cancel_token
        self.started.set()
        await cancel_token.wait()
        return "NORMAL"


@pytest.mark.asyncio
async def test_shutdown_token_reaches_in_flight_direct_llm_job() -> None:
    llm = _CancellationAwareLLM()
    module = ProactiveModule()
    module.llm_in = SimpleNamespace(llm_client=llm)
    module.build(
        {
            "proactive": {
                "enabled": True,
                "patrol_interval": 600,
            },
        },
        ModuleRegistry(),
    )
    assert module.agent._llm is llm
    module.agent._scene_history.append("baseline")

    await module.start()
    lifecycle_token = module._stop_event
    job = asyncio.create_task(module.agent._detect_anomaly("current scene"))
    try:
        await asyncio.wait_for(llm.started.wait(), timeout=1.0)
        await module.stop()
        result = await asyncio.wait_for(job, timeout=1.0)
    finally:
        await module.stop()
        if not job.done():
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)

    assert result is None
    assert llm.cancel_token is lifecycle_token
    assert lifecycle_token.is_set()
    assert llm.context.session_id == module._proactive_thread_id
    assert llm.context.channel == "proactive"
