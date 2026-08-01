"""Tests for the SkillDispatcher orchestration layer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from askme.pipeline.skill_dispatcher import MissionContext, SkillDispatcher, _AgentTaskTurn
from askme.tools.builtin_tools import DispatchSkillTool

from askme.conversation import (
    InteractionInput,
    InteractionTurnManager,
    TurnStatus,
    VoiceTurnLedger,
)
from askme.pipeline.core.protocols import SkillExecutionDisposition

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.execute_skill = AsyncMock(return_value="技能执行完成")
    pipeline.process = AsyncMock(return_value="LLM回复")
    return pipeline


@pytest.fixture()
def mock_skill_manager():
    mgr = MagicMock()
    skill = MagicMock()
    skill.name = "navigate"
    skill.description = "导航技能"
    mgr.get.return_value = skill
    mgr.get_enabled.return_value = [skill]
    mgr.get_skill_catalog.return_value = "navigate, get_time"
    return mgr


@pytest.fixture()
def mock_audio():
    return MagicMock()


@pytest.fixture()
def dispatcher(mock_pipeline, mock_skill_manager, mock_audio):
    return SkillDispatcher(
        pipeline=mock_pipeline,
        skill_manager=mock_skill_manager,
        audio=mock_audio,
    )


class CanonicalAgentPipeline:
    def __init__(self, ledger, result="后台任务完成"):
        self._turn_ledger = ledger
        self._interaction_turns = InteractionTurnManager(ledger)
        self.result = result
        self.skill_calls = []
        self.settle_calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def _open_direct_interaction(self, **kwargs):
        return self._interaction_turns.open(
            InteractionInput(
                user_text=kwargs["user_text"],
                source=kwargs["source"],
                thread_id=kwargs.get("conversation_session_id"),
                turn_id=kwargs.get("voice_turn_id"),
                channel=kwargs["source"],
                metadata=kwargs.get("metadata") or {},
                cancel_token=kwargs.get("turn_cancel_token"),
            )
        )

    def _settle_direct_interaction(self, interaction, outcome):
        self.settle_calls.append((interaction, outcome))
        return self._interaction_turns.settle(interaction, outcome)

    def _classify_skill_execution_result(self, result, skill_name):
        assert skill_name == "agent_task"
        if str(result).startswith("cancel:"):
            return SkillExecutionDisposition(status="cancelled", code="operator_cancelled")
        if str(result).startswith("fail:"):
            return SkillExecutionDisposition(status="failed", code="agent_failed")
        return SkillExecutionDisposition(status="succeeded", code="succeeded")

    async def execute_skill(
        self,
        skill_name,
        user_text,
        extra_context="",
        source="voice",
        **kwargs,
    ):
        self.skill_calls.append((skill_name, user_text, extra_context, source, kwargs))
        self.started.set()
        await self.release.wait()
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


# ── MissionContext tests ──────────────────────────────────────────


class TestMissionContext:
    def test_new_mission_has_no_steps(self):
        ctx = MissionContext()
        assert ctx.step_count == 0
        assert ctx.history_for_context() == ""

    def test_add_step(self):
        ctx = MissionContext(source="voice")
        ctx.add_step("navigate", "去仓库", "已开始导航")
        assert ctx.step_count == 1
        assert "navigate" in ctx.summary()
        assert "voice" in ctx.summary()

    def test_history_for_context(self):
        ctx = MissionContext()
        ctx.add_step("navigate", "去仓库", "已开始导航")
        ctx.add_step("environment_report", "查温度", "温度28度")
        history = ctx.history_for_context()
        assert "步骤1" in history
        assert "步骤2" in history
        assert "navigate" in history
        assert "environment_report" in history

    def test_mission_id_is_unique(self):
        a = MissionContext()
        b = MissionContext()
        assert a.mission_id != b.mission_id


# ── SkillDispatcher tests ─────────────────────────────────────────


class TestSkillDispatcher:
    async def test_dispatch_creates_mission(self, dispatcher):
        assert not dispatcher.has_active_mission
        await dispatcher.dispatch("navigate", "去仓库", source="voice")
        assert dispatcher.has_active_mission
        assert dispatcher.current_mission.step_count == 1

    async def test_dispatch_tracks_steps(self, dispatcher):
        await dispatcher.dispatch("navigate", "去仓库")
        await dispatcher.dispatch("environment_report", "查温度")
        mission = dispatcher.current_mission
        assert mission.step_count == 2
        assert mission.steps[0].skill_name == "navigate"
        assert mission.steps[1].skill_name == "environment_report"

    async def test_handle_general_completes_mission(self, dispatcher):
        await dispatcher.dispatch("navigate", "去仓库")
        assert dispatcher.has_active_mission
        await dispatcher.handle_general("今天天气怎么样")
        assert not dispatcher.has_active_mission

    async def test_complete_mission_returns_context(self, dispatcher):
        await dispatcher.dispatch("navigate", "去仓库")
        mission = dispatcher.complete_mission()
        assert mission is not None
        assert mission.step_count == 1
        assert not dispatcher.has_active_mission

    async def test_complete_empty_returns_none(self, dispatcher):
        assert dispatcher.complete_mission() is None

    async def test_dispatch_calls_pipeline(self, dispatcher, mock_pipeline):
        await dispatcher.dispatch("navigate", "去仓库")
        mock_pipeline.execute_skill.assert_called_once_with(
            "navigate", "去仓库", "", source="voice"
        )

    async def test_handle_general_calls_pipeline(self, dispatcher, mock_pipeline):
        await dispatcher.handle_general("你好", memory_task=None)
        mock_pipeline.process.assert_called_once_with("你好", memory_task=None, source="voice")

    async def test_source_tracking(self, dispatcher):
        await dispatcher.dispatch("navigate", "去仓库", source="text")
        assert dispatcher.current_mission.source == "text"

    async def test_dispatch_with_extra_context(self, dispatcher, mock_pipeline):
        await dispatcher.dispatch("navigate", "去仓库", extra_context="紧急任务")
        mock_pipeline.execute_skill.assert_called_once()

    async def test_skill_catalog_for_prompt(self, dispatcher):
        catalog = dispatcher.get_skill_catalog_for_prompt()
        assert "navigate" in catalog
        assert "导航技能" in catalog

    async def test_agent_task_canonical_ack_and_final_share_one_turn(
        self,
        tmp_path,
        mock_skill_manager,
        mock_audio,
    ):
        ledger = VoiceTurnLedger(tmp_path / "agent-task-ledger.jsonl")
        pipeline = CanonicalAgentPipeline(ledger, result="后台任务完成")
        skill = mock_skill_manager.get.return_value
        skill.name = "agent_task"
        skill.timeout = 120
        dispatcher = SkillDispatcher(
            pipeline=pipeline,
            skill_manager=mock_skill_manager,
            audio=mock_audio,
        )

        acknowledgement = await dispatcher.dispatch(
            "agent_task",
            "分析传感器日志",
            source="text",
            conversation_session_id="thread-agent",
            voice_turn_id="turn-agent",
        )
        await pipeline.started.wait()

        opened = ledger.get_turn("turn-agent")
        assert opened is not None
        assert opened.status is TurnStatus.SPEAKING
        assert opened.assistant_text == acknowledgement
        assert len(ledger.list_turns(thread_id="thread-agent")) == 1

        pipeline.release.set()
        assert dispatcher._active_agent_task is not None
        await dispatcher._active_agent_task

        settled = ledger.get_turn("turn-agent")
        assert settled is not None
        assert settled.status is TurnStatus.COMMITTED
        assert settled.assistant_text == "后台任务完成"
        assert len(ledger.list_turns(thread_id="thread-agent")) == 1
        assert pipeline.skill_calls[0][4]["record_turn"] is False
        assert len(pipeline.settle_calls) == 1

    async def test_agent_task_canonical_failure_settles_same_turn(
        self,
        tmp_path,
        mock_skill_manager,
        mock_audio,
    ):
        ledger = VoiceTurnLedger(tmp_path / "agent-task-failure-ledger.jsonl")
        pipeline = CanonicalAgentPipeline(ledger, result="fail: executor failed")
        skill = mock_skill_manager.get.return_value
        skill.name = "agent_task"
        skill.timeout = 120
        dispatcher = SkillDispatcher(
            pipeline=pipeline,
            skill_manager=mock_skill_manager,
            audio=mock_audio,
        )

        await dispatcher.dispatch(
            "agent_task",
            "分析传感器日志",
            source="text",
            conversation_session_id="thread-agent-fail",
            voice_turn_id="turn-agent-fail",
        )
        await pipeline.started.wait()
        pipeline.release.set()
        assert dispatcher._active_agent_task is not None
        await dispatcher._active_agent_task

        settled = ledger.get_turn("turn-agent-fail")
        assert settled is not None
        assert settled.status is TurnStatus.FAILED
        assert settled.failure_reason == "agent_failed"
        assert len(ledger.list_turns(thread_id="thread-agent-fail")) == 1

    async def test_agent_task_timeout_settles_same_turn_as_failed(
        self,
        tmp_path,
        mock_skill_manager,
        mock_audio,
    ):
        ledger = VoiceTurnLedger(tmp_path / "agent-task-timeout-ledger.jsonl")
        pipeline = CanonicalAgentPipeline(ledger)
        skill = mock_skill_manager.get.return_value
        skill.name = "agent_task"
        skill.timeout = -9.99
        dispatcher = SkillDispatcher(
            pipeline=pipeline,
            skill_manager=mock_skill_manager,
            audio=mock_audio,
        )

        await dispatcher.dispatch(
            "agent_task",
            "持续执行直到超时",
            source="text",
            conversation_session_id="thread-agent-timeout",
            voice_turn_id="turn-agent-timeout",
        )
        assert dispatcher._active_agent_task is not None
        await dispatcher._active_agent_task

        settled = ledger.get_turn("turn-agent-timeout")
        assert settled is not None
        assert settled.status is TurnStatus.FAILED
        assert settled.failure_reason == "execution_timeout"
        assert len(ledger.list_turns(thread_id="thread-agent-timeout")) == 1

    async def test_agent_task_cancel_settles_same_turn_once(
        self,
        tmp_path,
        mock_skill_manager,
        mock_audio,
    ):
        ledger = VoiceTurnLedger(tmp_path / "agent-task-cancel-ledger.jsonl")
        pipeline = CanonicalAgentPipeline(ledger)
        skill = mock_skill_manager.get.return_value
        skill.name = "agent_task"
        skill.timeout = 120
        dispatcher = SkillDispatcher(
            pipeline=pipeline,
            skill_manager=mock_skill_manager,
            audio=mock_audio,
        )

        await dispatcher.dispatch(
            "agent_task",
            "停止前一直执行",
            source="text",
            conversation_session_id="thread-agent-cancel",
            voice_turn_id="turn-agent-cancel",
        )
        await pipeline.started.wait()
        task = dispatcher._active_agent_task
        assert task is not None

        assert dispatcher.cancel_active_agent_task() is True
        with pytest.raises(asyncio.CancelledError):
            await task

        settled = ledger.get_turn("turn-agent-cancel")
        assert settled is not None
        assert settled.status is TurnStatus.CANCELLED
        assert settled.cancel_reason == "agent_task_cancelled"
        assert len(pipeline.settle_calls) == 1

    async def test_agent_task_duplicate_final_settlement_is_idempotent(
        self,
        tmp_path,
        mock_skill_manager,
        mock_audio,
    ):
        ledger = VoiceTurnLedger(tmp_path / "agent-task-idempotent-ledger.jsonl")
        pipeline = CanonicalAgentPipeline(ledger)
        dispatcher = SkillDispatcher(
            pipeline=pipeline,
            skill_manager=mock_skill_manager,
            audio=mock_audio,
        )
        context = pipeline._open_direct_interaction(
            user_text="分析日志",
            source="text",
            conversation_session_id="thread-agent-idempotent",
            voice_turn_id="turn-agent-idempotent",
            turn_cancel_token=None,
            metadata={"interaction": "agent_task"},
        )
        lifecycle = _AgentTaskTurn(context=context)

        dispatcher._settle_agent_task_result(lifecycle, "后台任务完成")
        dispatcher._settle_agent_task_result(lifecycle, "fail: duplicate")

        settled = ledger.get_turn("turn-agent-idempotent")
        assert settled is not None
        assert settled.status is TurnStatus.COMMITTED
        assert settled.assistant_text == "后台任务完成"
        assert len(pipeline.settle_calls) == 1

    async def test_agent_task_without_canonical_interaction_fails_closed(
        self,
        mock_skill_manager,
        mock_audio,
    ):
        class AgentPipeline:
            def __init__(self):
                self.direct_replies = []
                self.skill_calls = []

            async def record_direct_reply(
                self,
                user_text,
                assistant_text,
                **kwargs,
            ):
                self.direct_replies.append((user_text, assistant_text, kwargs))
                return assistant_text

            async def execute_skill(
                self,
                skill_name,
                user_text,
                extra_context="",
                source="voice",
                **kwargs,
            ):
                self.skill_calls.append((skill_name, user_text, extra_context, source, kwargs))
                return "后台任务完成"

        pipeline = AgentPipeline()
        skill = mock_skill_manager.get.return_value
        skill.name = "agent_task"
        skill.timeout = 120
        dispatcher = SkillDispatcher(
            pipeline=pipeline,
            skill_manager=mock_skill_manager,
            audio=mock_audio,
        )

        acknowledgement = await dispatcher.dispatch(
            "agent_task",
            "分析传感器日志",
            source="text",
            conversation_session_id="thread-agent",
            voice_turn_id="turn-agent",
        )

        assert "canonical_interaction_unavailable" in acknowledgement
        assert pipeline.direct_replies == []
        assert pipeline.skill_calls == []
        assert not dispatcher.has_active_agent_task
        assert not dispatcher.has_active_mission

    async def test_agent_task_null_canonical_context_fails_closed(
        self,
        mock_skill_manager,
        mock_audio,
    ):
        class NullContextPipeline:
            def __init__(self):
                self.skill_calls = []

            def _open_direct_interaction(self, **kwargs):
                return None

            def _settle_direct_interaction(self, interaction, outcome):
                raise AssertionError("settlement should not run without a context")

            async def execute_skill(
                self,
                skill_name,
                user_text,
                extra_context="",
                source="voice",
                **kwargs,
            ):
                self.skill_calls.append((skill_name, user_text, extra_context, source, kwargs))
                return "后台任务完成"

        pipeline = NullContextPipeline()
        skill = mock_skill_manager.get.return_value
        skill.name = "agent_task"
        skill.timeout = 120
        dispatcher = SkillDispatcher(
            pipeline=pipeline,
            skill_manager=mock_skill_manager,
            audio=mock_audio,
        )

        result = await dispatcher.dispatch(
            "agent_task",
            "分析传感器日志",
            source="text",
            conversation_session_id="thread-agent-null",
            voice_turn_id="turn-agent-null",
        )

        assert "canonical_interaction_unavailable" in result
        assert pipeline.skill_calls == []
        assert not dispatcher.has_active_agent_task
        assert not dispatcher.has_active_mission

    async def test_agent_task_swallowed_final_settlement_marks_visible_failure(
        self,
        tmp_path,
        mock_skill_manager,
        mock_audio,
    ):
        ledger = VoiceTurnLedger(tmp_path / "agent-task-settlement-failure-ledger.jsonl")

        class SwallowingSettlePipeline(CanonicalAgentPipeline):
            def _settle_direct_interaction(self, interaction, outcome):
                self.settle_calls.append((interaction, outcome))
                return None

        pipeline = SwallowingSettlePipeline(ledger, result="后台任务完成")
        skill = mock_skill_manager.get.return_value
        skill.name = "agent_task"
        skill.timeout = 120
        dispatcher = SkillDispatcher(
            pipeline=pipeline,
            skill_manager=mock_skill_manager,
            audio=mock_audio,
        )

        await dispatcher.dispatch(
            "agent_task",
            "分析传感器日志",
            source="text",
            conversation_session_id="thread-agent-settle-fail",
            voice_turn_id="turn-agent-settle-fail",
        )
        await pipeline.started.wait()
        pipeline.release.set()
        assert dispatcher._active_agent_task is not None
        await dispatcher._active_agent_task

        settled = ledger.get_turn("turn-agent-settle-fail")
        assert settled is not None
        assert settled.status is TurnStatus.FAILED
        assert settled.failure_reason == "agent_task_settlement_failed"
        assert len(ledger.list_turns(thread_id="thread-agent-settle-fail")) == 1


# ── DispatchSkillTool tests ───────────────────────────────────────


class TestDispatchSkillTool:
    def test_no_dispatcher_returns_error(self):
        tool = DispatchSkillTool()
        result = tool.execute(skill_name="navigate")
        assert "[Error]" in result

    def test_empty_skill_name_returns_error(self):
        tool = DispatchSkillTool()
        tool.set_dispatcher(MagicMock())
        result = tool.execute(skill_name="")
        assert "[Error]" in result

    def test_tool_definition_format(self):
        tool = DispatchSkillTool()
        defn = tool.get_definition()
        assert defn["type"] == "function"
        assert defn["function"]["name"] == "dispatch_skill"
        assert "skill_name" in defn["function"]["parameters"]["properties"]

    def test_dispatches_to_dispatcher(self):
        tool = DispatchSkillTool()
        mock_dispatcher = MagicMock()
        mock_dispatcher.execute_skill_sync.return_value = "导航已开始"
        tool.set_dispatcher(mock_dispatcher)
        result = tool.execute(skill_name="navigate", reason="用户想去仓库")
        assert result == "导航已开始"
        mock_dispatcher.execute_skill_sync.assert_called_once_with("navigate", "用户想去仓库")

    def test_nonexistent_skill(self):
        tool = DispatchSkillTool()
        mock_dispatcher = MagicMock()
        mock_dispatcher.execute_skill_sync.return_value = "[Error] 技能不存在: foo"
        tool.set_dispatcher(mock_dispatcher)
        result = tool.execute(skill_name="foo")
        assert "技能不存在" in result
