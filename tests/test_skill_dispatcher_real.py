"""Real execution-path tests for SkillDispatcher.

Gap analysis (what existing mock-heavy tests missed):
  P0  execute_skill_sync() — sync bridge path, tool execution must walk here
  P1  TimeoutError actually triggered → "[超时]" result + step recorded
  P1  dispatch() step ≥2 audio feedback (speak "继续执行第N步")
  P1  complete_mission() multi-step vs single-step audio
  P2  non-TimeoutError propagates from dispatch()
  P2  history_for_context truncates result at 200 chars
  P2  source locked to first dispatch
  P2  get_skill_catalog_for_prompt: empty / multi-skill format
  P2  handle_general() completes active mission and calls pipeline.process()

Status BEFORE: all code paths untested. Status AFTER: fully covered.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.pipeline.skill_dispatcher import SkillDispatcher


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_skill(name: str = "navigate", timeout: int = 30) -> MagicMock:
    skill = MagicMock()
    skill.name = name
    skill.description = f"{name}技能"
    skill.timeout = timeout
    return skill


def _make_dispatcher(
    *,
    skill: MagicMock | None = None,
    execute_skill_result: str = "技能执行完成",
) -> tuple[SkillDispatcher, MagicMock, MagicMock, MagicMock]:
    """Return (dispatcher, mock_pipeline, mock_skill_manager, mock_audio)."""
    mock_pipeline = MagicMock()
    mock_pipeline.execute_skill = AsyncMock(return_value=execute_skill_result)
    mock_pipeline.process = AsyncMock(return_value="LLM回复")

    mock_skill_manager = MagicMock()
    mock_skill_manager.get.return_value = skill
    mock_skill_manager.get_skill_catalog.return_value = "navigate, get_time"
    mock_skill_manager.get_enabled.return_value = [skill] if skill else []

    mock_audio = MagicMock()

    dispatcher = SkillDispatcher(
        pipeline=mock_pipeline,
        skill_manager=mock_skill_manager,
        audio=mock_audio,
    )
    return dispatcher, mock_pipeline, mock_skill_manager, mock_audio


# ── Group 1: execute_skill_sync() ─────────────────────────────────────────────


class TestExecuteSkillSync:
    """P0 — sync bridge path used by dispatch_skill tool via asyncio.to_thread."""

    def test_sync_no_loop_returns_error(self):
        """_loop is None → [Error] 事件循环未就绪."""
        skill = _make_skill()
        dispatcher, _, _, _ = _make_dispatcher(skill=skill)
        assert dispatcher._loop is None

        result = dispatcher.execute_skill_sync("navigate", "去仓库")
        assert "[Error]" in result
        assert "事件循环" in result

    def test_sync_loop_not_running_returns_error(self):
        """Loop exists but is_running()=False → [Error] 事件循环未就绪."""
        skill = _make_skill()
        dispatcher, _, _, _ = _make_dispatcher(skill=skill)
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        dispatcher._loop = mock_loop

        result = dispatcher.execute_skill_sync("navigate", "去仓库")
        assert "[Error]" in result
        assert "事件循环" in result

    async def test_sync_with_running_loop_via_thread(self):
        """Real path: asyncio.to_thread → execute_skill_sync → run_coroutine_threadsafe."""
        skill = _make_skill()
        dispatcher, _, _, _ = _make_dispatcher(skill=skill)

        await dispatcher.dispatch("navigate", "去仓库A")
        assert dispatcher._loop is not None

        result = await asyncio.to_thread(
            dispatcher.execute_skill_sync, "navigate", "再去仓库B"
        )
        assert result == "技能执行完成"

    def test_sync_unknown_skill_returns_error(self):
        """skill_manager.get() returns None → [Error] 技能不存在."""
        dispatcher, _, mock_sm, _ = _make_dispatcher(skill=None)
        mock_sm.get.return_value = None
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        dispatcher._loop = mock_loop

        result = dispatcher.execute_skill_sync("nonexistent", "")
        assert "[Error]" in result
        assert "技能不存在" in result

    def test_sync_future_exception_returns_error(self):
        """future.result() raises Exception → [Error] 技能执行失败.

        Closes the dispatch() coroutine explicitly to prevent RuntimeWarning
        about an unawaited coroutine being passed to the patched mock.
        """
        skill = _make_skill()
        dispatcher, _, _, _ = _make_dispatcher(skill=skill)
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        dispatcher._loop = mock_loop

        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("pipeline crashed")

        def _mock_rcf(coro, loop):
            coro.close()  # close unawaited coroutine — prevents RuntimeWarning
            return mock_future

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_mock_rcf):
            result = dispatcher.execute_skill_sync("navigate", "去仓库")

        assert "[Error]" in result
        assert "技能执行失败" in result

    def test_sync_timeout_is_skill_timeout_plus_15(self):
        """sync_timeout = skill.timeout + 15 (differs from dispatch's +10).

        Closes the dispatch() coroutine to suppress RuntimeWarning about an
        unawaited coroutine being discarded by the patched mock.
        """
        skill = _make_skill(timeout=20)
        dispatcher, _, _, _ = _make_dispatcher(skill=skill)
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        dispatcher._loop = mock_loop

        mock_future = MagicMock()
        mock_future.result.return_value = "技能执行完成"

        def _mock_rcf(coro, loop):
            coro.close()  # close unawaited coroutine — prevents RuntimeWarning
            return mock_future

        with patch("asyncio.run_coroutine_threadsafe", side_effect=_mock_rcf):
            dispatcher.execute_skill_sync("navigate", "去仓库")

        mock_future.result.assert_called_once_with(timeout=35.0)  # 20 + 15


# ── Group 2: dispatch() audio feedback ────────────────────────────────────────


class TestDispatchAudioFeedback:

    async def test_first_dispatch_no_audio(self):
        """First skill in a mission must NOT speak."""
        skill = _make_skill()
        dispatcher, _, _, mock_audio = _make_dispatcher(skill=skill)

        await dispatcher.dispatch("navigate", "去仓库")
        mock_audio.speak.assert_not_called()
        mock_audio.start_playback.assert_not_called()

    async def test_second_dispatch_triggers_audio(self):
        """Second step must call speak() and start_playback()."""
        skill = _make_skill()
        dispatcher, _, _, mock_audio = _make_dispatcher(skill=skill)

        await dispatcher.dispatch("navigate", "去仓库")
        mock_audio.speak.reset_mock()
        mock_audio.start_playback.reset_mock()

        await dispatcher.dispatch("navigate", "再去仓库")
        mock_audio.speak.assert_called_once()
        mock_audio.start_playback.assert_called_once()

    async def test_audio_message_includes_step_number_and_skill(self):
        """speak() for step 2 must contain the step number AND the skill name."""
        skill = _make_skill("navigate")
        dispatcher, _, _, mock_audio = _make_dispatcher(skill=skill)

        await dispatcher.dispatch("navigate", "去仓库")
        await dispatcher.dispatch("navigate", "再去仓库")

        spoken = mock_audio.speak.call_args[0][0]
        assert "2" in spoken, f"Step number '2' missing in: {spoken!r}"
        assert "navigate" in spoken, f"Skill name missing in: {spoken!r}"


# ── Group 3: complete_mission() audio feedback ────────────────────────────────


class TestCompleteMissionAudio:

    async def test_complete_multistep_mission_triggers_audio(self):
        """≥2 steps → speak("多步任务完成:...") + start_playback()."""
        skill = _make_skill()
        dispatcher, _, _, mock_audio = _make_dispatcher(skill=skill)

        await dispatcher.dispatch("navigate", "去仓库A")
        await dispatcher.dispatch("navigate", "去仓库B")
        mock_audio.speak.reset_mock()
        mock_audio.start_playback.reset_mock()

        dispatcher.complete_mission()

        mock_audio.speak.assert_called_once()
        spoken = mock_audio.speak.call_args[0][0]
        assert "多步" in spoken or "完成" in spoken, f"Unexpected message: {spoken!r}"
        mock_audio.start_playback.assert_called_once()

    async def test_complete_single_step_mission_no_audio(self):
        """1-step mission → complete silently."""
        skill = _make_skill()
        dispatcher, _, _, mock_audio = _make_dispatcher(skill=skill)

        await dispatcher.dispatch("navigate", "去仓库")
        mock_audio.speak.reset_mock()

        dispatcher.complete_mission()
        mock_audio.speak.assert_not_called()

    def test_complete_no_active_mission_no_audio(self):
        """No active mission → complete_mission() is a no-op."""
        skill = _make_skill()
        dispatcher, _, _, mock_audio = _make_dispatcher(skill=skill)

        dispatcher.complete_mission()
        mock_audio.speak.assert_not_called()


# ── Group 4: TimeoutError actually triggered ──────────────────────────────────


class TestDispatchTimeoutActual:

    async def test_dispatch_timeout_returns_timeout_message(self):
        """TimeoutError from wait_for → '[超时]' in result (not re-raised).

        Closes the execute_skill coroutine inside the mock to prevent
        RuntimeWarning about an unawaited coroutine.
        """
        skill = _make_skill("navigate")
        dispatcher, _, _, _ = _make_dispatcher(skill=skill)

        def _mock_wait_for(coro, timeout):
            coro.close()  # prevent RuntimeWarning
            raise asyncio.TimeoutError()

        with patch(
            "askme.pipeline.skill_dispatcher.asyncio.wait_for",
            side_effect=_mock_wait_for,
        ):
            result = await dispatcher.dispatch("navigate", "去仓库")

        assert "[超时]" in result, f"Expected '[超时]' in: {result!r}"
        assert "navigate" in result

    async def test_dispatch_timeout_step_still_recorded(self):
        """After timeout, step is recorded in mission history with '[超时]' result."""
        skill = _make_skill()
        dispatcher, _, _, _ = _make_dispatcher(skill=skill)

        def _mock_wait_for(coro, timeout):
            coro.close()
            raise asyncio.TimeoutError()

        with patch(
            "askme.pipeline.skill_dispatcher.asyncio.wait_for",
            side_effect=_mock_wait_for,
        ):
            await dispatcher.dispatch("navigate", "去仓库")

        assert dispatcher.current_mission.step_count == 1
        step_result = dispatcher.current_mission.steps[0].result
        assert "[超时]" in step_result, (
            f"Step result must contain '[超时]', got: {step_result!r}"
        )


# ── Group 5: Edge cases and P2 coverage ───────────────────────────────────────


class TestDispatchEdgeCases:

    async def test_dispatch_non_timeout_exception_propagates(self):
        """Non-TimeoutError propagates to caller — only TimeoutError is caught.

        Sets side_effect on the existing AsyncMock instead of replacing it,
        which avoids creating a dangling unawaited coroutine from the old mock.
        """
        skill = _make_skill()
        dispatcher, mock_pipeline, _, _ = _make_dispatcher(skill=skill)
        # Mutate existing AsyncMock — do NOT assign a new one (avoids dangling coros)
        mock_pipeline.execute_skill.side_effect = ValueError("LLM crashed")

        with pytest.raises(ValueError, match="LLM crashed"):
            await dispatcher.dispatch("navigate", "去仓库")

    async def test_history_truncates_long_result_at_200_chars(self):
        """step result >200 chars → history_for_context uses only [:200].

        Captures combined_context via *args to avoid brittle positional coupling;
        if execute_skill's signature changes, the test will still capture correctly.
        """
        skill = _make_skill()
        long_result = "X" * 300
        dispatcher, mock_pipeline, _, _ = _make_dispatcher(
            skill=skill, execute_skill_result=long_result
        )
        await dispatcher.dispatch("navigate", "去仓库A")

        captured_contexts: list[str] = []

        async def capture(*args, **kwargs):
            # combined_context is the third positional arg to execute_skill
            ctx = args[2] if len(args) > 2 else kwargs.get("combined_context", "")
            captured_contexts.append(ctx)
            return "第二步结果"

        mock_pipeline.execute_skill = capture
        await dispatcher.dispatch("navigate", "去仓库B")

        assert len(captured_contexts) == 1
        x_count = captured_contexts[0].count("X")
        assert x_count == 200, (
            f"history_for_context must truncate to 200 chars; found {x_count}"
        )

    async def test_source_locked_to_first_dispatch(self):
        """Mission source is set by first dispatch and never overridden."""
        skill = _make_skill()
        dispatcher, _, _, _ = _make_dispatcher(skill=skill)

        await dispatcher.dispatch("navigate", "去仓库", source="voice")
        await dispatcher.dispatch("navigate", "再去",   source="text")

        assert dispatcher.current_mission.source == "voice", (
            f"Source must stay 'voice', got {dispatcher.current_mission.source!r}"
        )

    def test_skill_catalog_empty_returns_empty_string(self):
        """0 enabled skills → ''."""
        skill = _make_skill()
        dispatcher, _, mock_sm, _ = _make_dispatcher(skill=skill)
        mock_sm.get_enabled.return_value = []

        assert dispatcher.get_skill_catalog_for_prompt() == ""

    def test_skill_catalog_multiple_skills_multiline(self):
        """3 skills → 3 lines in '- name: description' format."""
        skill = _make_skill()
        dispatcher, _, mock_sm, _ = _make_dispatcher(skill=skill)

        s1 = _make_skill("navigate"); s1.description = "导航"
        s2 = _make_skill("search");   s2.description = "搜索"
        s3 = _make_skill("get_time"); s3.description = "获取时间"
        mock_sm.get_enabled.return_value = [s1, s2, s3]

        result = dispatcher.get_skill_catalog_for_prompt()
        lines = [ln for ln in result.strip().splitlines() if ln.strip()]
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}: {lines}"

        # Verify exact "- name: description" format, not just prefix presence
        for skill_obj in (s1, s2, s3):
            expected_line = f"- {skill_obj.name}: {skill_obj.description}"
            assert expected_line in result, (
                f"Expected line {expected_line!r} not found in:\n{result}"
            )

    async def test_handle_general_completes_active_mission(self):
        """handle_general() must end the active mission and call pipeline.process().

        This is a distinct code path from dispatch() that was previously untested.
        """
        skill = _make_skill()
        dispatcher, mock_pipeline, _, _ = _make_dispatcher(skill=skill)

        await dispatcher.dispatch("navigate", "去仓库")
        assert dispatcher.has_active_mission

        await dispatcher.handle_general("查个时间")

        assert not dispatcher.has_active_mission, (
            "handle_general() must complete the active mission"
        )
        mock_pipeline.process.assert_called_once()


class TestBindRuntimeMission:
    async def test_bind_sets_runtime_mission_id(self):
        skill = _make_skill()
        dispatcher, _, _, _ = _make_dispatcher(skill=skill)

        await dispatcher.dispatch("navigate", "去仓库")
        dispatcher.bind_runtime_mission("rt-abc123")

        assert dispatcher.current_mission is not None
        assert dispatcher.current_mission.runtime_mission_id == "rt-abc123"

    async def test_bind_shows_in_summary(self):
        skill = _make_skill()
        dispatcher, _, _, _ = _make_dispatcher(skill=skill)

        await dispatcher.dispatch("navigate", "去仓库")
        dispatcher.bind_runtime_mission("rt-xyz")

        summary = dispatcher.current_mission.summary()
        assert "rt-xyz" in summary

    def test_bind_noop_when_no_mission(self):
        dispatcher, _, _, _ = _make_dispatcher()
        # Should not raise
        dispatcher.bind_runtime_mission("rt-noop")
        assert dispatcher.current_mission is None

    async def test_bind_persisted_in_json(self, tmp_path, monkeypatch):
        import json
        from askme.config import project_root as _pr
        monkeypatch.setattr("askme.pipeline.skill_dispatcher.project_root", lambda: tmp_path)

        skill = _make_skill()
        dispatcher, _, _, _ = _make_dispatcher(skill=skill)
        await dispatcher.dispatch("navigate", "去仓库")
        dispatcher.bind_runtime_mission("rt-persist-test")

        missions_dir = tmp_path / "data" / "missions"
        files = list(missions_dir.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["runtime_mission_id"] == "rt-persist-test"


class TestCompleteMissionFailedState:
    async def test_failed_mission_no_success_audio(self):
        """complete_mission() must NOT speak 多步任务完成 when state is FAILED."""
        from askme.pipeline.skill_dispatcher import MissionState
        skill = _make_skill()
        dispatcher, _, _, mock_audio = _make_dispatcher(skill=skill)

        await dispatcher.dispatch("navigate", "去仓库")
        await dispatcher.dispatch("navigate", "去B点")
        # Manually mark mission as failed (simulates a timeout step)
        dispatcher._current_mission.state = MissionState.FAILED
        mock_audio.speak.reset_mock()

        dispatcher.complete_mission()

        # Must not say "多步任务完成"
        for call in mock_audio.speak.call_args_list:
            text = call[0][0]
            assert "多步任务完成" not in text, f"Should not announce success for failed mission: {text!r}"


class TestHandleGeneralPlanOriginalContext:
    async def test_plan_steps_receive_original_user_text(self):
        """Each plan step's extra_context must contain the original user_text."""
        from unittest.mock import AsyncMock as _AsyncMock, MagicMock as _MagicMock
        from askme.pipeline.planner_agent import PlanStep

        skill = _make_skill()
        dispatcher, mock_pipeline, _, _ = _make_dispatcher(skill=skill)

        planner = _MagicMock()
        planner.plan = _AsyncMock(return_value=[
            PlanStep(skill_name="navigate", intent="前往仓库"),
            PlanStep(skill_name="navigate", intent="前往B点"),
        ])
        dispatcher._planner = planner

        original_text = "去东区仓库取货然后送到B点"
        await dispatcher.handle_general(original_text, source="text")

        # execute_skill is called as (skill_name, user_text, extra_context)
        calls = mock_pipeline.execute_skill.call_args_list
        assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
        for call in calls:
            args = call[0]
            extra_context = args[2] if len(args) > 2 else ""
            assert original_text in extra_context, (
                f"original user_text not found in extra_context: {extra_context!r}"
            )
