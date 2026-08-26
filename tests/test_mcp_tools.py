"""Tests for MCP tool and resource modules."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_mcp_registration_manifest_is_explicit_tool_and_resource_boundary():
    from askme.mcp.registration import mcp_module_manifest

    modules = mcp_module_manifest()

    assert len(modules) == len(set(modules))
    assert modules == [
        "askme.mcp.resources.contract_resources",
        "askme.mcp.resources.health_resources",
        "askme.mcp.resources.perception_resources",
        "askme.mcp.resources.robot_resources",
        "askme.mcp.resources.skill_resources",
        "askme.mcp.tools.memory_tools",
        "askme.mcp.tools.robot_tools",
        "askme.mcp.tools.skill_tools",
        "askme.mcp.tools.vision_tools",
        "askme.mcp.tools.voice_tools",
    ]
    assert all(
        module.startswith(("askme.mcp.resources.", "askme.mcp.tools.")) for module in modules
    )


def test_mcp_and_pipeline_share_turn_ledger_path_resolution(tmp_path, monkeypatch):
    from askme.conversation.paths import resolve_turn_ledger_path
    from askme.runtime.modules import pipeline_module

    monkeypatch.delenv("ASKME_TURN_LEDGER_PATH", raising=False)
    monkeypatch.setattr(pipeline_module, "_project_root", lambda: tmp_path)
    config = {"conversation": {"turn_ledger_path": "state/turns.jsonl"}}
    expected = tmp_path / "state" / "turns.jsonl"

    assert resolve_turn_ledger_path(config, project_root=tmp_path) == expected
    assert pipeline_module._turn_ledger_path(config) == expected

    environment_path = tmp_path / "environment-turns.jsonl"
    monkeypatch.setenv("ASKME_TURN_LEDGER_PATH", str(environment_path))
    assert resolve_turn_ledger_path(config, project_root=tmp_path) == environment_path
    assert pipeline_module._turn_ledger_path(config) == environment_path


@pytest.mark.asyncio
async def test_mcp_lifespan_reuses_one_ledger_for_the_canonical_turn_manager(
    tmp_path,
    monkeypatch,
):
    import askme.conversation as conversation_core
    import askme.conversation.paths as conversation_paths
    import askme.providers as providers
    from askme.mcp import context as mcp_context
    from askme.mcp import resource_surface
    from askme.memory.core import conversation as legacy_conversation
    from askme.memory.core import episodic_memory, session
    from askme.memory.retrieval import bridge
    from askme.runtime.modules import llm_module as llm_module_module
    from askme.skills.core import skill_executor, skill_manager
    from askme.tools.core import tool_registry

    config = {"brain": {"model": "test-model"}, "runtime": {}}
    ledger_path = tmp_path / "turns.jsonl"
    ledger = object()
    manager = object()
    ledger_constructor = MagicMock(return_value=ledger)
    manager_constructor = MagicMock(return_value=manager)
    monkeypatch.setattr(mcp_context, "get_config", lambda: config)
    monkeypatch.setattr(mcp_context, "validate_config", lambda _config: [])
    monkeypatch.setattr(mcp_context, "get_section", lambda _name: {})
    monkeypatch.setattr(mcp_context, "register_runtime_tool_surface", MagicMock())
    monkeypatch.setattr(conversation_core, "VoiceTurnLedger", ledger_constructor)
    monkeypatch.setattr(conversation_core, "InteractionTurnManager", manager_constructor)
    monkeypatch.setattr(
        conversation_paths,
        "resolve_turn_ledger_path",
        lambda _config: ledger_path,
    )
    llm_client = object()
    llm_owner = SimpleNamespace(
        name="llm",
        llm_client=llm_client,
        build=MagicMock(),
        stop=AsyncMock(),
    )
    llm_module_constructor = MagicMock(return_value=llm_owner)
    monkeypatch.setattr(llm_module_module, "LLMModule", llm_module_constructor)
    monkeypatch.setattr(session, "SessionMemory", MagicMock(return_value=object()))
    monkeypatch.setattr(
        legacy_conversation,
        "ConversationManager",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(episodic_memory, "EpisodicMemory", MagicMock(return_value=object()))
    monkeypatch.setattr(bridge, "MemoryBridge", MagicMock(return_value=object()))
    monkeypatch.setattr(
        providers,
        "build_perception",
        MagicMock(return_value=SimpleNamespace(vision=None)),
    )
    for builder_name in (
        "build_navigation",
        "build_robot_control",
        "build_scene_intelligence",
        "build_temporal_memory",
    ):
        monkeypatch.setattr(providers, builder_name, MagicMock(return_value=None))
    tool_registry_instance = MagicMock()
    monkeypatch.setattr(
        tool_registry,
        "ToolRegistry",
        MagicMock(return_value=tool_registry_instance),
    )
    skill_manager_instance = MagicMock()
    skill_manager_instance.get_enabled.return_value = []
    monkeypatch.setattr(
        skill_manager,
        "SkillManager",
        MagicMock(return_value=skill_manager_instance),
    )
    monkeypatch.setattr(skill_executor, "SkillExecutor", MagicMock(return_value=object()))
    monkeypatch.setattr(resource_surface, "resource_surface_from_context", MagicMock())
    monkeypatch.setattr(
        resource_surface,
        "set_resource_surface",
        MagicMock(side_effect=["previous", None]),
    )

    async with mcp_context.app_lifespan(MagicMock()) as app:
        assert app.llm_module is llm_owner
        assert app.llm_client is llm_client
        assert app.turn_ledger is ledger
        assert app.interaction_turn_manager is manager

    llm_module_constructor.assert_called_once_with()
    llm_owner.build.assert_called_once()
    llm_owner.stop.assert_awaited_once_with()
    ledger_constructor.assert_called_once_with(ledger_path)
    manager_constructor.assert_called_once_with(ledger)


# ── Helpers ────────────────────────────────────────────────────


def _make_ctx(app_context, *, client_id: str | None = None):
    """Build a mock MCP Context whose lifespan_context is *app_context*."""
    ctx = AsyncMock()
    ctx.request_context = MagicMock()
    ctx.request_context.lifespan_context = app_context
    ctx.client_id = client_id
    ctx.info = AsyncMock()
    return ctx


def _assert_error_json(result: str, expected_code: str | None = None):
    """Assert the result is a valid error_response JSON."""
    data = json.loads(result)
    assert "error" in data
    assert "code" in data["error"]
    assert "message" in data["error"]
    if expected_code:
        assert data["error"]["code"] == expected_code


@pytest.fixture()
def allow_lab_unsafe(monkeypatch):
    """Enable direct arm tools for tests that exercise the lab-only fallback."""
    from askme.mcp.tools import robot_tools

    monkeypatch.setattr(robot_tools, "_LAB_UNSAFE", True)
    return robot_tools


@pytest.mark.asyncio
async def test_direct_robot_motion_tools_are_blocked_by_default(app_context, monkeypatch):
    from askme.mcp.tools import robot_tools

    monkeypatch.setattr(robot_tools, "_LAB_UNSAFE", False)
    ctx = _make_ctx(app_context)

    data = json.loads(await robot_tools.robot_move(100.0, 200.0, 50.0, ctx))

    assert data["error"] == "unsafe_direct_arm_access_blocked"
    assert "robot_submit_task" in data["message"]


# ── Robot tool tests: no connection ──────────────────────────


class TestRobotToolsNoConnection:
    """All robot tools should return an error when no arm_controller."""

    @pytest.fixture(autouse=True)
    def _allow_lab_unsafe(self, allow_lab_unsafe):
        return None

    @pytest.mark.asyncio
    async def test_robot_move_no_controller(self, app_context):
        from askme.mcp.tools.robot_tools import robot_move

        ctx = _make_ctx(app_context)
        result = await robot_move(100.0, 200.0, 50.0, ctx)
        _assert_error_json(result, "robot_not_connected")

    @pytest.mark.asyncio
    async def test_robot_pick_no_controller(self, app_context):
        from askme.mcp.tools.robot_tools import robot_pick

        ctx = _make_ctx(app_context)
        result = await robot_pick("cup", ctx)
        _assert_error_json(result, "robot_not_connected")

    @pytest.mark.asyncio
    async def test_robot_home_no_controller(self, app_context):
        from askme.mcp.tools.robot_tools import robot_home

        ctx = _make_ctx(app_context)
        result = await robot_home(ctx)
        _assert_error_json(result, "robot_not_connected")

    @pytest.mark.asyncio
    async def test_robot_state_no_controller(self, app_context):
        from askme.mcp.tools.robot_tools import robot_state

        ctx = _make_ctx(app_context)
        result = await robot_state(ctx)
        _assert_error_json(result, "robot_not_connected")

    @pytest.mark.asyncio
    async def test_robot_estop_no_controller(self, app_context):
        from askme.mcp.tools.robot_tools import robot_estop

        ctx = _make_ctx(app_context)
        result = await robot_estop(ctx)
        _assert_error_json(result, "robot_not_connected")

    @pytest.mark.asyncio
    async def test_robot_wave_no_controller(self, app_context):
        from askme.mcp.tools.robot_tools import robot_wave

        ctx = _make_ctx(app_context)
        result = await robot_wave(ctx)
        _assert_error_json(result, "robot_not_connected")

    @pytest.mark.asyncio
    async def test_robot_place_no_controller(self, app_context):
        from askme.mcp.tools.robot_tools import robot_place

        ctx = _make_ctx(app_context)
        result = await robot_place("table", ctx)
        _assert_error_json(result, "robot_not_connected")


# ── Robot tool tests: happy path (mocked) ────────────────────


class TestRobotToolsWithMock:
    """Happy-path: robot tools with a mocked ArmController."""

    @pytest.fixture(autouse=True)
    def _allow_lab_unsafe(self, allow_lab_unsafe):
        return None

    @pytest.fixture
    def robot_context(self):
        from askme.mcp.server import AppContext

        ctx = AppContext()
        ctx.arm_controller = MagicMock()
        ctx.arm_controller.execute = MagicMock(return_value={"status": "ok", "action": [0] * 16})
        ctx.arm_controller.get_state = MagicMock(
            return_value={"connected": True, "estopped": False}
        )
        ctx.arm_controller.emergency_stop = MagicMock()
        ctx.robot_enabled = True
        return ctx

    @pytest.mark.asyncio
    async def test_robot_move_success(self, robot_context):
        from askme.mcp.tools.robot_tools import robot_move

        ctx = _make_ctx(robot_context)
        result = await robot_move(100.0, 200.0, 50.0, ctx)
        data = json.loads(result)
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_robot_move_offloads_sync_execute(self, robot_context, monkeypatch):
        from askme.mcp.tools import robot_tools

        calls = []

        async def fake_to_thread(func, *args, **kwargs):
            calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        monkeypatch.setattr(robot_tools.asyncio, "to_thread", fake_to_thread)
        ctx = _make_ctx(robot_context)

        result = await robot_tools.robot_move(100.0, 200.0, 50.0, ctx)

        data = json.loads(result)
        assert data["status"] == "ok"
        assert calls == [
            (
                robot_context.arm_controller.execute,
                ("move", {"x": 100.0, "y": 200.0, "z": 50.0}),
                {},
            )
        ]

    @pytest.mark.asyncio
    async def test_robot_home_supports_awaitable_execute(self, robot_context, monkeypatch):
        from askme.mcp.tools import robot_tools

        robot_context.arm_controller.execute = AsyncMock(return_value={"status": "ok"})
        to_thread = AsyncMock()
        monkeypatch.setattr(robot_tools.asyncio, "to_thread", to_thread)

        ctx = _make_ctx(robot_context)
        result = await robot_tools.robot_home(ctx)

        data = json.loads(result)
        assert data["status"] == "ok"
        robot_context.arm_controller.execute.assert_awaited_once_with("home")
        to_thread.assert_not_called()

    @pytest.mark.asyncio
    async def test_robot_state_success(self, robot_context):
        from askme.mcp.tools.robot_tools import robot_state

        ctx = _make_ctx(robot_context)
        result = await robot_state(ctx)
        data = json.loads(result)
        assert data["connected"] is True

    @pytest.mark.asyncio
    async def test_robot_estop_success(self, robot_context):
        from askme.mcp.tools.robot_tools import robot_estop

        ctx = _make_ctx(robot_context)
        result = await robot_estop(ctx)
        data = json.loads(result)
        assert data["status"] == "emergency_stop_activated"
        robot_context.arm_controller.emergency_stop.assert_called_once()


# ── Voice tool tests: no engine ──────────────────────────────


class TestVoiceToolsNoEngine:
    """Voice tools should return error when engines are not initialised."""

    @pytest.mark.asyncio
    async def test_voice_listen_no_engine(self, app_context):
        from askme.mcp.tools.voice_tools import voice_listen

        ctx = _make_ctx(app_context)
        result = await voice_listen(ctx)
        _assert_error_json(result, "voice_not_available")

    @pytest.mark.asyncio
    async def test_voice_speak_no_engine(self, app_context):
        from askme.mcp.tools.voice_tools import voice_speak

        ctx = _make_ctx(app_context)
        result = await voice_speak("hello", ctx)
        _assert_error_json(result, "voice_not_available")


# ── Voice tool tests: happy path (mocked) ────────────────────


class TestVoiceToolsWithMock:
    """Happy-path: voice tools with mocked TTS engine."""

    @pytest.fixture
    def voice_context(self):
        from askme.mcp.server import AppContext

        ctx = AppContext()
        ctx.tts_engine = MagicMock()
        ctx.tts_engine.speak = MagicMock()
        ctx.tts_engine.start_playback = MagicMock()
        ctx.tts_engine.wait_done = MagicMock()
        ctx.tts_engine.stop_playback = MagicMock()
        ctx.voice_enabled = True
        return ctx

    @pytest.mark.asyncio
    async def test_voice_speak_success(self, voice_context):
        from askme.mcp.tools.voice_tools import voice_speak

        ctx = _make_ctx(voice_context)
        result = await voice_speak("hello world", ctx)
        assert "[Spoken]" in result
        voice_context.tts_engine.speak.assert_called_once_with("hello world")


class TestVisionToolsStableContext:
    """Vision/text MCP tools should use AppContext stable fields."""

    @pytest.fixture
    def vision_context(self):
        from askme.mcp.server import AppContext

        ctx = AppContext()
        ctx.vision_bridge = AsyncMock()
        ctx.vision_bridge.describe_scene = AsyncMock(return_value="clear hallway")
        ctx.vision_bridge.describe_scene_with_question = AsyncMock(return_value="a red cup")
        ctx.vision_bridge.find_object = AsyncMock(
            return_value={
                "class_id": "cup",
                "confidence": 0.876,
                "bbox": [1, 2, 3, 4],
                "distance_m": 1.2,
            }
        )
        return ctx

    @pytest.mark.asyncio
    async def test_look_around_uses_vision_bridge(self, vision_context):
        from askme.mcp.tools.vision_tools import look_around

        ctx = _make_ctx(vision_context)
        result = await look_around(ctx=ctx)
        data = json.loads(result)
        assert data["scene"] == "clear hallway"
        vision_context.vision_bridge.describe_scene.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_find_target_uses_vision_bridge(self, vision_context):
        from askme.mcp.tools.vision_tools import find_target

        ctx = _make_ctx(vision_context)
        result = await find_target("cup", ctx=ctx)
        data = json.loads(result)
        assert data["found"] is True
        assert data["object"] == "cup"
        assert data["confidence"] == 0.88
        vision_context.vision_bridge.find_object.assert_awaited_once_with("cup")

    @pytest.mark.asyncio
    async def test_look_around_no_vision_bridge(self, app_context):
        from askme.mcp.tools.vision_tools import look_around

        ctx = _make_ctx(app_context)
        result = await look_around(ctx=ctx)
        data = json.loads(result)
        assert data["error"] == "vision not available"


class TestChatToolStableContext:
    @pytest.mark.asyncio
    async def test_chat_uses_llm_client_and_conversation(self):
        from askme.mcp.server import AppContext
        from askme.mcp.tools.vision_tools import chat

        app = AppContext()
        app.llm_client = AsyncMock()
        app.llm_client.chat = AsyncMock(return_value="hello")
        app.conversation = MagicMock()
        app.conversation.get_messages.return_value = []

        ctx = _make_ctx(app)
        result = await chat("hi", ctx=ctx)
        data = json.loads(result)
        assert data == {"reply": "hello", "text": "hi"}
        app.conversation.add_user_message.assert_called_once_with(
            "hi",
            conversation_session_id=app.process_session_id,
        )
        app.conversation.get_messages.assert_called_once_with(
            system_prompt="You are Askme, a helpful robot assistant.",
            conversation_session_id=app.process_session_id,
        )
        call = app.llm_client.chat.await_args
        assert call.args == ([{"role": "user", "content": "hi"}],)
        context = call.kwargs["context"]
        assert context.purpose == "assistant_response"
        assert context.channel == "text"
        assert context.request_class == "text"
        assert context.call_id
        assert context.session_id == app.process_session_id
        app.conversation.add_assistant_message.assert_called_once_with(
            "hello",
            conversation_session_id=app.process_session_id,
        )

    @pytest.mark.asyncio
    async def test_chat_prefers_explicit_session_then_mcp_client_id(self):
        from askme.mcp.server import AppContext
        from askme.mcp.tools.vision_tools import chat

        app = AppContext()
        app.llm_client = AsyncMock()
        app.llm_client.chat = AsyncMock(side_effect=["explicit reply", "client reply"])
        app.conversation = MagicMock()
        app.conversation.get_messages.return_value = []
        ctx = _make_ctx(app, client_id="client-session")

        await chat("explicit", ctx=ctx, conversation_session_id=" explicit-session ")
        await chat("implicit", ctx=ctx)

        assert [
            call.kwargs["conversation_session_id"]
            for call in app.conversation.add_user_message.call_args_list
        ] == ["explicit-session", "client-session"]
        assert [
            call.kwargs["conversation_session_id"]
            for call in app.conversation.get_messages.call_args_list
        ] == ["explicit-session", "client-session"]
        assert [
            call.kwargs["context"].session_id for call in app.llm_client.chat.await_args_list
        ] == ["explicit-session", "client-session"]
        assert [
            call.kwargs["conversation_session_id"]
            for call in app.conversation.add_assistant_message.call_args_list
        ] == ["explicit-session", "client-session"]

    @pytest.mark.asyncio
    async def test_chat_records_one_canonical_turn_correlated_to_the_llm_call(self):
        from askme.conversation import TurnStatus
        from askme.mcp.server import AppContext
        from askme.mcp.tools.vision_tools import chat

        app = AppContext()
        app.llm_client = AsyncMock()
        app.llm_client.chat = AsyncMock(return_value="canonical reply")
        app.conversation = None
        app.turn_ledger = MagicMock()
        app.interaction_turn_manager = MagicMock()
        opened = SimpleNamespace(thread_id="thread-mcp", turn_id="turn-mcp")
        generating = SimpleNamespace(thread_id="thread-mcp", turn_id="turn-mcp")
        app.interaction_turn_manager.open.return_value = opened
        app.interaction_turn_manager.advance.return_value = generating

        result = await chat(
            "canonical question",
            ctx=_make_ctx(app, client_id="client-ignored"),
            conversation_session_id="thread-mcp",
        )

        assert json.loads(result)["reply"] == "canonical reply"
        app.interaction_turn_manager.open.assert_called_once()
        interaction = app.interaction_turn_manager.open.call_args.args[0]
        assert interaction.user_text == "canonical question"
        assert interaction.source == "mcp"
        assert interaction.channel == "text"
        assert interaction.thread_id == "thread-mcp"

        app.interaction_turn_manager.advance.assert_called_once()
        assert app.interaction_turn_manager.advance.call_args.args[0] is opened
        generation = app.interaction_turn_manager.advance.call_args.args[1]
        llm_context = app.llm_client.chat.await_args.kwargs["context"]
        assert generation.generation_id == llm_context.call_id
        assert llm_context.session_id == "thread-mcp"
        assert llm_context.turn_id == "turn-mcp"

        app.interaction_turn_manager.settle.assert_called_once()
        assert app.interaction_turn_manager.settle.call_args.args[0] is generating
        outcome = app.interaction_turn_manager.settle.call_args.args[1]
        assert outcome.status is TurnStatus.COMMITTED
        assert outcome.assistant_text == "canonical reply"
        assert app.turn_ledger.method_calls == []

    @pytest.mark.asyncio
    async def test_chat_fails_the_canonical_turn_when_generation_raises(self):
        from askme.conversation import TurnStatus
        from askme.mcp.server import AppContext
        from askme.mcp.tools.vision_tools import chat

        app = AppContext()
        app.llm_client = AsyncMock()
        app.llm_client.chat = AsyncMock(side_effect=RuntimeError("provider unavailable"))
        app.conversation = None
        app.interaction_turn_manager = MagicMock()
        opened = SimpleNamespace(thread_id="thread-fail", turn_id="turn-fail")
        generating = SimpleNamespace(thread_id="thread-fail", turn_id="turn-fail")
        app.interaction_turn_manager.open.return_value = opened
        app.interaction_turn_manager.advance.return_value = generating

        with pytest.raises(RuntimeError, match="provider unavailable"):
            await chat(
                "question that fails",
                ctx=_make_ctx(app),
                conversation_session_id="thread-fail",
            )

        app.interaction_turn_manager.settle.assert_called_once()
        assert app.interaction_turn_manager.settle.call_args.args[0] is generating
        outcome = app.interaction_turn_manager.settle.call_args.args[1]
        assert outcome.status is TurnStatus.FAILED
        assert outcome.reason == "mcp_chat_failed"
        assert outcome.metadata == {"error_type": "RuntimeError"}

    @pytest.mark.asyncio
    async def test_chat_failure_does_not_project_a_ghost_legacy_message(self):
        from askme.mcp.server import AppContext
        from askme.mcp.tools.vision_tools import chat

        app = AppContext()
        app.llm_client = AsyncMock()
        app.llm_client.chat = AsyncMock(side_effect=RuntimeError("provider unavailable"))
        app.conversation = MagicMock()
        app.conversation.get_messages.return_value = []

        with pytest.raises(RuntimeError, match="provider unavailable"):
            await chat("failed question", ctx=_make_ctx(app))

        app.conversation.add_user_message.assert_not_called()
        app.conversation.add_assistant_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_no_llm_client(self, app_context):
        from askme.mcp.tools.vision_tools import chat

        ctx = _make_ctx(app_context)
        result = await chat("hi", ctx=ctx)
        data = json.loads(result)
        assert data["error"] == "llm client not available"


# ── Skill tool tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_skill_passes_operational_context_to_executor():
    from askme.mcp.server import AppContext
    from askme.mcp.tools.skill_tools import execute_skill

    app = AppContext()
    skill = SimpleNamespace(enabled=True)
    app.skill_manager = MagicMock()
    app.skill_manager.get.return_value = skill
    app.skill_executor = MagicMock()
    app.skill_executor.execute = AsyncMock(return_value="done")
    ctx = _make_ctx(app)

    assert await execute_skill("patrol", "go", ctx) == "done"

    call = app.skill_executor.execute.await_args
    llm_context = call.kwargs["llm_call_context"]
    assert llm_context.purpose == "tool_followup"
    assert llm_context.channel == "text"
    assert llm_context.request_class == "robot_action"
    assert llm_context.call_id


class TestSkillToolNoManager:
    """Skill tool should return error when manager is not initialised."""

    @pytest.mark.asyncio
    async def test_execute_skill_no_manager(self, app_context):
        from askme.mcp.tools.skill_tools import execute_skill

        ctx = _make_ctx(app_context)
        result = await execute_skill("test_skill", "hello", ctx)
        _assert_error_json(result, "internal_error")


# ── Resource tests ────────────────────────────────────────────


class TestResources:
    """MCP resources should return valid JSON."""

    def test_robot_status_resource(self):
        from askme.mcp.resources.robot_resources import robot_status

        result = robot_status()
        data = json.loads(result)
        assert "enabled" in data

    def test_robot_joint_info_valid(self):
        from askme.mcp.resources.robot_resources import robot_joint_info

        result = robot_joint_info("0")
        data = json.loads(result)
        assert data["joint_id"] == 0
        assert data["name"] == "shoulder_pan"
        assert data["type"] == "arm"

    def test_robot_joint_info_finger(self):
        from askme.mcp.resources.robot_resources import robot_joint_info

        result = robot_joint_info("7")
        data = json.loads(result)
        assert data["name"] == "finger_2"
        assert data["type"] == "finger"

    def test_robot_joint_info_invalid(self):
        from askme.mcp.resources.robot_resources import robot_joint_info

        result = robot_joint_info("abc")
        data = json.loads(result)
        assert "error" in data

    def test_robot_joint_info_out_of_range(self):
        from askme.mcp.resources.robot_resources import robot_joint_info

        result = robot_joint_info("20")
        data = json.loads(result)
        assert "error" in data

    def test_robot_safety_config_resource(self):
        from askme.mcp.resources.robot_resources import robot_safety_config

        result = robot_safety_config()
        data = json.loads(result)
        assert "estop_keywords" in data
        assert "停" in data["estop_keywords"]

    def test_askme_config_resource(self):
        from askme.mcp.resources.skill_resources import askme_config

        result = askme_config()
        data = json.loads(result)
        assert isinstance(data, dict)
        # API keys should be stripped
        for section in data.values():
            if isinstance(section, dict):
                for key in section:
                    assert "key" not in key.lower() or "api" not in key.lower()
