"""Tests for MCP tool and resource modules."""

import json
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
        module.startswith(("askme.mcp.resources.", "askme.mcp.tools."))
        for module in modules
    )

# ── Helpers ────────────────────────────────────────────────────

def _make_ctx(app_context):
    """Build a mock MCP Context whose lifespan_context is *app_context*."""
    ctx = AsyncMock()
    ctx.request_context = MagicMock()
    ctx.request_context.lifespan_context = app_context
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
        ctx.arm_controller.get_state = MagicMock(return_value={"connected": True, "estopped": False})
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
        assert calls == [(
            robot_context.arm_controller.execute,
            ("move", {"x": 100.0, "y": 200.0, "z": 50.0}),
            {},
        )]

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
        ctx.vision_bridge.find_object = AsyncMock(return_value={
            "class_id": "cup",
            "confidence": 0.876,
            "bbox": [1, 2, 3, 4],
            "distance_m": 1.2,
        })
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
        app.conversation.get_messages.return_value = [{"role": "user", "content": "hi"}]

        ctx = _make_ctx(app)
        result = await chat("hi", ctx=ctx)
        data = json.loads(result)
        assert data == {"reply": "hello", "text": "hi"}
        app.conversation.add_user_message.assert_called_once_with("hi")
        app.llm_client.chat.assert_awaited_once_with([{"role": "user", "content": "hi"}])
        app.conversation.add_assistant_message.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_chat_no_llm_client(self, app_context):
        from askme.mcp.tools.vision_tools import chat

        ctx = _make_ctx(app_context)
        result = await chat("hi", ctx=ctx)
        data = json.loads(result)
        assert data["error"] == "llm client not available"


# ── Skill tool tests ──────────────────────────────────────────

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
