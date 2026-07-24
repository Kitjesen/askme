"""Product contract for cooperative cancellation of long-running tools."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from askme.tools.builtin_tools import NavDispatchTool
from askme.tools.move_tool import MoveRobotTool
from askme.tools.tool_registry import BaseTool, ToolRegistry


class _BlockingCancelableTool(BaseTool):
    name = "blocking_cancelable"
    description = "test tool"
    parameters = {"type": "object", "properties": {}}
    cancelable = True
    cancel_on_turn_interrupt = True
    side_effect_class = "external_operation"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_reasons: list[str] = []

    def execute(self, **kwargs) -> str:
        self.started.set()
        self.release.wait(timeout=2.0)
        return "late result"

    def request_cancel(self, reason: str) -> bool:
        self.cancel_reasons.append(reason)
        self.release.set()
        return True


class _BlockingNonCancelableTool(BaseTool):
    name = "blocking_non_cancelable"
    description = "test tool"
    parameters = {"type": "object", "properties": {}}
    side_effect_class = "external_operation"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.completed = threading.Event()

    def execute(self, **kwargs) -> str:
        self.started.set()
        self.release.wait(timeout=2.0)
        self.completed.set()
        return "late result"

    def request_cancel(self, reason: str) -> bool:
        raise AssertionError("non-cancelable tools must not receive callbacks")


class _DangerousBlockingCancelableTool(_BlockingCancelableTool):
    name = "dangerous_blocking_cancelable"
    safety_level = "dangerous"


def test_running_opt_in_tool_receives_turn_cancel_signal() -> None:
    registry = ToolRegistry({"default_timeout": 2.0})
    tool = _BlockingCancelableTool()
    registry.register(tool)
    token = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as caller:
        result_future = caller.submit(
            registry.execute,
            tool.name,
            None,
            cancel_token=token,
        )
        assert tool.started.wait(timeout=0.5)
        token.set()
        result = result_future.result(timeout=0.5)

    registry.shutdown()
    assert result.startswith("[Cancelled]")
    assert tool.cancel_reasons == ["turn_interrupted"]


def test_non_cancelable_tool_continues_but_late_result_is_isolated() -> None:
    registry = ToolRegistry({"default_timeout": 2.0})
    tool = _BlockingNonCancelableTool()
    registry.register(tool)
    token = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as caller:
        result_future = caller.submit(
            registry.execute,
            tool.name,
            None,
            cancel_token=token,
        )
        assert tool.started.wait(timeout=0.5)
        token.set()
        result = result_future.result(timeout=0.5)

    assert result.startswith("[Cancelled]")
    assert "late result" not in result
    assert not tool.completed.is_set()
    tool.release.set()
    assert tool.completed.wait(timeout=0.5)
    registry.shutdown()


def test_navigation_turn_cancel_uses_graceful_endpoint_not_emergency_stop(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:8088")
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.read.return_value = b'{"status":"cancelled"}'
    captured_requests = []

    def capture(request, timeout):
        captured_requests.append(request)
        return response

    tool = NavDispatchTool()
    with patch("urllib.request.urlopen", side_effect=capture):
        accepted = tool.request_cancel("barge_in")

    assert accepted is True
    assert tool.cancelable is True
    assert tool.cancel_on_turn_interrupt is True
    assert tool.side_effect_class == "physical_motion"
    assert captured_requests[0].full_url.endswith("/api/v1/navigation/cancel")
    assert not captured_requests[0].full_url.endswith("/api/v1/stop")


def test_move_robot_navigation_forwards_graceful_cancel_signal() -> None:
    class NavigationClient:
        def __init__(self) -> None:
            self.cancel_reasons: list[str] = []

        def dispatch_navigation(self, *args, **kwargs):
            return {"session": {"mission_id": "nav-1"}}

        def cancel_navigation(self, *, reason: str):
            self.cancel_reasons.append(reason)
            return {"status": "cancelled"}

    navigation = NavigationClient()
    tool = MoveRobotTool(navigation_client=navigation)
    tool.execute(action="go_to", target="dock")

    assert tool.request_cancel("barge_in") is True
    assert navigation.cancel_reasons == ["barge_in"]
    assert tool.side_effect_class == "physical_motion"


def test_approved_dangerous_tool_keeps_turn_cancel_contract() -> None:
    registry = ToolRegistry(
        {
            "default_timeout": 2.0,
            "dangerous_timeout": 2.0,
            "require_confirmation_levels": ["dangerous"],
        }
    )
    tool = _DangerousBlockingCancelableTool()
    registry.register(tool)
    token = threading.Event()
    assert "Approval Required" in registry.execute(tool.name)

    with ThreadPoolExecutor(max_workers=1) as caller:
        result_future = caller.submit(
            registry.handle_pending_input,
            "确认执行",
            cancel_token=token,
        )
        assert tool.started.wait(timeout=0.5)
        token.set()
        result = result_future.result(timeout=0.5)

    registry.shutdown()
    assert result.startswith("[Cancelled]")
    assert tool.cancel_reasons == ["turn_interrupted"]
