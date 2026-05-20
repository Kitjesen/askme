"""Tests for runtime-to-tool dependency binding."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from askme.runtime.core.module import ModuleRegistry


class _FakeRobotControl:
    def __init__(self, execution_id: str = "runtime-001") -> None:
        self.execution_id = execution_id
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def is_configured(self) -> bool:
        return True

    def dispatch_capability(
        self,
        capability: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((capability, params))
        return {"status": "ok", "execution_id": self.execution_id}


class _FakeNavigation:
    def __init__(self, mission_id: str = "nav-runtime-001") -> None:
        self.mission_id = mission_id
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def is_configured(self) -> bool:
        return True

    def dispatch_navigation(
        self,
        capability: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((capability, params))
        return {"session": {"mission_id": self.mission_id, "state": "submitted"}}

    def status(self) -> dict[str, Any]:
        return {"sessions": [{"mission_id": self.mission_id}]}


class _FakeTemporalMemory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query_temporal_observations(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(params)
        return {"observations": []}


def test_tools_module_can_bind_runtime_robot_control_client() -> None:
    from askme.runtime.modules.tools_module import ToolsModule

    mod = ToolsModule()
    mod.build({}, ModuleRegistry())
    client = _FakeRobotControl()

    assert mod.bind_robot_control_client(client) is True
    dispatch_tool = mod.registry.get("dog_control_dispatch")
    move_tool = mod.registry.get("move_robot")
    scan_tool = mod.registry.get("scan_around")
    assert dispatch_tool is not None
    assert move_tool is not None
    assert scan_tool is not None
    assert "runtime-001" in dispatch_tool.execute(capability="stand")
    assert move_tool._robot_control_client is client
    assert scan_tool._robot_control_client is client


def test_tools_module_can_bind_navigation_client() -> None:
    from askme.runtime.modules.tools_module import ToolsModule

    mod = ToolsModule()
    mod.build({}, ModuleRegistry())
    client = _FakeNavigation()

    assert mod.navigation_client is not None
    assert mod.bind_navigation_client(client) is True
    assert mod.navigation_client is client
    nav_tool = mod.registry.get("nav_dispatch")
    nav_status_tool = mod.registry.get("nav_status")
    move_tool = mod.registry.get("move_robot")
    assert nav_tool is not None
    assert nav_status_tool is not None
    assert move_tool is not None
    assert "nav-runtime-001" in nav_tool.execute(destination="dock")
    assert "nav-runtime-001" in nav_status_tool.execute()
    assert move_tool._navigation_client is client


def test_navigation_binding_does_not_replace_temporal_memory_client() -> None:
    from askme.runtime.modules.tools_module import ToolsModule

    mod = ToolsModule()
    mod.build({}, ModuleRegistry())
    client = _FakeNavigation()

    assert mod.bind_navigation_client(client) is True
    temporal_tool = mod.registry.get("temporal_query")
    assert temporal_tool is not None
    assert temporal_tool._temporal_memory_client is mod.temporal_memory_client


def test_tools_module_can_bind_temporal_memory_client() -> None:
    from askme.runtime.modules.tools_module import ToolsModule

    mod = ToolsModule()
    mod.build({}, ModuleRegistry())
    client = _FakeTemporalMemory()

    assert mod.bind_temporal_memory_client(client) is True
    assert mod.temporal_memory_client is client
    temporal_tool = mod.registry.get("temporal_query")
    assert temporal_tool is not None
    assert temporal_tool._temporal_memory_client is client
    assert "未发现" in temporal_tool.execute(label="person")
    assert client.calls == [{"since": "1h", "limit": 20, "label": "person"}]


def test_control_module_injects_robot_control_into_existing_tools_module() -> None:
    from askme.runtime.modules.control_module import ControlModule
    from askme.runtime.modules.tools_module import ToolsModule

    registry = ModuleRegistry()
    tools_mod = ToolsModule()
    tools_mod.build({}, registry)
    registry.register(tools_mod)
    client = _FakeRobotControl("control-001")

    with patch(
        "askme.runtime.modules.control_module.build_robot_control",
        return_value=client,
    ):
        ControlModule().build({"runtime": {"dog_control": {}}}, registry)

    tool = tools_mod.registry.get("dog_control_dispatch")
    move_tool = tools_mod.registry.get("move_robot")
    scan_tool = tools_mod.registry.get("scan_around")
    assert tool is not None
    assert move_tool is not None
    assert scan_tool is not None
    assert "control-001" in tool.execute(capability="sit")
    assert move_tool._robot_control_client is client
    assert scan_tool._robot_control_client is client
    assert client.calls == [("sit", None)]
