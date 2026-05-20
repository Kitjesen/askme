from __future__ import annotations

from pathlib import Path

from askme.agent_shell.agent_profile import AgentProfileRegistry
from askme.api.services.agent_profile_tools import agent_profile_known_tools


def test_agent_profile_registry_exposes_product_roles() -> None:
    registry = AgentProfileRegistry()
    catalog = registry.catalog({"read_file", "web_search", "create_skill", "move_robot"})

    names = {profile["name"] for profile in catalog["profiles"]}

    assert "field_operator" in names
    assert "knowledge_curator" in names
    assert "skill_growth_manager" in names
    assert catalog["profile_count"] >= 5


def test_agent_profile_resolves_tool_boundaries() -> None:
    registry = AgentProfileRegistry()
    field = registry.get("field_operator")
    growth = registry.get("skill_growth_manager")

    field_tools = field.resolve_tools({"read_file", "move_robot", "spawn_agent"})
    growth_tools = growth.resolve_tools({"create_skill", "web_search", "robot_api", "write_file"})

    assert "move_robot" not in field_tools
    assert "spawn_agent" in field_tools
    assert growth_tools == {"create_skill", "web_search"}


def test_agent_profile_known_tools_include_runtime_vision_tools() -> None:
    assert {"find_target", "look_around"} <= agent_profile_known_tools()


def test_agent_profile_registry_loads_project_markdown_profiles(tmp_path: Path) -> None:
    profile_dir = tmp_path / ".askme" / "agents"
    profile_dir.mkdir(parents=True)
    (profile_dir / "route-service.md").write_text(
        """---
name: route_service
display_name: Route service agent
description: Handles customer wayfinding requests
tools: read_file, http_request
disallowedTools: http_request
spawnableProfiles:
  - safety_reviewer
skills:
  - wayfinding
mcpServers:
  - map
hooks:
  PreToolUse:
    - matcher: robot_api
effort: high
isolation: worktree
color: green
memory: project
model: inherit
permissionMode: default
risk_level: medium
maxTurns: 7
timeoutSeconds: 45
---
Only answer park-space route questions.
""",
        encoding="utf-8",
    )

    registry = AgentProfileRegistry(project_dir=tmp_path)
    profile = registry.get("route_service")
    catalog = registry.catalog({"read_file", "http_request", "spawn_agent"})
    route = next(item for item in catalog["profiles"] if item["name"] == "route_service")

    assert profile.instructions == "Only answer park-space route questions."
    assert profile.resolve_tools({"read_file", "http_request", "spawn_agent"}) == {
        "read_file",
        "spawn_agent",
    }
    assert route["source"] == "project"
    assert route["preloaded_skills"] == ["wayfinding"]
    assert route["mcp_servers"] == ["map"]
    assert route["hooks_configured"] == ["PreToolUse"]
    assert route["effort"] == "high"
    assert route["isolation"] == "worktree"
    assert route["color"] == "green"
    assert route["memory_scope"] == "project"
    assert route["max_iterations"] == 7
    assert route["timeout_seconds"] == 45.0


def test_agent_profile_markdown_can_disable_builtin_profile(tmp_path: Path) -> None:
    profile_dir = tmp_path / ".askme" / "managed" / "agents"
    profile_dir.mkdir(parents=True)
    (profile_dir / "skill-growth-manager.md").write_text(
        """---
name: skill_growth_manager
description: Temporarily disabled for production site
disabled: true
---
Disabled by managed policy.
""",
        encoding="utf-8",
    )

    registry = AgentProfileRegistry(project_dir=tmp_path)
    names = {profile.name for profile in registry.all()}

    assert "skill_growth_manager" not in names
    assert "field_operator" in names


def test_agent_profile_registry_writes_project_profile_with_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import askme.agent_shell.agent_profile as profile_module

    audit_path = tmp_path / "skill-audit.jsonl"
    monkeypatch.setattr(
        profile_module,
        "SkillAuditLog",
        lambda: __import__("askme.skills.audit", fromlist=["SkillAuditLog"]).SkillAuditLog(audit_path),
    )
    registry = AgentProfileRegistry(project_dir=tmp_path)

    result = registry.write_project_profile(
        name="Parking Detector PM",
        display_name="Parking Detector PM",
        description="Turns illegal-parking customer demand into reviewed delivery tasks.",
        instructions="Only propose parking detection delivery tasks with safety evidence and acceptance criteria.",
        tools=["create_skill", "read_file"],
        disallowed_tools=["move_robot"],
        spawnable_profiles=["safety_reviewer"],
        skills=["detect_illegal_parking"],
        risk_level="medium",
        operator_id="pm-1",
        known_tools={"create_skill", "read_file", "move_robot", "spawn_agent"},
    )

    assert result["ok"] is True
    assert result["profile"]["name"] == "parking_detector_pm"
    assert Path(result["path"]).exists()
    assert "parking_detector_pm" in {profile.name for profile in registry.all()}
    preview = registry.preview("parking_detector_pm")
    assert preview["ok"] is True
    assert "Parking Detector PM" in preview["raw_body"]
    audit_text = audit_path.read_text(encoding="utf-8")
    assert "agent_profile:parking_detector_pm" in audit_text


def test_agent_profile_registry_rejects_unknown_tools(tmp_path: Path) -> None:
    registry = AgentProfileRegistry(project_dir=tmp_path)

    result = registry.write_project_profile(
        name="unsafe",
        description="bad profile",
        instructions="This profile tries to use unknown tools.",
        tools=["unknown_tool"],
        known_tools={"read_file"},
    )

    assert result["ok"] is False
    assert result["error"] == "unknown tools requested"
    assert result["unknown_tools"] == ["unknown_tool"]
