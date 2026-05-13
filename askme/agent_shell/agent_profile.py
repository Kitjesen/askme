"""Product-level agent profiles for ThunderAgentShell.

The profile layer mirrors the useful part of Claude Code subagents: a named
role has a clear purpose, an instruction overlay, and an explicit tool
boundary.  It is intentionally small and deterministic so product teams can
review what each robot brain lane is allowed to do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from askme.skills.audit import SkillAuditLog


@dataclass(frozen=True)
class AgentProfile:
    """A customer-reviewable agent lane."""

    name: str
    display_name: str
    description: str
    instructions: str
    allowed_tools: frozenset[str] | None = None
    disallowed_tools: frozenset[str] = field(default_factory=frozenset)
    spawnable_profiles: tuple[str, ...] = ()
    max_iterations: int | None = None
    timeout_seconds: float | None = None
    model: str = "inherit"
    permission_mode: str = "default"
    preloaded_skills: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()
    hooks: dict[str, Any] = field(default_factory=dict)
    memory_scope: str = ""
    background: bool = False
    effort: str = ""
    isolation: str = ""
    color: str = ""
    disabled: bool = False
    customer_visible: bool = True
    risk_level: str = "medium"
    source: str = "builtin"

    def resolve_tools(self, inherited_tools: set[str]) -> set[str]:
        """Apply this profile's allow/deny policy to inherited tool names."""
        tools = set(inherited_tools)
        if self.allowed_tools is not None:
            tools &= set(self.allowed_tools)
        tools -= set(self.disallowed_tools)
        if self.spawnable_profiles:
            tools.add("spawn_agent")
        else:
            tools.discard("spawn_agent")
        return tools

    def summary(self, inherited_tools: set[str] | None = None) -> dict[str, Any]:
        allowed = sorted(self.resolve_tools(set(inherited_tools or ())))
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "risk_level": self.risk_level,
            "customer_visible": self.customer_visible,
            "source": self.source,
            "model": self.model,
            "permission_mode": self.permission_mode,
            "preloaded_skills": list(self.preloaded_skills),
            "mcp_servers": list(self.mcp_servers),
            "hooks_configured": sorted(self.hooks.keys()),
            "memory_scope": self.memory_scope,
            "background": self.background,
            "effort": self.effort,
            "isolation": self.isolation,
            "color": self.color,
            "disabled": self.disabled,
            "allowed_tools": allowed,
            "disallowed_tools": sorted(self.disallowed_tools),
            "spawnable_profiles": list(self.spawnable_profiles),
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
        }


_BUILTIN_PROFILES: dict[str, AgentProfile] = {
    "field_operator": AgentProfile(
        name="field_operator",
        display_name="现场任务总控",
        description="面向巡检、事件处置和客户演示的默认任务代理。",
        instructions=(
            "你是现场任务总控。先判断用户是在问路、问知识、创建任务还是处理异常；"
            "涉及机器人移动、通知或现场事件时，必须优先检查安全和可审计证据。"
        ),
        disallowed_tools=frozenset({"move_robot"}),
        spawnable_profiles=("knowledge_curator", "skill_growth_manager"),
        risk_level="medium",
    ),
    "knowledge_curator": AgentProfile(
        name="knowledge_curator",
        display_name="知识运营代理",
        description="负责知识导入、冲突检查、证据整理和过期知识处理。",
        instructions=(
            "你只处理知识库和证据问题。回答必须说明证据来源；过期、冲突或未审批知识不能直接用于结论。"
        ),
        allowed_tools=frozenset({"read_file", "list_directory", "web_search", "web_fetch", "http_request"}),
        risk_level="low",
    ),
    "wayfinding_guide": AgentProfile(
        name="wayfinding_guide",
        display_name="园区问路代理",
        description="负责访客问路、目的地解析、路线说明和带路前确认。",
        instructions=(
            "你只回答园区地点、路线、商户和服务点相关问题。目的地不明确时先确认；"
            "带路前必须二次确认并交给任务运行链路。"
        ),
        allowed_tools=frozenset({"temporal_query", "http_request", "speak_progress", "robot_api"}),
        risk_level="medium",
    ),
    "safety_reviewer": AgentProfile(
        name="safety_reviewer",
        display_name="安全复核代理",
        description="用于高风险操作前的结构化复核。",
        instructions=(
            "你只做风险复核。输出是否允许、阻断原因、缺失证据和建议的低风险替代动作。"
        ),
        allowed_tools=frozenset({"read_file", "http_request", "robot_api"}),
        risk_level="high",
    ),
    "skill_growth_manager": AgentProfile(
        name="skill_growth_manager",
        display_name="在线技能增长代理",
        description="把重复出现的客户需求沉淀成可审核、可启停、可审计的新技能。",
        instructions=(
            "你负责把稳定的重复需求转成技能草稿。新增技能必须写清触发词、适用场景、工具边界、"
            "安全等级、验收用例和回滚方式；没有审批前不能承诺生产可用。"
        ),
        allowed_tools=frozenset({
            "create_skill",
            "create_agent_profile",
            "read_file",
            "list_directory",
            "web_search",
            "web_fetch",
        }),
        risk_level="medium",
    ),
}


class AgentProfileRegistry:
    """Registry for product-reviewable agent profiles.

    Built-ins provide a safe baseline. User/project Markdown files can override
    or add profiles without code changes, mirroring the useful Claude Code
    subagent pattern while keeping robot permissions explicit.
    """

    def __init__(
        self,
        profiles: dict[str, AgentProfile] | None = None,
        *,
        project_dir: str | Path | None = None,
    ) -> None:
        self._project_dir = Path(project_dir) if project_dir is not None else Path.cwd()
        self._profiles = dict(_BUILTIN_PROFILES)
        if profiles is not None:
            self._profiles.update(profiles)
        else:
            self._load_file_profiles()

    def get(self, name: str | None) -> AgentProfile:
        if name and name in self._profiles:
            return self._profiles[name]
        return self._profiles["field_operator"]

    def all(self) -> list[AgentProfile]:
        return list(self._profiles.values())

    def catalog(self, inherited_tools: set[str] | None = None) -> dict[str, Any]:
        tools = set(inherited_tools or ())
        profiles = [profile.summary(tools) for profile in self.all()]
        return {
            "title": "机器人 Agent 体系",
            "mechanism": "profile markdown + tool allowlist/denylist + spawn boundary + review hooks + audit",
            "profiles": profiles,
            "profile_count": len(profiles),
            "inherited_tool_count": len(tools),
            "profile_locations": [
                str(path) for path, _source in self._profile_locations()
            ],
            "profile_scopes": [
                {"source": source, "path": str(path), "priority": index + 1}
                for index, (path, source) in enumerate(self._profile_locations())
            ],
            "policy": {
                "file_format": "Markdown frontmatter + instructions body",
                "project_profiles_override_user_profiles": True,
                "managed_profiles_override_project_profiles": True,
                "subagent_spawn_requires_profile_allowlist": True,
                "tool_access_is_profile_scoped": True,
            },
        }

    def write_project_profile(
        self,
        *,
        name: str,
        description: str,
        instructions: str,
        display_name: str = "",
        tools: Any = None,
        disallowed_tools: Any = None,
        spawnable_profiles: Any = None,
        skills: Any = None,
        mcp_servers: Any = None,
        hooks: dict[str, Any] | None = None,
        model: str = "inherit",
        permission_mode: str = "default",
        risk_level: str = "medium",
        customer_visible: bool = True,
        memory_scope: str = "",
        max_iterations: int | None = None,
        timeout_seconds: float | None = None,
        operator_id: str = "",
        known_tools: set[str] | None = None,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Create or update a project-level agent profile Markdown file.

        This is the productized equivalent of Claude Code's project subagent
        files: it is file-based, reviewable in git, scoped to this project, and
        every write is audited.
        """

        clean_name = _sanitize_profile_name(name)
        if not clean_name:
            return {"ok": False, "error": "invalid agent profile name"}
        if not str(description or "").strip():
            return {"ok": False, "error": "description is required", "profile_name": clean_name}
        if len(str(instructions or "").strip()) < 10:
            return {"ok": False, "error": "instructions are too short", "profile_name": clean_name}

        tool_list = _list(tools)
        disallowed_list = _list(disallowed_tools)
        unknown_tools: list[str] = []
        if known_tools is not None:
            known = set(known_tools)
            unknown_tools = sorted((set(tool_list) | set(disallowed_list)) - known)
        if unknown_tools:
            return {
                "ok": False,
                "error": "unknown tools requested",
                "profile_name": clean_name,
                "unknown_tools": unknown_tools,
            }

        spawnable = _list(spawnable_profiles)
        known_profiles = set(self._profiles) | {clean_name}
        unknown_spawnable = sorted(set(spawnable) - known_profiles)
        if unknown_spawnable:
            return {
                "ok": False,
                "error": "unknown spawnable profiles requested",
                "profile_name": clean_name,
                "unknown_spawnable_profiles": unknown_spawnable,
            }

        target_dir = self.project_profile_dir
        target_path = (target_dir / f"{clean_name}.md").resolve()
        if target_dir.resolve() not in target_path.parents:
            return {"ok": False, "error": "profile path escaped project profile directory"}
        if target_path.exists() and not overwrite:
            return {
                "ok": False,
                "error": "agent profile already exists",
                "profile_name": clean_name,
                "path": str(target_path),
            }

        body = str(instructions or "").strip()
        meta: dict[str, Any] = {
            "name": clean_name,
            "display_name": display_name or clean_name.replace("_", " ").title(),
            "description": str(description).strip(),
            "tools": tool_list,
            "disallowedTools": disallowed_list,
            "spawnableProfiles": spawnable,
            "skills": _list(skills),
            "mcpServers": _list(mcp_servers),
            "hooks": hooks or {},
            "model": model or "inherit",
            "permissionMode": permission_mode or "default",
            "risk_level": risk_level or "medium",
            "customer_visible": bool(customer_visible),
            "memory": memory_scope or "",
        }
        if max_iterations is not None:
            meta["maxTurns"] = int(max_iterations)
        if timeout_seconds is not None:
            meta["timeoutSeconds"] = float(timeout_seconds)

        target_dir.mkdir(parents=True, exist_ok=True)
        content = (
            "---\n"
            + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
            + "\n---\n"
            + body
            + "\n"
        )
        target_path.write_text(content, encoding="utf-8")

        profile = self._parse_profile_file(target_path, "project")
        if profile is None:
            return {
                "ok": False,
                "error": "written profile could not be parsed",
                "profile_name": clean_name,
                "path": str(target_path),
            }
        self._profiles[profile.name] = profile
        SkillAuditLog().append(
            skill_name=f"agent_profile:{profile.name}",
            status="profile_written",
            event_type="governance",
            source="agent_profile",
            operator_id=operator_id,
            action="write_project_profile",
            result_preview=f"agent profile {profile.name} written",
            metadata={
                "path": str(target_path),
                "tools": ",".join(tool_list),
                "spawnable_profiles": ",".join(spawnable),
                "risk_level": profile.risk_level,
            },
        )
        return {
            "ok": True,
            "profile": profile.summary(set(known_tools or ())),
            "path": str(target_path),
            "catalog": self.catalog(known_tools),
        }

    def preview(self, name: str, inherited_tools: set[str] | None = None) -> dict[str, Any]:
        clean_name = _sanitize_profile_name(name)
        profile = self._profiles.get(clean_name)
        if profile is None:
            return {"ok": False, "error": "agent profile not found", "profile_name": clean_name}
        raw_body = ""
        path = ""
        for directory, _source in self._profile_locations():
            candidate = directory / f"{clean_name}.md"
            if candidate.is_file():
                path = str(candidate)
                try:
                    raw_body = candidate.read_text(encoding="utf-8")
                except OSError:
                    raw_body = ""
                break
        return {
            "ok": True,
            "profile": profile.summary(set(inherited_tools or ())),
            "path": path,
            "raw_body": raw_body,
        }

    @property
    def project_profile_dir(self) -> Path:
        return self._project_dir / ".askme" / "agents"

    def _load_file_profiles(self) -> None:
        for directory, source in self._profile_locations():
            if not directory.is_dir():
                continue
            for file_path in sorted(directory.glob("*.md")):
                profile = self._parse_profile_file(file_path, source)
                if profile is not None:
                    if profile.disabled:
                        self._profiles.pop(profile.name, None)
                    else:
                        self._profiles[profile.name] = profile

    def _profile_locations(self) -> list[tuple[Path, str]]:
        return [
            (Path.home() / ".askme" / "agents", "user"),
            (self._project_dir / ".askme" / "agents", "project"),
            (self._project_dir / "agents", "project"),
            (Path.home() / ".askme" / "managed" / "agents", "managed"),
            (self._project_dir / ".askme" / "managed" / "agents", "managed"),
        ]

    def _parse_profile_file(self, file_path: Path, source: str) -> AgentProfile | None:
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            return None
        match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n?", content, re.DOTALL)
        if match is None:
            return None
        meta = _yaml_dict(match.group(1))
        body = content[match.end():].strip()
        name = _clean(meta.get("name") or file_path.stem)
        if not name:
            return None
        return AgentProfile(
            name=name,
            display_name=_clean(meta.get("display_name") or meta.get("displayName") or name),
            description=str(meta.get("description") or ""),
            instructions=body or str(meta.get("instructions") or ""),
            allowed_tools=_optional_set(meta.get("tools", meta.get("allowed_tools"))),
            disallowed_tools=frozenset(
                _list(meta.get("disallowedTools", meta.get("disallowed_tools")))
            ),
            spawnable_profiles=tuple(
                _list(meta.get("spawnableProfiles", meta.get("spawnable_profiles")))
            ),
            max_iterations=_optional_int(meta.get("maxTurns", meta.get("max_iterations"))),
            timeout_seconds=_optional_float(meta.get("timeoutSeconds", meta.get("timeout_seconds"))),
            model=_clean(meta.get("model") or "inherit"),
            permission_mode=_clean(meta.get("permissionMode", meta.get("permission_mode")) or "default"),
            preloaded_skills=tuple(_list(meta.get("skills", meta.get("preloaded_skills")))),
            mcp_servers=tuple(_list(meta.get("mcpServers", meta.get("mcp_servers")))),
            hooks=_dict(meta.get("hooks")),
            memory_scope=_clean(meta.get("memory", meta.get("memory_scope"))),
            background=bool(meta.get("background", False)),
            effort=_clean(meta.get("effort")),
            isolation=_clean(meta.get("isolation")),
            color=_clean(meta.get("color")),
            disabled=bool(meta.get("disabled", False)),
            customer_visible=bool(meta.get("customer_visible", meta.get("customerVisible", True))),
            risk_level=_clean(meta.get("risk_level", meta.get("riskLevel")) or "medium"),
            source=source,
        )


def _yaml_dict(text: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_set(value: Any) -> frozenset[str] | None:
    if value is None:
        return None
    return frozenset(_list(value))


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _sanitize_profile_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text and text[0].isdigit():
        text = f"profile_{text}"
    return text[:64]
