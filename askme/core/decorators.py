"""Declarative decorators for tools and skills.

Usage::

    @tool(name="look_around", description="观察环境", agent_allowed=True)
    def look_around(question: str = "") -> str:
        ...

    @skill(name="find_object", triggers=["帮我找", "找一下"], agent_shell=True)
    async def find_object(user_input: str, context: dict) -> str:
        ...

Decorated functions are collected in module-level registries and
auto-discovered at startup by ``askme.core.registry.auto_discover()``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

# Global registries — populated by decorators, consumed by auto_discover()
_TOOL_REGISTRY: list[ToolSpec] = []
_SKILL_REGISTRY: list[SkillSpec] = []


@dataclass
class ToolSpec:
    """Metadata collected by @tool decorator."""

    name: str
    description: str
    fn: Callable
    safety_level: str = "normal"
    agent_allowed: bool = False
    voice_label: str | None = None
    requires: list[str] = field(default_factory=list)
    parameters: dict[str, Any] | None = None  # auto-generated from signature if None
    dev_only: bool = False


@dataclass
class SkillSpec:
    """Metadata collected by @skill decorator."""

    name: str
    fn: Callable
    triggers: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    agent_shell: bool = False
    timeout: int = 30
    safety_level: str = "normal"
    description: str = ""
    tags: list[str] = field(default_factory=list)


def tool(
    name: str,
    description: str = "",
    *,
    safety: str = "normal",
    agent_allowed: bool = False,
    voice_label: str | None = None,
    requires: list[str] | None = None,
    dev_only: bool = False,
) -> Callable:
    """Decorator that marks a function as a tool.

    The function's type annotations are used to auto-generate the
    OpenAI-compatible parameters schema. The decorated function is
    collected into ``_TOOL_REGISTRY`` for auto-discovery.

    Args:
        name: Tool name (e.g. "look_around").
        description: Chinese description for LLM.
        safety: "normal", "dangerous", or "critical".
        agent_allowed: If True, added to ThunderAgentShell whitelist.
        voice_label: Chinese TTS label (e.g. "观察环境"). None = no announce.
        requires: List of service names for dependency injection.
        dev_only: If True, excluded in production mode.
    """

    def decorator(fn: Callable) -> Callable:
        desc = description or fn.__doc__ or ""
        params = _fn_to_schema(fn)
        spec = ToolSpec(
            name=name,
            description=desc.strip(),
            fn=fn,
            safety_level=safety,
            agent_allowed=agent_allowed,
            voice_label=voice_label,
            requires=requires or [],
            parameters=params,
            dev_only=dev_only,
        )
        fn.__tool_spec__ = spec
        _TOOL_REGISTRY.append(spec)
        return fn

    return decorator


def skill(
    name: str,
    *,
    triggers: list[str] | None = None,
    tools: list[str] | None = None,
    agent_shell: bool = False,
    timeout: int = 30,
    safety: str = "normal",
    tags: list[str] | None = None,
) -> Callable:
    """Decorator that marks an async function as a skill.

    Args:
        name: Skill name (e.g. "find_object").
        triggers: Voice trigger phrases (e.g. ["帮我找", "找一下"]).
        tools: Allowed tool names for this skill.
        agent_shell: If True, routed to ThunderAgentShell.
        timeout: Max execution time in seconds.
        safety: Safety level.
        tags: Classification tags.
    """

    def decorator(fn: Callable) -> Callable:
        desc = fn.__doc__ or ""
        spec = SkillSpec(
            name=name,
            fn=fn,
            triggers=triggers or [],
            tools=tools or [],
            agent_shell=agent_shell,
            timeout=timeout,
            safety_level=safety,
            description=desc.strip(),
            tags=tags or [],
        )
        fn.__skill_spec__ = spec
        _SKILL_REGISTRY.append(spec)
        return fn

    return decorator


def get_registered_tools() -> list[ToolSpec]:
    """Return all @tool-decorated functions found so far."""
    return list(_TOOL_REGISTRY)


def get_registered_skills() -> list[SkillSpec]:
    """Return all @skill-decorated functions found so far."""
    return list(_SKILL_REGISTRY)


def _fn_to_schema(fn: Callable) -> dict[str, Any]:
    """Generate OpenAI-compatible parameters JSON Schema from function signature.

    Inspects type annotations and defaults to build the schema automatically.
    Skips parameters starting with _ (reserved for dependency injection).
    """
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    _type_map = {
        str: "string", "str": "string",
        int: "integer", "int": "integer",
        float: "number", "float": "number",
        bool: "boolean", "bool": "boolean",
    }

    # Resolve string annotations from `from __future__ import annotations`
    hints = {}
    try:
        hints = inspect.get_annotations(fn, eval_str=True)
    except Exception:
        try:
            import typing
            hints = typing.get_type_hints(fn)
        except Exception:
            pass

    for param_name, param in sig.parameters.items():
        # Skip self, cls, and injected dependencies (prefixed with _)
        if param_name in ("self", "cls") or param_name.startswith("_"):
            continue
        # Skip **kwargs
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue

        annotation = hints.get(param_name, param.annotation)
        json_type = "string"  # default
        if annotation != inspect.Parameter.empty:
            json_type = _type_map.get(annotation, "string")

        prop: dict[str, Any] = {"type": json_type}

        # Use docstring or param name as description
        prop["description"] = param_name

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema
