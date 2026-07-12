"""Askme CLI — thin dispatch over cli/ submodules."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any

import requests  # kept at module level for test monkeypatching compat

from askme.cli._app import build_parser

# ── Fallback to submodules for monkeypatching compat ───────────────────────
# Tests do ``monkeypatch.setattr(cli, "_run_field_*", mock)`` which requires
# every public symbol from the original cli.py to be resolvable as ``cli.xxx``.
# We resolve them lazily through the submodule packages.

_SUB_MODULES: dict[str, str] = {
    "utils": "askme.cli.utils",
    "runtime": "askme.cli.runtime",
    "skills": "askme.cli.skills",
    "agent": "askme.cli.agent",
    "mission": "askme.cli.mission",
    "memory": "askme.cli.memory",
    "voice": "askme.cli.voice",
    "field": "askme.cli.field",
    "field_audit": "askme.cli.field_audit",
    "audit_cmd": "askme.cli.audit_cmd",
}

_EXPLICIT = {
    k: v
    for k, v in globals().items()
    if not k.startswith("_") and k not in ("annotations", "Any")
}


def __getattr__(name: str) -> Any:
    # Check already-imported names first
    if name in _EXPLICIT:
        return _EXPLICIT[name]
    # Search submodules
    for sub_name, sub_path in _SUB_MODULES.items():
        try:
            mod = sys.modules.get(sub_path) or importlib.import_module(sub_path)
        except ImportError:
            continue
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    result = list(_EXPLICIT)
    for sub_path in _SUB_MODULES.values():
        try:
            mod = sys.modules.get(sub_path) or importlib.import_module(sub_path)
        except ImportError:
            continue
        if hasattr(mod, "__all__"):
            result.extend(mod.__all__)
        else:
            result.extend(n for n in dir(mod) if not n.startswith("_"))
    return sorted(set(result))


# ── Dispatch helpers ───────────────────────────────────────────────────────


def _apply_common_options(args: Any) -> None:
    """Apply global CLI options (config path, log level)."""
    if getattr(args, "config", None):
        os.environ["ASKME_CONFIG_PATH"] = args.config
    if getattr(args, "log_level", None):
        logging.getLogger().setLevel(getattr(logging, args.log_level))


def _dispatch_compat_mode(args: Any, *, raw_args: list[str]) -> None:
    """Handle no-subcommand invocation (legacy / TUI / MCP compat)."""
    from askme.cli.runtime import (
        _looks_like_mcp_request,
        _run_interactive_runtime,
        _run_mcp_server,
        _run_terminal_tui,
    )
    from askme.cli.utils import _cli_root_override, _resolve_runtime_flags

    resolve_runtime_flags = _cli_root_override(
        "_resolve_runtime_flags", _resolve_runtime_flags
    )
    looks_like_mcp_request = _cli_root_override(
        "_looks_like_mcp_request", _looks_like_mcp_request
    )
    run_interactive_runtime = _cli_root_override(
        "_run_interactive_runtime", _run_interactive_runtime
    )
    run_mcp_server = _cli_root_override("_run_mcp_server", _run_mcp_server)
    run_terminal_tui = _cli_root_override("_run_terminal_tui", _run_terminal_tui)

    if args.legacy:
        voice_mode, robot_mode = resolve_runtime_flags(args)
        run_interactive_runtime(voice_mode=voice_mode, robot_mode=robot_mode)
        return
    if args.voice:
        run_interactive_runtime(voice_mode=True, robot_mode=args.robot)
        return
    if args.text:
        run_interactive_runtime(voice_mode=False, robot_mode=args.robot)
        return
    if looks_like_mcp_request(raw_args):
        run_mcp_server(transport=args.transport, host=args.host, port=args.port)
        return
    run_terminal_tui(robot_mode=args.robot)


def _handle_mcp_command(args: Any) -> None:
    from askme.cli.runtime import _run_mcp_server
    from askme.cli.utils import _cli_root_override

    if args.mcp_command != "serve":
        raise SystemExit(f"Unknown mcp command: {args.mcp_command}")
    run_mcp_server = _cli_root_override("_run_mcp_server", _run_mcp_server)
    run_mcp_server(transport=args.transport, host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the askme CLI."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    _apply_common_options(args)

    if not getattr(args, "command", None):
        _dispatch_compat_mode(args, raw_args=raw_args)
        return

    if args.command == "runtime":
        from askme.cli.runtime import _handle_runtime_command
        from askme.cli.utils import _cli_root_override

        handle_runtime_command = _cli_root_override(
            "_handle_runtime_command", _handle_runtime_command
        )
        handle_runtime_command(args)
    elif args.command == "tui":
        from askme.cli.runtime import _run_terminal_tui
        from askme.cli.utils import _cli_root_override

        run_terminal_tui = _cli_root_override("_run_terminal_tui", _run_terminal_tui)
        run_terminal_tui(robot_mode=args.robot)
    elif args.command == "skills":
        from askme.cli.skills import _handle_skills_command
        from askme.cli.utils import _cli_root_override

        handle_skills_command = _cli_root_override(
            "_handle_skills_command", _handle_skills_command
        )
        handle_skills_command(args)
    elif args.command == "agent":
        from askme.cli.agent import _handle_agent_command
        from askme.cli.utils import _cli_root_override

        handle_agent_command = _cli_root_override("_handle_agent_command", _handle_agent_command)
        handle_agent_command(args)
    elif args.command == "mission":
        from askme.cli.mission import _handle_mission_command
        from askme.cli.utils import _cli_root_override

        handle_mission_command = _cli_root_override(
            "_handle_mission_command", _handle_mission_command
        )
        handle_mission_command(args)
    elif args.command == "memory":
        from askme.cli.memory import _handle_memory_command
        from askme.cli.utils import _cli_root_override

        handle_memory_command = _cli_root_override(
            "_handle_memory_command", _handle_memory_command
        )
        handle_memory_command(args)
    elif args.command == "mcp":
        _handle_mcp_command(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")
