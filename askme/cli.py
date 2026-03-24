"""Structured CLI for askme with dimos-style command groups."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

import requests

DEFAULT_RUNTIME_URL = "http://127.0.0.1:8765"


def build_parser() -> argparse.ArgumentParser:
    """Build the askme CLI parser."""
    parser = argparse.ArgumentParser(
        prog="askme",
        description="Askme CLI",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (overrides default)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level",
    )

    # Legacy compatibility flags.
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport mode (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for SSE transport (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE transport (default: 8080)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run the interactive runtime instead of the MCP server",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Use text mode for the interactive runtime",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Use voice mode for the interactive runtime",
    )
    parser.add_argument(
        "--robot",
        action="store_true",
        help="Enable robot APIs for the interactive runtime",
    )

    subparsers = parser.add_subparsers(dest="command")

    tui_parser = subparsers.add_parser("tui", help="Run the full-screen terminal UI")
    tui_parser.add_argument(
        "--robot",
        action="store_true",
        help="Enable robot APIs in the terminal UI",
    )

    runtime_parser = subparsers.add_parser("runtime", help="Run or inspect askme runtimes")
    runtime_subparsers = runtime_parser.add_subparsers(dest="runtime_command")

    runtime_run = runtime_subparsers.add_parser("run", help="Run the interactive runtime")
    _add_runtime_selection_args(runtime_run)

    runtime_status = runtime_subparsers.add_parser("status", help="Query a running runtime health endpoint")
    runtime_status.add_argument(
        "--server",
        default=DEFAULT_RUNTIME_URL,
        help=f"Base URL for the running runtime (default: {DEFAULT_RUNTIME_URL})",
    )
    runtime_status.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON",
    )

    runtime_capabilities = runtime_subparsers.add_parser(
        "capabilities",
        help="Show runtime capabilities from a local profile or a running server",
    )
    _add_runtime_selection_args(runtime_capabilities)
    runtime_capabilities.add_argument(
        "--server",
        default="",
        help="Read capabilities from a running runtime instead of building locally",
    )
    runtime_capabilities.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON",
    )

    skills_parser = subparsers.add_parser("skills", help="Inspect loaded skills and generated contracts")
    skills_subparsers = skills_parser.add_subparsers(dest="skills_command")

    skills_list = skills_subparsers.add_parser("list", help="List loaded skills")
    skills_list.add_argument("--json", action="store_true", help="Print raw JSON")

    skills_show = skills_subparsers.add_parser("show", help="Show a single skill contract")
    skills_show.add_argument("skill_name", help="Skill name")
    skills_show.add_argument("--json", action="store_true", help="Print raw JSON")

    skills_openapi = skills_subparsers.add_parser("openapi", help="Print generated OpenAPI for skills")
    skills_openapi.add_argument("--json", action="store_true", help="Print raw JSON")

    agent_parser = subparsers.add_parser("agent", help="Interact with askme as an agent shell")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command")

    agent_send = agent_subparsers.add_parser(
        "send",
        help="Send a single message to askme",
    )
    agent_send.add_argument("message", help="Message to send")
    agent_send.add_argument(
        "--server",
        default="",
        help="Use a running runtime via HTTP instead of local execution",
    )
    agent_send.add_argument(
        "--local",
        action="store_true",
        help="Force local one-shot execution even if a runtime is already running",
    )
    agent_send.add_argument(
        "--robot",
        action="store_true",
        help="Enable robot APIs for local execution",
    )
    agent_send.add_argument("--json", action="store_true", help="Print raw JSON")

    mcp_parser = subparsers.add_parser("mcp", help="Serve askme over MCP")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")

    mcp_serve = mcp_subparsers.add_parser("serve", help="Run the MCP server")
    mcp_serve.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport mode (default: stdio)",
    )
    mcp_serve.add_argument(
        "--host",
        default="localhost",
        help="Host for SSE transport (default: localhost)",
    )
    mcp_serve.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE transport (default: 8080)",
    )

    return parser


def _add_runtime_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=["voice", "text", "edge_robot"],
        default="",
        help="Named runtime profile",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Force text mode",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Force voice mode",
    )
    parser.add_argument(
        "--robot",
        action="store_true",
        help="Enable robot APIs",
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the askme CLI."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    _apply_common_options(args)

    if not getattr(args, "command", None):
        _dispatch_compat_mode(args, raw_args=raw_args)
        return

    if args.command == "runtime":
        _handle_runtime_command(args)
        return
    if args.command == "tui":
        _run_terminal_tui(robot_mode=args.robot)
        return
    if args.command == "skills":
        _handle_skills_command(args)
        return
    if args.command == "agent":
        _handle_agent_command(args)
        return
    if args.command == "mcp":
        _handle_mcp_command(args)
        return

    raise SystemExit(f"Unknown command: {args.command}")


def _apply_common_options(args: argparse.Namespace) -> None:
    if getattr(args, "config", None):
        os.environ["ASKME_CONFIG_PATH"] = args.config
    if getattr(args, "log_level", None):
        logging.getLogger().setLevel(getattr(logging, args.log_level))


def _dispatch_compat_mode(args: argparse.Namespace, *, raw_args: list[str]) -> None:
    if args.legacy:
        voice_mode, robot_mode = _resolve_runtime_flags(args)
        _run_interactive_runtime(voice_mode=voice_mode, robot_mode=robot_mode)
        return
    if args.voice:
        _run_interactive_runtime(voice_mode=True, robot_mode=args.robot)
        return
    if args.text:
        _run_interactive_runtime(voice_mode=False, robot_mode=args.robot)
        return
    if _looks_like_mcp_request(raw_args):
        _run_mcp_server(
            transport=args.transport,
            host=args.host,
            port=args.port,
        )
        return
    _run_terminal_tui(robot_mode=args.robot)


def _handle_runtime_command(args: argparse.Namespace) -> None:
    if args.runtime_command == "run":
        voice_mode, robot_mode = _resolve_runtime_flags(args)
        _run_interactive_runtime(voice_mode=voice_mode, robot_mode=robot_mode)
        return

    if args.runtime_command == "status":
        payload = _get_json(f"{_normalise_server_url(args.server)}/health")
        _emit_payload(payload, json_output=args.json)
        return

    if args.runtime_command == "capabilities":
        if args.server:
            payload = _get_json(f"{_normalise_server_url(args.server)}/api/capabilities")
        else:
            voice_mode, robot_mode = _resolve_runtime_flags(args)
            payload = _load_local_capabilities(
                voice_mode=voice_mode,
                robot_mode=robot_mode,
            )
        _emit_payload(payload, json_output=args.json)
        return

    raise SystemExit(f"Unknown runtime command: {args.runtime_command}")


def _handle_skills_command(args: argparse.Namespace) -> None:
    manager = _load_skill_manager()

    if args.skills_command == "list":
        payload = {
            "skills": manager.get_contract_catalog(),
            "count": len(manager.get_all()),
        }
        _emit_payload(payload, json_output=args.json)
        return

    if args.skills_command == "show":
        skill = manager.get(args.skill_name)
        contract = manager.get_contract(args.skill_name)
        if skill is None or contract is None:
            raise SystemExit(f"Unknown skill: {args.skill_name}")
        payload = {
            "name": skill.name,
            "enabled": skill.enabled,
            "trigger": skill.trigger,
            "voice_trigger": skill.voice_trigger,
            "source": skill.source,
            "contract": contract.summary(),
            "parameters": [
                {
                    "name": parameter.name,
                    "type": parameter.type,
                    "description": parameter.description,
                    "required": parameter.required,
                    "default": parameter.default,
                    "enum": list(parameter.enum),
                }
                for parameter in contract.parameters
            ],
        }
        _emit_payload(payload, json_output=args.json)
        return

    if args.skills_command == "openapi":
        _emit_payload(manager.openapi_document(), json_output=True)
        return

    raise SystemExit(f"Unknown skills command: {args.skills_command}")


def _handle_agent_command(args: argparse.Namespace) -> None:
    if args.agent_command != "send":
        raise SystemExit(f"Unknown agent command: {args.agent_command}")

    if args.local:
        payload = _run_local_agent_turn_sync(args.message, robot_mode=args.robot)
        _emit_agent_payload(payload, json_output=args.json)
        return

    if args.server:
        payload = _send_agent_message_via_server(args.message, args.server)
        _emit_agent_payload(payload, json_output=args.json)
        return

    try:
        payload = _send_agent_message_via_server(args.message, DEFAULT_RUNTIME_URL)
    except requests.RequestException:
        payload = _run_local_agent_turn_sync(args.message, robot_mode=args.robot)
    _emit_agent_payload(payload, json_output=args.json)


def _handle_mcp_command(args: argparse.Namespace) -> None:
    if args.mcp_command != "serve":
        raise SystemExit(f"Unknown mcp command: {args.mcp_command}")
    _run_mcp_server(
        transport=args.transport,
        host=args.host,
        port=args.port,
    )


def _load_skill_manager():
    from askme.skills.skill_manager import SkillManager

    manager = SkillManager()
    manager.load()
    return manager


def _resolve_runtime_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    voice_mode = True
    robot_mode = bool(getattr(args, "robot", False))
    profile = getattr(args, "profile", "") or ""

    if profile == "text":
        voice_mode = False
    elif profile in {"voice", "edge_robot"}:
        voice_mode = True

    if getattr(args, "text", False):
        voice_mode = False
    if getattr(args, "voice", False):
        voice_mode = True

    if profile == "edge_robot":
        robot_mode = True

    return voice_mode, robot_mode


def _looks_like_mcp_request(raw_args: list[str]) -> bool:
    return any(arg in {"--transport", "--host", "--port"} for arg in raw_args)


def _run_interactive_runtime(*, voice_mode: bool, robot_mode: bool) -> None:
    from askme.main import run_legacy_app

    asyncio.run(run_legacy_app(voice_mode=voice_mode, robot_mode=robot_mode))


def _run_terminal_tui(*, robot_mode: bool) -> None:
    from askme.tui import run_terminal_ui

    asyncio.run(run_terminal_ui(robot_mode=robot_mode))


def _run_mcp_server(*, transport: str, host: str, port: int) -> None:
    from askme.mcp.server import mcp

    if transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
        return
    mcp.run()


def _load_local_capabilities(*, voice_mode: bool, robot_mode: bool) -> dict[str, Any]:
    return asyncio.run(
        _load_local_capabilities_async(voice_mode=voice_mode, robot_mode=robot_mode)
    )


async def _load_local_capabilities_async(
    *, voice_mode: bool, robot_mode: bool,
) -> dict[str, Any]:
    from askme.config import get_config
    from askme.main import _select_blueprint
    from askme.runtime.profiles import legacy_profile_for

    cfg = get_config()
    blueprint = _select_blueprint(voice_mode=voice_mode, robot_mode=robot_mode)
    app = await blueprint.build(cfg)
    profile = legacy_profile_for(voice_mode=voice_mode, robot_mode=robot_mode)

    skill_mod = app.modules.get("skill")
    sm = getattr(skill_mod, "skill_manager", None) if skill_mod else None
    contracts = sm.get_contracts() if sm else []
    openapi_doc = sm.openapi_document() if sm else {"info": {"title": "", "version": ""}, "paths": {}}

    from askme import __version__ as ASKME_VERSION

    app_name = cfg.get("app", {}).get("name", "askme")
    app_version = cfg.get("app", {}).get("version") or ASKME_VERSION

    components: dict[str, dict[str, Any]] = {}
    for name, mod in app.modules.items():
        components[name] = {
            "health": mod.health(),
            "capabilities": mod.capabilities(),
        }

    return {
        "app": {
            "name": app_name,
            "version": app_version,
            "voice_mode": voice_mode,
            "robot_mode": robot_mode,
        },
        "profile": profile.snapshot(),
        "components": components,
        "skills": {
            "count": len(sm.get_all()) if sm else 0,
            "enabled_count": len(sm.get_enabled()) if sm else 0,
            "contract_count": len(contracts),
            "code_contract_count": sum(
                1 for c in contracts if c.source == "code"
            ),
            "legacy_contract_count": sum(
                1 for c in contracts if c.source != "code"
            ),
            "catalog": sm.get_contract_catalog() if sm else [],
        },
        "openapi": {
            "title": openapi_doc["info"]["title"],
            "version": openapi_doc["info"]["version"],
            "path_count": len(openapi_doc["paths"]),
        },
    }


def _run_local_agent_turn_sync(message: str, *, robot_mode: bool) -> dict[str, Any]:
    return asyncio.run(_run_local_agent_turn(message, robot_mode=robot_mode))


async def _run_local_agent_turn(message: str, *, robot_mode: bool) -> dict[str, Any]:
    from askme.config import get_config
    from askme.main import _select_blueprint
    from askme.runtime.profiles import legacy_profile_for

    cfg = get_config()
    blueprint = _select_blueprint(voice_mode=False, robot_mode=robot_mode)
    app = await blueprint.build(cfg)
    profile = legacy_profile_for(voice_mode=False, robot_mode=robot_mode)
    await app.start()
    try:
        text_mod = app.modules.get("text")
        text_loop = getattr(text_mod, "text_loop", None) if text_mod else None
        reply = await text_loop.process_turn(message) if text_loop else ""
        return {
            "mode": "local",
            "profile": profile.name,
            "reply": reply,
            "message": message,
        }
    finally:
        await app.stop()


def _send_agent_message_via_server(message: str, server: str) -> dict[str, Any]:
    base_url = _normalise_server_url(server)
    response = requests.post(
        f"{base_url}/api/chat",
        json={"text": message},
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "mode": "server",
        "server": base_url,
        "reply": payload.get("reply", ""),
        "message": payload.get("text", message),
    }


def _get_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()


def _normalise_server_url(server: str) -> str:
    return server.rstrip("/")


def _emit_agent_payload(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(_json(payload))  # noqa: T201
        return
    print(payload.get("reply", ""))  # noqa: T201


def _emit_payload(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(_json(payload))  # noqa: T201
        return

    if "profile" in payload and "components" in payload:
        profile = payload["profile"]
        print(f"profile: {profile.get('name')} ({profile.get('primary_loop')})")  # noqa: T201
        for name, component in payload.get("components", {}).items():
            health = component.get("health", {})
            print(f"{name}: {health.get('status', 'unknown')}")  # noqa: T201
        return

    if "skills" in payload and isinstance(payload["skills"], list):
        for skill in payload["skills"]:
            state = "enabled" if skill.get("enabled", False) else "disabled"
            execution = skill.get("execution", "?")
            name = skill.get("name", "?")
            description = skill.get("description", "")
            print(f"{name:20} {state:8} {execution:14} {description}")  # noqa: T201
        return

    print(_json(payload))  # noqa: T201


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1:])
