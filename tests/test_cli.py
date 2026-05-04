"""Tests for the structured askme CLI."""

from __future__ import annotations

import json

import requests

from askme import cli


def test_cli_compat_legacy_routes_to_runtime(monkeypatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_run_interactive_runtime",
        lambda *, voice_mode, robot_mode: seen.update(
            {"voice_mode": voice_mode, "robot_mode": robot_mode}
        ),
    )

    cli.main(["--legacy", "--text", "--robot"])

    assert seen == {"voice_mode": False, "robot_mode": True}


def test_cli_defaults_to_terminal_tui(monkeypatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_run_terminal_tui",
        lambda *, robot_mode: seen.update(
            {"robot_mode": robot_mode}
        ),
    )

    cli.main([])

    assert seen == {"robot_mode": False}


def test_cli_text_flag_routes_to_plain_runtime(monkeypatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_run_interactive_runtime",
        lambda *, voice_mode, robot_mode: seen.update(
            {"voice_mode": voice_mode, "robot_mode": robot_mode}
        ),
    )

    cli.main(["--text", "--robot"])

    assert seen == {"voice_mode": False, "robot_mode": True}


def test_cli_robot_flag_routes_to_tui(monkeypatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_run_terminal_tui",
        lambda *, robot_mode: seen.update({"robot_mode": robot_mode}),
    )

    cli.main(["--robot"])

    assert seen == {"robot_mode": True}


def test_cli_transport_flags_still_route_to_mcp(monkeypatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_run_mcp_server",
        lambda *, transport, host, port: seen.update(
            {"transport": transport, "host": host, "port": port}
        ),
    )

    cli.main(["--transport", "sse", "--host", "0.0.0.0", "--port", "9999"])

    assert seen == {"transport": "sse", "host": "0.0.0.0", "port": 9999}


def test_cli_tui_subcommand(monkeypatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_run_terminal_tui",
        lambda *, robot_mode: seen.update({"robot_mode": robot_mode}),
    )

    cli.main(["tui", "--robot"])

    assert seen == {"robot_mode": True}


def test_cli_agent_send_uses_server(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_send_agent_message_via_server",
        lambda message, server: {
            "mode": "server",
            "server": server,
            "reply": f"server:{message}",
        },
    )

    cli.main(["agent", "send", "hello", "--server", "http://runtime"])

    assert capsys.readouterr().out.strip() == "server:hello"


def test_cli_agent_send_falls_back_to_local(monkeypatch, capsys) -> None:
    def _fail(message: str, server: str) -> dict[str, str]:
        raise requests.RequestException("offline")

    monkeypatch.setattr(cli, "_send_agent_message_via_server", _fail)
    monkeypatch.setattr(
        cli,
        "_run_local_agent_turn_sync",
        lambda message, robot_mode: {
            "mode": "local",
            "profile": "text",
            "reply": f"local:{message}:{robot_mode}",
        },
    )

    cli.main(["agent", "send", "hello", "--robot", "--json"])

    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "local"
    assert data["reply"] == "local:hello:True"


def test_cli_mission_draft_routes_to_local_adapter(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def _fake_draft(text, **kwargs):
        seen["text"] = text
        seen.update(kwargs)
        return {"mission": {"mission_id": "mission-1", "goal": text}, "drafted": True}

    monkeypatch.setattr(cli, "_draft_mission_sync", _fake_draft)

    cli.main([
        "mission",
        "draft",
        "inspect",
        "area-a",
        "--operator-id",
        "operator-1",
        "--json",
    ])

    data = json.loads(capsys.readouterr().out)
    assert data["drafted"] is True
    assert seen["text"] == "inspect area-a"
    assert seen["operator_id"] == "operator-1"


def test_cli_mission_run_defaults_to_dry_run(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def _fake_run(source, **kwargs):
        seen["source"] = source
        seen.update(kwargs)
        return {
            "mission": {"mission_id": "mission-1", "status": "dry_run"},
            "submission": {"dry_run": kwargs["dry_run"]},
        }

    monkeypatch.setattr(cli, "_run_mission_sync", _fake_run)

    cli.main(["mission", "run", "inspect", "area-a", "--confirm", "--json"])

    data = json.loads(capsys.readouterr().out)
    assert data["submission"]["dry_run"] is True
    assert seen["source"] == "inspect area-a"
    assert seen["confirmed"] is True


def test_cli_mission_report_uses_server_when_requested(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def _fake_report(mission_id, **kwargs):
        seen["mission_id"] = mission_id
        seen.update(kwargs)
        return {"report": {"mission_id": mission_id, "status": "dry_run"}}

    monkeypatch.setattr(cli, "_mission_report_sync", _fake_report)

    cli.main([
        "mission",
        "report",
        "mission-1",
        "--server",
        "http://runtime",
        "--json",
    ])

    data = json.loads(capsys.readouterr().out)
    assert data["report"]["mission_id"] == "mission-1"
    assert seen["server"] == "http://runtime"


def test_cli_skills_show_returns_code_contract(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    cli.main(["skills", "show", "navigate", "--json"])

    data = json.loads(capsys.readouterr().out)
    assert data["name"] == "navigate"
    assert data["contract"]["contract_source"] == "code"
