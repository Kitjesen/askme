"""Tests for optional spoken replies from ``askme agent send``."""

from __future__ import annotations

from askme import cli


def test_send_agent_message_via_server_forwards_speak_flag(monkeypatch) -> None:
    """The server path asks the running service to own playback."""
    seen: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"reply": "ok", "text": "hello", "spoken": True}

    def post(url: str, *, json: dict[str, object], timeout: int) -> Response:
        seen["url"] = url
        seen["json"] = json
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(cli.requests, "post", post)

    payload = cli._send_agent_message_via_server(
        "hello",
        "http://runtime/",
        speak=True,
    )

    assert seen == {
        "url": "http://runtime/api/chat",
        "json": {"text": "hello", "speak": True},
        "timeout": 90,
    }
    assert payload["spoken"] is True
    assert payload["server_speak_requested"] is True


def test_cli_agent_send_does_not_speak_reply_by_default(monkeypatch, capsys) -> None:
    """agent send prints the reply without playing audio by default."""
    spoken: list[str] = []

    monkeypatch.setattr(
        cli,
        "_send_agent_message_via_server",
        lambda message, server, *, speak=False: {
            "mode": "server",
            "server": server,
            "reply": f"server:{message}",
            "server_speak_requested": speak,
        },
    )
    monkeypatch.setattr(
        cli,
        "_speak_agent_reply",
        lambda reply: spoken.append(reply),
        raising=False,
    )

    cli.main(["agent", "send", "hello", "--server", "http://runtime"])

    assert capsys.readouterr().out.strip() == "server:hello"
    assert spoken == []


def test_cli_agent_send_speaks_reply_when_requested(monkeypatch, capsys) -> None:
    """agent send --speak asks the active service to play the reply."""
    spoken: list[str] = []
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_send_agent_message_via_server",
        lambda message, server, *, speak=False: seen.update(
            {"message": message, "server": server, "speak": speak}
        ) or {
            "mode": "server",
            "server": server,
            "reply": f"server:{message}",
            "spoken": True,
            "server_speak_requested": speak,
        },
    )
    monkeypatch.setattr(
        cli,
        "_speak_agent_reply",
        lambda reply: spoken.append(reply),
        raising=False,
    )

    cli.main(["agent", "send", "hello", "--server", "http://runtime", "--speak"])

    assert capsys.readouterr().out.strip() == "server:hello"
    assert seen == {"message": "hello", "server": "http://runtime", "speak": True}
    assert spoken == []


def test_cli_agent_send_does_not_speak_empty_reply(monkeypatch, capsys) -> None:
    """agent send --speak skips audio when the turn returns no reply."""
    spoken: list[str] = []

    monkeypatch.setattr(
        cli,
        "_send_agent_message_via_server",
        lambda message, server, *, speak=False: {
            "mode": "server",
            "server": server,
            "reply": "",
            "server_speak_requested": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_speak_agent_reply",
        lambda reply: spoken.append(reply),
        raising=False,
    )

    cli.main(["agent", "send", "hello", "--server", "http://runtime", "--speak"])

    assert capsys.readouterr().out == "\n"
    assert spoken == []


def test_cli_agent_send_reports_speak_error_without_hiding_reply(
    monkeypatch,
    capsys,
) -> None:
    """agent send --speak still prints the reply when playback fails."""

    def fail_to_speak(reply: str) -> None:
        raise RuntimeError("audio offline")

    monkeypatch.setattr(
        cli,
        "_send_agent_message_via_server",
        lambda message, server, *, speak=False: {
            "mode": "server",
            "server": server,
            "reply": f"server:{message}",
            "server_speak_requested": False,
        },
    )
    monkeypatch.setattr(cli, "_speak_agent_reply", fail_to_speak, raising=False)

    cli.main(["agent", "send", "hello", "--server", "http://runtime", "--speak"])

    captured = capsys.readouterr()
    assert captured.out.strip() == "server:hello"
    assert "speak" in captured.err.lower()
    assert "audio offline" in captured.err
