"""Agent CLI commands — send one-shot runtime chat messages."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import requests

from askme.cli.utils import (
    _cli_root_override,
    _emit_agent_payload,
    _normalise_server_url,
    _server_auth_headers,
)

DEFAULT_RUNTIME_URL = "http://127.0.0.1:8765"


def _handle_agent_command(args: Any) -> None:
    """Handle the 'agent' command group: send."""
    if args.agent_command != "send":
        raise SystemExit(f"Unknown agent command: {args.agent_command}")

    run_local_turn = _cli_root_override("_run_local_agent_turn_for_cli", _run_local_agent_turn_for_cli)
    send_server_message = _cli_root_override(
        "_send_agent_message_via_server",
        _send_agent_message_via_server,
    )
    speak_agent_payload = _cli_root_override("_speak_agent_payload", _speak_agent_payload)

    if args.local:
        payload = run_local_turn(
            args.message,
            robot_mode=args.robot,
            speak=args.speak,
        )
        _emit_agent_payload(payload, json_output=args.json)
        return

    if args.server:
        payload = send_server_message(
            args.message,
            args.server,
            speak=args.speak,
        )
        speak_agent_payload(
            payload,
            enabled=args.speak and not bool(payload.get("server_speak_requested")),
        )
        _emit_agent_payload(payload, json_output=args.json)
        return

    try:
        payload = send_server_message(
            args.message,
            DEFAULT_RUNTIME_URL,
            speak=args.speak,
        )
    except requests.RequestException:
        payload = run_local_turn(
            args.message,
            robot_mode=args.robot,
            speak=args.speak,
        )
    else:
        speak_agent_payload(
            payload,
            enabled=args.speak and not bool(payload.get("server_speak_requested")),
        )
    _emit_agent_payload(payload, json_output=args.json)


def _run_local_agent_turn_sync(
    message: str,
    *,
    robot_mode: bool,
    speak: bool = False,
) -> dict[str, Any]:
    """Run a synchronous local text-runtime turn."""
    return asyncio.run(_run_local_agent_turn(message, robot_mode=robot_mode, speak=speak))


def _run_local_agent_turn_for_cli(
    message: str,
    *,
    robot_mode: bool,
    speak: bool,
) -> dict[str, Any]:
    """Run a local text-runtime turn for the CLI, optionally with TTS."""
    run_sync = _cli_root_override("_run_local_agent_turn_sync", _run_local_agent_turn_sync)
    if speak:
        return run_sync(message, robot_mode=robot_mode, speak=True)
    return run_sync(message, robot_mode=robot_mode)


async def _run_local_agent_turn(
    message: str,
    *,
    robot_mode: bool,
    speak: bool = False,
) -> dict[str, Any]:
    """Build a local runtime, process one turn, and return the reply payload."""
    from askme.config import get_config
    from askme.main import _select_blueprint
    from askme.runtime.core.profiles import legacy_profile_for

    cfg = get_config()
    blueprint = _select_blueprint(voice_mode=False, robot_mode=robot_mode)
    app = await blueprint.build(cfg)
    profile = legacy_profile_for(voice_mode=False, robot_mode=robot_mode)
    await app.start()
    try:
        text_mod = app.modules.get("text")
        text_loop = getattr(text_mod, "text_loop", None) if text_mod else None
        reply = await text_loop.process_turn(message) if text_loop else ""
        payload = {
            "mode": "local",
            "profile": profile.name,
            "reply": reply,
            "message": message,
        }
        if speak:
            try:
                payload["spoken"] = await _speak_local_text_reply(text_loop, reply)
            except Exception as exc:
                payload["spoken"] = False
                payload["speak_error"] = str(exc)
                _report_speak_error(exc)
        return payload
    finally:
        await app.stop()


async def _speak_local_text_reply(text_loop: Any, reply: str) -> bool:
    """Wait for the TextLoop audio queue or play *reply* if it has not queued."""
    if not isinstance(reply, str) or not reply.strip():
        return False
    audio = getattr(text_loop, "_audio", None) if text_loop is not None else None
    if audio is None:
        raise RuntimeError("local text loop has no audio output")

    if not bool(getattr(audio, "is_busy", False)):
        audio.speak(reply.strip())
        audio.start_playback()
    try:
        done = await asyncio.to_thread(audio.wait_speaking_done)
        if done is False:
            raise TimeoutError("TTS playback did not finish within timeout")
    finally:
        audio.stop_playback()
    return True


def _speak_agent_payload(payload: dict[str, Any], *, enabled: bool) -> None:
    """Play the agent reply through TTS if enabled and a reply is present."""
    if not enabled:
        return
    reply = payload.get("reply", "")
    if not isinstance(reply, str) or not reply.strip():
        return
    try:
        speak_reply = _cli_root_override("_speak_agent_reply", _speak_agent_reply)
        speak_reply(reply.strip())
    except Exception as exc:
        _report_speak_error(exc)


def _speak_agent_reply(reply: str) -> None:
    """Play a one-shot agent reply using the local configured TTS output."""
    from askme.config import get_config
    from askme.providers import build_audio_frontend

    audio = build_audio_frontend(get_config(), voice_mode=False).audio
    audio.speak(reply)
    audio.start_playback()
    try:
        done = audio.wait_speaking_done()
        if done is False:
            raise TimeoutError("TTS playback did not finish within timeout")
    finally:
        audio.stop_playback()
        audio.shutdown()


def _report_speak_error(exc: Exception) -> None:
    """Print a speak error to stderr."""
    print(f"[askme] speak failed: {exc}", file=sys.stderr)


def _send_agent_message_via_server(
    message: str,
    server: str,
    *,
    speak: bool = False,
) -> dict[str, Any]:
    """Send a message to a running askme runtime via HTTP and return the reply."""
    base_url = _normalise_server_url(server)
    request_payload: dict[str, Any] = {"text": message}
    if speak:
        request_payload["speak"] = True
    kwargs: dict[str, Any] = {
        "json": request_payload,
        "timeout": 90 if speak else 5,
    }
    headers = _server_auth_headers()
    if headers:
        kwargs["headers"] = headers
    http_requests = _cli_root_override("requests", requests)
    response = http_requests.post(f"{base_url}/api/chat", **kwargs)
    response.raise_for_status()
    payload = response.json()
    result: dict[str, Any] = {
        "mode": "server",
        "server": base_url,
        "reply": payload.get("reply", ""),
        "message": payload.get("text", message),
        "server_speak_requested": bool(speak),
    }
    for key in ("spoken", "speak_error"):
        if key in payload:
            result[key] = payload[key]
    return result
