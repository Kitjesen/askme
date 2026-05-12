"""Aggregate Sunrise voice readiness checks into one operator-facing gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from askme.config import get_config
from askme.voice.health_check import run_voice_health
from askme.voice.sunrise_audio_doctor import (
    DEFAULT_FIRST_TOKEN_GUARD_SECONDS,
    print_sunrise_audio_doctor_summary,
    run_sunrise_audio_doctor,
)

DEFAULT_ROOM_LOOP_TEXT = "\u4e00\u4e8c\u4e09\u56db\u4e94\uff0c\u8bed\u97f3\u5c31\u7eea\u6d4b\u8bd5\u3002"
DEFAULT_ROOM_LOOP_EXPECT_PREFIX = "\u4e00"

HealthRunner = Callable[[dict[str, Any]], dict[str, Any]]
AudioDoctorRunner = Callable[[dict[str, Any], float], dict[str, Any]]
RoomLoopRunner = Callable[[str, str, int, bool, str], dict[str, Any]]


def run_sunrise_voice_readiness(
    config: dict[str, Any] | None = None,
    *,
    guard_min_seconds: float = DEFAULT_FIRST_TOKEN_GUARD_SECONDS,
    include_room_loop: bool = False,
    room_loop_text: str = DEFAULT_ROOM_LOOP_TEXT,
    room_loop_expect_prefix: str = DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
    room_loop_trials: int = 3,
    live_tts_room_loop: bool = False,
    room_loop_asr: str = "auto",
    require_cloud_asr: bool = False,
    voice_health_runner: HealthRunner | None = None,
    audio_doctor_runner: AudioDoctorRunner | None = None,
    room_loop_runner: RoomLoopRunner | None = None,
) -> dict[str, Any]:
    """Run the non-playing Sunrise readiness gate, optionally adding room loopback."""
    cfg = get_config(reload=True) if config is None else config
    health_runner = voice_health_runner or _run_voice_health
    doctor_runner = audio_doctor_runner or _run_audio_doctor
    loop_runner = room_loop_runner or _run_room_loop_sentinel

    voice_health = health_runner(cfg)
    audio_doctor = doctor_runner(cfg, guard_min_seconds)
    cloud_asr = _check_cloud_asr_requirement(cfg, required=require_cloud_asr)
    effective_room_loop_asr = (
        "cloud" if room_loop_asr == "auto" and require_cloud_asr else room_loop_asr
    )
    if effective_room_loop_asr == "auto":
        effective_room_loop_asr = "local"
    if include_room_loop:
        room_loop = loop_runner(
            room_loop_text,
            room_loop_expect_prefix,
            max(1, int(room_loop_trials)),
            live_tts_room_loop,
            effective_room_loop_asr,
        )
    else:
        room_loop = {
            "status": "skipped",
            "required": False,
            "errors": [],
            "warnings": ["room loopback was skipped; pass --with-room-loop for acoustic verification"],
        }

    required_checks = {
        "voice_health": voice_health.get("status") == "ok",
        "sunrise_audio_doctor": audio_doctor.get("status") == "ok",
    }
    if require_cloud_asr:
        required_checks["cloud_asr"] = cloud_asr.get("status") == "ok"
    room_loop_ok = room_loop.get("status") == "ok"
    room_loop_required = bool(include_room_loop)
    status_ok = all(required_checks.values()) and (room_loop_ok or not room_loop_required)

    errors: list[str] = []
    warnings: list[str] = []
    _extend_messages(errors, "voice_health", voice_health.get("errors", []))
    _extend_messages(warnings, "voice_health", voice_health.get("warnings", []))
    _extend_messages(errors, "sunrise_audio_doctor", audio_doctor.get("errors", []))
    _extend_messages(warnings, "sunrise_audio_doctor", audio_doctor.get("warnings", []))
    _extend_messages(errors, "cloud_asr", cloud_asr.get("errors", []))
    _extend_messages(warnings, "cloud_asr", cloud_asr.get("warnings", []))
    _extend_messages(errors, "room_loop", room_loop.get("errors", []))
    _extend_messages(warnings, "room_loop", room_loop.get("warnings", []))

    return {
        "status": "ok" if status_ok else "degraded",
        "target": "sunrise-voice-readiness",
        "summary": {
            "required_checks_ok": all(required_checks.values()),
            "room_loop_required": room_loop_required,
            "room_loop_ok": room_loop_ok if include_room_loop else None,
            "cloud_asr_required": bool(require_cloud_asr),
            "cloud_asr_ok": cloud_asr.get("status") == "ok" if require_cloud_asr else None,
            "room_loop_asr": effective_room_loop_asr if include_room_loop else None,
            "guard_min_seconds": guard_min_seconds,
        },
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "voice_health": voice_health,
            "sunrise_audio_doctor": audio_doctor,
            "cloud_asr": cloud_asr,
            "room_loop": room_loop,
        },
    }


def print_sunrise_voice_readiness_summary(payload: dict[str, Any]) -> None:
    """Print a compact readiness summary."""
    checks = payload.get("checks", {})
    voice_health = checks.get("voice_health", {})
    audio_doctor = checks.get("sunrise_audio_doctor", {})
    cloud_asr = checks.get("cloud_asr", {})
    room_loop = checks.get("room_loop", {})

    print(f"sunrise-voice-readiness: {payload.get('status', 'unknown')}")  # noqa: T201
    print(f"  voice-health: {voice_health.get('status', 'unknown')}")  # noqa: T201
    print_sunrise_audio_doctor_summary(audio_doctor)
    print(f"  cloud-asr: {cloud_asr.get('status', 'unknown')}")  # noqa: T201
    print(f"  room-loop: {room_loop.get('status', 'unknown')}")  # noqa: T201
    for warning in payload.get("warnings", []):
        print(f"  warn: {warning}")  # noqa: T201
    for error in payload.get("errors", []):
        print(f"  error: {error}")  # noqa: T201


def _run_voice_health(config: dict[str, Any]) -> dict[str, Any]:
    return run_voice_health(config)


def _run_audio_doctor(config: dict[str, Any], guard_min_seconds: float) -> dict[str, Any]:
    return run_sunrise_audio_doctor(config, guard_min_seconds=guard_min_seconds)


def _check_cloud_asr_requirement(config: dict[str, Any], *, required: bool) -> dict[str, Any]:
    voice_cfg = config.get("voice", {}) if isinstance(config, dict) else {}
    cloud_cfg = voice_cfg.get("cloud_asr", {}) if isinstance(voice_cfg, dict) else {}
    if not isinstance(cloud_cfg, dict):
        cloud_cfg = {}

    enabled = bool(cloud_cfg.get("enabled", False))
    api_key = str(cloud_cfg.get("api_key", "") or "").strip()
    api_key_present = bool(api_key) and not api_key.startswith("${")
    websocket_ok = _websocket_client_available()
    errors: list[str] = []

    if required:
        if not enabled:
            errors.append("voice.cloud_asr.enabled is not true")
        if not api_key_present:
            errors.append("voice.cloud_asr.api_key is empty")
        if not websocket_ok:
            errors.append("Cloud ASR dependency missing: websocket-client")

    return {
        "status": "ok" if not errors and required else "skipped" if not required else "degraded",
        "required": bool(required),
        "enabled": enabled,
        "api_key_present": api_key_present,
        "dependency_ok": websocket_ok,
        "model": str(cloud_cfg.get("model", "paraformer-realtime-v2")),
        "errors": errors,
        "warnings": [],
    }


def _dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _websocket_client_available() -> bool:
    """Return True only for the websocket-client package API."""
    if importlib.util.find_spec("websocket") is None:
        return False
    try:
        import websocket  # type: ignore[import-not-found]
    except Exception:
        return False
    return callable(getattr(websocket, "create_connection", None))


def _run_room_loop_sentinel(
    text: str,
    expect_prefix: str,
    trials: int,
    live_tts: bool,
    asr_backend: str = "local",
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "bench" / "test_sunrise_audio_sentinel.py"
    if not script.is_file():
        return {
            "status": "degraded",
            "required": True,
            "errors": [f"room loop sentinel script missing: {script}"],
            "warnings": [],
        }

    artifact_dir = Path(tempfile.mkdtemp(prefix="sunrise_voice_readiness_room_loop_"))
    json_out = artifact_dir / "room_loop.json"
    wav_out = artifact_dir / "room_loop.wav"
    command = [
        sys.executable,
        str(script),
        "--text",
        text,
        "--expect-prefix",
        expect_prefix,
        "--trials",
        str(trials),
        "--json-out",
        str(json_out),
        "--wav-out",
        str(wav_out),
        "--asr-backend",
        asr_backend,
    ]
    if live_tts:
        command.append("--live-tts")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(90, trials * 45),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "degraded",
            "required": True,
            "artifact_dir": str(artifact_dir),
            "command": command,
            "returncode": 124,
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or "timeout"),
            "errors": ["room loop sentinel timed out"],
            "warnings": [],
        }

    sentinel_payload: dict[str, Any] = {}
    if json_out.is_file():
        try:
            sentinel_payload = json.loads(json_out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            sentinel_payload = {"read_error": str(exc)}

    passed = bool(sentinel_payload.get("summary", {}).get("passed"))
    status_ok = result.returncode == 0 and passed
    errors = []
    if not status_ok:
        errors.append("room loop sentinel did not pass")
    return {
        "status": "ok" if status_ok else "degraded",
        "required": True,
        "artifact_dir": str(artifact_dir),
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
        "payload": sentinel_payload,
        "errors": errors,
        "warnings": [],
    }


def _extend_messages(target: list[str], prefix: str, messages: Any) -> None:
    for message in messages or []:
        target.append(f"{prefix}: {message}")


def _tail(text: str, *, lines: int = 16) -> str:
    return "\n".join(str(text).splitlines()[-lines:])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    parser.add_argument("--json-out", default="", help="Also write the readiness JSON to this path")
    parser.add_argument(
        "--guard-min-seconds",
        type=float,
        default=DEFAULT_FIRST_TOKEN_GUARD_SECONDS,
        help="Minimum sacrificial lead-in+cushion before real speech",
    )
    parser.add_argument("--with-room-loop", action="store_true", help="Also run acoustic room loopback")
    parser.add_argument("--room-loop-trials", type=int, default=3)
    parser.add_argument("--room-loop-text", default=DEFAULT_ROOM_LOOP_TEXT)
    parser.add_argument("--room-loop-expect-prefix", default=DEFAULT_ROOM_LOOP_EXPECT_PREFIX)
    parser.add_argument("--live-tts-room-loop", action="store_true")
    parser.add_argument(
        "--room-loop-asr",
        choices=("auto", "local", "cloud", "both"),
        default="auto",
        help="ASR backend used by the acoustic room-loop transcript gate",
    )
    parser.add_argument(
        "--require-cloud-asr",
        action="store_true",
        help="Fail readiness unless Cloud ASR is enabled and configured",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_sunrise_voice_readiness(
        guard_min_seconds=args.guard_min_seconds,
        include_room_loop=args.with_room_loop,
        room_loop_text=args.room_loop_text,
        room_loop_expect_prefix=args.room_loop_expect_prefix,
        room_loop_trials=args.room_loop_trials,
        live_tts_room_loop=args.live_tts_room_loop,
        room_loop_asr=args.room_loop_asr,
        require_cloud_asr=args.require_cloud_asr,
    )
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))  # noqa: T201
    else:
        print_sunrise_voice_readiness_summary(payload)
    return 0 if payload.get("status") == "ok" else 1
