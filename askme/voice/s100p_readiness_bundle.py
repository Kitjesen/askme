"""Collect S100P field-readiness evidence into one auditable bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import request

from askme.voice.sunrise_audio_doctor import DEFAULT_FIRST_TOKEN_GUARD_SECONDS
from askme.voice.sunrise_readiness import (
    DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
    DEFAULT_ROOM_LOOP_TEXT,
)

DEFAULT_HEALTH_URL = "http://127.0.0.1:8765"
DEFAULT_CHANGE_EVENT_FILE = "/tmp/askme_events.jsonl"
DEFAULT_JOURNAL_SINCE = "30 minutes ago"
DEFAULT_COMMAND_TIMEOUT_SECONDS = 180.0
DEFAULT_CONCURRENCY_REQUEST_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[[list[str], float], CommandResult]
HttpGetter = Callable[[str, float], tuple[int, str]]
Clock = Callable[[], datetime]


def collect_s100p_readiness_bundle(
    output_dir: str | Path | None = None,
    *,
    field: bool = False,
    include_room_loop: bool = False,
    room_loop_trials: int = 3,
    room_loop_text: str = DEFAULT_ROOM_LOOP_TEXT,
    room_loop_expect_prefix: str = DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
    live_tts_room_loop: bool = False,
    require_cloud_asr: bool = False,
    guard_min_seconds: float = DEFAULT_FIRST_TOKEN_GUARD_SECONDS,
    health_url: str = DEFAULT_HEALTH_URL,
    change_event_file: str | Path = DEFAULT_CHANGE_EVENT_FILE,
    journal_since: str = DEFAULT_JOURNAL_SINCE,
    skip_health: bool = False,
    skip_service_log: bool = False,
    command_timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    command_runner: CommandRunner | None = None,
    http_getter: HttpGetter | None = None,
    now: Clock | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Collect automated readiness artifacts and write a manifest."""
    if field:
        include_room_loop = True
        require_cloud_asr = True

    runner = command_runner or _default_command_runner
    getter = http_getter or _default_http_getter
    host = hostname or socket.gethostname()
    clock = now or (lambda: datetime.now(UTC))
    created_at = clock()
    bundle_dir = _resolve_bundle_dir(output_dir, created_at=created_at, hostname=host)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    _append_command_step(
        steps,
        name="git-commit",
        command=["git", "rev-parse", "HEAD"],
        artifact=bundle_dir / "commit.txt",
        required=False,
        runner=runner,
        timeout=command_timeout,
        warnings=warnings,
        errors=errors,
    )
    _append_command_step(
        steps,
        name="voice-health",
        command=[sys.executable, "-m", "askme", "runtime", "voice-health", "--json"],
        artifact=bundle_dir / "voice-health.json",
        required=True,
        runner=runner,
        timeout=command_timeout,
        warnings=warnings,
        errors=errors,
    )

    base_readiness = _readiness_command(
        json_out=bundle_dir / "sunrise-readiness.json",
        guard_min_seconds=guard_min_seconds,
    )
    _append_command_step(
        steps,
        name="sunrise-readiness",
        command=base_readiness,
        artifact=bundle_dir / "sunrise-readiness.json",
        required=True,
        runner=runner,
        timeout=command_timeout,
        warnings=warnings,
        errors=errors,
    )

    if include_room_loop:
        room_loop_artifact = bundle_dir / "room-loop-readiness.json"
        room_loop_command = _readiness_command(
            json_out=room_loop_artifact,
            guard_min_seconds=guard_min_seconds,
            include_room_loop=True,
            room_loop_trials=room_loop_trials,
            room_loop_text=room_loop_text,
            room_loop_expect_prefix=room_loop_expect_prefix,
            live_tts_room_loop=live_tts_room_loop,
        )
        _append_command_step(
            steps,
            name="room-loop-readiness",
            command=room_loop_command,
            artifact=bundle_dir / "room-loop-readiness.json",
            required=True,
            runner=runner,
            timeout=max(command_timeout, max(1, int(room_loop_trials)) * 45.0),
            warnings=warnings,
            errors=errors,
        )
        _append_room_loop_artifact_index(
            steps,
            readiness_artifact=room_loop_artifact,
            artifact=bundle_dir / "room-loop-artifacts.txt",
            bundled_dir=bundle_dir / "room-loop",
            required=field,
            warnings=warnings,
            errors=errors,
        )

    if require_cloud_asr:
        cloud_command = _readiness_command(
            json_out=bundle_dir / "cloud-asr-readiness.json",
            guard_min_seconds=guard_min_seconds,
            include_room_loop=include_room_loop,
            room_loop_trials=room_loop_trials,
            room_loop_text=room_loop_text,
            room_loop_expect_prefix=room_loop_expect_prefix,
            live_tts_room_loop=live_tts_room_loop,
            require_cloud_asr=True,
        )
        _append_command_step(
            steps,
            name="cloud-asr-readiness",
            command=cloud_command,
            artifact=bundle_dir / "cloud-asr-readiness.json",
            required=True,
            runner=runner,
            timeout=max(command_timeout, max(1, int(room_loop_trials)) * 45.0),
            warnings=warnings,
            errors=errors,
        )

    if not skip_health:
        _append_http_step(
            steps,
            name="health",
            url=_join_url(health_url, "/health"),
            artifact=bundle_dir / "health.json",
            required=True,
            validator=_validate_json_health_ok,
            getter=getter,
            timeout=command_timeout,
            warnings=warnings,
            errors=errors,
        )
        _append_http_step(
            steps,
            name="healthz",
            url=_join_url(health_url, "/healthz"),
            artifact=bundle_dir / "healthz.json",
            required=True,
            validator=_validate_json_health_ok,
            getter=getter,
            timeout=command_timeout,
            warnings=warnings,
            errors=errors,
        )
        _append_http_step(
            steps,
            name="prometheus",
            url=_join_url(health_url, "/metrics/prometheus"),
            artifact=bundle_dir / "prometheus.txt",
            required=True,
            validator=_validate_prometheus_health_metric,
            getter=getter,
            timeout=command_timeout,
            warnings=warnings,
            errors=errors,
        )
    elif field:
        for name in ("health", "healthz", "prometheus"):
            _append_manual_required_step(
                steps,
                name=name,
                artifact=bundle_dir / f"{name}.manual-required.txt",
                message=f"{name} endpoint evidence is required in field mode",
                errors=errors,
            )

    if not skip_service_log:
        _append_command_step(
            steps,
            name="systemctl-cat",
            command=["systemctl", "cat", "askme.service"],
            artifact=bundle_dir / "askme.service.cat.txt",
            required=field,
            runner=runner,
            timeout=command_timeout,
            warnings=warnings,
            errors=errors,
        )
        _append_command_step(
            steps,
            name="service-log",
            command=["journalctl", "-u", "askme.service", "--since", journal_since],
            artifact=bundle_dir / "askme.service.log",
            required=field,
            runner=runner,
            timeout=command_timeout,
            warnings=warnings,
            errors=errors,
        )
    elif field:
        for name in ("systemctl-cat", "service-log"):
            _append_manual_required_step(
                steps,
                name=name,
                artifact=bundle_dir / f"{name}.manual-required.txt",
                message=f"{name} evidence is required in field mode",
                errors=errors,
            )

    _append_event_file_step(
        steps,
        source=Path(change_event_file),
        artifact=bundle_dir / "change-events.jsonl",
        required=field,
        warnings=warnings,
        errors=errors,
    )
    if field:
        _append_otrev_closed_loop_step(
            steps,
            event_artifact=bundle_dir / "change-events.jsonl",
            log_artifact=bundle_dir / "askme.service.log",
            artifact=bundle_dir / "otrev-proactive-closed-loop.txt",
            errors=errors,
        )

    notes_path = bundle_dir / "notes.md"
    _write_notes_template(notes_path, field=field)

    status = "ok" if all(step["ok"] for step in steps if step.get("required")) else "degraded"
    manifest = {
        "status": status,
        "target": "s100p-readiness-bundle",
        "created_at": created_at.astimezone(UTC).isoformat(),
        "hostname": host,
        "bundle_dir": str(bundle_dir.resolve()),
        "notes": str(notes_path.resolve()),
        "summary": {
            "required_steps_ok": status == "ok",
            "required_step_count": sum(1 for step in steps if step.get("required")),
            "manual_required_step_count": sum(
                1 for step in steps if step.get("requirement") == "manual_required"
            ),
            "step_count": len(steps),
            "field": field,
        },
        "errors": errors,
        "warnings": warnings,
        "steps": steps,
    }
    _write_json(bundle_dir / "manifest.json", manifest)
    return manifest


def print_s100p_readiness_bundle_summary(payload: dict[str, Any]) -> None:
    """Print a compact operator-facing bundle summary."""
    print(f"s100p-readiness-bundle: {payload.get('status', 'unknown')}")  # noqa: T201
    print(f"  bundle-dir: {payload.get('bundle_dir', '')}")  # noqa: T201
    for step in payload.get("steps", []):
        marker = "ok" if step.get("ok") else "degraded"
        requirement = step.get("requirement") or _requirement_label(bool(step.get("required")))
        print(f"  {marker}: {step.get('name', 'unknown')} ({requirement})")  # noqa: T201
    for warning in payload.get("warnings", []):
        print(f"  warn: {warning}")  # noqa: T201
    for error in payload.get("errors", []):
        print(f"  error: {error}")  # noqa: T201


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Bundle output directory (default: artifacts/s100p/<timestamp>-<hostname>)",
    )
    parser.add_argument(
        "--field",
        action="store_true",
        help="Require S100P field evidence, acoustic room-loop artifacts, and Cloud ASR as hard gates",
    )
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
        "--require-cloud-asr",
        action="store_true",
        help="Fail readiness unless Cloud ASR is enabled and configured",
    )
    parser.add_argument(
        "--health-url",
        default=DEFAULT_HEALTH_URL,
        help=f"Runtime health base URL (default: {DEFAULT_HEALTH_URL})",
    )
    parser.add_argument(
        "--change-event-file",
        default=DEFAULT_CHANGE_EVENT_FILE,
        help=f"Change-event JSONL path (default: {DEFAULT_CHANGE_EVENT_FILE})",
    )
    parser.add_argument(
        "--journal-since",
        default=DEFAULT_JOURNAL_SINCE,
        help=f"journalctl --since value (default: {DEFAULT_JOURNAL_SINCE})",
    )
    parser.add_argument("--skip-health", action="store_true", help="Skip runtime health endpoints")
    parser.add_argument("--skip-service-log", action="store_true", help="Skip systemd log collection")
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        help=f"Per-command timeout seconds (default: {DEFAULT_COMMAND_TIMEOUT_SECONDS:g})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = collect_s100p_readiness_bundle(
        args.output_dir or None,
        field=args.field,
        include_room_loop=args.with_room_loop,
        room_loop_trials=args.room_loop_trials,
        room_loop_text=args.room_loop_text,
        room_loop_expect_prefix=args.room_loop_expect_prefix,
        live_tts_room_loop=args.live_tts_room_loop,
        require_cloud_asr=args.require_cloud_asr,
        guard_min_seconds=args.guard_min_seconds,
        health_url=args.health_url,
        change_event_file=args.change_event_file,
        journal_since=args.journal_since,
        skip_health=args.skip_health,
        skip_service_log=args.skip_service_log,
        command_timeout=args.command_timeout,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))  # noqa: T201
    else:
        print_s100p_readiness_bundle_summary(payload)
    return 0 if payload.get("status") == "ok" else 1


def _resolve_bundle_dir(
    output_dir: str | Path | None,
    *,
    created_at: datetime,
    hostname: str,
) -> Path:
    if output_dir:
        return Path(output_dir)
    label = f"{created_at.astimezone(UTC).strftime('%Y%m%d-%H%M%S')}-{hostname}"
    return Path("artifacts") / "s100p" / label


def _readiness_command(
    *,
    json_out: Path,
    guard_min_seconds: float,
    include_room_loop: bool = False,
    room_loop_trials: int = 3,
    room_loop_text: str = DEFAULT_ROOM_LOOP_TEXT,
    room_loop_expect_prefix: str = DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
    live_tts_room_loop: bool = False,
    require_cloud_asr: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "askme",
        "runtime",
        "sunrise-voice-readiness",
        "--json",
        "--json-out",
        str(json_out),
        "--guard-min-seconds",
        str(guard_min_seconds),
    ]
    if include_room_loop:
        command.extend(
            [
                "--with-room-loop",
                "--room-loop-trials",
                str(max(1, int(room_loop_trials))),
                "--room-loop-text",
                room_loop_text,
                "--room-loop-expect-prefix",
                room_loop_expect_prefix,
            ]
        )
        if live_tts_room_loop:
            command.append("--live-tts-room-loop")
    if require_cloud_asr:
        command.append("--require-cloud-asr")
    return command


def _append_command_step(
    steps: list[dict[str, Any]],
    *,
    name: str,
    command: list[str],
    artifact: Path,
    required: bool,
    runner: CommandRunner,
    timeout: float,
    warnings: list[str],
    errors: list[str],
) -> None:
    result = runner(command, timeout)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    content = result.stdout if result.stdout else result.stderr
    artifact.write_text(content, encoding="utf-8")
    ok = result.returncode == 0
    step = {
        "name": name,
        "ok": ok,
        "required": required,
        "requirement": _requirement_label(required),
        "artifact": str(artifact.resolve()),
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }
    _record_step_status(step, warnings=warnings, errors=errors)
    steps.append(step)


def _append_http_step(
    steps: list[dict[str, Any]],
    *,
    name: str,
    url: str,
    artifact: Path,
    required: bool,
    validator: Callable[[str], tuple[bool, str]] | None = None,
    getter: HttpGetter,
    timeout: float,
    warnings: list[str],
    errors: list[str],
) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    try:
        status_code, body = getter(url, timeout)
    except Exception as exc:  # pragma: no cover - exercised through injected fakes.
        error_artifact = artifact.with_suffix(f"{artifact.suffix}.error.txt")
        error_artifact.write_text(str(exc), encoding="utf-8")
        step = {
            "name": name,
            "ok": False,
            "required": required,
            "requirement": _requirement_label(required),
            "artifact": str(error_artifact.resolve()),
            "url": url,
            "error": str(exc),
        }
        _record_step_status(step, warnings=warnings, errors=errors)
        steps.append(step)
        return

    ok = 200 <= status_code < 300
    validation_error = ""
    if ok and validator is not None:
        ok, validation_error = validator(body)
    artifact.write_text(body, encoding="utf-8")
    step = {
        "name": name,
        "ok": ok,
        "required": required,
        "requirement": _requirement_label(required),
        "artifact": str(artifact.resolve()),
        "url": url,
        "status_code": status_code,
        "body_tail": _tail(body),
    }
    if validation_error:
        step["error"] = validation_error
    _record_step_status(step, warnings=warnings, errors=errors)
    steps.append(step)


def _append_event_file_step(
    steps: list[dict[str, Any]],
    *,
    source: Path,
    artifact: Path,
    required: bool,
    warnings: list[str],
    errors: list[str],
) -> None:
    if source.is_file():
        artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, artifact)
        steps.append(
            {
                "name": "change-events",
                "ok": True,
                "required": required,
                "requirement": _requirement_label(required),
                "artifact": str(artifact.resolve()),
                "source": str(source),
            }
        )
        return

    missing = artifact.with_suffix(".missing.txt")
    missing.write_text(f"missing change event file: {source}\n", encoding="utf-8")
    message = f"change-events missing {'required' if required else 'optional'} source: {source}"
    if required:
        errors.append(message)
    else:
        warnings.append(message)
    steps.append(
        {
            "name": "change-events",
            "ok": False,
            "required": required,
            "requirement": _requirement_label(required),
            "artifact": str(missing.resolve()),
            "source": str(source),
            "error": f"missing source: {source}",
        }
    )


def _append_room_loop_artifact_index(
    steps: list[dict[str, Any]],
    *,
    readiness_artifact: Path,
    artifact: Path,
    bundled_dir: Path,
    required: bool,
    warnings: list[str],
    errors: list[str],
) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = ""
    copied_files: list[Path] = []
    parse_error = ""
    validation_errors: list[str] = []
    try:
        payload = json.loads(readiness_artifact.read_text(encoding="utf-8"))
        room_loop = payload.get("checks", {}).get("room_loop", {})
        artifact_dir = str(room_loop.get("artifact_dir", "") or "")
        if artifact_dir:
            source = Path(artifact_dir)
            if source.is_dir():
                bundled_dir.mkdir(parents=True, exist_ok=True)
                for source_path in sorted(source.iterdir(), key=lambda path: path.name):
                    target_path = bundled_dir / source_path.name
                    if source_path.is_dir():
                        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                    elif source_path.is_file():
                        shutil.copy2(source_path, target_path)
                    else:
                        continue
                    copied_files.append(target_path)
            else:
                parse_error = f"room_loop artifact_dir does not exist: {artifact_dir}"
        else:
            parse_error = "room_loop artifact_dir missing from readiness JSON"
    except (OSError, json.JSONDecodeError) as exc:
        parse_error = f"unable to read room-loop artifact index: {exc}"

    if not parse_error and required:
        validation_errors = _validate_room_loop_bundled_artifacts(copied_files)

    lines = [
        "Room-loop artifact index",
        f"readiness_json: {readiness_artifact.resolve()}",
        f"source_artifact_dir: {artifact_dir or '<missing>'}",
        f"bundled_artifact_dir: {bundled_dir.resolve()}",
    ]
    if copied_files:
        lines.append("files:")
        lines.extend(f"- {path.resolve()}" for path in copied_files)
    if parse_error:
        lines.append(f"error: {parse_error}")
    for validation_error in validation_errors:
        lines.append(f"error: {validation_error}")
    artifact.write_text("\n".join(lines) + "\n", encoding="utf-8")

    step = {
        "name": "room-loop-artifacts",
        "ok": bool(artifact_dir) and not parse_error and not validation_errors,
        "required": required,
        "requirement": _requirement_label(required),
        "artifact": str(artifact.resolve()),
        "source": artifact_dir,
        "bundled_dir": str(bundled_dir.resolve()),
        "artifacts": [str(path.resolve()) for path in copied_files],
    }
    if parse_error or validation_errors:
        step["error"] = "; ".join([message for message in [parse_error, *validation_errors] if message])
    _record_step_status(step, warnings=warnings, errors=errors)
    steps.append(step)


def _validate_room_loop_bundled_artifacts(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    files = [path for path in paths if path.is_file()]
    if not files:
        return ["room_loop artifact_dir contains no files"]

    empty_files = [path.name for path in files if path.stat().st_size <= 0]
    if empty_files:
        errors.append(f"room_loop artifact files are empty: {', '.join(empty_files)}")

    json_files = [path for path in files if path.suffix.lower() == ".json" and path.stat().st_size > 0]
    wav_files = [path for path in files if path.suffix.lower() == ".wav" and path.stat().st_size > 0]
    if not json_files:
        errors.append("room_loop artifact_dir must include at least one non-empty JSON artifact")
    if not wav_files:
        errors.append("room_loop artifact_dir must include at least one non-empty WAV artifact")

    invalid_json = []
    for json_file in json_files:
        try:
            json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_json.append(json_file.name)
    if invalid_json:
        errors.append(f"room_loop JSON artifacts are invalid: {', '.join(invalid_json)}")
    return errors


def _append_otrev_closed_loop_step(
    steps: list[dict[str, Any]],
    *,
    event_artifact: Path,
    log_artifact: Path,
    artifact: Path,
    errors: list[str],
) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    correlated, reason = _detect_otrev_closed_loop(event_artifact, log_artifact)
    if correlated:
        artifact.write_text(f"Automated OTREV/proactive closed-loop correlation passed: {reason}\n", encoding="utf-8")
        steps.append(
            {
                "name": "otrev-proactive-closed-loop",
                "ok": True,
                "required": True,
                "requirement": "required",
                "artifact": str(artifact.resolve()),
                "event_artifact": str(event_artifact.resolve()),
                "log_artifact": str(log_artifact.resolve()),
            }
        )
        return

    message = (
        "OTREV/proactive closed-loop evidence requires manual field signoff unless "
        f"automated event-to-log correlation exists: {reason}"
    )
    artifact.write_text(
        "\n".join(
            [
                message,
                "",
                "Required manual evidence:",
                "- controlled event trigger",
                "- ProactiveAgent consumption in service log",
                "- auto-solve/tool-chain or operator-approved no-action decision",
                "- robot voice/action/status feedback matching the event",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    step = {
        "name": "otrev-proactive-closed-loop",
        "ok": False,
        "required": True,
        "requirement": "manual_required",
        "artifact": str(artifact.resolve()),
        "event_artifact": str(event_artifact.resolve()),
        "log_artifact": str(log_artifact.resolve()),
        "error": message,
    }
    errors.append("otrev-proactive-closed-loop failed")
    steps.append(step)


def _detect_otrev_closed_loop(event_artifact: Path, log_artifact: Path) -> tuple[bool, str]:
    if not event_artifact.is_file() or event_artifact.stat().st_size <= 0:
        return False, "change-events artifact is missing or empty"
    if not log_artifact.is_file() or log_artifact.stat().st_size <= 0:
        return False, "service log artifact is missing or empty"

    event_lines = [line.strip() for line in event_artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
    log_text = log_artifact.read_text(encoding="utf-8", errors="replace")
    if not event_lines:
        return False, "change-events artifact has no JSONL entries"

    parsed_events = []
    for line in event_lines:
        try:
            parsed_events.append(json.loads(line))
        except json.JSONDecodeError:
            return False, "change-events artifact contains invalid JSONL"

    log_lower = log_text.lower()
    consumed = "[proactive] change event" in log_lower or "change event:" in log_lower
    closed = (
        "auto-solving change event" in log_lower
        or "auto-solve from change event" in log_lower
        or "solve_problem" in log_lower
        or "closed-loop" in log_lower
    )
    if not consumed:
        return False, "service log does not show ProactiveAgent consuming the change event"
    if not closed:
        return False, "service log does not show auto-solve/tool-chain or explicit closed-loop result"

    event_hints = _event_correlation_hints(parsed_events)
    if event_hints and any(hint.lower() in log_lower for hint in event_hints):
        return True, "service log includes ProactiveAgent consumption, closed-loop handling, and event text"
    if not event_hints:
        return True, "service log includes ProactiveAgent consumption and closed-loop handling"
    return False, "service log lacks text that correlates to the captured event"


def _event_correlation_hints(events: list[Any]) -> list[str]:
    hints: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        for key in ("id", "type", "kind", "label", "description", "description_zh"):
            value = event.get(key)
            if isinstance(value, str) and len(value.strip()) >= 3:
                hints.append(value.strip())
        for key in ("object", "person", "source"):
            nested = event.get(key)
            if isinstance(nested, dict):
                for value in nested.values():
                    if isinstance(value, str) and len(value.strip()) >= 3:
                        hints.append(value.strip())
    return hints


def _append_manual_required_step(
    steps: list[dict[str, Any]],
    *,
    name: str,
    artifact: Path,
    message: str,
    errors: list[str],
) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(f"{message}\n", encoding="utf-8")
    step = {
        "name": name,
        "ok": False,
        "required": True,
        "requirement": "manual_required",
        "artifact": str(artifact.resolve()),
        "error": message,
    }
    errors.append(f"{name} failed")
    steps.append(step)


def _record_step_status(
    step: dict[str, Any],
    *,
    warnings: list[str],
    errors: list[str],
) -> None:
    if step.get("ok"):
        return
    message = f"{step.get('name', 'unknown')} failed"
    if step.get("required"):
        errors.append(message)
    else:
        warnings.append(message)


def _requirement_label(required: bool) -> str:
    return "required" if required else "optional"


def _validate_json_health_ok(body: str) -> tuple[bool, str]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON health body: {exc.msg}"
    if not isinstance(payload, dict):
        return False, "health body must be a JSON object"
    if payload.get("status") != "ok":
        return False, "health status must be ok"
    return True, ""


def _validate_prometheus_health_metric(body: str) -> tuple[bool, str]:
    metric_values = {
        "askme_up": _prometheus_metric_values(body, "askme_up"),
        "askme_health_status": _prometheus_metric_values(body, "askme_health_status"),
    }
    if not metric_values["askme_up"] and not metric_values["askme_health_status"]:
        return False, "missing askme_up or askme_health_status metric"
    for metric_name, values in metric_values.items():
        for value in values:
            if value != 1.0:
                return False, f"{metric_name} unhealthy value: {value:g}"
    return True, ""


def _prometheus_metric_values(body: str, metric_name: str) -> list[float]:
    values: list[float] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == metric_name or line.startswith(f"{metric_name} ") or line.startswith(f"{metric_name}{{"):
            parts = line.split()
            if len(parts) < 2:
                values.append(0.0)
                continue
            try:
                values.append(float(parts[1]))
            except ValueError:
                values.append(0.0)
    return values


def _default_command_runner(command: list[str], timeout: float) -> CommandResult:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return CommandResult(returncode=127, stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            returncode=124,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or "timeout"),
        )
    return CommandResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _default_http_getter(url: str, timeout: float) -> tuple[int, str]:
    with request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return response.status, body


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_notes_template(path: Path, *, field: bool) -> None:
    if path.exists():
        return
    mode = "field" if field else "standard"
    path.write_text(
        "\n".join(
            [
                "# S100P Field Notes",
                "",
                f"- Mode: {mode}",
                "- Robot ID:",
                "- Location:",
                "- Operator:",
                "- Reviewer:",
                "- Evidence dir:",
                "- Start/end time:",
                "",
                "## OTREV",
                "- Trigger:",
                "- Expected behavior:",
                "- Actual behavior:",
                "- Log timestamp:",
                "- Result:",
                "",
                "## Concurrency",
                "- Voice turn:",
                "- Concurrent low-risk request:",
                f"- Repetitions/timeout: 3 requests, {DEFAULT_CONCURRENCY_REQUEST_TIMEOUT_SECONDS:g}s each unless site SLA differs",
                "- Busy/resource observation:",
                "- Result:",
                "",
                "## Cloud ASR",
                "- Network condition:",
                "- Noise condition:",
                "- Latency/quality:",
                "- Fallback behavior:",
                "- Privacy/key leak check:",
                "",
                "## Conclusion",
                "- PASS/BLOCKED/FAIL:",
                "- Owner and next step:",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _tail(text: str, *, lines: int = 16) -> str:
    return "\n".join(str(text).splitlines()[-lines:])


if __name__ == "__main__":
    raise SystemExit(main())
