"""Tests for S100P readiness evidence bundle collection."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from askme import cli
from askme.voice import s100p_readiness_bundle as bundle


def test_bundle_collects_manifest_and_core_artifacts(tmp_path: Path) -> None:
    event_file = tmp_path / "events.jsonl"
    event_file.write_text('{"kind":"change"}\n', encoding="utf-8")
    commands: list[list[str]] = []
    urls: list[str] = []

    def fake_runner(command: list[str], _timeout: float) -> bundle.CommandResult:
        commands.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return bundle.CommandResult(0, stdout="abc123\n")
        return bundle.CommandResult(0, stdout='{"status":"ok"}\n')

    def fake_http_getter(url: str, _timeout: float) -> tuple[int, str]:
        urls.append(url)
        body = "askme_up 1\n" if url.endswith("/metrics/prometheus") else '{"status":"ok"}\n'
        return 200, body

    payload = bundle.collect_s100p_readiness_bundle(
        tmp_path / "bundle",
        include_room_loop=True,
        room_loop_trials=3,
        require_cloud_asr=True,
        change_event_file=event_file,
        command_runner=fake_runner,
        http_getter=fake_http_getter,
        now=lambda: datetime(2026, 5, 5, 1, 2, 3, tzinfo=UTC),
        hostname="s100p-test",
    )

    bundle_dir = Path(payload["bundle_dir"])
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    flattened = [part for command in commands for part in command]

    assert payload["status"] == "ok"
    assert manifest["status"] == "ok"
    assert manifest["summary"]["field"] is False
    assert (bundle_dir / "voice-health.json").is_file()
    assert (bundle_dir / "sunrise-readiness.json").is_file()
    assert (bundle_dir / "room-loop-readiness.json").is_file()
    assert (bundle_dir / "cloud-asr-readiness.json").is_file()
    assert (bundle_dir / "health.json").is_file()
    assert (bundle_dir / "healthz.json").is_file()
    assert (bundle_dir / "prometheus.txt").is_file()
    assert (bundle_dir / "askme.service.log").is_file()
    assert (bundle_dir / "askme.service.cat.txt").is_file()
    assert (bundle_dir / "notes.md").is_file()
    assert manifest["notes"].endswith("notes.md")
    assert (bundle_dir / "change-events.jsonl").read_text(encoding="utf-8") == '{"kind":"change"}\n'
    assert _removed_room_loop_flag() not in flattened
    assert "--with-room-loop" in flattened
    assert "--require-cloud-asr" in flattened
    assert urls == [
        "http://127.0.0.1:8765/health",
        "http://127.0.0.1:8765/healthz",
        "http://127.0.0.1:8765/metrics/prometheus",
    ]
    assert all(step["requirement"] in {"required", "optional"} for step in payload["steps"])


def test_default_bundle_keeps_service_and_change_events_optional(tmp_path: Path) -> None:
    def fake_runner(_command: list[str], _timeout: float) -> bundle.CommandResult:
        return bundle.CommandResult(0, stdout='{"status":"ok"}\n')

    def fake_http_getter(url: str, _timeout: float) -> tuple[int, str]:
        if url.endswith("/metrics/prometheus"):
            return 200, "askme_health_status 1\n"
        return 200, '{"status":"ok"}\n'

    payload = bundle.collect_s100p_readiness_bundle(
        tmp_path / "bundle",
        change_event_file=tmp_path / "missing-events.jsonl",
        command_runner=fake_runner,
        http_getter=fake_http_getter,
    )

    service_log = next(step for step in payload["steps"] if step["name"] == "service-log")
    change_events = next(step for step in payload["steps"] if step["name"] == "change-events")

    assert payload["status"] == "ok"
    assert service_log["required"] is False
    assert service_log["requirement"] == "optional"
    assert change_events["required"] is False
    assert change_events["requirement"] == "optional"


def test_bundle_marks_degraded_when_required_step_fails(tmp_path: Path) -> None:
    def fake_runner(command: list[str], _timeout: float) -> bundle.CommandResult:
        if "sunrise-voice-readiness" in command:
            return bundle.CommandResult(1, stdout='{"status":"degraded"}\n', stderr="audio missing")
        return bundle.CommandResult(0, stdout='{"status":"ok"}\n')

    def fake_http_getter(url: str, _timeout: float) -> tuple[int, str]:
        if url.endswith("/healthz"):
            raise RuntimeError("healthz down")
        return 200, '{"status":"ok"}\n'

    payload = bundle.collect_s100p_readiness_bundle(
        tmp_path / "bundle",
        change_event_file=tmp_path / "missing-events.jsonl",
        command_runner=fake_runner,
        http_getter=fake_http_getter,
    )

    bundle_dir = Path(payload["bundle_dir"])
    healthz_step = next(step for step in payload["steps"] if step["name"] == "healthz")

    assert payload["status"] == "degraded"
    assert "sunrise-readiness failed" in payload["errors"]
    assert "healthz failed" in payload["errors"]
    assert "change-events missing optional source" in "\n".join(payload["warnings"])
    assert (bundle_dir / "sunrise-readiness.json").is_file()
    assert healthz_step["artifact"].endswith("healthz.json.error.txt")
    assert Path(healthz_step["artifact"]).read_text(encoding="utf-8") == "healthz down"


def test_field_mode_requires_room_loop_cloud_service_and_change_events(tmp_path: Path) -> None:
    event_file = tmp_path / "events.jsonl"
    event_file.write_text('{"kind":"change"}\n', encoding="utf-8")
    room_loop_dir = tmp_path / "room-loop"
    room_loop_dir.mkdir()
    (room_loop_dir / "room_loop.json").write_text(
        '{"summary":{"passed":true},"description":"door event"}\n',
        encoding="utf-8",
    )
    (room_loop_dir / "room_loop.wav").write_bytes(b"RIFFfakeWAVE")
    commands: list[list[str]] = []

    def fake_runner(command: list[str], _timeout: float) -> bundle.CommandResult:
        commands.append(command)
        if command[:2] == ["journalctl", "-u"]:
            return bundle.CommandResult(
                0,
                stdout=(
                    "[Proactive] Change event: door event (importance=0.90)\n"
                    "[Proactive] Auto-solving change event: door event\n"
                ),
            )
        if "--with-room-loop" in command:
            return bundle.CommandResult(
                0,
                stdout=json.dumps(
                    {
                        "status": "ok",
                        "checks": {"room_loop": {"artifact_dir": str(room_loop_dir)}},
                    }
                ),
            )
        return bundle.CommandResult(0, stdout='{"status":"ok"}\n')

    def fake_http_getter(url: str, _timeout: float) -> tuple[int, str]:
        if url.endswith("/metrics/prometheus"):
            return 200, "askme_up 1\n"
        return 200, '{"status":"ok"}\n'

    payload = bundle.collect_s100p_readiness_bundle(
        tmp_path / "bundle",
        field=True,
        change_event_file=event_file,
        command_runner=fake_runner,
        http_getter=fake_http_getter,
    )

    flattened = [part for command in commands for part in command]
    by_name = {step["name"]: step for step in payload["steps"]}

    assert payload["status"] == "ok"
    assert payload["summary"]["field"] is True
    assert "--with-room-loop" in flattened
    assert "--require-cloud-asr" in flattened
    assert by_name["room-loop-readiness"]["required"] is True
    assert by_name["room-loop-artifacts"]["required"] is True
    assert by_name["room-loop-artifacts"]["ok"] is True
    assert by_name["room-loop-artifacts"]["bundled_dir"].endswith("room-loop")
    assert any(path.endswith("room_loop.json") for path in by_name["room-loop-artifacts"]["artifacts"])
    assert any(path.endswith("room_loop.wav") for path in by_name["room-loop-artifacts"]["artifacts"])
    assert by_name["cloud-asr-readiness"]["required"] is True
    assert by_name["systemctl-cat"]["required"] is True
    assert by_name["service-log"]["required"] is True
    assert by_name["change-events"]["required"] is True
    assert by_name["otrev-proactive-closed-loop"]["ok"] is True
    assert (Path(payload["bundle_dir"]) / "notes.md").is_file()


def test_field_mode_records_manual_required_when_evidence_is_skipped(tmp_path: Path) -> None:
    def fake_runner(_command: list[str], _timeout: float) -> bundle.CommandResult:
        return bundle.CommandResult(0, stdout='{"status":"ok"}\n')

    payload = bundle.collect_s100p_readiness_bundle(
        tmp_path / "bundle",
        field=True,
        skip_health=True,
        skip_service_log=True,
        command_runner=fake_runner,
        http_getter=lambda _url, _timeout: (200, '{"status":"ok"}\n'),
    )

    manual_required = [step for step in payload["steps"] if step["requirement"] == "manual_required"]

    assert payload["status"] == "degraded"
    assert payload["summary"]["manual_required_step_count"] == 6
    assert {step["name"] for step in manual_required} == {
        "health",
        "healthz",
        "prometheus",
        "systemctl-cat",
        "service-log",
        "otrev-proactive-closed-loop",
    }


def test_health_step_degrades_on_non_ok_json_body(tmp_path: Path) -> None:
    def fake_runner(_command: list[str], _timeout: float) -> bundle.CommandResult:
        return bundle.CommandResult(0, stdout='{"status":"ok"}\n')

    def fake_http_getter(url: str, _timeout: float) -> tuple[int, str]:
        if url.endswith("/health"):
            return 200, '{"status":"degraded"}\n'
        if url.endswith("/metrics/prometheus"):
            return 200, "askme_up 1\n"
        return 200, '{"status":"ok"}\n'

    payload = bundle.collect_s100p_readiness_bundle(
        tmp_path / "bundle",
        skip_service_log=True,
        command_runner=fake_runner,
        http_getter=fake_http_getter,
    )
    health = next(step for step in payload["steps"] if step["name"] == "health")

    assert payload["status"] == "degraded"
    assert health["ok"] is False
    assert health["error"] == "health status must be ok"


def test_prometheus_step_degrades_when_core_health_metric_is_missing(tmp_path: Path) -> None:
    def fake_runner(_command: list[str], _timeout: float) -> bundle.CommandResult:
        return bundle.CommandResult(0, stdout='{"status":"ok"}\n')

    def fake_http_getter(url: str, _timeout: float) -> tuple[int, str]:
        if url.endswith("/metrics/prometheus"):
            return 200, "askme_requests_total 1\n"
        return 200, '{"status":"ok"}\n'

    payload = bundle.collect_s100p_readiness_bundle(
        tmp_path / "bundle",
        skip_service_log=True,
        command_runner=fake_runner,
        http_getter=fake_http_getter,
    )
    prometheus = next(step for step in payload["steps"] if step["name"] == "prometheus")

    assert payload["status"] == "degraded"
    assert prometheus["ok"] is False
    assert prometheus["error"] == "missing askme_up or askme_health_status metric"


@pytest.mark.parametrize(
    ("body", "expected_error"),
    [
        ("askme_up 0\n", "askme_up unhealthy value: 0"),
        ("askme_health_status 0\n", "askme_health_status unhealthy value: 0"),
        ("askme_up 1\naskme_health_status 0\n", "askme_health_status unhealthy value: 0"),
        ("askme_up 0\naskme_health_status 1\n", "askme_up unhealthy value: 0"),
    ],
)
def test_prometheus_step_degrades_on_unhealthy_health_metric_values(
    tmp_path: Path,
    body: str,
    expected_error: str,
) -> None:
    def fake_runner(_command: list[str], _timeout: float) -> bundle.CommandResult:
        return bundle.CommandResult(0, stdout='{"status":"ok"}\n')

    def fake_http_getter(url: str, _timeout: float) -> tuple[int, str]:
        if url.endswith("/metrics/prometheus"):
            return 200, body
        return 200, '{"status":"ok"}\n'

    payload = bundle.collect_s100p_readiness_bundle(
        tmp_path / "bundle",
        skip_service_log=True,
        command_runner=fake_runner,
        http_getter=fake_http_getter,
    )
    prometheus = next(step for step in payload["steps"] if step["name"] == "prometheus")

    assert payload["status"] == "degraded"
    assert prometheus["ok"] is False
    assert prometheus["error"] == expected_error


def test_field_room_loop_artifacts_fail_without_required_json_and_wav(tmp_path: Path) -> None:
    event_file = tmp_path / "events.jsonl"
    event_file.write_text('{"description":"door event"}\n', encoding="utf-8")
    room_loop_dir = tmp_path / "room-loop"
    room_loop_dir.mkdir()
    (room_loop_dir / "room_loop.json").write_text('{"summary":{"passed":true}}\n', encoding="utf-8")

    def fake_runner(command: list[str], _timeout: float) -> bundle.CommandResult:
        if command[:2] == ["journalctl", "-u"]:
            return bundle.CommandResult(
                0,
                stdout=(
                    "[Proactive] Change event: door event (importance=0.90)\n"
                    "[Proactive] Auto-solving change event: door event\n"
                ),
            )
        if "--with-room-loop" in command:
            return bundle.CommandResult(
                0,
                stdout=json.dumps(
                    {
                        "status": "ok",
                        "checks": {"room_loop": {"artifact_dir": str(room_loop_dir)}},
                    }
                ),
            )
        return bundle.CommandResult(0, stdout='{"status":"ok"}\n')

    def fake_http_getter(url: str, _timeout: float) -> tuple[int, str]:
        if url.endswith("/metrics/prometheus"):
            return 200, "askme_up 1\n"
        return 200, '{"status":"ok"}\n'

    payload = bundle.collect_s100p_readiness_bundle(
        tmp_path / "bundle",
        field=True,
        change_event_file=event_file,
        command_runner=fake_runner,
        http_getter=fake_http_getter,
    )
    room_loop = next(step for step in payload["steps"] if step["name"] == "room-loop-artifacts")

    assert payload["status"] == "degraded"
    assert room_loop["ok"] is False
    assert "non-empty WAV artifact" in room_loop["error"]
    assert (Path(payload["bundle_dir"]) / "room-loop" / "room_loop.json").is_file()


def test_field_mode_requires_manual_otrev_when_closed_loop_correlation_is_absent(tmp_path: Path) -> None:
    event_file = tmp_path / "events.jsonl"
    event_file.write_text('{"description":"door event"}\n', encoding="utf-8")
    room_loop_dir = tmp_path / "room-loop"
    room_loop_dir.mkdir()
    (room_loop_dir / "room_loop.json").write_text('{"summary":{"passed":true}}\n', encoding="utf-8")
    (room_loop_dir / "room_loop.wav").write_bytes(b"RIFFfakeWAVE")

    def fake_runner(command: list[str], _timeout: float) -> bundle.CommandResult:
        if command[:2] == ["journalctl", "-u"]:
            return bundle.CommandResult(0, stdout="service started\n")
        if "--with-room-loop" in command:
            return bundle.CommandResult(
                0,
                stdout=json.dumps(
                    {
                        "status": "ok",
                        "checks": {"room_loop": {"artifact_dir": str(room_loop_dir)}},
                    }
                ),
            )
        return bundle.CommandResult(0, stdout='{"status":"ok"}\n')

    def fake_http_getter(url: str, _timeout: float) -> tuple[int, str]:
        if url.endswith("/metrics/prometheus"):
            return 200, "askme_health_status 1\n"
        return 200, '{"status":"ok"}\n'

    payload = bundle.collect_s100p_readiness_bundle(
        tmp_path / "bundle",
        field=True,
        change_event_file=event_file,
        command_runner=fake_runner,
        http_getter=fake_http_getter,
    )
    otrev = next(step for step in payload["steps"] if step["name"] == "otrev-proactive-closed-loop")

    assert payload["status"] == "degraded"
    assert otrev["requirement"] == "manual_required"
    assert "manual field signoff" in otrev["error"]


def test_cli_runtime_s100p_readiness_bundle_json(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_bundle(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "ok", "target": "s100p-readiness-bundle", "steps": []}

    monkeypatch.setattr(cli, "_run_s100p_readiness_bundle", fake_bundle)

    cli.main(
        [
            "runtime",
            "s100p-readiness-bundle",
            "--json",
            "--output-dir",
            "artifacts/s100p/manual",
            "--field",
            "--with-room-loop",
            "--room-loop-trials",
            "2",
            "--room-loop-text",
            "ping",
            "--room-loop-expect-prefix",
            "pi",
            "--require-cloud-asr",
            "--skip-service-log",
            "--command-timeout",
            "12",
        ]
    )

    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "ok"
    assert seen["output_dir"] == "artifacts/s100p/manual"
    assert seen["field"] is True
    assert seen["include_room_loop"] is True
    assert seen["room_loop_trials"] == 2
    assert seen["room_loop_text"] == "ping"
    assert seen["room_loop_expect_prefix"] == "pi"
    assert seen["require_cloud_asr"] is True
    assert seen["skip_service_log"] is True
    assert seen["command_timeout"] == 12.0


def test_cli_runtime_s100p_readiness_bundle_exits_nonzero_when_degraded(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_run_s100p_readiness_bundle",
        lambda **_kwargs: {"status": "degraded", "steps": [], "errors": [], "warnings": []},
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "s100p-readiness-bundle", "--skip-health", "--skip-service-log"])

    assert exc.value.code == 1


def test_readiness_command_uses_current_python_and_known_flags(tmp_path: Path) -> None:
    command = bundle._readiness_command(  # noqa: SLF001
        json_out=tmp_path / "readiness.json",
        guard_min_seconds=1.25,
        include_room_loop=True,
        room_loop_trials=2,
        require_cloud_asr=True,
    )

    assert command[:5] == [sys.executable, "-m", "askme", "runtime", "sunrise-voice-readiness"]
    assert "--json-out" in command
    assert "--with-room-loop" in command
    assert _removed_room_loop_flag() not in command


def _removed_room_loop_flag() -> str:
    return "--room-loop-" + "required-" + "passes"
