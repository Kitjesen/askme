"""Tests for the structured askme CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_cli_runtime_without_subcommand_prints_help(capsys) -> None:
    cli.main(["runtime"])

    output = capsys.readouterr().out
    assert "usage: askme runtime" in output
    assert "blueprints" in output
    assert "s100p-readiness-bundle" in output


def test_cli_runtime_blueprints_outputs_catalog(capsys) -> None:
    cli.main(["runtime", "blueprints", "--name", "park"])

    output = capsys.readouterr().out
    assert "blueprints=1" in output
    assert "edge_robot: 园区巡检机器人运行时" in output
    assert "valid=True" in output


def test_cli_runtime_blueprints_json(capsys) -> None:
    cli.main(["runtime", "blueprints", "--customer-visible", "--json"])

    output = capsys.readouterr().out
    assert "园区巡检机器人运行时" in output
    payload = json.loads(output)
    names = {item["name"] for item in payload["items"]}
    assert "edge_robot" in names
    assert "mcp" not in names
    assert payload["summary"]["valid_count"] == payload["summary"]["blueprint_count"]


def test_cli_json_falls_back_to_ascii_when_stdout_cannot_encode(monkeypatch) -> None:
    class AsciiStdout:
        encoding = "ascii"

    monkeypatch.setattr(cli.sys, "stdout", AsciiStdout())

    output = cli._json({"title": "园区巡检机器人运行时"})

    assert "\\u56ed\\u533a" in output
    assert "园区巡检机器人运行时" not in output


def test_cli_json_preserves_chinese_for_local_gbk_terminal(monkeypatch) -> None:
    class GbkTerminal:
        encoding = "gbk"

        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(cli.sys, "stdout", GbkTerminal())

    output = cli._json({"title": "园区巡检机器人运行时"})

    assert "园区巡检机器人运行时" in output
    assert "\\u56ed\\u533a" not in output


def test_cli_json_escapes_chinese_for_non_utf8_pipe(monkeypatch) -> None:
    class GbkPipe:
        encoding = "gbk"

        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(cli.sys, "stdout", GbkPipe())

    output = cli._json({"title": "园区巡检机器人运行时"})

    assert "\\u56ed\\u533a" in output
    assert "园区巡检机器人运行时" not in output


def test_cli_runtime_blueprints_delivery_package_writes_json(tmp_path: Path, capsys) -> None:
    output = tmp_path / "blueprint-package.json"

    cli.main([
        "runtime",
        "blueprints",
        "--name",
        "park",
        "--delivery-package",
        "--output",
        str(output),
    ])

    console = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert "delivery-package: blueprint.edge_robot" in console
    assert payload["package_id"] == "blueprint.edge_robot"
    assert payload["deliverables"]["scenario_acceptance"]
    assert payload["operator_runbook"]["start"] == "python -m askme.blueprints.presets.edge_robot"


def test_cli_runtime_blueprints_delivery_package_json_is_direct_package(capsys) -> None:
    cli.main([
        "runtime",
        "blueprints",
        "--name",
        "park",
        "--delivery-package",
        "--json",
    ])

    payload = json.loads(capsys.readouterr().out)

    assert payload["package_id"] == "blueprint.edge_robot"
    assert payload["blueprint"] == "edge_robot"
    assert "items" not in payload
    assert payload["operator_runbook"]["start"] == "python -m askme.blueprints.presets.edge_robot"


def test_cli_runtime_s100p_readiness_bundle_help_lists_field_flags(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "s100p-readiness-bundle", "--help"])

    output = capsys.readouterr().out
    assert exc.value.code == 0
    assert "--health-url" in output
    assert "--change-event-file" in output
    assert "--journal-since" in output
    assert "--skip-health" in output
    assert "--live-tts-room-loop" in output


def test_cli_runtime_field_eval_writes_report(monkeypatch, tmp_path: Path) -> None:
    out_path = tmp_path / "field-eval.json"

    monkeypatch.setattr(
        cli,
        "_run_field_operations_eval",
        lambda *, output: {
            "status": "passed",
            "scenario_count": 2,
            "passed": 2,
            "failed": 0,
            "report_path": output,
        },
    )

    cli.main(["runtime", "field-eval", "--output", str(out_path)])


def test_cli_runtime_field_eval_prints_product_demo(monkeypatch, tmp_path: Path, capsys) -> None:
    out_path = tmp_path / "field-eval.json"

    monkeypatch.setattr(
        cli,
        "_run_field_operations_eval",
        lambda *, output: {
            "status": "passed",
            "scenario_count": 1,
            "passed": 1,
            "failed": 0,
            "report_path": output,
            "product_demo": {
                "suite_name": "园区机器狗场景演示包",
                "demo_ready": True,
                "real_integration_ready": False,
                "customer_scenario_count": 1,
                "passed": 1,
                "customer_scenarios": [
                    {
                        "customer_name": "车辆违停检测",
                        "passed": True,
                        "expected_robot_action": "拍照记录违停位置，播报提醒，通知保安处理。",
                        "actual": {
                            "notification_group": "security",
                            "delivery_status": "sent",
                        },
                        "evidence": {"event_id": "field-demo"},
                    }
                ],
                "blocked_on_real_integrations": ["真实摄像头/VMS 事件流"],
            },
        },
    )

    cli.main(["runtime", "field-eval", "--output", str(out_path)])

    output = capsys.readouterr().out
    assert "product-demo: 园区机器狗场景演示包 ready=True" in output
    assert "车辆违停检测" in output
    assert "real-integration-gaps:" in output
    assert "真实摄像头/VMS 事件流" in output


def test_cli_runtime_field_ingest_file_dry_run_normalizes_camera_frame(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps({
            "source": "camera",
            "timestamp": 1770000000,
            "detections": [{"class_id": "2", "confidence": 0.91}],
            "zone": {"id": "main-road-1", "type": "main_channel", "parking_allowed": False},
        })
        + "\n",
        encoding="utf-8",
    )

    payload = cli._run_field_ingest_file(
        source=str(path),
        server="http://runtime.local:8765",
        dry_run=True,
        limit=0,
        device_secrets=None,
    )

    assert payload["status"] == "ok"
    assert payload["signed"] == 0
    assert payload["results"][0]["status"] == "dry_run"
    normalized = payload["results"][0]["normalized"]
    assert normalized["detections"][0]["label"] == "vehicle"
    assert normalized["zone_id"] == "main-road-1"


def test_cli_runtime_field_ingest_file_accepts_utf8_bom_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\ufeff" + json.dumps({"source": "sensor", "temperature_c": 72}),
        encoding="utf-8",
    )

    payload = cli._run_field_ingest_file(
        source=str(path),
        server="http://runtime.local:8765",
        dry_run=True,
        limit=0,
        device_secrets=None,
    )

    assert payload["status"] == "ok"
    assert payload["results"][0]["normalized"]["source"] == "sensor"


def test_cli_runtime_field_ingest_file_can_sign_normalized_events(tmp_path: Path) -> None:
    from askme.pipeline.field_operations import sign_field_device_payload

    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps({
            "source": "sensor",
            "device_id": "smoke-01",
            "timestamp": 1770000000,
            "sensor": {"temperature_c": 72, "smoke_level": 0.9},
            "location": "Power Room",
        })
        + "\n",
        encoding="utf-8",
    )

    payload = cli._run_field_ingest_file(
        source=str(path),
        server="http://runtime.local:8765",
        dry_run=True,
        limit=0,
        device_secrets={"smoke-01": "device-secret"},
    )

    normalized = payload["results"][0]["normalized"]
    assert payload["status"] == "ok"
    assert payload["signed"] == 1
    assert payload["results"][0]["device_signing"]["signed"] is True
    assert normalized["device_signature_alg"] == "hmac-sha256"
    assert normalized["device_signature"]
    assert normalized["device_signature"] == sign_field_device_payload(
        normalized,
        secret="device-secret",
    )


def test_cli_runtime_field_ingest_file_forwards_device_secret_args(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    seen: dict[str, object] = {}
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"source": "sensor", "device_id": "smoke-01"}) + "\n", encoding="utf-8")

    def fake_ingest(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "status": "ok",
            "target": "field-ingest-file",
            "count": 1,
            "failed": 0,
            "signed": 1,
            "dry_run": True,
            "results": [{"index": 1, "status": "dry_run", "normalized": {"scenario_id": "fire_or_smoke"}}],
        }

    monkeypatch.setattr(cli, "_run_field_ingest_file", fake_ingest)

    cli.main([
        "runtime",
        "field-ingest-file",
        str(path),
        "--dry-run",
        "--device-secret",
        "smoke-01=device-secret",
    ])

    output = capsys.readouterr().out
    assert seen["device_secrets"] == {"smoke-01": "device-secret"}
    assert "signed=1" in output


def test_cli_runtime_field_ingest_file_loads_device_secrets_from_site_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ASKME_SMOKE_SECRET", "smoke-secret")
    profile = tmp_path / "site.yaml"
    profile.write_text(
        """
devices:
  smoke-01:
    source: sensor
    secret_env: ASKME_SMOKE_SECRET
  camera-01:
    source: camera
    secret_env: ASKME_MISSING_CAMERA_SECRET
""".strip(),
        encoding="utf-8",
    )
    source = tmp_path / "events.jsonl"
    source.write_text(
        json.dumps({"source": "sensor", "device_id": "smoke-01", "sensor": {"smoke_level": 0.9}})
        + "\n",
        encoding="utf-8",
    )

    payload = cli._run_field_ingest_file(
        source=str(source),
        server="http://runtime.local:8765",
        dry_run=True,
        limit=0,
        device_secrets=cli._resolve_field_device_secrets([], site_profile=str(profile)),
    )

    assert payload["signed"] == 1
    assert payload["results"][0]["device_signing"]["device_id"] == "smoke-01"
    assert cli._resolve_field_device_secrets([], site_profile=str(profile)) == {
        "smoke-01": "smoke-secret"
    }


def test_cli_runtime_field_ingest_bridge_loads_site_profile_secrets_and_allows_override(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setenv("ASKME_SMOKE_SECRET", "smoke-secret")
    seen: dict[str, object] = {}
    profile = tmp_path / "site.yaml"
    profile.write_text(
        """
devices:
  smoke-01:
    source: sensor
    secret_env: ASKME_SMOKE_SECRET
""".strip(),
        encoding="utf-8",
    )
    source = tmp_path / "events.jsonl"
    source.write_text(json.dumps({"source": "sensor", "device_id": "smoke-01"}) + "\n", encoding="utf-8")

    def fake_bridge(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "status": "ok",
            "target": "field-ingest-bridge",
            "count": 1,
            "failed": 0,
            "dry_run": True,
            "state_path": str(tmp_path / "state.json"),
            "summary": {"signed": 1},
            "results": [],
        }

    monkeypatch.setattr(cli, "_run_field_ingest_bridge", fake_bridge)

    cli.main([
        "runtime",
        "field-ingest-bridge",
        str(source),
        "--site-profile",
        str(profile),
        "--device-secret",
        "smoke-01=override-secret",
        "--dry-run",
    ])

    _ = capsys.readouterr()
    assert seen["device_secrets"] == {"smoke-01": "override-secret"}


def test_cli_runtime_field_ingest_bridge_forwards_args(monkeypatch, tmp_path: Path, capsys) -> None:
    seen: dict[str, object] = {}
    source = tmp_path / "events.jsonl"
    source.write_text(json.dumps({"detections": [{"class_id": "2"}]}) + "\n", encoding="utf-8")

    def fake_bridge(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "status": "ok",
            "target": "field-ingest-bridge",
            "count": 1,
            "failed": 0,
            "dry_run": True,
            "state_path": str(tmp_path / "state.json"),
            "summary": {
                "posted": 0,
                "accepted": 0,
                "signed": 0,
                "source_format": "jsonl",
                "scenario_counts": {"illegal_parking": 1},
                "source_counts": {"camera": 1},
                "device_counts": {"camera-main-road-1": 1},
            },
            "results": [{"index": 1, "status": "dry_run", "normalized": {"scenario_id": "illegal_parking"}}],
        }

    monkeypatch.setattr(cli, "_run_field_ingest_bridge", fake_bridge)

    cli.main([
        "runtime",
        "field-ingest-bridge",
        str(source),
        "--server",
        "http://runtime.local:8765",
        "--state-path",
        str(tmp_path / "state.json"),
        "--dry-run",
        "--limit",
        "3",
        "--timeout",
        "2",
    ])

    output = capsys.readouterr().out

    assert seen == {
        "source": str(source),
        "server": "http://runtime.local:8765",
        "state_path": str(tmp_path / "state.json"),
        "dry_run": True,
        "limit": 3,
        "timeout_s": 2.0,
        "device_secrets": {},
    }
    assert "sources=camera:1" in output
    assert "devices=camera-main-road-1:1" in output


def test_cli_runtime_field_sign_device_payload_writes_signed_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from askme.pipeline.field_operations import sign_field_device_payload

    monkeypatch.setenv("ASKME_UNIT_DEVICE_SECRET", "unit-device-secret")
    source = tmp_path / "event.json"
    output = tmp_path / "signed.json"
    source.write_text(
        json.dumps({
            "source": "sensor",
            "device_id": "smoke-01",
            "temperature_c": 72,
        }),
        encoding="utf-8",
    )

    payload = cli._run_field_sign_device_payload(
        source=str(source),
        output=str(output),
        device_id="",
        secret="",
        secret_env="ASKME_UNIT_DEVICE_SECRET",
        timestamp=1770000000.0,
    )

    signed = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "signed"
    assert payload["count"] == 1
    assert payload["secret_source"] == "env:ASKME_UNIT_DEVICE_SECRET"
    assert signed["device_signature_alg"] == "hmac-sha256"
    assert signed["device_signature_timestamp"] == 1770000000.0
    assert signed["device_signature"] == sign_field_device_payload(
        signed,
        secret="unit-device-secret",
    )


def test_cli_runtime_field_sign_device_payload_can_override_device_id(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    output = tmp_path / "signed.jsonl"
    source.write_text(
        json.dumps({"source": "camera", "zone_id": "main-road-1"}) + "\n",
        encoding="utf-8",
    )

    cli.main([
        "runtime",
        "field-sign-device-payload",
        str(source),
        "--output",
        str(output),
        "--device-id",
        "camera-main-road-1",
        "--secret",
        "camera-secret",
        "--timestamp",
        "1770000001",
    ])

    signed = json.loads(output.read_text(encoding="utf-8").strip())
    assert signed["device_id"] == "camera-main-road-1"
    assert signed["device_signature"]


def test_cli_runtime_field_sign_device_payload_exits_when_secret_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "event.json"
    source.write_text(json.dumps({"source": "sensor", "device_id": "smoke-01"}), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "field-sign-device-payload", str(source), "--secret-env", "MISSING_SECRET"])

    assert exc.value.code == 2


def test_cli_runtime_field_device_trust_reports_missing_secret(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.delenv("ASKME_FIELD_SMOKE_SECRET", raising=False)
    profile = tmp_path / "site.yaml"
    profile.write_text(
        """
site:
  site_id: unit-park
  name: Unit Park
responder_groups:
  security: {webhook_env: SECURITY_WEBHOOK, secret_env: SECURITY_SECRET}
  cleaning: {webhook_env: CLEANING_WEBHOOK, secret_env: CLEANING_SECRET}
  operations: {webhook_env: OPS_WEBHOOK, secret_env: OPS_SECRET}
zones:
  main-road: {type: main_channel, parking_allowed: false, location: Main Road}
  help-1: {type: help_point, help_point_id: help-1, location: Help Point}
  smoke-zone: {type: smoke_risk_area, location: Power Room}
devices:
  smoke-01:
    source: sensor
    sensor_type: smoke_temperature
    zone_id: smoke-zone
    secret_env: ASKME_FIELD_SMOKE_SECRET
  camera-01:
    source: camera
    camera_id: cam-01
    zone_id: main-road
    secret_env: ASKME_FIELD_CAMERA_SECRET
  robot-01:
    source: robot
    robot_id: thunder-1
    zone_id: main-road
    secret_env: ASKME_FIELD_ROBOT_SECRET
thresholds:
  parking_duration_s: 120
  night_stranger_dwell_s: 10
  fire_temperature_c: 60
  smoke_level: 0.7
  trash_fill_ratio: 0.8
  crowd_person_count: 5
  crowd_duration_min: 30
""".strip(),
        encoding="utf-8",
    )

    cli.main(["runtime", "field-device-trust", "--site-profile", str(profile), "--show-commands"])

    output = capsys.readouterr().out
    assert "field-device-trust: needs_secret registered=3 ready=0 missing=3" in output
    assert "secret_env=ASKME_FIELD_SMOKE_SECRET status=missing_secret" in output
    assert "field-sign-device-payload" in output


def test_cli_runtime_field_device_trust_reports_ready(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ASKME_FIELD_SMOKE_SECRET", "smoke-secret")
    monkeypatch.setenv("ASKME_FIELD_CAMERA_SECRET", "camera-secret")
    monkeypatch.setenv("ASKME_FIELD_ROBOT_SECRET", "robot-secret")
    profile = tmp_path / "site.yaml"
    profile.write_text(
        """
site:
  site_id: unit-park
  name: Unit Park
responder_groups:
  security: {webhook_env: SECURITY_WEBHOOK, secret_env: SECURITY_SECRET}
  cleaning: {webhook_env: CLEANING_WEBHOOK, secret_env: CLEANING_SECRET}
  operations: {webhook_env: OPS_WEBHOOK, secret_env: OPS_SECRET}
zones:
  main-road: {type: main_channel, parking_allowed: false, location: Main Road}
  help-1: {type: help_point, help_point_id: help-1, location: Help Point}
  smoke-zone: {type: smoke_risk_area, location: Power Room}
devices:
  smoke-01: {source: sensor, sensor_type: smoke_temperature, zone_id: smoke-zone, secret_env: ASKME_FIELD_SMOKE_SECRET}
  camera-01: {source: camera, camera_id: cam-01, zone_id: main-road, secret_env: ASKME_FIELD_CAMERA_SECRET}
  robot-01: {source: robot, robot_id: thunder-1, zone_id: main-road, secret_env: ASKME_FIELD_ROBOT_SECRET}
thresholds:
  parking_duration_s: 120
  night_stranger_dwell_s: 10
  fire_temperature_c: 60
  smoke_level: 0.7
  trash_fill_ratio: 0.8
  crowd_person_count: 5
  crowd_duration_min: 30
""".strip(),
        encoding="utf-8",
    )

    payload = cli._run_field_device_trust(site_profile=str(profile))

    assert payload["status"] == "ready"
    assert payload["summary"]["registered_device_count"] == 3
    assert payload["summary"]["signature_ready_count"] == 3
    assert payload["summary"]["missing_secret_count"] == 0


def test_cli_runtime_field_site_env_template_writes_output(tmp_path: Path, capsys) -> None:
    profile = tmp_path / "site.yaml"
    output_path = tmp_path / "field-site.env"
    profile.write_text(
        """
site:
  site_id: unit-park
  name: Unit Park
responder_groups:
  security: {webhook_env: SECURITY_WEBHOOK, secret_env: SECURITY_SECRET}
  cleaning: {webhook_env: CLEANING_WEBHOOK, secret_env: CLEANING_SECRET}
  operations: {webhook_env: OPS_WEBHOOK, secret_env: OPS_SECRET}
zones:
  main-road: {type: main_channel, parking_allowed: false, location: Main Road}
  help-1: {type: help_point, help_point_id: help-1, location: Help Point}
  smoke-zone: {type: smoke_risk_area, location: Power Room}
devices:
  smoke-01:
    source: sensor
    sensor_type: smoke_temperature
    zone_id: smoke-zone
    secret_env: ASKME_FIELD_SMOKE_SECRET
  camera-01:
    source: camera
    camera_id: cam-01
    zone_id: main-road
    secret_env: ASKME_FIELD_CAMERA_SECRET
  robot-01:
    source: robot
    robot_id: thunder-1
    zone_id: main-road
    secret_env: ASKME_FIELD_ROBOT_SECRET
thresholds:
  parking_duration_s: 120
  night_stranger_dwell_s: 10
  fire_temperature_c: 60
  smoke_level: 0.7
  trash_fill_ratio: 0.8
  crowd_person_count: 5
  crowd_duration_min: 30
""".strip(),
        encoding="utf-8",
    )

    cli.main([
        "runtime",
        "field-site-env-template",
        "--site-profile",
        str(profile),
        "--output",
        str(output_path),
    ])

    console = capsys.readouterr().out
    generated = output_path.read_text(encoding="utf-8")
    assert "field-site-env-template: ok envs=9 configured=" in console
    assert f"output: {output_path}" in console
    assert "SECURITY_WEBHOOK=" in generated
    assert "ASKME_FIELD_SMOKE_SECRET=" in generated
    assert "ASKME_FIELD_ROBOT_SECRET=" in generated


def test_cli_runtime_field_site_env_template_json_keeps_template_when_no_output() -> None:
    payload = cli._run_field_site_env_template(site_profile="deploy/site-profiles/park-demo.yaml")

    assert payload["status"] == "ok"
    assert payload["env_count"] >= 10
    assert payload["output"] == ""
    assert "ASKME_DINGTALK_SECURITY_WEBHOOK=" in payload["template"]
    assert "ASKME_FIELD_ROBOT_THUNDER_SECRET=" in payload["template"]


def test_cli_runtime_field_ingest_smoke_runs_local_http(tmp_path: Path) -> None:
    payload = cli._run_field_ingest_smoke(output_dir=str(tmp_path / "smoke"))

    assert payload["status"] == "passed"
    assert payload["local_server"] is True
    assert payload["event_count"] == 8
    assert payload["expected_bridge_count"] == 8
    assert payload["bridge"]["count"] == 8
    assert payload["bridge"]["summary"]["posted"] == 8
    assert payload["bridge"]["summary"]["accepted"] == 8
    assert payload["bridge"]["summary"]["events_created"] == 8
    assert payload["bridge"]["summary"]["source_counts"] == {"camera": 4, "robot": 2, "sensor": 2}
    assert set(payload["required_scenario_ids"]).issubset(set(payload["scenario_ids"]))
    assert "trash_bin_full" in payload["scenario_ids"]
    assert "crowd_gathering" in payload["scenario_ids"]
    assert Path(payload["source"]).exists()
    assert Path(payload["archive_path"]).exists()
    assert payload["operator_action"]["acknowledged"] is True
    assert payload["operator_action"]["event"]["action_audit"][-1]["action"] == "acknowledge"
    assert json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))["status"] == "passed"


def test_cli_runtime_field_ingest_smoke_can_require_device_signatures(tmp_path: Path) -> None:
    payload = cli._run_field_ingest_smoke(
        output_dir=str(tmp_path / "signed-smoke"),
        require_device_signatures=True,
    )

    summary = payload["bridge"]["summary"]
    assert payload["status"] == "passed"
    assert payload["require_device_signatures"] is True
    assert summary["posted"] == 8
    assert summary["accepted"] == 8
    assert summary["events_created"] == 8
    assert summary["signed"] == 8
    assert payload["bridge"]["results"][0]["device_signing"]["signed"] is True


def test_cli_runtime_field_ingest_smoke_produces_strict_audit_anchor(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    payload = cli._run_field_ingest_smoke(
        output_dir=str(output),
        audit_hmac_secret="unit-test-secret",
    )

    anchor = cli._run_field_audit_anchor(
        server="",
        archive_path=str(output / "field-events.jsonl"),
        audit_path=str(output / "field-action-audit.jsonl"),
        hmac_secret="unit-test-secret",
        output=str(output / "audit-checkpoint.json"),
        require_valid=True,
    )

    assert payload["status"] == "passed"
    assert anchor["status"] == "anchored"
    assert anchor["checkpoint"]["signed"] is True
    assert anchor["checkpoint"]["signature_alg"] == "hmac-sha256"
    assert anchor["checkpoint"]["checked_count"] == 1
    assert anchor["checkpoint"]["expected_count"] == 1
    assert json.loads((output / "audit-checkpoint.json").read_text(encoding="utf-8"))["status"] == "anchored"


def test_cli_runtime_field_disposition_smoke_closes_p0_with_report(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    payload = cli._run_field_disposition_smoke(
        output_dir=str(output),
        audit_hmac_secret="unit-test-secret",
    )

    assert payload["status"] == "passed"
    assert payload["local_server"] is True
    assert payload["created"]["accepted"] is True
    assert payload["acknowledged"]["acknowledged"] is True
    assert payload["close_requested"]["requested"] is True
    assert payload["closed"]["event"]["status"] == "closed"
    assert payload["closed"]["event"]["close_approval"]["supervisor_id"] == "supervisor-1"
    assert payload["timeline_count"] >= 3
    assert payload["action_audit_integrity"]["valid"] is True
    assert payload["action_audit_integrity"]["signed"] is True
    assert json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))["status"] == "passed"


def test_cli_runtime_field_voice_smoke_queues_recorded_voice(tmp_path: Path) -> None:
    payload = cli._run_field_voice_smoke(
        output_dir=str(tmp_path / "voice-smoke"),
        scenario="fire",
    )

    assert payload["status"] == "passed"
    assert payload["target"] == "field-voice-smoke"
    assert payload["voice_delivery"]["status"] == "queued"
    assert payload["voice_directive"]["resolved_profile"] == "emergency_short"
    assert payload["recorded_voice_handler"]["profiles"] == ["emergency_short"]
    assert payload["recorded_voice_handler"]["playback_started"] is True
    assert Path(payload["report_path"]).exists()


def test_cli_field_voice_smoke_event_gets_unique_dedupe_fields(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_field_smoke_run_id", lambda: "unit-run")

    payload = cli._field_voice_smoke_event("illegal_parking")
    original_location = payload["location"]
    original_plate = payload["plate_number"]

    cli._make_field_voice_smoke_event_unique(payload)

    assert payload["smoke_run_id"] == "unit-run"
    assert payload["location"] == f"{original_location}-unit-run"
    assert payload["plate_number"] == f"{original_plate}-unit-run"


def test_cli_runtime_field_voice_smoke_command_forwards_args(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_voice_smoke(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "status": "passed",
            "voice_delivery": {"status": "queued"},
            "voice_directive": {"resolved_profile": "emergency_short"},
        }

    monkeypatch.setattr(cli, "_run_field_voice_smoke", fake_voice_smoke)

    cli.main([
        "runtime",
        "field-voice-smoke",
        "--output-dir",
        str(tmp_path / "voice-smoke"),
        "--scenario",
        "joint_fault",
        "--live-tts",
    ])

    assert seen == {
        "output_dir": str(tmp_path / "voice-smoke"),
        "server": "",
        "scenario": "joint_fault",
        "live_tts": True,
    }


def test_cli_runtime_field_notification_smoke_uses_local_collector(tmp_path: Path) -> None:
    payload = cli._run_field_notification_smoke(
        output_dir=str(tmp_path / "notification-smoke"),
        groups="security,cleaning",
    )

    assert payload["status"] == "passed"
    assert payload["target"] == "field-notification-smoke"
    assert payload["local_webhook_collector"] is True
    assert payload["sent_groups"] == ["security", "cleaning"]
    assert payload["collector_request_count"] >= 2
    assert Path(payload["report_path"]).exists()


def test_cli_runtime_field_notification_smoke_command_forwards_args(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_notification_smoke(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "status": "passed",
            "sent_groups": ["security"],
            "collector_request_count": 1,
        }

    monkeypatch.setattr(cli, "_run_field_notification_smoke", fake_notification_smoke)

    cli.main([
        "runtime",
        "field-notification-smoke",
        "--output-dir",
        str(tmp_path / "notification-smoke"),
        "--groups",
        "security",
    ])

    assert seen == {
        "output_dir": str(tmp_path / "notification-smoke"),
        "server": "",
        "groups": "security",
    }


def test_cli_runtime_field_notification_preflight_reads_local_config(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_preflight(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "ready", "ready": True, "groups": {}, "next_actions": []}

    monkeypatch.setattr(cli, "_run_field_notification_preflight", fake_preflight)

    cli.main([
        "runtime",
        "field-notification-preflight",
        "--groups",
        "security,operations",
        "--allow-unsigned",
    ])

    assert seen == {
        "server": "",
        "groups": "security,operations",
        "require_secret": False,
    }


def test_cli_runtime_field_notification_preflight_exits_when_blocked(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_run_field_notification_preflight",
        lambda **_kwargs: {
            "status": "blocked",
            "ready": False,
            "groups": {},
            "next_actions": ["Configure security"],
        },
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "field-notification-preflight"])

    assert exc.value.code == 1


def test_cli_json_output_escapes_unicode_on_non_utf8_stdout(monkeypatch) -> None:
    class Stdout:
        encoding = "gbk"

    monkeypatch.setattr(cli.sys, "stdout", Stdout())

    text = cli._json({"message": "⚠ 中文"})

    assert "\\u26a0" in text
    assert "\\u4e2d\\u6587" in text


def test_cli_runtime_field_smoke_suite_aggregates_reports(monkeypatch, tmp_path: Path) -> None:
    def fake_eval(*, output: str) -> dict[str, object]:
        Path(output).write_text('{"status":"passed"}', encoding="utf-8")
        return {"status": "passed", "report_path": output}

    def fake_ingest(**kwargs: object) -> dict[str, object]:
        output_dir = str(kwargs["output_dir"])
        return {"status": "passed", "report_path": str(Path(output_dir) / "field-ingest-smoke.json")}

    def fake_voice(**kwargs: object) -> dict[str, object]:
        return {"status": "passed", "report_path": str(Path(str(kwargs["output_dir"])) / "field-voice-smoke.json")}

    def fake_notification(**kwargs: object) -> dict[str, object]:
        return {
            "status": "passed",
            "report_path": str(Path(str(kwargs["output_dir"])) / "field-notification-smoke.json"),
        }

    def fake_disposition(**kwargs: object) -> dict[str, object]:
        return {
            "status": "passed",
            "report_path": str(Path(str(kwargs["output_dir"])) / "field-disposition-smoke.json"),
        }

    def fake_readiness(**_kwargs: object) -> dict[str, object]:
        return {"status": "ready_for_lab", "blockers": [], "warnings": ["sample"]}

    def fake_audit_anchor(**kwargs: object) -> dict[str, object]:
        return {
            "status": "anchored",
            "target": "field-audit-anchor",
            "checkpoint": {"latest_hash": "hash-suite"},
            "kwargs": kwargs,
        }

    monkeypatch.setattr(cli, "_run_field_operations_eval", fake_eval)
    monkeypatch.setattr(cli, "_run_field_ingest_smoke", fake_ingest)
    monkeypatch.setattr(cli, "_run_field_voice_smoke", fake_voice)
    monkeypatch.setattr(cli, "_run_field_notification_smoke", fake_notification)
    monkeypatch.setattr(cli, "_run_field_disposition_smoke", fake_disposition)
    monkeypatch.setattr(cli, "_run_field_readiness", fake_readiness)
    monkeypatch.setattr(cli, "_run_field_audit_anchor", fake_audit_anchor)

    payload = cli._run_field_smoke_suite(
        output_dir=str(tmp_path / "suite"),
        voice_scenario="joint_fault",
        groups="security",
        audit_hmac_secret="secret",
        audit_webhook_url="http://siem.local/audit",
    )

    assert payload["status"] == "passed"
    assert payload["checks"] == {
        "scenario_eval": True,
        "field_ingest_smoke": True,
        "field_voice_smoke": True,
        "field_notification_smoke": True,
        "field_disposition_smoke": True,
        "readiness_unblocked": True,
        "audit_checkpoint_created": True,
    }
    assert payload["audit_anchor"]["checkpoint"]["latest_hash"] == "hash-suite"
    assert payload["audit_anchor"]["kwargs"]["hmac_secret"] == "secret"
    assert payload["audit_anchor"]["kwargs"]["webhook_url"] == "http://siem.local/audit"
    assert Path(payload["report_path"]).exists()
    html_report = Path(str(payload["html_report_path"]))
    assert html_report.exists()
    html = html_report.read_text(encoding="utf-8")
    assert "Askme 现场能力验收报告" in html
    assert "部署门禁" in html


def test_cli_runtime_field_smoke_suite_uses_env_audit_hmac_secret(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_eval(*, output: str) -> dict[str, object]:
        Path(output).write_text('{"status":"passed"}', encoding="utf-8")
        return {"status": "passed"}

    def fake_ingest(**kwargs: object) -> dict[str, object]:
        seen["ingest_hmac"] = kwargs.get("audit_hmac_secret")
        return {"status": "passed"}

    def fake_voice(**_kwargs: object) -> dict[str, object]:
        return {"status": "passed"}

    def fake_notification(**_kwargs: object) -> dict[str, object]:
        return {"status": "passed"}

    def fake_disposition(**kwargs: object) -> dict[str, object]:
        seen["disposition_hmac"] = kwargs.get("audit_hmac_secret")
        return {"status": "passed"}

    def fake_readiness(**kwargs: object) -> dict[str, object]:
        seen["readiness_hmac"] = kwargs.get("audit_hmac_secret")
        return {"status": "ready_for_lab", "blockers": [], "warnings": []}

    def fake_audit_anchor(**kwargs: object) -> dict[str, object]:
        seen["anchor_hmac"] = kwargs.get("hmac_secret")
        return {"status": "anchored", "target": "field-audit-anchor"}

    monkeypatch.setenv("ASKME_FIELD_ACTION_AUDIT_HMAC_SECRET", "env-audit-secret")
    monkeypatch.setattr(cli, "_run_field_operations_eval", fake_eval)
    monkeypatch.setattr(cli, "_run_field_ingest_smoke", fake_ingest)
    monkeypatch.setattr(cli, "_run_field_voice_smoke", fake_voice)
    monkeypatch.setattr(cli, "_run_field_notification_smoke", fake_notification)
    monkeypatch.setattr(cli, "_run_field_disposition_smoke", fake_disposition)
    monkeypatch.setattr(cli, "_run_field_readiness", fake_readiness)
    monkeypatch.setattr(cli, "_run_field_audit_anchor", fake_audit_anchor)

    payload = cli._run_field_smoke_suite(output_dir=str(tmp_path / "suite"))

    assert payload["status"] == "passed"
    assert seen == {
        "ingest_hmac": "env-audit-secret",
        "disposition_hmac": "env-audit-secret",
        "readiness_hmac": "env-audit-secret",
        "anchor_hmac": "env-audit-secret",
    }


def test_cli_runtime_field_smoke_suite_command_forwards_args(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_suite(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "passed", "checks": {}, "readiness": {}}

    monkeypatch.setattr(cli, "_run_field_smoke_suite", fake_suite)

    cli.main([
        "runtime",
        "field-smoke-suite",
        "--output-dir",
        str(tmp_path / "suite"),
        "--voice-scenario",
        "illegal_parking",
        "--groups",
        "security",
        "--live-tts",
        "--audit-hmac-secret",
        "secret",
        "--audit-webhook-url",
        "http://siem.local/audit",
        "--audit-webhook-retries",
        "2",
    ])

    assert seen == {
        "output_dir": str(tmp_path / "suite"),
        "voice_scenario": "illegal_parking",
        "groups": "security",
        "live_tts": True,
        "audit_hmac_secret": "secret",
        "audit_webhook_url": "http://siem.local/audit",
        "audit_webhook_retries": 2,
        "include_audit_anchor": True,
    }


def test_cli_runtime_field_deployed_smoke_runs_against_existing_server(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_get_json(url: str) -> dict[str, object]:
        calls.append(url)
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/api/field/readiness"):
            return {"status": "ready_for_lab"}
        raise AssertionError(url)

    monkeypatch.setattr(cli, "_get_json", fake_get_json)
    monkeypatch.setattr(
        cli,
        "_run_field_notification_preflight",
        lambda **_kwargs: {"status": "ready", "ready": True},
    )
    seen_ingest: dict[str, object] = {}

    def fake_ingest(**kwargs: object) -> dict[str, object]:
        seen_ingest.update(kwargs)
        return {"status": "passed", "bridge": {"summary": {"signed": 8}}}

    monkeypatch.setattr(cli, "_run_field_ingest_smoke", fake_ingest)
    monkeypatch.setattr(cli, "_run_field_voice_smoke", lambda **_kwargs: {"status": "passed"})
    monkeypatch.setattr(cli, "_run_field_notification_smoke", lambda **_kwargs: {"status": "passed"})

    payload = cli._run_field_deployed_smoke(
        server="http://runtime.local:8765",
        output_dir=str(tmp_path / "deployed"),
        voice_scenario="joint_fault",
        groups="security",
        require_device_signatures=True,
    )

    assert payload["status"] == "passed"
    assert payload["checks"]["health_reachable"] is True
    assert payload["checks"]["signed_device_ingest_smoke"] is True
    assert payload["checks"]["field_notification_smoke"] is True
    assert seen_ingest["require_device_signatures"] is True
    assert Path(payload["report_path"]).exists()
    assert calls == [
        "http://runtime.local:8765/health",
        "http://runtime.local:8765/api/field/readiness",
    ]


def test_cli_runtime_field_deployed_smoke_blocks_when_notification_preflight_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "_get_json",
        lambda url: {"status": "ok"} if url.endswith("/health") else {"status": "ready_for_lab"},
    )
    monkeypatch.setattr(
        cli,
        "_run_field_notification_preflight",
        lambda **_kwargs: {"status": "blocked", "ready": False},
    )
    monkeypatch.setattr(cli, "_run_field_ingest_smoke", lambda **_kwargs: {"status": "passed"})
    monkeypatch.setattr(cli, "_run_field_voice_smoke", lambda **_kwargs: {"status": "passed"})

    def fail_notification(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("notification smoke should be gated by preflight")

    monkeypatch.setattr(cli, "_run_field_notification_smoke", fail_notification)

    payload = cli._run_field_deployed_smoke(
        server="http://runtime.local:8765",
        output_dir=str(tmp_path / "deployed"),
    )

    assert payload["status"] == "failed"
    assert payload["notification_smoke"]["status"] == "skipped"
    assert payload["notification_smoke"]["reason"] == "notification_preflight_blocked"
    assert payload["checks"]["notification_preflight_ready"] is False


def test_cli_runtime_field_deployed_smoke_command_forwards_args(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_deployed(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "passed", "checks": {}}

    monkeypatch.setattr(cli, "_run_field_deployed_smoke", fake_deployed)

    cli.main([
        "runtime",
        "field-deployed-smoke",
        "--server",
        "http://runtime.local:8765",
        "--output-dir",
        str(tmp_path / "deployed"),
        "--voice-scenario",
        "illegal_parking",
        "--groups",
        "security",
        "--allow-notification-not-ready",
        "--require-device-signatures",
    ])

    assert seen == {
        "server": "http://runtime.local:8765",
        "output_dir": str(tmp_path / "deployed"),
        "voice_scenario": "illegal_parking",
        "groups": "security",
        "require_notification_ready": False,
        "require_device_signatures": True,
    }


def test_cli_runtime_field_readiness_reads_local_files(monkeypatch, tmp_path: Path, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_readiness(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "status": "ready_for_lab",
            "site_profile": {
                "configured": True,
                "valid": True,
                "summary": {"site_id": "demo", "zone_count": 6, "device_count": 4},
            },
            "device_trust": {
                "registered_device_count": 4,
                "signed_device_count": 3,
                "unsigned_device_count": 1,
                "unsigned_device_ids": ["legacy-smoke-1"],
                "all_registered_devices_signature_ready": False,
            },
            "delivery_brief": {
                "stage_code": "pilot_ready_pending_site_launch",
                "customer_status": "已达到试点演示标准，待完成现场上线项",
                "release_scope": "pilot_demo_and_site_integration",
                "release_claim": "可对外说明：当前版本适合试点演示和现场联调，正式上线项已在交付清单中跟踪",
            },
            "blockers": [],
            "warnings": ["sample"],
            "next_actions": [],
        }

    monkeypatch.setattr(cli, "_run_field_readiness", fake_readiness)

    cli.main([
        "runtime",
        "field-readiness",
        "--archive-path",
        str(tmp_path / "events.jsonl"),
        "--scenario-report",
        str(tmp_path / "scenario.json"),
        "--smoke-report",
        str(tmp_path / "smoke.json"),
        "--voice-smoke-report",
        str(tmp_path / "voice-smoke.json"),
        "--notification-smoke-report",
        str(tmp_path / "notification-smoke.json"),
        "--site-profile",
        "deploy/site-profiles/park-demo.yaml",
        "--check-site-env",
        "--audit-hmac-secret",
        "readiness-secret",
        "--review-path",
        str(tmp_path / "reviews.jsonl"),
    ])
    output = capsys.readouterr().out

    assert seen == {
        "server": "",
        "archive_path": str(tmp_path / "events.jsonl"),
        "scenario_report": str(tmp_path / "scenario.json"),
        "smoke_report": str(tmp_path / "smoke.json"),
        "voice_smoke_report": str(tmp_path / "voice-smoke.json"),
        "notification_smoke_report": str(tmp_path / "notification-smoke.json"),
        "site_profile": "deploy/site-profiles/park-demo.yaml",
        "check_site_env": True,
        "audit_hmac_secret": "readiness-secret",
        "review_path": str(tmp_path / "reviews.jsonl"),
    }
    assert "product-stage: pilot_ready_pending_site_launch" in output
    assert "release-scope: pilot_demo_and_site_integration" in output
    assert "site-profile: configured=True valid=True site=demo zones=6 devices=4" in output
    assert "device-trust: registered=4 signed=3 unsigned=1 all_ready=False unsigned_ids=legacy-smoke-1" in output


def test_cli_runtime_field_readiness_exits_nonzero_when_blocked(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_run_field_readiness",
        lambda **_kwargs: {
            "status": "blocked",
            "blockers": ["field scenario evaluation has not passed"],
            "warnings": [],
            "next_actions": ["Run field-eval"],
        },
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "field-readiness"])

    assert exc.value.code == 1


def test_cli_runtime_audit_events_prints_review_queue(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_audit_events(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "count": 1,
            "filtered_total": 1,
            "product_summary": {"status": "needs_review", "requires_review_count": 1},
            "customer_report": {
                "status_label": "待主管复核",
                "summary_sentence": "当前有 1 条记录需要主管复核。",
            },
            "records": [],
            "review_queue": [
                {
                    "record_id": "field:2",
                    "severity": "critical",
                    "source": "field",
                    "action": "acknowledge",
                    "outcome": "accepted",
                    "customer_copy": {"review_owner": "安全主管"},
                }
            ],
        }

    monkeypatch.setattr(cli, "_run_unified_audit_events", fake_audit_events)

    cli.main([
        "runtime",
        "audit-events",
        "--review-queue-only",
        "--limit",
        "10",
        "--source",
        "field",
    ])
    output = capsys.readouterr().out

    assert seen["limit"] == 10
    assert seen["source"] == "field"
    assert "audit-events: records=1 filtered=1 status=needs_review review_queue=1" in output
    assert "customer-status: 待主管复核" in output
    assert "field:2 critical field acknowledge accepted owner=安全主管" in output


def test_cli_runtime_audit_events_forwards_source_paths(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_audit_events(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "count": 0,
            "filtered_total": 0,
            "product_summary": {"status": "auditable", "requires_review_count": 0},
            "records": [],
            "review_queue": [],
        }

    monkeypatch.setattr(cli, "_run_unified_audit_events", fake_audit_events)

    cli.main([
        "runtime",
        "audit-events",
        "--field-action-audit",
        str(tmp_path / "field-action-audit.jsonl"),
        "--review-path",
        str(tmp_path / "reviews.jsonl"),
    ])

    assert seen["field_action_audit"] == str(tmp_path / "field-action-audit.jsonl")
    assert seen["review_path"] == str(tmp_path / "reviews.jsonl")


def test_cli_runtime_audit_review_submits_decision(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_review(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "ok": True,
            "record": {
                "record_id": "field:2",
                "decision": "waived",
                "clears_review": True,
            },
            "path": "artifacts/audit/reviews.jsonl",
            "post_review": {
                "review_queue_count": 0,
                "requires_review_count": 0,
                "customer_status_label": "可交付审计包",
            },
        }

    monkeypatch.setattr(cli, "_run_unified_audit_review", fake_review)

    cli.main([
        "runtime",
        "audit-review",
        "field:2",
        "waived",
        "--reviewer-id",
        "supervisor-1",
        "--note",
        "duplicate smoke evidence",
    ])
    output = capsys.readouterr().out

    assert seen == {
        "record_id": "field:2",
        "reviewer_id": "supervisor-1",
        "decision": "waived",
        "note": "duplicate smoke evidence",
        "skill_audit": "",
        "field_action_audit": "",
        "field_event_archive": "",
        "runtime_audit": "",
        "review_path": "",
    }
    assert "audit-review: ok=True record=field:2 decision=waived clears_review=True" in output
    assert "post-review: queue=0 requires_review=0 status=可交付审计包" in output


def test_cli_runtime_audit_review_exits_nonzero_when_record_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_run_unified_audit_review",
        lambda **_kwargs: {"ok": False, "reason": "audit_record_not_found"},
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "audit-review", "field:missing", "accepted"])

    assert exc.value.code == 2


def test_cli_runtime_field_live_demo_forwards_args(monkeypatch, tmp_path: Path, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_live_demo(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "status": "passed",
            "accepted": 5,
            "scenario_count": 5,
            "mode": "remote_server",
            "report_path": "artifacts/field_operations/live-demo/live-field-demo.json",
            "guide_path": "artifacts/field_operations/live-demo/live-field-demo.md",
            "html_report_path": "artifacts/field_operations/live-demo/live-field-demo.html",
            "readiness": {"status": "ready_for_lab"},
            "scenarios": [
                {
                    "scenario_id": "fire_or_smoke",
                    "http_status": 200,
                    "accepted": True,
                    "event_id": "field-evt-1",
                }
            ],
        }

    monkeypatch.setattr(cli, "_run_field_live_demo", fake_live_demo)

    cli.main([
        "runtime",
        "field-live-demo",
        "--output-dir",
        str(tmp_path / "live-demo"),
        "--site-profile",
        "deploy/site-profiles/park-demo.yaml",
        "--server",
        "http://runtime.local:8765",
        "--scenario-file",
        str(tmp_path / "customer-scenarios.json"),
        "--refresh-scenario-timestamps",
        "--timeout",
        "3.5",
    ])
    output = capsys.readouterr().out

    assert seen == {
        "output_dir": str(tmp_path / "live-demo"),
        "site_profile": "deploy/site-profiles/park-demo.yaml",
        "server": "http://runtime.local:8765",
        "timeout_s": 3.5,
        "scenario_file": str(tmp_path / "customer-scenarios.json"),
        "refresh_scenario_timestamps": True,
    }
    assert "field-live-demo: passed accepted=5/5 mode=remote_server" in output
    assert "readiness: ready_for_lab" in output
    assert "html: artifacts/field_operations/live-demo/live-field-demo.html" in output
    assert "fire_or_smoke: http=200 accepted=True event=field-evt-1" in output


def test_cli_runtime_field_live_demo_exits_nonzero_when_failed(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_run_field_live_demo",
        lambda **_kwargs: {
            "status": "failed",
            "accepted": 3,
            "scenario_count": 5,
            "mode": "inprocess_http",
            "scenarios": [],
        },
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "field-live-demo"])

    assert exc.value.code == 1


def test_cli_runtime_field_audit_integrity_reads_server(monkeypatch, capsys) -> None:
    seen: dict[str, str] = {}

    def fake_get_json(url: str) -> dict[str, object]:
        seen["url"] = url
        return {
            "enabled": True,
            "valid": True,
            "checked_count": 2,
            "expected_count": 2,
            "signed": True,
            "failures": [],
        }

    monkeypatch.setattr(cli, "_get_json", fake_get_json)

    cli.main(["runtime", "field-audit-integrity", "--server", "http://runtime.local:8765", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert seen["url"] == "http://runtime.local:8765/api/field/audit/integrity"
    assert payload["valid"] is True
    assert payload["signed"] is True


def test_cli_runtime_field_audit_integrity_verifies_local_signed_chain(tmp_path: Path) -> None:
    from askme.pipeline.field_operations import FieldOperationsService

    archive_path = tmp_path / "events.jsonl"
    audit_path = tmp_path / "field-action-audit.jsonl"
    service = FieldOperationsService(
        config={
            "archive_path": str(archive_path),
            "action_audit": {
                "enabled": True,
                "path": str(audit_path),
                "swallow_errors": False,
                "hmac_secret": "unit-test-secret",
            },
        }
    )
    created = asyncio.run(
        service.trigger_payload(
            {
                "scenario_id": "illegal_parking",
                "source": "camera",
                "location": "main-road",
                "zone_name": "main-road",
                "duration_s": 180,
            }
        )
    )
    service.acknowledge_payload(
        created["event"]["event_id"],
        {"operator_id": "security-1", "note": "seen"},
    )

    payload = cli._run_field_audit_integrity(
        server="",
        archive_path=str(archive_path),
        audit_path=str(audit_path),
        hmac_secret="unit-test-secret",
    )

    assert payload["valid"] is True
    assert payload["signed"] is True
    assert payload["checked_count"] == 1
    assert payload["expected_count"] == 1


def test_cli_runtime_field_audit_integrity_exits_nonzero_when_invalid(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_run_field_audit_integrity",
        lambda **_kwargs: {
            "enabled": True,
            "valid": False,
            "path": "audit.jsonl",
            "checked_count": 1,
            "expected_count": 2,
            "signed": False,
            "latest_hash": "abc",
            "failures": [{"line": 0, "reason": "audit_count_mismatch"}],
        },
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "field-audit-integrity"])

    output = capsys.readouterr().out
    assert exc.value.code == 2
    assert "field-audit-integrity: invalid" in output
    assert "audit_count_mismatch" in output


def test_cli_runtime_field_audit_anchor_writes_checkpoint(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "audit-checkpoint.json"
    monkeypatch.setattr(
        cli,
        "_run_field_audit_integrity",
        lambda **_kwargs: {
            "enabled": True,
            "valid": True,
            "path": str(tmp_path / "field-action-audit.jsonl"),
            "checked_count": 2,
            "expected_count": 2,
            "latest_hash": "hash-123",
            "hash_alg": "sha256",
            "signed": True,
            "signature_alg": "hmac-sha256",
            "failures": [],
        },
    )

    payload = cli._run_field_audit_anchor(
        server="",
        archive_path=str(tmp_path / "events.jsonl"),
        audit_path=str(tmp_path / "field-action-audit.jsonl"),
        output=str(output),
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "anchored"
    assert payload["checkpoint"]["latest_hash"] == "hash-123"
    assert written["checkpoint"]["signed"] is True


def test_cli_runtime_field_audit_anchor_posts_webhook(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "audit-checkpoint.json"
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_run_field_audit_integrity",
        lambda **_kwargs: {
            "enabled": True,
            "valid": True,
            "path": "field-action-audit.jsonl",
            "checked_count": 1,
            "expected_count": 1,
            "latest_hash": "hash-abc",
            "hash_alg": "sha256",
            "signed": False,
            "signature_alg": "",
            "failures": [],
        },
    )
    monkeypatch.setattr(
        cli,
        "_post_json",
        lambda url, payload: seen.update({"url": url, "payload": payload}) or {"ok": True},
    )

    cli.main([
        "runtime",
        "field-audit-anchor",
        "--output",
        str(output),
        "--webhook-url",
        "http://siem.local/audit",
    ])

    written = json.loads(output.read_text(encoding="utf-8"))
    assert seen["url"] == "http://siem.local/audit"
    assert seen["payload"]["checkpoint"]["latest_hash"] == "hash-abc"
    assert written["webhook_delivery"] == {
        "status": "sent",
        "attempts": 1,
        "response": {"ok": True},
    }


def test_cli_runtime_field_audit_anchor_reports_webhook_failure(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "audit-checkpoint.json"

    monkeypatch.setattr(
        cli,
        "_run_field_audit_integrity",
        lambda **_kwargs: {
            "enabled": True,
            "valid": True,
            "path": "field-action-audit.jsonl",
            "checked_count": 1,
            "expected_count": 1,
            "latest_hash": "hash-abc",
            "hash_alg": "sha256",
            "signed": False,
            "signature_alg": "",
            "failures": [],
        },
    )
    monkeypatch.setattr(
        cli,
        "_post_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.RequestException("offline")),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main([
            "runtime",
            "field-audit-anchor",
            "--output",
            str(output),
            "--webhook-url",
            "http://siem.local/audit",
            "--webhook-retries",
            "2",
            "--retry-queue",
            str(tmp_path / "retry.jsonl"),
        ])

    written = json.loads(output.read_text(encoding="utf-8"))
    queue = tmp_path / "retry.jsonl"
    assert exc.value.code == 3
    assert queue.exists()
    assert written["status"] == "delivery_failed"
    assert written["webhook_delivery"]["attempts"] == 2
    assert "offline" in written["webhook_delivery"]["error"]


def test_cli_runtime_field_audit_retry_delivery_replays_queue(monkeypatch, tmp_path: Path) -> None:
    queue = tmp_path / "retry.jsonl"
    queue.write_text(
        json.dumps({
            "queued_at": 123,
            "webhook_url": "http://siem.local/audit",
            "payload": {"target": "field-audit-anchor", "checkpoint": {"latest_hash": "hash-abc"}},
        })
        + "\n",
        encoding="utf-8",
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "_post_json",
        lambda url, payload: seen.update({"url": url, "payload": payload}) or {"ok": True},
    )

    payload = cli._run_field_audit_delivery_retry(queue=str(queue), webhook_retries=2)

    assert payload["status"] == "sent"
    assert payload["attempted"] == 1
    assert payload["sent"] == 1
    assert payload["remaining"] == 0
    assert not queue.exists()
    assert not (tmp_path / "retry.jsonl.lock").exists()
    assert seen["url"] == "http://siem.local/audit"


def test_cli_runtime_field_audit_retry_delivery_exits_when_locked(monkeypatch, tmp_path: Path) -> None:
    queue = tmp_path / "retry.jsonl"
    queue.write_text(
        json.dumps({
            "queued_at": 123,
            "webhook_url": "http://siem.local/audit",
            "payload": {"target": "field-audit-anchor", "checkpoint": {"latest_hash": "hash-abc"}},
        })
        + "\n",
        encoding="utf-8",
    )
    lock = tmp_path / "retry.jsonl.lock"
    lock.write_text(
        json.dumps({"pid": 9999, "queue": str(queue), "acquired_at": 123, "expires_at": 9999999999}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_post_json",
        lambda *_args, **_kwargs: pytest.fail("locked delivery should not post webhooks"),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "field-audit-retry-delivery", "--queue", str(queue)])

    assert exc.value.code == 4
    assert queue.exists()
    assert lock.exists()


def test_cli_runtime_field_audit_retry_delivery_takes_stale_lock(monkeypatch, tmp_path: Path) -> None:
    queue = tmp_path / "retry.jsonl"
    queue.write_text(
        json.dumps({
            "queued_at": 123,
            "webhook_url": "http://siem.local/audit",
            "payload": {"target": "field-audit-anchor", "checkpoint": {"latest_hash": "hash-abc"}},
        })
        + "\n",
        encoding="utf-8",
    )
    lock = tmp_path / "retry.jsonl.lock"
    lock.write_text(
        json.dumps({"pid": 9999, "queue": str(queue), "acquired_at": 1, "expires_at": 1}),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "_post_json",
        lambda url, payload: seen.update({"url": url, "payload": payload}) or {"ok": True},
    )

    payload = cli._run_field_audit_delivery_retry(queue=str(queue), webhook_retries=1, lock_timeout_s=60)

    assert payload["status"] == "sent"
    assert payload["lock"]["acquired"] is True
    assert seen["url"] == "http://siem.local/audit"
    assert not queue.exists()
    assert not lock.exists()


def test_cli_runtime_field_audit_retry_delivery_keeps_failed_items(monkeypatch, tmp_path: Path) -> None:
    queue = tmp_path / "retry.jsonl"
    queue.write_text(
        json.dumps({
            "queued_at": 123,
            "webhook_url": "http://siem.local/audit",
            "payload": {"target": "field-audit-anchor", "checkpoint": {"latest_hash": "hash-abc"}},
        })
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_post_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.RequestException("offline")),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main([
            "runtime",
            "field-audit-retry-delivery",
            "--queue",
            str(queue),
            "--webhook-retries",
            "2",
        ])

    assert exc.value.code == 3
    assert queue.exists()
    assert "hash-abc" in queue.read_text(encoding="utf-8")


def test_cli_runtime_field_audit_retry_status_reports_empty_missing_queue(tmp_path: Path) -> None:
    payload = cli._run_field_audit_retry_status(queue=str(tmp_path / "missing.jsonl"))

    assert payload["status"] == "empty"
    assert payload["pending"] == 0
    assert payload["invalid"] == 0
    assert payload["items"] == []


def test_cli_runtime_field_audit_retry_status_reports_pending_and_invalid_queue(tmp_path: Path) -> None:
    queue = tmp_path / "retry.jsonl"
    queue.write_text(
        json.dumps({
            "queued_at": 123,
            "webhook_url": "http://siem.local/audit",
            "payload": {
                "target": "field-audit-anchor",
                "checkpoint": {"latest_hash": "hash-abc", "checked_count": 2},
            },
        })
        + "\n"
        + "{bad-json\n",
        encoding="utf-8",
    )

    payload = cli._run_field_audit_retry_status(queue=str(queue))

    assert payload["status"] == "pending"
    assert payload["pending"] == 1
    assert payload["invalid"] == 1
    assert payload["items"][0]["webhook_url"] == "http://siem.local/audit"
    assert payload["items"][0]["latest_hash"] == "hash-abc"
    assert payload["items"][0]["checked_count"] == 2
    assert payload["items"][1]["status"] == "invalid_json"


def test_cli_runtime_field_audit_retry_status_fail_on_pending_exits_nonzero(tmp_path: Path) -> None:
    queue = tmp_path / "retry.jsonl"
    queue.write_text(
        json.dumps({
            "queued_at": 123,
            "webhook_url": "http://siem.local/audit",
            "payload": {"target": "field-audit-anchor", "checkpoint": {"latest_hash": "hash-abc"}},
        })
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        cli.main([
            "runtime",
            "field-audit-retry-status",
            "--queue",
            str(queue),
            "--fail-on-pending",
        ])

    assert exc.value.code == 3


def test_cli_runtime_field_audit_anchor_exits_nonzero_when_invalid(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_run_field_audit_integrity",
        lambda **_kwargs: {
            "enabled": True,
            "valid": False,
            "path": "field-action-audit.jsonl",
            "checked_count": 0,
            "expected_count": 1,
            "latest_hash": "GENESIS",
            "failures": [{"line": 0, "reason": "audit_count_mismatch"}],
        },
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "field-audit-anchor", "--output", ""])

    assert exc.value.code == 2


def test_cli_runtime_mic_calibration_forwards_args(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}
    out_path = tmp_path / "mic.json"

    def fake_mic_calibration(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "status": "ok",
            "target": "runtime-mic-calibration",
            "summary": {"observed_peak_max": 200},
        }

    monkeypatch.setattr(cli, "_run_mic_calibration", fake_mic_calibration)

    cli.main(
        [
            "runtime",
            "mic-calibration",
            "--server",
            "http://runtime.local:18765/",
            "--duration",
            "1.5",
            "--interval",
            "0.25",
            "--min-signal-peak",
            "42",
            "--json-out",
            str(out_path),
        ]
    )

    assert seen == {
        "server": "http://runtime.local:18765/",
        "duration_s": 1.5,
        "interval_s": 0.25,
        "min_signal_peak": 42,
    }
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "ok"


def test_cli_memory_import_outputs_summary(monkeypatch, capsys, tmp_path: Path) -> None:
    path = tmp_path / "site.md"
    path.write_text("- 洗手间在一楼", encoding="utf-8")
    seen: dict[str, object] = {}

    async def fake_import_knowledge_file(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return SimpleNamespace(
            to_dict=lambda: {
                "source": str(path),
                "parsed": 1,
                "imported": 1,
                "skipped": 0,
                "errors": [],
                "dry_run": False,
            }
        )

    monkeypatch.setattr(
        "askme.memory.importer.import_knowledge_file",
        fake_import_knowledge_file,
    )

    cli.main(["memory", "import", str(path), "--category", "location"])

    output = capsys.readouterr().out
    assert "imported=1" in output
    assert seen["kwargs"]["category"] == "location"


def test_cli_memory_search_outputs_results(monkeypatch, capsys) -> None:
    class FakeBridge:
        async def retrieve(self, query: str) -> str:
            assert query == "洗手间 在哪"
            return "- 洗手间在一楼东侧"

        def health(self) -> dict[str, object]:
            return {"enabled": True, "backend": "vector"}

    monkeypatch.setattr("askme.memory.bridge.MemoryBridge", FakeBridge)

    cli.main(["memory", "search", "洗手间", "在哪"])

    output = capsys.readouterr().out
    assert "- 洗手间在一楼东侧" in output


def test_cli_runtime_mic_calibration_exits_nonzero_when_degraded(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_run_mic_calibration",
        lambda **_kwargs: {
            "status": "degraded",
            "target": "runtime-mic-calibration",
            "summary": {},
            "warnings": ["observed_peak_below_noise_gate:5<80"],
            "errors": [],
        },
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["runtime", "mic-calibration", "--duration", "0"])

    assert exc.value.code == 1


def test_cli_runtime_s100p_readiness_bundle_forwards_field_args(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_bundle(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "ok", "target": "s100p-readiness-bundle", "steps": []}

    monkeypatch.setattr(cli, "_run_s100p_readiness_bundle", fake_bundle)

    cli.main(
        [
            "runtime",
            "s100p-readiness-bundle",
            "--health-url",
            "http://runtime.local:8765/",
            "--change-event-file",
            "events.jsonl",
            "--journal-since",
            "10 minutes ago",
            "--skip-health",
            "--live-tts-room-loop",
            "--guard-min-seconds",
            "2.5",
            "--with-room-loop",
            "--room-loop-trials",
            "4",
            "--room-loop-text",
            "ping sunrise",
            "--room-loop-expect-prefix",
            "ping",
            "--require-cloud-asr",
        ]
    )

    assert seen["health_url"] == "http://runtime.local:8765/"
    assert seen["change_event_file"] == "events.jsonl"
    assert seen["journal_since"] == "10 minutes ago"
    assert seen["skip_health"] is True
    assert seen["live_tts_room_loop"] is True
    assert seen["guard_min_seconds"] == 2.5
    assert seen["include_room_loop"] is True
    assert seen["room_loop_trials"] == 4
    assert seen["room_loop_text"] == "ping sunrise"
    assert seen["room_loop_expect_prefix"] == "ping"
    assert seen["require_cloud_asr"] is True


def test_cli_agent_send_uses_server(monkeypatch, capsys) -> None:
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

    cli.main(["agent", "send", "hello", "--server", "http://runtime"])

    assert capsys.readouterr().out.strip() == "server:hello"


def test_cli_agent_send_falls_back_to_local(monkeypatch, capsys) -> None:
    def _fail(message: str, server: str, *, speak: bool = False) -> dict[str, str]:
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
