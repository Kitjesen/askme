"""Runtime CLI command handlers extracted from askme.cli."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

from askme.cli.audit_cmd import (
    _emit_unified_audit_events_payload,
    _emit_unified_audit_review_payload,
)
from askme.cli.field import (
    _emit_field_deployed_smoke_payload,
    _emit_field_device_trust_payload,
    _emit_field_disposition_smoke_payload,
    _emit_field_ingest_bridge_payload,
    _emit_field_ingest_file_payload,
    _emit_field_ingest_smoke_payload,
    _emit_field_live_demo_payload,
    _emit_field_notification_preflight_payload,
    _emit_field_notification_smoke_payload,
    _emit_field_operations_eval_payload,
    _emit_field_readiness_payload,
    _emit_field_sign_device_payload,
    _emit_field_site_env_template_payload,
    _emit_field_smoke_suite_payload,
    _emit_field_voice_smoke_payload,
    _watch_field_ingest_bridge,
)
from askme.cli.field_audit import (
    _emit_field_audit_anchor_payload,
    _emit_field_audit_delivery_retry_payload,
    _emit_field_audit_integrity_payload,
    _emit_field_audit_retry_status_payload,
)
from askme.cli.utils import (
    _acquire_field_audit_retry_lock,
    _append_field_audit_retry_queue,
    _emit_payload,
    _emit_runtime_blueprints_summary,
    _field_action_audit_config,
    _field_signed_payload_text,
    _get_json,
    _load_field_ingest_events,
    _load_local_capabilities,
    _normalise_server_url,
    _parse_csv_ints,
    _post_json,
    _post_json_with_retries,
    _read_field_audit_retry_lock,
    _resolve_field_action_audit_hmac_secret,
    _resolve_field_device_secrets,
    _resolve_field_device_signing_secret,
    _resolve_runtime_flags,
    _runtime_blueprints_payload,
    _single_device_id,
    _start_field_smoke_server,
    _start_local_webhook_collector,
    _write_field_smoke_events,
)
from askme.cli.voice import (
    _emit_mic_calibration_payload,
    _emit_s100p_readiness_bundle_payload,
    _emit_sunrise_audio_doctor_payload,
    _emit_sunrise_voice_readiness_payload,
    _emit_voice_health_payload,
)

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_URL = "http://127.0.0.1:8765"


# ---------------------------------------------------------------------------
# 基本运行时函数
# ---------------------------------------------------------------------------


def _run_interactive_runtime(*, voice_mode: bool, robot_mode: bool) -> None:
    from askme.main import run_app

    asyncio.run(run_app(voice_mode=voice_mode, robot_mode=robot_mode))


def _run_terminal_tui(*, robot_mode: bool) -> None:
    from askme.tui import run_terminal_ui

    asyncio.run(run_terminal_ui(robot_mode=robot_mode))


def _run_mcp_server(*, transport: str, host: str, port: int) -> None:
    from askme.mcp.server import mcp

    if transport in {"sse", "streamable-http"}:
        mcp.settings.host = host
        mcp.settings.port = int(port)
        mcp.run(transport=transport)
        return
    mcp.run()


def _looks_like_mcp_request(raw_args: list[str]) -> bool:
    return any(arg in {"--transport", "--host", "--port"} for arg in raw_args)


def _run_dialogue_smoke(args: argparse.Namespace) -> dict[str, Any]:
    from askme.runtime.diagnostics.dialogue_smoke import run_dialogue_smoke_sync

    return run_dialogue_smoke_sync(
        message=args.message,
        memory_text=args.memory_text,
        memory_query=args.memory_query,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        token=args.token,
        chat_timeout_s=args.chat_timeout,
        memory_timeout_s=args.memory_timeout,
        vector_min_similarity=args.vector_min_similarity,
        fake_llm=bool(args.fake_llm),
        require_reply_token=not bool(args.allow_reply_without_token),
    )


def _run_dialogue_burst(args: argparse.Namespace) -> dict[str, Any]:
    from askme.runtime.diagnostics.dialogue_smoke import run_dialogue_burst_sync

    return run_dialogue_burst_sync(
        fake_runs=args.fake_runs,
        real_runs=args.real_runs,
        output_dir=args.output_dir,
        token_prefix=args.token_prefix,
        chat_timeout_s=args.chat_timeout,
        memory_timeout_s=args.memory_timeout,
        vector_min_similarity=args.vector_min_similarity,
        allow_reply_without_token=bool(args.allow_reply_without_token),
    )


# ---------------------------------------------------------------------------
# 主调度器
# ---------------------------------------------------------------------------


def _handle_runtime_command(args: argparse.Namespace) -> None:
    # Deferred import through cli module for monkeypatch compatibility
    from askme import cli as _cli_mod

    # Resolve _run_* names through cli module so tests can monkeypatch
    _run_field_operations_eval = _cli_mod._run_field_operations_eval
    _run_field_ingest_file = _cli_mod._run_field_ingest_file
    _run_field_ingest_bridge = _cli_mod._run_field_ingest_bridge
    _run_field_sign_device_payload = _cli_mod._run_field_sign_device_payload
    _run_field_ingest_smoke = _cli_mod._run_field_ingest_smoke
    _run_field_voice_smoke = _cli_mod._run_field_voice_smoke
    _run_field_notification_smoke = _cli_mod._run_field_notification_smoke
    _run_field_notification_preflight = _cli_mod._run_field_notification_preflight
    _run_field_disposition_smoke = _cli_mod._run_field_disposition_smoke
    _run_field_smoke_suite = _cli_mod._run_field_smoke_suite
    _run_field_deployed_smoke = _cli_mod._run_field_deployed_smoke
    _run_field_readiness = _cli_mod._run_field_readiness
    _run_field_device_trust = _cli_mod._run_field_device_trust
    _run_field_site_env_template = _cli_mod._run_field_site_env_template
    _run_field_live_demo = _cli_mod._run_field_live_demo
    _run_field_audit_integrity = _cli_mod._run_field_audit_integrity
    _run_field_audit_anchor = _cli_mod._run_field_audit_anchor
    _run_field_audit_delivery_retry = _cli_mod._run_field_audit_delivery_retry
    _run_field_audit_retry_status = _cli_mod._run_field_audit_retry_status
    _run_unified_audit_events = _cli_mod._run_unified_audit_events
    _run_unified_audit_review = _cli_mod._run_unified_audit_review
    _run_voice_health_check = _cli_mod._run_voice_health_check
    _run_mic_calibration = _cli_mod._run_mic_calibration
    _run_sunrise_audio_doctor = _cli_mod._run_sunrise_audio_doctor
    _run_sunrise_voice_readiness = _cli_mod._run_sunrise_voice_readiness
    _run_s100p_readiness_bundle = _cli_mod._run_s100p_readiness_bundle

    if args.runtime_command is None:
        runtime_parser = getattr(args, "_runtime_parser", None)
        if runtime_parser is not None:
            runtime_parser.print_help()
            return

    if args.runtime_command == "run":
        voice_mode, robot_mode = _resolve_runtime_flags(args)
        _run_interactive_runtime(voice_mode=voice_mode, robot_mode=robot_mode)
        return

    if args.runtime_command == "blueprints":
        payload = _runtime_blueprints_payload(
            name=args.name,
            customer_visible=True if args.customer_visible else None,
            delivery_package=args.delivery_package,
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_runtime_blueprints_summary(payload)
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

    if args.runtime_command == "dialogue-smoke":
        from askme.runtime.diagnostics.dialogue_smoke import (
            print_dialogue_smoke_summary,
        )

        payload = _run_dialogue_smoke(args)
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            print_dialogue_smoke_summary(payload)
        if payload.get("status") != "passed":
            raise SystemExit(1)
        return

    if args.runtime_command == "dialogue-burst":
        from askme.runtime.diagnostics.dialogue_smoke import (
            print_dialogue_burst_summary,
        )

        payload = _run_dialogue_burst(args)
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            print_dialogue_burst_summary(payload)
        if payload.get("status") != "passed":
            raise SystemExit(1)
        return

    if args.runtime_command == "voice-health":
        payload = _run_voice_health_check(live=args.live)
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_voice_health_payload(payload)
        if payload.get("status") != "ok":
            raise SystemExit(1)
        return

    if args.runtime_command == "audio-devices":
        from askme.voice.diagnostics.audio_devices import (
            print_audio_devices_summary,
            query_audio_devices,
        )

        payload = query_audio_devices()
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            print_audio_devices_summary(payload)
        if payload.get("status") != "ok":
            raise SystemExit(1)
        return

    if args.runtime_command == "audio-loopback":
        from askme.voice.diagnostics.audio_devices import (
            print_audio_loopback_summary,
            run_audio_loopback,
        )

        payload = run_audio_loopback(
            input_device=args.input_device,
            output_device=args.output_device,
            sample_rate=args.sample_rate,
            record_seconds=args.record_seconds,
            tone_seconds=args.tone_seconds,
            frequency_hz=args.frequency_hz,
            output_gain=args.output_gain,
            min_capture_peak=args.min_capture_peak,
            wav_out=args.wav_out,
            play_recording=args.play_recording,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            print_audio_loopback_summary(payload)
        if payload.get("status") != "ok":
            raise SystemExit(1)
        return

    if args.runtime_command == "audio-beep-loopback":
        from askme.voice.diagnostics.audio_devices import (
            print_windows_beep_loopback_summary,
            run_windows_beep_loopback,
        )

        payload = run_windows_beep_loopback(
            input_device=args.input_device,
            sample_rate=args.sample_rate,
            record_seconds=args.record_seconds,
            tone_seconds=args.tone_seconds,
            frequency_hz=args.frequency_hz,
            min_capture_peak=args.min_capture_peak,
            wav_out=args.wav_out,
            play_recording=args.play_recording,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            print_windows_beep_loopback_summary(payload)
        if payload.get("status") != "ok":
            raise SystemExit(1)
        return

    if args.runtime_command == "audio-route-scan":
        from askme.voice.diagnostics.audio_devices import (
            print_audio_route_scan_summary,
            run_audio_route_scan,
        )

        payload = run_audio_route_scan(
            input_devices=_parse_csv_ints(args.input_devices),
            output_devices=_parse_csv_ints(args.output_devices),
            sample_rates=_parse_csv_ints(args.sample_rates) or [48000, 44100],
            record_seconds=args.record_seconds,
            tone_seconds=args.tone_seconds,
            frequency_hz=args.frequency_hz,
            output_gain=args.output_gain,
            min_capture_peak=args.min_capture_peak,
            include_all_pairs=args.all_pairs,
            max_routes=args.max_routes,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            print_audio_route_scan_summary(payload)
        if payload.get("status") != "ok":
            raise SystemExit(1)
        return

    if args.runtime_command == "voice-online-smoke":
        from askme.voice.diagnostics.online_smoke import (
            print_voice_online_smoke_summary,
            run_voice_online_smoke_sync,
        )

        payload = run_voice_online_smoke_sync(
            text=args.text,
            silence_seconds=args.silence_seconds,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            print_voice_online_smoke_summary(payload)
        if payload.get("status") != "ok":
            raise SystemExit(1)
        return

    if args.runtime_command == "mic-calibration":
        payload = _run_mic_calibration(
            server=args.server,
            duration_s=args.duration,
            interval_s=args.interval,
            min_signal_peak=args.min_signal_peak,
        )
        if args.json_out:
            from askme.voice.diagnostics.mic_calibration import write_mic_calibration_json

            write_mic_calibration_json(payload, args.json_out)
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_mic_calibration_payload(payload)
        if payload.get("status") != "ok":
            raise SystemExit(1)
        return

    if args.runtime_command == "sunrise-audio-doctor":
        payload = _run_sunrise_audio_doctor(
            include_command_probes=not args.skip_command_probes,
            include_output_probe=not args.skip_output_probe,
            guard_min_seconds=args.guard_min_seconds,
        )
        if args.json_out:
            path = Path(args.json_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_sunrise_audio_doctor_payload(payload)
        if payload.get("status") != "ok":
            raise SystemExit(1)
        return

    if args.runtime_command == "sunrise-voice-readiness":
        payload = _run_sunrise_voice_readiness(
            guard_min_seconds=args.guard_min_seconds,
            include_room_loop=args.with_room_loop,
            room_loop_text=args.room_loop_text or None,
            room_loop_expect_prefix=args.room_loop_expect_prefix or None,
            room_loop_trials=args.room_loop_trials,
            live_tts_room_loop=args.live_tts_room_loop,
            room_loop_asr=args.room_loop_asr,
            require_cloud_asr=args.require_cloud_asr,
        )
        if args.json_out:
            path = Path(args.json_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_sunrise_voice_readiness_payload(payload)
        if payload.get("status") != "ok":
            raise SystemExit(1)
        return

    if args.runtime_command == "s100p-readiness-bundle":
        payload = _run_s100p_readiness_bundle(
            output_dir=args.output_dir or None,
            field=args.field,
            guard_min_seconds=args.guard_min_seconds,
            include_room_loop=args.with_room_loop,
            room_loop_text=args.room_loop_text or None,
            room_loop_expect_prefix=args.room_loop_expect_prefix or None,
            room_loop_trials=args.room_loop_trials,
            live_tts_room_loop=args.live_tts_room_loop,
            require_cloud_asr=args.require_cloud_asr,
            health_url=args.health_url,
            change_event_file=args.change_event_file,
            journal_since=args.journal_since,
            skip_health=args.skip_health,
            skip_service_log=args.skip_service_log,
            command_timeout=args.command_timeout,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_s100p_readiness_bundle_payload(payload)
        if payload.get("status") != "ok":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-eval":
        payload = _run_field_operations_eval(output=args.output)
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_operations_eval_payload(payload)
        if payload.get("status") != "passed":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-ingest-file":
        payload = _run_field_ingest_file(
            source=args.source,
            server=args.server,
            dry_run=args.dry_run,
            limit=args.limit,
            device_secrets=_resolve_field_device_secrets(
                args.device_secret,
                site_profile=args.site_profile,
            ),
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_ingest_file_payload(payload)
        if payload.get("status") == "failed":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-ingest-bridge":
        if args.watch:
            _watch_field_ingest_bridge(
                source=args.source,
                server=args.server,
                state_path=args.state_path or None,
                interval_s=args.interval,
                dry_run=args.dry_run,
                limit=args.limit,
                timeout_s=args.timeout,
                device_secrets=_resolve_field_device_secrets(
                    args.device_secret,
                    site_profile=args.site_profile,
                ),
            )
            return
        payload = _run_field_ingest_bridge(
            source=args.source,
            server=args.server,
            state_path=args.state_path or None,
            dry_run=args.dry_run,
            limit=args.limit,
            timeout_s=args.timeout,
            device_secrets=_resolve_field_device_secrets(
                args.device_secret,
                site_profile=args.site_profile,
            ),
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_ingest_bridge_payload(payload)
        if payload.get("status") == "failed":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-sign-device-payload":
        payload = _run_field_sign_device_payload(
            source=args.source,
            output=args.output,
            device_id=args.device_id,
            secret=args.secret,
            secret_env=args.secret_env,
            timestamp=args.timestamp,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_sign_device_payload(payload)
        if payload.get("status") != "signed":
            raise SystemExit(2)
        return

    if args.runtime_command == "field-ingest-smoke":
        payload = _run_field_ingest_smoke(
            output_dir=args.output_dir,
            server=args.server or "",
            audit_hmac_secret=args.audit_hmac_secret,
            require_device_signatures=args.require_device_signatures,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_ingest_smoke_payload(payload)
        if payload.get("status") != "passed":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-voice-smoke":
        payload = _run_field_voice_smoke(
            output_dir=args.output_dir,
            server=args.server or "",
            scenario=args.scenario,
            live_tts=bool(args.live_tts),
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_voice_smoke_payload(payload)
        if payload.get("status") != "passed":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-notification-smoke":
        payload = _run_field_notification_smoke(
            output_dir=args.output_dir,
            server=args.server or "",
            groups=args.groups,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_notification_smoke_payload(payload)
        if payload.get("status") != "passed":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-notification-preflight":
        payload = _run_field_notification_preflight(
            server=args.server or "",
            groups=args.groups,
            require_secret=not args.allow_unsigned,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_notification_preflight_payload(payload)
        if payload.get("ready") is not True:
            raise SystemExit(1)
        return

    if args.runtime_command == "field-disposition-smoke":
        payload = _run_field_disposition_smoke(
            output_dir=args.output_dir,
            server=args.server or "",
            audit_hmac_secret=args.audit_hmac_secret,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_disposition_smoke_payload(payload)
        if payload.get("status") != "passed":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-smoke-suite":
        payload = _run_field_smoke_suite(
            output_dir=args.output_dir,
            voice_scenario=args.voice_scenario,
            groups=args.groups,
            live_tts=bool(args.live_tts),
            audit_hmac_secret=args.audit_hmac_secret,
            audit_webhook_url=args.audit_webhook_url,
            audit_webhook_retries=args.audit_webhook_retries,
            include_audit_anchor=not args.skip_audit_anchor,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_smoke_suite_payload(payload)
        if payload.get("status") != "passed":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-deployed-smoke":
        payload = _run_field_deployed_smoke(
            server=args.server,
            output_dir=args.output_dir,
            voice_scenario=args.voice_scenario,
            groups=args.groups,
            require_notification_ready=not args.allow_notification_not_ready,
            require_device_signatures=args.require_device_signatures,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_deployed_smoke_payload(payload)
        if payload.get("status") != "passed":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-readiness":
        payload = _run_field_readiness(
            server=args.server or "",
            archive_path=args.archive_path,
            scenario_report=args.scenario_report,
            smoke_report=args.smoke_report,
            voice_smoke_report=args.voice_smoke_report,
            notification_smoke_report=args.notification_smoke_report,
            site_profile=args.site_profile,
            check_site_env=args.check_site_env,
            audit_hmac_secret=args.audit_hmac_secret,
            review_path=args.review_path,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_readiness_payload(payload)
        if payload.get("status") == "blocked":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-device-trust":
        payload = _run_field_device_trust(
            site_profile=args.site_profile,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_device_trust_payload(
                payload,
                show_commands=bool(args.show_commands),
            )
        if payload.get("status") == "invalid_profile":
            raise SystemExit(2)
        return

    if args.runtime_command == "field-site-env-template":
        payload = _run_field_site_env_template(
            site_profile=args.site_profile,
            output=args.output or "",
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_site_env_template_payload(payload)
        if payload.get("status") != "ok":
            raise SystemExit(2)
        return

    if args.runtime_command == "audit-events":
        payload = _run_unified_audit_events(
            limit=args.limit,
            source=args.source,
            operator_id=args.operator_id,
            action=args.action,
            outcome=args.outcome,
            q=args.q,
            since=args.since,
            until=args.until,
            skill_audit=args.skill_audit,
            field_action_audit=args.field_action_audit,
            field_event_archive=args.field_event_archive,
            runtime_audit=args.runtime_audit,
            review_path=args.review_path,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_unified_audit_events_payload(
                payload,
                review_queue_only=bool(args.review_queue_only),
            )
        return

    if args.runtime_command == "audit-review":
        payload = _run_unified_audit_review(
            record_id=args.record_id,
            reviewer_id=args.reviewer_id,
            decision=args.decision,
            note=args.note,
            skill_audit=args.skill_audit,
            field_action_audit=args.field_action_audit,
            field_event_archive=args.field_event_archive,
            runtime_audit=args.runtime_audit,
            review_path=args.review_path,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_unified_audit_review_payload(payload)
        if payload.get("ok") is not True:
            raise SystemExit(2)
        return

    if args.runtime_command == "field-live-demo":
        payload = _run_field_live_demo(
            output_dir=args.output_dir,
            site_profile=args.site_profile,
            server=args.server or "",
            timeout_s=args.timeout,
            scenario_file=args.scenario_file or "",
            refresh_scenario_timestamps=args.refresh_scenario_timestamps,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_live_demo_payload(payload)
        if payload.get("status") != "passed":
            raise SystemExit(1)
        return

    if args.runtime_command == "field-audit-integrity":
        payload = _run_field_audit_integrity(
            server=args.server or "",
            archive_path=args.archive_path,
            audit_path=args.audit_path,
            hmac_secret=args.hmac_secret,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_audit_integrity_payload(payload)
        if payload.get("enabled") is not False and payload.get("valid") is not True:
            raise SystemExit(2)
        return

    if args.runtime_command == "field-audit-anchor":
        payload = _run_field_audit_anchor(
            server=args.server or "",
            archive_path=args.archive_path,
            audit_path=args.audit_path,
            hmac_secret=args.hmac_secret,
            output=args.output,
            webhook_url=args.webhook_url,
            webhook_retries=args.webhook_retries,
            retry_queue=args.retry_queue,
            require_valid=not args.allow_invalid,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_audit_anchor_payload(payload)
        if payload.get("status") == "blocked":
            raise SystemExit(2)
        if payload.get("status") == "delivery_failed":
            raise SystemExit(3)
        return

    if args.runtime_command == "field-audit-retry-delivery":
        payload = _run_field_audit_delivery_retry(
            queue=args.queue,
            webhook_retries=args.webhook_retries,
            lock_timeout_s=args.lock_timeout,
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_audit_delivery_retry_payload(payload)
        if payload.get("status") == "failed":
            raise SystemExit(3)
        if payload.get("status") == "locked":
            raise SystemExit(4)
        return

    if args.runtime_command == "field-audit-retry-status":
        payload = _run_field_audit_retry_status(queue=args.queue)
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_audit_retry_status_payload(payload)
        if args.fail_on_pending and payload.get("pending", 0):
            raise SystemExit(3)
        return

    raise SystemExit(f"Unknown runtime command: {args.runtime_command}")
