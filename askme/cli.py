"""Structured CLI for askme with dimos-style command groups."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
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
    runtime_parser.set_defaults(_runtime_parser=runtime_parser)
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

    runtime_voice_health = runtime_subparsers.add_parser(
        "voice-health",
        help="Run an offline voice pipeline health check",
    )
    runtime_voice_health.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_voice_health.add_argument(
        "--live",
        action="store_true",
        help="Mark the check as live preflight without opening audio devices",
    )
    runtime_mic_calibration = runtime_subparsers.add_parser(
        "mic-calibration",
        help="Poll a running runtime and summarize microphone gate levels",
    )
    runtime_mic_calibration.add_argument(
        "--server",
        default=DEFAULT_RUNTIME_URL,
        help=f"Base URL for the running runtime (default: {DEFAULT_RUNTIME_URL})",
    )
    runtime_mic_calibration.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Seconds to sample runtime /health input diagnostics",
    )
    runtime_mic_calibration.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between health samples",
    )
    runtime_mic_calibration.add_argument(
        "--min-signal-peak",
        type=int,
        default=100,
        help="Minimum peak considered a useful speech/input signal",
    )
    runtime_mic_calibration.add_argument(
        "--json-out",
        default="",
        help="Also write the calibration JSON to this path",
    )
    runtime_mic_calibration.add_argument("--json", action="store_true", help="Print raw JSON")

    runtime_sunrise_audio_doctor = runtime_subparsers.add_parser(
        "sunrise-audio-doctor",
        help="Run the Sunrise MCP01 USB audio diagnostic",
    )
    runtime_sunrise_audio_doctor.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_sunrise_audio_doctor.add_argument(
        "--json-out",
        default="",
        help="Also write the diagnostic JSON to this path",
    )
    runtime_sunrise_audio_doctor.add_argument(
        "--guard-min-seconds",
        type=float,
        default=1.5,
        help="Minimum sacrificial lead-in+cushion before real speech",
    )
    runtime_sunrise_audio_doctor.add_argument(
        "--skip-command-probes",
        action="store_true",
        help="Skip lsusb and /proc/asound probes",
    )
    runtime_sunrise_audio_doctor.add_argument(
        "--skip-output-probe",
        action="store_true",
        help="Skip non-playing TTSEngine USB output-shape probe",
    )
    runtime_sunrise_voice_readiness = runtime_subparsers.add_parser(
        "sunrise-voice-readiness",
        help="Run the aggregate Sunrise voice readiness gate",
    )
    runtime_sunrise_voice_readiness.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_sunrise_voice_readiness.add_argument(
        "--json-out",
        default="",
        help="Also write the readiness JSON to this path",
    )
    runtime_sunrise_voice_readiness.add_argument(
        "--guard-min-seconds",
        type=float,
        default=1.5,
        help="Minimum sacrificial lead-in+cushion before real speech",
    )
    runtime_sunrise_voice_readiness.add_argument(
        "--with-room-loop",
        action="store_true",
        help="Also run acoustic room loopback",
    )
    runtime_sunrise_voice_readiness.add_argument("--room-loop-trials", type=int, default=3)
    runtime_sunrise_voice_readiness.add_argument("--room-loop-text", default="")
    runtime_sunrise_voice_readiness.add_argument("--room-loop-expect-prefix", default="")
    runtime_sunrise_voice_readiness.add_argument("--live-tts-room-loop", action="store_true")
    runtime_sunrise_voice_readiness.add_argument(
        "--room-loop-asr",
        choices=("auto", "local", "cloud", "both"),
        default="auto",
        help="ASR backend used by the acoustic room-loop transcript gate",
    )
    runtime_sunrise_voice_readiness.add_argument(
        "--require-cloud-asr",
        action="store_true",
        help="Fail readiness unless Cloud ASR is enabled and configured",
    )
    runtime_s100p_readiness_bundle = runtime_subparsers.add_parser(
        "s100p-readiness-bundle",
        help="Collect S100P field-readiness evidence into a bundle",
    )
    runtime_s100p_readiness_bundle.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_s100p_readiness_bundle.add_argument(
        "--output-dir",
        default="",
        help="Bundle output directory (default: artifacts/s100p/<timestamp>-<hostname>)",
    )
    runtime_s100p_readiness_bundle.add_argument(
        "--field",
        action="store_true",
        help="Require S100P field evidence, room loopback, and Cloud ASR gates",
    )
    runtime_s100p_readiness_bundle.add_argument(
        "--guard-min-seconds",
        type=float,
        default=1.5,
        help="Minimum sacrificial lead-in+cushion before real speech",
    )
    runtime_s100p_readiness_bundle.add_argument(
        "--with-room-loop",
        action="store_true",
        help="Also run acoustic room loopback",
    )
    runtime_s100p_readiness_bundle.add_argument("--room-loop-trials", type=int, default=3)
    runtime_s100p_readiness_bundle.add_argument("--room-loop-text", default="")
    runtime_s100p_readiness_bundle.add_argument("--room-loop-expect-prefix", default="")
    runtime_s100p_readiness_bundle.add_argument("--live-tts-room-loop", action="store_true")
    runtime_s100p_readiness_bundle.add_argument(
        "--require-cloud-asr",
        action="store_true",
        help="Fail readiness unless Cloud ASR is enabled and configured",
    )
    runtime_s100p_readiness_bundle.add_argument(
        "--health-url",
        default=DEFAULT_RUNTIME_URL,
        help=f"Runtime health base URL (default: {DEFAULT_RUNTIME_URL})",
    )
    runtime_s100p_readiness_bundle.add_argument(
        "--change-event-file",
        default="/tmp/askme_events.jsonl",
        help="Change-event JSONL path (default: /tmp/askme_events.jsonl)",
    )
    runtime_s100p_readiness_bundle.add_argument(
        "--journal-since",
        default="30 minutes ago",
        help='journalctl --since value (default: "30 minutes ago")',
    )
    runtime_s100p_readiness_bundle.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip runtime health endpoint collection",
    )
    runtime_s100p_readiness_bundle.add_argument(
        "--skip-service-log",
        action="store_true",
        help="Skip systemd log collection",
    )
    runtime_s100p_readiness_bundle.add_argument(
        "--command-timeout",
        type=float,
        default=180.0,
        help="Per-command timeout seconds (default: 180)",
    )
    runtime_field_eval = runtime_subparsers.add_parser(
        "field-eval",
        help="Run field-operation product scenarios and write a readiness report",
    )
    runtime_field_eval.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_eval.add_argument(
        "--output",
        default="artifacts/field_operations/scenario-evaluation.json",
        help="Report path (default: artifacts/field_operations/scenario-evaluation.json)",
    )
    runtime_field_ingest_file = runtime_subparsers.add_parser(
        "field-ingest-file",
        help="Forward JSON/JSONL camera, sensor, or robot events to /api/field/ingest",
    )
    runtime_field_ingest_file.add_argument("source", help="JSON object/array or JSONL file")
    runtime_field_ingest_file.add_argument(
        "--server",
        default=DEFAULT_RUNTIME_URL,
        help=f"Runtime base URL (default: {DEFAULT_RUNTIME_URL})",
    )
    runtime_field_ingest_file.add_argument(
        "--dry-run",
        action="store_true",
        help="Normalize events locally without posting to the runtime",
    )
    runtime_field_ingest_file.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of events to process (0 means all)",
    )
    runtime_field_ingest_file.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_ingest_bridge = runtime_subparsers.add_parser(
        "field-ingest-bridge",
        help="Incrementally bridge device JSON/JSONL events to /api/field/ingest",
    )
    runtime_field_ingest_bridge.add_argument("source", help="JSON object/array or append-only JSONL/NDJSON file")
    runtime_field_ingest_bridge.add_argument(
        "--server",
        default=DEFAULT_RUNTIME_URL,
        help=f"Runtime base URL (default: {DEFAULT_RUNTIME_URL})",
    )
    runtime_field_ingest_bridge.add_argument("--state-path", default="", help="Offset/fingerprint state file")
    runtime_field_ingest_bridge.add_argument("--watch", action="store_true", help="Keep polling for new events")
    runtime_field_ingest_bridge.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Watch polling interval seconds",
    )
    runtime_field_ingest_bridge.add_argument(
        "--dry-run",
        action="store_true",
        help="Normalize events locally without posting to the runtime",
    )
    runtime_field_ingest_bridge.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of events per pass (0 means all)",
    )
    runtime_field_ingest_bridge.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="HTTP timeout seconds",
    )
    runtime_field_ingest_bridge.add_argument(
        "--device-secret",
        action="append",
        default=[],
        metavar="DEVICE_ID=SECRET",
        help="Sign bridged events for a registered device id, source, or * wildcard",
    )
    runtime_field_ingest_bridge.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_ingest_smoke = runtime_subparsers.add_parser(
        "field-ingest-smoke",
        help="Run an end-to-end smoke test for device JSONL -> bridge -> /api/field/ingest",
    )
    runtime_field_ingest_smoke.add_argument(
        "--server",
        default="",
        help="Existing Askme runtime base URL. If omitted, a temporary local server is started.",
    )
    runtime_field_ingest_smoke.add_argument(
        "--output-dir",
        default="artifacts/field_operations/smoke",
        help="Directory for generated sample JSONL, bridge state, archive, and report",
    )
    runtime_field_ingest_smoke.add_argument(
        "--audit-hmac-secret",
        default="",
        help="HMAC secret used by the temporary local field action audit writer",
    )
    runtime_field_ingest_smoke.add_argument(
        "--require-device-signatures",
        action="store_true",
        help="Require signed sample device events and trusted-device admission",
    )
    runtime_field_ingest_smoke.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON",
    )
    runtime_field_voice_smoke = runtime_subparsers.add_parser(
        "field-voice-smoke",
        help="Trigger a field incident and verify voice_directive -> TTS playback wiring",
    )
    runtime_field_voice_smoke.add_argument(
        "--server",
        default="",
        help="Existing Askme runtime base URL. If omitted, a temporary local server is started.",
    )
    runtime_field_voice_smoke.add_argument(
        "--output-dir",
        default="artifacts/field_operations/smoke",
        help="Directory for generated report and local archive",
    )
    runtime_field_voice_smoke.add_argument(
        "--scenario",
        choices=("fire", "joint_fault", "illegal_parking"),
        default="fire",
        help="Field event scenario to trigger",
    )
    runtime_field_voice_smoke.add_argument(
        "--live-tts",
        action="store_true",
        help="Use the configured real TTSEngine in the temporary local server",
    )
    runtime_field_voice_smoke.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_notification_smoke = runtime_subparsers.add_parser(
        "field-notification-smoke",
        help="Verify DingTalk notification delivery wiring for field responder groups",
    )
    runtime_field_notification_smoke.add_argument(
        "--server",
        default="",
        help="Existing Askme runtime base URL. If omitted, a temporary server and webhook collector are started.",
    )
    runtime_field_notification_smoke.add_argument(
        "--output-dir",
        default="artifacts/field_operations/smoke",
        help="Directory for generated notification smoke report",
    )
    runtime_field_notification_smoke.add_argument(
        "--groups",
        default="security,cleaning,operations",
        help="Comma-separated responder groups to test",
    )
    runtime_field_notification_smoke.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_notification_preflight = runtime_subparsers.add_parser(
        "field-notification-preflight",
        help="Check real DingTalk responder webhook/secret configuration without sending",
    )
    runtime_field_notification_preflight.add_argument(
        "--server",
        default="",
        help="Existing Askme runtime base URL. If omitted, local config/env is checked.",
    )
    runtime_field_notification_preflight.add_argument(
        "--groups",
        default="security,cleaning,operations",
        help="Comma-separated responder groups to check",
    )
    runtime_field_notification_preflight.add_argument(
        "--allow-unsigned",
        action="store_true",
        help="Do not require DingTalk signing secrets during preflight",
    )
    runtime_field_notification_preflight.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_disposition_smoke = runtime_subparsers.add_parser(
        "field-disposition-smoke",
        help="Verify P0 acknowledge -> close approval -> report timeline workflow",
    )
    runtime_field_disposition_smoke.add_argument(
        "--server",
        default="",
        help="Existing Askme runtime base URL. If omitted, a temporary local server is started.",
    )
    runtime_field_disposition_smoke.add_argument(
        "--output-dir",
        default="artifacts/field_operations/smoke",
        help="Directory for generated disposition smoke report and local archive",
    )
    runtime_field_disposition_smoke.add_argument(
        "--audit-hmac-secret",
        default="",
        help="HMAC secret used by the temporary local field action audit writer",
    )
    runtime_field_disposition_smoke.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_smoke_suite = runtime_subparsers.add_parser(
        "field-smoke-suite",
        help="Run field scenario, ingest, voice, notification, and readiness smoke checks",
    )
    runtime_field_smoke_suite.add_argument(
        "--output-dir",
        default="artifacts/field_operations/smoke",
        help="Directory for generated field smoke reports",
    )
    runtime_field_smoke_suite.add_argument(
        "--voice-scenario",
        choices=("fire", "joint_fault", "illegal_parking"),
        default="fire",
        help="Voice smoke scenario",
    )
    runtime_field_smoke_suite.add_argument(
        "--groups",
        default="security,cleaning,operations",
        help="Comma-separated responder groups for notification smoke",
    )
    runtime_field_smoke_suite.add_argument(
        "--live-tts",
        action="store_true",
        help="Use configured real TTSEngine for the voice smoke step",
    )
    runtime_field_smoke_suite.add_argument(
        "--audit-hmac-secret",
        default="",
        help="HMAC secret used when creating an audit checkpoint for the suite",
    )
    runtime_field_smoke_suite.add_argument(
        "--audit-webhook-url",
        default="",
        help="Optional external SIEM/WORM webhook URL for the suite audit checkpoint",
    )
    runtime_field_smoke_suite.add_argument(
        "--audit-webhook-retries",
        type=int,
        default=3,
        help="Webhook delivery attempts for the suite audit checkpoint",
    )
    runtime_field_smoke_suite.add_argument(
        "--skip-audit-anchor",
        action="store_true",
        help="Do not create the suite audit checkpoint artifact",
    )
    runtime_field_smoke_suite.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_deployed_smoke = runtime_subparsers.add_parser(
        "field-deployed-smoke",
        help="Run field smoke checks against an already running Askme deployment",
    )
    runtime_field_deployed_smoke.add_argument(
        "--server",
        default=DEFAULT_RUNTIME_URL,
        help=f"Deployed Askme runtime base URL (default: {DEFAULT_RUNTIME_URL})",
    )
    runtime_field_deployed_smoke.add_argument(
        "--output-dir",
        default="artifacts/field_operations/deployed-smoke",
        help="Directory for generated deployed smoke reports",
    )
    runtime_field_deployed_smoke.add_argument(
        "--voice-scenario",
        choices=("fire", "joint_fault", "illegal_parking"),
        default="fire",
        help="Voice smoke scenario",
    )
    runtime_field_deployed_smoke.add_argument(
        "--groups",
        default="security,cleaning,operations",
        help="Comma-separated responder groups to check/test",
    )
    runtime_field_deployed_smoke.add_argument(
        "--allow-notification-not-ready",
        action="store_true",
        help="Continue deployed smoke without real notification readiness",
    )
    runtime_field_deployed_smoke.add_argument(
        "--require-device-signatures",
        action="store_true",
        help="Require signed bridge events during deployed ingest smoke",
    )
    runtime_field_deployed_smoke.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_readiness = runtime_subparsers.add_parser(
        "field-readiness",
        help="Check field-operations deployment readiness gates",
    )
    runtime_field_readiness.add_argument(
        "--server",
        default="",
        help="Read readiness from a running Askme runtime instead of local files",
    )
    runtime_field_readiness.add_argument(
        "--archive-path",
        default="artifacts/field_operations/smoke/field-events.jsonl",
        help="Local field event archive path",
    )
    runtime_field_readiness.add_argument(
        "--scenario-report",
        default="artifacts/field_operations/scenario-evaluation.json",
        help="Local field scenario report path",
    )
    runtime_field_readiness.add_argument(
        "--smoke-report",
        default="artifacts/field_operations/smoke/field-ingest-smoke.json",
        help="Local field ingest smoke report path",
    )
    runtime_field_readiness.add_argument(
        "--voice-smoke-report",
        default="artifacts/field_operations/smoke/field-voice-smoke.json",
        help="Local field voice smoke report path",
    )
    runtime_field_readiness.add_argument(
        "--notification-smoke-report",
        default="artifacts/field_operations/smoke/field-notification-smoke.json",
        help="Local field DingTalk notification smoke report path",
    )
    runtime_field_readiness.add_argument(
        "--site-profile",
        default="deploy/site-profiles/park-demo.yaml",
        help="Field site profile YAML used for local readiness checks",
    )
    runtime_field_readiness.add_argument(
        "--check-site-env",
        action="store_true",
        help="Warn when site profile webhook/device secret environment variables are unset",
    )
    runtime_field_readiness.add_argument(
        "--audit-hmac-secret",
        default="",
        help="HMAC secret for local field action audit verification",
    )
    runtime_field_readiness.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_live_demo = runtime_subparsers.add_parser(
        "field-live-demo",
        help="Run customer field-operation scenarios through the real HTTP API",
    )
    runtime_field_live_demo.add_argument(
        "--output-dir",
        default="artifacts/field_operations/live-demo",
        help="Directory for live demo report artifacts",
    )
    runtime_field_live_demo.add_argument(
        "--site-profile",
        default="deploy/site-profiles/park-demo.yaml",
        help="Field site profile YAML used for the local in-process demo",
    )
    runtime_field_live_demo.add_argument(
        "--server",
        default="",
        help="Existing Askme runtime base URL. If omitted, an in-process HTTP app is used.",
    )
    runtime_field_live_demo.add_argument(
        "--scenario-file",
        default="",
        help="Replay customer/device scenario JSON instead of the built-in demo scenarios",
    )
    runtime_field_live_demo.add_argument(
        "--refresh-scenario-timestamps",
        action="store_true",
        help="Refresh observed_at in replayed scenarios for demo-only freshness gates",
    )
    runtime_field_live_demo.add_argument("--timeout", type=float, default=8.0)
    runtime_field_live_demo.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_audit_integrity = runtime_subparsers.add_parser(
        "field-audit-integrity",
        help="Verify field action audit hash/signature integrity",
    )
    runtime_field_audit_integrity.add_argument(
        "--server",
        default="",
        help="Read audit integrity from a running Askme runtime instead of local files",
    )
    runtime_field_audit_integrity.add_argument(
        "--archive-path",
        default="artifacts/field_ops/events.jsonl",
        help="Local field event archive path",
    )
    runtime_field_audit_integrity.add_argument(
        "--audit-path",
        default="artifacts/field_ops/field-action-audit.jsonl",
        help="Local field action audit JSONL path",
    )
    runtime_field_audit_integrity.add_argument(
        "--hmac-secret",
        default="",
        help="HMAC secret for local signature verification",
    )
    runtime_field_audit_integrity.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_audit_anchor = runtime_subparsers.add_parser(
        "field-audit-anchor",
        help="Write or deliver a field action audit checkpoint for external anchoring",
    )
    runtime_field_audit_anchor.add_argument(
        "--server",
        default="",
        help="Read audit integrity from a running Askme runtime instead of local files",
    )
    runtime_field_audit_anchor.add_argument(
        "--archive-path",
        default="artifacts/field_ops/events.jsonl",
        help="Local field event archive path",
    )
    runtime_field_audit_anchor.add_argument(
        "--audit-path",
        default="artifacts/field_ops/field-action-audit.jsonl",
        help="Local field action audit JSONL path",
    )
    runtime_field_audit_anchor.add_argument(
        "--hmac-secret",
        default="",
        help="HMAC secret for local signature verification",
    )
    runtime_field_audit_anchor.add_argument(
        "--output",
        default="artifacts/field_ops/audit-checkpoint.json",
        help="Checkpoint JSON output path",
    )
    runtime_field_audit_anchor.add_argument(
        "--webhook-url",
        default="",
        help="Optional external SIEM/WORM webhook URL to receive the checkpoint",
    )
    runtime_field_audit_anchor.add_argument(
        "--webhook-retries",
        type=int,
        default=3,
        help="Webhook delivery attempts before marking delivery failed",
    )
    runtime_field_audit_anchor.add_argument(
        "--retry-queue",
        default="artifacts/field_ops/audit-delivery-retry.jsonl",
        help="Append failed webhook delivery checkpoints to this JSONL retry queue",
    )
    runtime_field_audit_anchor.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Write the checkpoint even when integrity is invalid and exit zero",
    )
    runtime_field_audit_anchor.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_audit_retry = runtime_subparsers.add_parser(
        "field-audit-retry-delivery",
        help="Retry queued field audit checkpoint webhook deliveries",
    )
    runtime_field_audit_retry.add_argument(
        "--queue",
        default="artifacts/field_ops/audit-delivery-retry.jsonl",
        help="JSONL retry queue written by field-audit-anchor",
    )
    runtime_field_audit_retry.add_argument(
        "--webhook-retries",
        type=int,
        default=3,
        help="Webhook delivery attempts per queued checkpoint",
    )
    runtime_field_audit_retry.add_argument(
        "--lock-timeout",
        type=float,
        default=300.0,
        help="Seconds before an existing retry delivery lock is considered stale",
    )
    runtime_field_audit_retry.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_audit_retry_status = runtime_subparsers.add_parser(
        "field-audit-retry-status",
        help="Inspect queued field audit checkpoint webhook deliveries without sending",
    )
    runtime_field_audit_retry_status.add_argument(
        "--queue",
        default="artifacts/field_ops/audit-delivery-retry.jsonl",
        help="JSONL retry queue written by field-audit-anchor",
    )
    runtime_field_audit_retry_status.add_argument(
        "--fail-on-pending",
        action="store_true",
        help="Exit non-zero when queued deliveries remain",
    )
    runtime_field_audit_retry_status.add_argument("--json", action="store_true", help="Print raw JSON")

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
    agent_send.add_argument(
        "--speak",
        action="store_true",
        help="Play the assistant reply through the local configured TTS output",
    )
    agent_send.add_argument("--json", action="store_true", help="Print raw JSON")

    mission_parser = subparsers.add_parser(
        "mission",
        help="Draft and dry-run industrial inspection missions",
    )
    mission_subparsers = mission_parser.add_subparsers(dest="mission_command")

    mission_draft = mission_subparsers.add_parser(
        "draft",
        help="Draft a high-level mission from operator text",
    )
    mission_draft.add_argument("text", nargs="+", help="Mission request text")
    _add_mission_context_args(mission_draft)

    mission_run = mission_subparsers.add_parser(
        "run",
        help="Dry-run or submit a mission plan through the runtime arbiter",
    )
    mission_run.add_argument("source", nargs="+", help="Mission text or JSON/YAML plan path")
    mission_run.add_argument(
        "--submit",
        action="store_true",
        help="Request live submission; requires runtime.mission.submit_enabled",
    )
    mission_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run even when --submit is present",
    )
    mission_run.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm an operator-reviewed mission before live submission",
    )
    _add_mission_context_args(mission_run)

    mission_report = mission_subparsers.add_parser(
        "report",
        help="Build an inspection report shell for a mission",
    )
    mission_report.add_argument("mission_id", help="Mission id")
    mission_report.add_argument(
        "--server",
        default="",
        help="Read from a running runtime mission endpoint",
    )
    mission_report.add_argument("--json", action="store_true", help="Print raw JSON")

    memory_parser = subparsers.add_parser(
        "memory",
        help="Import and query robot long-term memory/RAG knowledge",
    )
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command")

    memory_import = memory_subparsers.add_parser(
        "import",
        help="Import Markdown, JSON, JSONL, or CSV knowledge into memory",
    )
    memory_import.add_argument("path", help="Knowledge file path")
    memory_import.add_argument("--source", default="", help="Override source label")
    memory_import.add_argument(
        "--category",
        choices=["location", "equipment", "route", "faq", "note"],
        default="",
        help="Override knowledge category",
    )
    memory_import.add_argument("--dry-run", action="store_true", help="Parse without saving")
    memory_import.add_argument("--json", action="store_true", help="Print raw JSON")

    memory_search = memory_subparsers.add_parser(
        "search",
        help="Search configured memory/RAG backend",
    )
    memory_search.add_argument("query", nargs="+", help="Search query")
    memory_search.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Override memory retrieval timeout seconds for cold CLI searches",
    )
    memory_search.add_argument("--json", action="store_true", help="Print raw JSON")

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


def _add_mission_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--operator-id",
        default="",
        help="Operator id for audit and runtime submission headers",
    )
    parser.add_argument(
        "--robot-id",
        default="",
        help="Target robot id",
    )
    parser.add_argument(
        "--site-id",
        default="",
        help="Target site id",
    )
    parser.add_argument(
        "--server",
        default="",
        help="Use a running runtime mission endpoint instead of local dry-run logic",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON")


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
    if args.command == "mission":
        _handle_mission_command(args)
        return
    if args.command == "memory":
        _handle_memory_command(args)
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
    if args.runtime_command is None:
        runtime_parser = getattr(args, "_runtime_parser", None)
        if runtime_parser is not None:
            runtime_parser.print_help()
            return

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

    if args.runtime_command == "voice-health":
        payload = _run_voice_health_check(live=args.live)
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_voice_health_payload(payload)
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
            from askme.voice.mic_calibration import write_mic_calibration_json

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
                device_secrets=_parse_device_secret_args(args.device_secret),
            )
            return
        payload = _run_field_ingest_bridge(
            source=args.source,
            server=args.server,
            state_path=args.state_path or None,
            dry_run=args.dry_run,
            limit=args.limit,
            timeout_s=args.timeout,
            device_secrets=_parse_device_secret_args(args.device_secret),
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_ingest_bridge_payload(payload)
        if payload.get("status") == "failed":
            raise SystemExit(1)
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
        )
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            _emit_field_readiness_payload(payload)
        if payload.get("status") == "blocked":
            raise SystemExit(1)
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
        payload = _run_local_agent_turn_for_cli(
            args.message,
            robot_mode=args.robot,
            speak=args.speak,
        )
        _emit_agent_payload(payload, json_output=args.json)
        return

    if args.server:
        payload = _send_agent_message_via_server(
            args.message,
            args.server,
            speak=args.speak,
        )
        _speak_agent_payload(
            payload,
            enabled=args.speak and not bool(payload.get("server_speak_requested")),
        )
        _emit_agent_payload(payload, json_output=args.json)
        return

    try:
        payload = _send_agent_message_via_server(
            args.message,
            DEFAULT_RUNTIME_URL,
            speak=args.speak,
        )
    except requests.RequestException:
        payload = _run_local_agent_turn_for_cli(
            args.message,
            robot_mode=args.robot,
            speak=args.speak,
        )
    else:
        _speak_agent_payload(
            payload,
            enabled=args.speak and not bool(payload.get("server_speak_requested")),
        )
    _emit_agent_payload(payload, json_output=args.json)


def _handle_mission_command(args: argparse.Namespace) -> None:
    if args.mission_command == "draft":
        payload = _draft_mission_sync(
            " ".join(args.text),
            operator_id=args.operator_id,
            robot_id=args.robot_id,
            site_id=args.site_id,
            server=args.server,
        )
        _emit_payload(payload, json_output=args.json)
        return

    if args.mission_command == "run":
        payload = _run_mission_sync(
            " ".join(args.source),
            dry_run=(not args.submit) or args.dry_run,
            confirmed=args.confirm,
            operator_id=args.operator_id,
            robot_id=args.robot_id,
            site_id=args.site_id,
            server=args.server,
        )
        _emit_payload(payload, json_output=args.json)
        return

    if args.mission_command == "report":
        payload = _mission_report_sync(args.mission_id, server=args.server)
        _emit_payload(payload, json_output=args.json)
        return

    raise SystemExit(f"Unknown mission command: {args.mission_command}")


def _handle_memory_command(args: argparse.Namespace) -> None:
    if args.memory_command is None:
        raise SystemExit("Missing memory command. Use: askme memory import|search")

    if args.memory_command == "import":
        from askme.memory.importer import import_knowledge_file

        payload = asyncio.run(
            import_knowledge_file(
                args.path,
                source=args.source or None,
                category=args.category or None,
                dry_run=bool(args.dry_run),
            )
        ).to_dict()
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            print(  # noqa: T201
                "imported={imported} parsed={parsed} skipped={skipped} errors={errors} source={source}".format(
                    **payload
                )
            )
        if payload.get("errors"):
            raise SystemExit(1)
        return

    if args.memory_command == "search":
        from askme.memory.bridge import MemoryBridge

        query = " ".join(args.query)
        bridge = MemoryBridge()
        if args.timeout and args.timeout > 0:
            bridge._retrieve_timeout = float(args.timeout)
        text = asyncio.run(bridge.retrieve(query))
        payload = {
            "query": query,
            "results": [
                line.strip().lstrip("- ").strip()
                for line in text.splitlines()
                if line.strip()
            ],
            "rag": bridge.health(),
        }
        if args.json:
            _emit_payload(payload, json_output=True)
        else:
            if not payload["results"]:
                print("No matching memories found")  # noqa: T201
            for item in payload["results"]:
                print(f"- {item}")  # noqa: T201
        return

    raise SystemExit(f"Unknown memory command: {args.memory_command}")


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
    from askme.main import run_app

    asyncio.run(run_app(voice_mode=voice_mode, robot_mode=robot_mode))


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


def _run_voice_health_check(*, live: bool) -> dict[str, Any]:
    from askme.voice.health_check import run_voice_health

    return run_voice_health(live=live)


def _emit_voice_health_payload(payload: dict[str, Any]) -> None:
    from askme.voice.health_check import print_voice_health_summary

    print_voice_health_summary(payload)


def _run_mic_calibration(
    *,
    server: str,
    duration_s: float,
    interval_s: float,
    min_signal_peak: int,
) -> dict[str, Any]:
    from askme.voice.mic_calibration import collect_runtime_mic_calibration

    return collect_runtime_mic_calibration(
        server=server,
        duration_s=duration_s,
        interval_s=interval_s,
        min_signal_peak=min_signal_peak,
    )


def _emit_mic_calibration_payload(payload: dict[str, Any]) -> None:
    from askme.voice.mic_calibration import print_mic_calibration_summary

    print_mic_calibration_summary(payload)


def _run_sunrise_audio_doctor(
    *,
    include_command_probes: bool,
    include_output_probe: bool,
    guard_min_seconds: float,
) -> dict[str, Any]:
    from askme.voice.sunrise_audio_doctor import run_sunrise_audio_doctor

    return run_sunrise_audio_doctor(
        include_command_probes=include_command_probes,
        include_output_probe=include_output_probe,
        guard_min_seconds=guard_min_seconds,
    )


def _emit_sunrise_audio_doctor_payload(payload: dict[str, Any]) -> None:
    from askme.voice.sunrise_audio_doctor import print_sunrise_audio_doctor_summary

    print_sunrise_audio_doctor_summary(payload)


def _run_sunrise_voice_readiness(
    *,
    guard_min_seconds: float,
    include_room_loop: bool,
    room_loop_text: str | None,
    room_loop_expect_prefix: str | None,
    room_loop_trials: int,
    live_tts_room_loop: bool,
    room_loop_asr: str,
    require_cloud_asr: bool,
) -> dict[str, Any]:
    from askme.voice.sunrise_readiness import (
        DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
        DEFAULT_ROOM_LOOP_TEXT,
        run_sunrise_voice_readiness,
    )

    return run_sunrise_voice_readiness(
        guard_min_seconds=guard_min_seconds,
        include_room_loop=include_room_loop,
        room_loop_text=room_loop_text or DEFAULT_ROOM_LOOP_TEXT,
        room_loop_expect_prefix=room_loop_expect_prefix or DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
        room_loop_trials=room_loop_trials,
        live_tts_room_loop=live_tts_room_loop,
        room_loop_asr=room_loop_asr,
        require_cloud_asr=require_cloud_asr,
    )


def _emit_sunrise_voice_readiness_payload(payload: dict[str, Any]) -> None:
    from askme.voice.sunrise_readiness import print_sunrise_voice_readiness_summary

    print_sunrise_voice_readiness_summary(payload)


def _run_s100p_readiness_bundle(
    *,
    output_dir: str | None,
    field: bool,
    guard_min_seconds: float,
    include_room_loop: bool,
    room_loop_text: str | None,
    room_loop_expect_prefix: str | None,
    room_loop_trials: int,
    live_tts_room_loop: bool,
    require_cloud_asr: bool,
    health_url: str,
    change_event_file: str,
    journal_since: str,
    skip_health: bool,
    skip_service_log: bool,
    command_timeout: float,
) -> dict[str, Any]:
    from askme.voice.s100p_readiness_bundle import collect_s100p_readiness_bundle
    from askme.voice.sunrise_readiness import (
        DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
        DEFAULT_ROOM_LOOP_TEXT,
    )

    return collect_s100p_readiness_bundle(
        output_dir,
        field=field,
        guard_min_seconds=guard_min_seconds,
        include_room_loop=include_room_loop,
        room_loop_text=room_loop_text or DEFAULT_ROOM_LOOP_TEXT,
        room_loop_expect_prefix=room_loop_expect_prefix or DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
        room_loop_trials=room_loop_trials,
        live_tts_room_loop=live_tts_room_loop,
        require_cloud_asr=require_cloud_asr,
        health_url=health_url,
        change_event_file=change_event_file,
        journal_since=journal_since,
        skip_health=skip_health,
        skip_service_log=skip_service_log,
        command_timeout=command_timeout,
    )


def _emit_s100p_readiness_bundle_payload(payload: dict[str, Any]) -> None:
    from askme.voice.s100p_readiness_bundle import print_s100p_readiness_bundle_summary

    print_s100p_readiness_bundle_summary(payload)


def _run_field_operations_eval(*, output: str) -> dict[str, Any]:
    from scripts.eval.evaluate_field_operations_scenarios import evaluate_scenarios, write_report

    payload = asyncio.run(evaluate_scenarios())
    report_path = write_report(payload, Path(output))
    payload["report_path"] = str(report_path)
    return payload


def _emit_field_operations_eval_payload(payload: dict[str, Any]) -> None:
    print(
        "field-operations: "
        f"{payload.get('status')} "
        f"{payload.get('passed', 0)}/{payload.get('scenario_count', 0)} scenarios "
        f"failed={payload.get('failed', 0)}"
    )
    print(f"report: {payload.get('report_path') or '-'}")
    product_demo = payload.get("product_demo")
    if not isinstance(product_demo, dict):
        return
    print(
        "product-demo: "
        f"{product_demo.get('suite_name') or 'field operations'} "
        f"ready={bool(product_demo.get('demo_ready'))} "
        f"real-integration-ready={bool(product_demo.get('real_integration_ready'))} "
        f"{product_demo.get('passed', 0)}/{product_demo.get('customer_scenario_count', 0)} scenes"
    )
    scenarios = product_demo.get("customer_scenarios")
    if isinstance(scenarios, list):
        print("customer-scenes:")
        for item in scenarios[:20]:
            if not isinstance(item, dict):
                continue
            actual = item.get("actual") if isinstance(item.get("actual"), dict) else {}
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            marker = "ok" if item.get("passed") else "fail"
            print(
                "  - "
                f"[{marker}] {item.get('customer_name') or item.get('name')}: "
                f"{item.get('expected_robot_action') or '-'} "
                f"notify={actual.get('notification_group') or '-'} "
                f"delivery={actual.get('delivery_status') or '-'} "
                f"event={evidence.get('event_id') or '-'}"
            )
    gaps = product_demo.get("blocked_on_real_integrations")
    if isinstance(gaps, list) and gaps:
        print("real-integration-gaps:")
        for gap in gaps[:10]:
            print(f"  - {gap}")


def _run_field_ingest_file(
    *,
    source: str,
    server: str,
    dry_run: bool,
    limit: int,
) -> dict[str, Any]:
    from askme.pipeline.field_ingest_adapters import normalize_field_ingest_payload

    events = _load_field_ingest_events(Path(source))
    if limit > 0:
        events = events[:limit]
    base_url = _normalise_server_url(server)
    results: list[dict[str, Any]] = []
    failures = 0
    for index, event in enumerate(events, start=1):
        normalized = normalize_field_ingest_payload(event)
        item: dict[str, Any] = {
            "index": index,
            "normalized": normalized,
            "posted": False,
        }
        if dry_run:
            item["status"] = "dry_run"
        else:
            try:
                response = _post_json(f"{base_url}/api/field/ingest", normalized)
            except Exception as exc:
                failures += 1
                item["status"] = "failed"
                item["error"] = str(exc)
            else:
                item["posted"] = True
                item["status"] = str(response.get("status") or "unknown")
                item["accepted"] = bool(response.get("accepted"))
                item["scenario_id"] = (
                    (response.get("normalized") or {}).get("scenario_id")
                    or normalized.get("scenario_id")
                    or ""
                )
                item["event_id"] = (response.get("event") or {}).get("event_id") or ""
        results.append(item)
    return {
        "status": "failed" if failures else "ok",
        "target": "field-ingest-file",
        "server": base_url,
        "dry_run": dry_run,
        "count": len(events),
        "failed": failures,
        "results": results,
    }


def _emit_field_ingest_file_payload(payload: dict[str, Any]) -> None:
    print(
        "field-ingest-file: "
        f"{payload.get('status')} count={payload.get('count', 0)} "
        f"failed={payload.get('failed', 0)} "
        f"dry_run={payload.get('dry_run')}"
    )
    for item in payload.get("results", [])[:20]:
        print(
            f"- #{item.get('index')} {item.get('status')} "
            f"scenario={item.get('scenario_id') or item.get('normalized', {}).get('scenario_id') or '-'} "
            f"event={item.get('event_id') or '-'}"
        )


def _run_field_ingest_bridge(
    *,
    source: str,
    server: str,
    state_path: str | None,
    dry_run: bool,
    limit: int,
    timeout_s: float,
    device_secrets: dict[str, str] | None = None,
) -> dict[str, Any]:
    from askme.pipeline.field_ingest_bridge import run_field_ingest_bridge_once

    return run_field_ingest_bridge_once(
        source=source,
        server=server,
        state_path=state_path,
        dry_run=dry_run,
        limit=limit,
        timeout_s=timeout_s,
        device_secrets=device_secrets,
    )


def _watch_field_ingest_bridge(
    *,
    source: str,
    server: str,
    state_path: str | None,
    interval_s: float,
    dry_run: bool,
    limit: int,
    timeout_s: float,
    device_secrets: dict[str, str] | None = None,
) -> None:
    from askme.pipeline.field_ingest_bridge import watch_field_ingest_bridge

    watch_field_ingest_bridge(
        source=source,
        server=server,
        state_path=state_path,
        interval_s=interval_s,
        dry_run=dry_run,
        limit=limit,
        timeout_s=timeout_s,
        device_secrets=device_secrets,
    )


def _parse_device_secret_args(values: list[str] | tuple[str, ...] | None) -> dict[str, str]:
    secrets: dict[str, str] = {}
    for item in values or []:
        raw = str(item or "")
        if "=" not in raw:
            raise SystemExit("--device-secret must use DEVICE_ID=SECRET")
        key, secret = raw.split("=", 1)
        key = key.strip()
        if not key or not secret:
            raise SystemExit("--device-secret must include a non-empty device id and secret")
        secrets[key] = secret
    return secrets


def _emit_field_ingest_bridge_payload(payload: dict[str, Any]) -> None:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    print(
        "field-ingest-bridge: "
        f"{payload.get('status')} count={payload.get('count', 0)} "
        f"failed={payload.get('failed', 0)} "
        f"dry_run={payload.get('dry_run')} "
        f"state={payload.get('state_path') or '-'}"
    )
    if summary:
        scenarios = summary.get("scenario_counts") if isinstance(summary.get("scenario_counts"), dict) else {}
        sources = summary.get("source_counts") if isinstance(summary.get("source_counts"), dict) else {}
        devices = summary.get("device_counts") if isinstance(summary.get("device_counts"), dict) else {}
        scenario_text = ", ".join(f"{key}:{value}" for key, value in sorted(scenarios.items())) or "-"
        source_text = ", ".join(f"{key}:{value}" for key, value in sorted(sources.items())) or "-"
        device_text = ", ".join(f"{key}:{value}" for key, value in sorted(devices.items())) or "-"
        print(
            "summary: "
            f"posted={summary.get('posted', 0)} "
            f"accepted={summary.get('accepted', 0)} "
            f"signed={summary.get('signed', 0)} "
            f"format={summary.get('source_format') or '-'} "
            f"scenarios={scenario_text} "
            f"sources={source_text} "
            f"devices={device_text}"
        )
    for item in payload.get("results", [])[:20]:
        print(
            f"- #{item.get('index')} {item.get('status')} "
            f"scenario={item.get('scenario_id') or item.get('normalized', {}).get('scenario_id') or '-'} "
            f"event={item.get('event_id') or '-'}"
        )


def _resolve_field_action_audit_hmac_secret(hmac_secret: str = "") -> str:
    return str(hmac_secret or os.getenv("ASKME_FIELD_ACTION_AUDIT_HMAC_SECRET") or "").strip()


def _field_action_audit_config(path: Path, *, hmac_secret: str = "") -> dict[str, Any]:
    resolved_secret = _resolve_field_action_audit_hmac_secret(hmac_secret)
    config: dict[str, Any] = {
        "enabled": True,
        "path": str(path),
        "swallow_errors": False,
    }
    if resolved_secret:
        config["hmac_secret"] = resolved_secret
    return config


def _run_field_ingest_smoke(
    *,
    output_dir: str,
    server: str = "",
    audit_hmac_secret: str = "",
    require_device_signatures: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = output / "device-events.jsonl"
    state_path = output / "device-events.state.json"
    archive_path = output / "field-events.jsonl"
    action_audit_path = output / "field-action-audit.jsonl"
    report_path = output / "field-ingest-smoke.json"
    _write_field_smoke_events(source)
    state_path.unlink(missing_ok=True)
    archive_path.unlink(missing_ok=True)
    action_audit_path.unlink(missing_ok=True)

    local_server = None
    base_url = server.strip()
    operator_action_payload: dict[str, Any] = {}
    if not base_url:
        field_config: dict[str, Any] = {"action_audit": _field_action_audit_config(
            action_audit_path,
            hmac_secret=audit_hmac_secret,
        )}
        if require_device_signatures:
            field_config.update(_field_ingest_smoke_trusted_device_config())
        local_server = _start_field_smoke_server(
            archive_path=archive_path,
            field_config=field_config,
        )
        base_url = str(local_server["base_url"])

    device_secrets = _field_ingest_smoke_device_secrets() if require_device_signatures else None
    try:
        bridge_payload = _run_field_ingest_bridge(
            source=str(source),
            server=base_url,
            state_path=str(state_path),
            dry_run=False,
            limit=0,
            timeout_s=8.0,
            device_secrets=device_secrets,
        )
        events_payload = _get_json(f"{_normalise_server_url(base_url)}/api/field/events?limit=20")
        events = events_payload.get("events") if isinstance(events_payload, dict) else []
        first_event = next((item for item in events if isinstance(item, dict) and item.get("event_id")), None)
        if first_event:
            operator_action_payload = _post_json(
                f"{_normalise_server_url(base_url)}/api/field/events/{first_event['event_id']}/acknowledge",
                {
                    "operator_id": "security-1",
                    "note": "field-smoke-suite acknowledges first incident for audit evidence",
                },
            )
            events_payload = _get_json(f"{_normalise_server_url(base_url)}/api/field/events?limit=20")
    finally:
        if local_server:
            local_server["server"].should_exit = True
            local_server["thread"].join(timeout=5)

    events = events_payload.get("events") if isinstance(events_payload, dict) else []
    scenario_ids = {
        str(item.get("scenario_id") or "")
        for item in events
        if isinstance(item, dict)
    }
    required = {
        "illegal_parking",
        "fire_or_smoke",
        "robot_abnormal_incident",
        "trash_bin_full",
        "crowd_gathering",
    }
    bridge_summary = bridge_payload.get("summary") if isinstance(bridge_payload.get("summary"), dict) else {}
    passed = (
        bridge_payload.get("status") == "ok"
        and int(bridge_payload.get("count") or 0) == 8
        and int(bridge_summary.get("posted") or 0) == 8
        and int(bridge_summary.get("accepted") or 0) == 8
        and int(bridge_summary.get("events_created") or 0) == 8
        and (not require_device_signatures or int(bridge_summary.get("signed") or 0) == 8)
        and required.issubset(scenario_ids)
        and operator_action_payload.get("acknowledged") is True
    )
    payload = {
        "status": "passed" if passed else "failed",
        "target": "field-ingest-smoke",
        "server": _normalise_server_url(base_url),
        "local_server": bool(local_server),
        "source": str(source),
        "state_path": str(state_path),
        "archive_path": str(archive_path),
        "report_path": str(report_path),
        "require_device_signatures": require_device_signatures,
        "bridge": bridge_payload,
        "operator_action": operator_action_payload,
        "event_count": len(events) if isinstance(events, list) else 0,
        "expected_bridge_count": 8,
        "scenario_ids": sorted(scenario_ids),
        "required_scenario_ids": sorted(required),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _emit_field_ingest_smoke_payload(payload: dict[str, Any]) -> None:
    bridge = payload.get("bridge") if isinstance(payload.get("bridge"), dict) else {}
    summary = bridge.get("summary") if isinstance(bridge.get("summary"), dict) else {}
    print(
        "field-ingest-smoke: "
        f"{payload.get('status')} "
        f"events={payload.get('event_count', 0)} "
        f"server={payload.get('server')}"
    )
    print(f"source: {payload.get('source')}")
    print(f"archive: {payload.get('archive_path')}")
    print(f"report: {payload.get('report_path')}")
    if summary:
        print(
            "bridge: "
            f"posted={summary.get('posted', 0)} "
            f"accepted={summary.get('accepted', 0)} "
            f"events_created={summary.get('events_created', 0)} "
            f"signed={summary.get('signed', 0)}"
        )
    print("scenarios: " + ", ".join(payload.get("scenario_ids", [])))


class _RecordingVoiceHandler:
    """Small voice handler used by field-voice-smoke without audio hardware."""

    def __init__(self) -> None:
        self.profiles: list[str] = []
        self.spoken: list[str] = []
        self.playback_started = False

    def set_voice_profile_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(body.get("profile_id") or "")
        self.profiles.append(profile_id)
        return {
            "updated": True,
            "active_profile": profile_id,
            "requested_profile": profile_id,
            "resolved_profile": profile_id,
            "profile": {"profile_id": profile_id},
        }

    def speak(self, text: str) -> None:
        self.spoken.append(str(text))

    def start_playback(self) -> None:
        self.playback_started = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "profiles": list(self.profiles),
            "spoken": list(self.spoken),
            "playback_started": self.playback_started,
        }


def _run_field_voice_smoke(
    *,
    output_dir: str,
    server: str = "",
    scenario: str = "fire",
    live_tts: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "field-voice-events.jsonl"
    report_path = output / "field-voice-smoke.json"
    archive_path.unlink(missing_ok=True)

    local_server = None
    voice_handler: Any | None = None
    base_url = server.strip()
    if not base_url:
        voice_handler = _build_field_voice_smoke_handler(live_tts=live_tts)
        local_server = _start_field_smoke_server(
            archive_path=archive_path,
            voice_handler=voice_handler,
            voice_enabled=True,
        )
        base_url = str(local_server["base_url"])

    request_payload = _field_voice_smoke_event(scenario)
    _make_field_voice_smoke_event_unique(request_payload)
    response_payload: dict[str, Any] = {}
    status_code = 0
    try:
        response = requests.post(
            f"{_normalise_server_url(base_url)}/api/field/events",
            json=request_payload,
            timeout=10,
        )
        status_code = response.status_code
        response_payload = response.json()
    finally:
        if local_server:
            local_server["server"].should_exit = True
            local_server["thread"].join(timeout=5)
        if live_tts and voice_handler is not None and hasattr(voice_handler, "shutdown"):
            voice_handler.shutdown()

    delivery = response_payload.get("voice_delivery") if isinstance(response_payload, dict) else {}
    event = response_payload.get("event") if isinstance(response_payload, dict) else {}
    directive = event.get("voice_directive") if isinstance(event, dict) else {}
    recorded = voice_handler.snapshot() if isinstance(voice_handler, _RecordingVoiceHandler) else {}
    passed = (
        status_code == 200
        and response_payload.get("accepted") is True
        and isinstance(delivery, dict)
        and delivery.get("status") == "queued"
        and isinstance(directive, dict)
        and bool(directive.get("resolved_profile"))
    )
    payload = {
        "status": "passed" if passed else "failed",
        "target": "field-voice-smoke",
        "server": _normalise_server_url(base_url),
        "local_server": bool(local_server),
        "live_tts": bool(live_tts),
        "scenario": scenario,
        "http_status": status_code,
        "archive_path": str(archive_path),
        "report_path": str(report_path),
        "request": request_payload,
        "response": response_payload,
        "voice_delivery": delivery if isinstance(delivery, dict) else {},
        "voice_directive": directive if isinstance(directive, dict) else {},
        "recorded_voice_handler": recorded,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _run_field_notification_smoke(
    *,
    output_dir: str,
    server: str = "",
    groups: str = "security,cleaning,operations",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "field-notification-events.jsonl"
    report_path = output / "field-notification-smoke.json"
    archive_path.unlink(missing_ok=True)

    group_names = [item.strip() for item in str(groups).split(",") if item.strip()]
    local_server = None
    collector = None
    base_url = server.strip()
    webhook_url = ""
    if not base_url:
        collector = _start_local_webhook_collector()
        webhook_url = str(collector["url"])
        local_server = _start_field_smoke_server(
            archive_path=archive_path,
            field_config={
                "dingtalk_webhooks": {group: webhook_url for group in group_names},
            },
        )
        base_url = str(local_server["base_url"])

    results: list[dict[str, Any]] = []
    try:
        for group in group_names:
            response = _post_json(
                f"{_normalise_server_url(base_url)}/api/field/notification-test",
                {
                    "notification_group": group,
                    "operator_id": "field-notification-smoke",
                    "message": f"Askme现场通知联调：{group}响应组。",
                },
            )
            results.append(response)
    finally:
        if local_server:
            local_server["server"].should_exit = True
            local_server["thread"].join(timeout=5)
        if collector:
            collector["server"].shutdown()
            collector["thread"].join(timeout=5)

    collector_requests = list(collector["requests"]) if collector else []
    sent_groups = [
        str(item.get("notification_group") or "")
        for item in results
        if item.get("sent") is True
    ]
    passed = set(group_names).issubset(set(sent_groups))
    if collector is not None:
        passed = passed and len(collector_requests) >= len(group_names)
    payload = {
        "status": "passed" if passed else "failed",
        "target": "field-notification-smoke",
        "server": _normalise_server_url(base_url),
        "local_server": bool(local_server),
        "local_webhook_collector": bool(collector),
        "external_services": bool(server.strip()),
        "groups": group_names,
        "sent_groups": sent_groups,
        "result_count": len(results),
        "collector_request_count": len(collector_requests),
        "collector_url": webhook_url,
        "archive_path": str(archive_path),
        "report_path": str(report_path),
        "results": results,
        "collector_requests": collector_requests[:10],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _run_field_notification_preflight(
    *,
    server: str = "",
    groups: str = "security,cleaning,operations",
    require_secret: bool = True,
) -> dict[str, Any]:
    group_names = [item.strip() for item in str(groups).split(",") if item.strip()]
    if server.strip():
        return _get_json(f"{_normalise_server_url(server)}/api/field/notification-preflight")

    from askme.config import get_config
    from askme.pipeline.field_operations import FieldOperationsService

    cfg = get_config()
    field_cfg = dict(cfg.get("field_operations", {}) if isinstance(cfg.get("field_operations"), dict) else {})
    service = FieldOperationsService(config=field_cfg)
    return service.notification_preflight_payload(
        groups=group_names,
        require_secret=require_secret,
    )


def _run_field_disposition_smoke(
    *,
    output_dir: str,
    server: str = "",
    audit_hmac_secret: str = "",
) -> dict[str, Any]:
    import time

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "field-events.jsonl"
    report_path = output / "field-disposition-smoke.json"

    local_server = None
    base_url = server.strip()
    if not base_url:
        local_server = _start_field_smoke_server(
            archive_path=archive_path,
            field_config={"action_audit": _field_action_audit_config(
                output / "field-action-audit.jsonl",
                hmac_secret=audit_hmac_secret,
            )},
        )
        base_url = str(local_server["base_url"])

    unique_location = f"配电间门口-处置验收-{int(time.time() * 1000)}"
    event_request = {
        "scenario_id": "fire_or_smoke",
        "source": "sensor",
        "observed_at": time.time(),
        "location": unique_location,
        "temperature_c": 68,
        "smoke_level": 0.82,
        "image_path": "artifacts/evidence/smoke-disposition.jpg",
    }
    created: dict[str, Any] = {}
    acknowledged: dict[str, Any] = {}
    close_requested: dict[str, Any] = {}
    closed: dict[str, Any] = {}
    report: dict[str, Any] = {}
    integrity: dict[str, Any] = {}
    try:
        created = _post_json(f"{_normalise_server_url(base_url)}/api/field/events", event_request)
        event = created.get("event") if isinstance(created.get("event"), dict) else {}
        event_id = str(event.get("event_id") or "")
        if event_id:
            acknowledged = _post_json(
                f"{_normalise_server_url(base_url)}/api/field/events/{event_id}/acknowledge",
                {"operator_id": "security-1", "note": "field disposition smoke acknowledged"},
            )
            close_requested = _post_json(
                f"{_normalise_server_url(base_url)}/api/field/events/{event_id}/request-close",
                {"operator_id": "security-1", "note": "request supervisor close approval"},
            )
            closed = _post_json(
                f"{_normalise_server_url(base_url)}/api/field/events/{event_id}/close",
                {
                    "operator_id": "security-1",
                    "note": "现场已复核并完成处置",
                    "supervisor_approved": True,
                    "supervisor_id": "supervisor-1",
                },
            )
            report = _get_json(f"{_normalise_server_url(base_url)}/api/field/events/{event_id}/report")
            integrity = _get_json(f"{_normalise_server_url(base_url)}/api/field/audit/integrity")
    finally:
        if local_server:
            local_server["server"].should_exit = True
            local_server["thread"].join(timeout=5)

    closed_event = closed.get("event") if isinstance(closed.get("event"), dict) else {}
    report_body = report.get("report") if isinstance(report.get("report"), dict) else {}
    timeline = report_body.get("timeline") if isinstance(report_body.get("timeline"), list) else []
    passed = (
        created.get("accepted") is True
        and acknowledged.get("acknowledged") is True
        and close_requested.get("requested") is True
        and closed_event.get("status") == "closed"
        and (closed_event.get("close_approval") or {}).get("approved") is True
        and len(timeline) >= 3
        and integrity.get("valid") is True
    )
    payload = {
        "status": "passed" if passed else "failed",
        "target": "field-disposition-smoke",
        "server": _normalise_server_url(base_url),
        "local_server": bool(local_server),
        "archive_path": str(archive_path),
        "report_path": str(report_path),
        "created": created,
        "acknowledged": acknowledged,
        "close_requested": close_requested,
        "closed": closed,
        "event_report": report,
        "action_audit_integrity": integrity,
        "timeline_count": len(timeline),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _run_field_smoke_suite(
    *,
    output_dir: str,
    voice_scenario: str = "fire",
    groups: str = "security,cleaning,operations",
    live_tts: bool = False,
    audit_hmac_secret: str = "",
    audit_webhook_url: str = "",
    audit_webhook_retries: int = 3,
    include_audit_anchor: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_audit_hmac_secret = _resolve_field_action_audit_hmac_secret(audit_hmac_secret)
    scenario_report_path = output / "scenario-evaluation.json"
    suite_report_path = output / "field-smoke-suite.json"
    html_report_path = output / "field-smoke-suite.html"

    scenario = _run_field_operations_eval(output=str(scenario_report_path))
    ingest = _run_field_ingest_smoke(
        output_dir=str(output),
        audit_hmac_secret=resolved_audit_hmac_secret,
    )
    voice = _run_field_voice_smoke(
        output_dir=str(output),
        scenario=voice_scenario,
        live_tts=live_tts,
    )
    notification = _run_field_notification_smoke(
        output_dir=str(output),
        groups=groups,
    )
    disposition = _run_field_disposition_smoke(
        output_dir=str(output),
        audit_hmac_secret=resolved_audit_hmac_secret,
    )
    readiness = _run_field_readiness(
        server="",
        archive_path=str(output / "field-events.jsonl"),
        scenario_report=str(scenario_report_path),
        smoke_report=str(output / "field-ingest-smoke.json"),
        voice_smoke_report=str(output / "field-voice-smoke.json"),
        notification_smoke_report=str(output / "field-notification-smoke.json"),
        audit_hmac_secret=resolved_audit_hmac_secret,
    )
    audit_anchor = (
        _run_field_audit_anchor(
            server="",
            archive_path=str(output / "field-events.jsonl"),
            audit_path=str(output / "field-action-audit.jsonl"),
            hmac_secret=resolved_audit_hmac_secret,
            output=str(output / "audit-checkpoint.json"),
            webhook_url=audit_webhook_url,
            webhook_retries=audit_webhook_retries,
            require_valid=True,
        )
        if include_audit_anchor
        else {"status": "skipped", "target": "field-audit-anchor"}
    )
    checks = {
        "scenario_eval": scenario.get("status") == "passed",
        "field_ingest_smoke": ingest.get("status") == "passed",
        "field_voice_smoke": voice.get("status") == "passed",
        "field_notification_smoke": notification.get("status") == "passed",
        "field_disposition_smoke": disposition.get("status") == "passed",
        "readiness_unblocked": not readiness.get("blockers"),
        "audit_checkpoint_created": audit_anchor.get("status") in {"anchored", "skipped"},
    }
    passed = all(checks.values())
    payload = {
        "status": "passed" if passed else "failed",
        "target": "field-smoke-suite",
        "output_dir": str(output),
        "report_path": str(suite_report_path),
        "html_report_path": str(html_report_path),
        "customer_summary": _field_smoke_customer_summary(
            checks=checks,
            readiness=readiness,
            notification=notification,
            voice=voice,
        ),
        "checks": checks,
        "scenario_report": scenario,
        "ingest_smoke": ingest,
        "voice_smoke": voice,
        "notification_smoke": notification,
        "disposition_smoke": disposition,
        "readiness": readiness,
        "audit_anchor": audit_anchor,
    }
    suite_report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_report_path.write_text(_field_smoke_suite_html(payload), encoding="utf-8")
    return payload


def _run_field_deployed_smoke(
    *,
    server: str,
    output_dir: str,
    voice_scenario: str = "fire",
    groups: str = "security,cleaning,operations",
    require_notification_ready: bool = True,
    require_device_signatures: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "field-deployed-smoke.json"
    base_url = _normalise_server_url(server)

    health = _get_json(f"{base_url}/health")
    notification_preflight = _run_field_notification_preflight(
        server=base_url,
        groups=groups,
        require_secret=True,
    )
    notification_ready = notification_preflight.get("ready") is True
    ingest = _run_field_ingest_smoke(
        output_dir=str(output),
        server=base_url,
        require_device_signatures=require_device_signatures,
    )
    voice = _run_field_voice_smoke(
        output_dir=str(output),
        server=base_url,
        scenario=voice_scenario,
    )
    if require_notification_ready and not notification_ready:
        notification = {
            "status": "skipped",
            "target": "field-notification-smoke",
            "reason": "notification_preflight_blocked",
        }
    else:
        notification = _run_field_notification_smoke(
            output_dir=str(output),
            server=base_url,
            groups=groups,
        )
    readiness = _get_json(f"{base_url}/api/field/readiness")
    checks = {
        "health_reachable": health.get("status") in {"ok", "degraded"},
        "notification_preflight_ready": notification_ready or not require_notification_ready,
        "field_ingest_smoke": ingest.get("status") == "passed",
        "signed_device_ingest_smoke": (
            not require_device_signatures
            or (
                isinstance(ingest.get("bridge"), dict)
                and isinstance(ingest["bridge"].get("summary"), dict)
                and int(ingest["bridge"]["summary"].get("signed") or 0) >= 1
            )
        ),
        "field_voice_smoke": voice.get("status") == "passed",
        "field_notification_smoke": notification.get("status") == "passed" or (
            not require_notification_ready and notification.get("status") == "skipped"
        ),
        "readiness_reachable": bool(readiness.get("status")),
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "target": "field-deployed-smoke",
        "server": base_url,
        "output_dir": str(output),
        "report_path": str(report_path),
        "checks": checks,
        "health": health,
        "notification_preflight": notification_preflight,
        "ingest_smoke": ingest,
        "voice_smoke": voice,
        "notification_smoke": notification,
        "readiness": readiness,
        "require_notification_ready": bool(require_notification_ready),
        "require_device_signatures": bool(require_device_signatures),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _field_smoke_customer_summary(
    *,
    checks: dict[str, bool],
    readiness: dict[str, Any],
    notification: dict[str, Any],
    voice: dict[str, Any],
) -> dict[str, Any]:
    warnings = [str(item) for item in readiness.get("warnings", []) if item]
    blockers = [str(item) for item in readiness.get("blockers", []) if item]
    return {
        "headline": "现场能力链路已通过本地实验室验证" if all(checks.values()) else "现场能力链路仍有未通过项",
        "readiness_status": readiness.get("status", "unknown"),
        "passed_checks": [name for name, passed in checks.items() if passed],
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "voice_verified": voice.get("status") == "passed",
        "voice_live_tts": bool(voice.get("live_tts")),
        "notification_verified": notification.get("status") == "passed",
        "notification_external_services": bool(notification.get("external_services")),
        "blockers": blockers,
        "warnings": warnings,
        "next_actions": [str(item) for item in readiness.get("next_actions", []) if item],
    }


def _field_smoke_suite_html(payload: dict[str, Any]) -> str:
    summary = payload.get("customer_summary") if isinstance(payload.get("customer_summary"), dict) else {}
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    warnings = summary.get("warnings") if isinstance(summary.get("warnings"), list) else []
    blockers = summary.get("blockers") if isinstance(summary.get("blockers"), list) else []
    actions = summary.get("next_actions") if isinstance(summary.get("next_actions"), list) else []
    gates = readiness.get("gates") if isinstance(readiness.get("gates"), dict) else {}
    check_rows = "".join(
        f"<li><strong>{_html_escape(name)}</strong>: {'通过' if passed else '未通过'}</li>"
        for name, passed in checks.items()
    )
    gate_rows = "".join(
        f"<li><strong>{_html_escape(name)}</strong>: {'通过' if value else '未通过'}</li>"
        for name, value in gates.items()
    )
    blocker_rows = "".join(f"<li>{_html_escape(item)}</li>" for item in blockers) or "<li>无阻塞项</li>"
    warning_rows = "".join(f"<li>{_html_escape(item)}</li>" for item in warnings) or "<li>无提醒项</li>"
    action_rows = "".join(f"<li>{_html_escape(item)}</li>" for item in actions) or "<li>无需额外动作</li>"
    status = str(payload.get("status") or "unknown")
    readiness_status = str(summary.get("readiness_status") or readiness.get("status") or "unknown")
    headline = str(summary.get("headline") or "现场验收报告")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Askme 现场能力验收报告</title>
  <style>
    body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:32px;color:#16352c;background:#f6fbf8;}}
    h1,h2{{color:#0b5a3f;}}
    .card{{background:#fff;border:1px solid #dbe9e2;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 8px 24px rgba(15,50,35,.08);}}
    .status{{display:inline-block;border-radius:999px;padding:6px 12px;background:#e2f7ea;color:#0a6a44;font-weight:700;}}
    .warn{{background:#fff6d8;color:#815500;}}
    li{{margin:6px 0;}}
    code{{background:#eef7f2;padding:2px 5px;border-radius:5px;}}
  </style>
</head>
<body>
  <h1>Askme 现场能力验收报告</h1>
  <div class="card">
    <p class="status{' warn' if status != 'passed' else ''}">Suite: {_html_escape(status)}</p>
    <p class="status{' warn' if readiness_status != 'production_ready' else ''}">Readiness: {_html_escape(readiness_status)}</p>
    <h2>{_html_escape(headline)}</h2>
    <p>这份报告面向演示、实验室验收和部署前自检。它证明本地链路是否打通，同时明确哪些能力仍未接入真实设备或真实外部服务。</p>
  </div>
  <div class="card"><h2>验收检查</h2><ul>{check_rows}</ul></div>
  <div class="card"><h2>部署门禁</h2><ul>{gate_rows}</ul></div>
  <div class="card"><h2>阻塞项</h2><ul>{blocker_rows}</ul></div>
  <div class="card"><h2>提醒项</h2><ul>{warning_rows}</ul></div>
  <div class="card"><h2>下一步</h2><ul>{action_rows}</ul></div>
  <div class="card">
    <h2>原始证据</h2>
    <p>JSON 报告：<code>{_html_escape(str(payload.get('report_path') or '-'))}</code></p>
  </div>
</body>
</html>
"""


def _html_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _emit_field_voice_smoke_payload(payload: dict[str, Any]) -> None:
    directive = payload.get("voice_directive") if isinstance(payload.get("voice_directive"), dict) else {}
    delivery = payload.get("voice_delivery") if isinstance(payload.get("voice_delivery"), dict) else {}
    print(
        "field-voice-smoke: "
        f"{payload.get('status')} "
        f"scenario={payload.get('scenario')} "
        f"delivery={delivery.get('status', '-')}"
    )
    print(f"server: {payload.get('server')}")
    print(f"voice: {directive.get('requested_profile', '-')} -> {directive.get('resolved_profile', '-')}")
    print(f"report: {payload.get('report_path')}")


def _emit_field_notification_smoke_payload(payload: dict[str, Any]) -> None:
    print(
        "field-notification-smoke: "
        f"{payload.get('status')} "
        f"groups={','.join(payload.get('sent_groups') or [])} "
        f"collector_requests={payload.get('collector_request_count', 0)}"
    )
    print(f"server: {payload.get('server')}")
    print(f"report: {payload.get('report_path')}")


def _emit_field_notification_preflight_payload(payload: dict[str, Any]) -> None:
    print(f"field-notification-preflight: {payload.get('status')}")
    groups = payload.get("groups") if isinstance(payload.get("groups"), dict) else {}
    for group, result in groups.items():
        if not isinstance(result, dict):
            continue
        print(
            f"- {group}: "
            f"{'ready' if result.get('ready') else 'blocked'} "
            f"webhook={bool(result.get('webhook_configured'))} "
            f"secret={bool(result.get('secret_configured'))}"
        )
    for action in payload.get("next_actions", [])[:5]:
        print(f"next: {action}")


def _emit_field_disposition_smoke_payload(payload: dict[str, Any]) -> None:
    closed = payload.get("closed") if isinstance(payload.get("closed"), dict) else {}
    event = closed.get("event") if isinstance(closed.get("event"), dict) else {}
    print(
        "field-disposition-smoke: "
        f"{payload.get('status')} "
        f"event={event.get('event_id', '-')} "
        f"timeline={payload.get('timeline_count', 0)}"
    )
    print(f"server: {payload.get('server')}")
    print(f"report: {payload.get('report_path')}")


def _emit_field_smoke_suite_payload(payload: dict[str, Any]) -> None:
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    print(f"field-smoke-suite: {payload.get('status')}")  # noqa: T201
    for name, passed in checks.items():
        print(f"- {name}: {'passed' if passed else 'failed'}")  # noqa: T201
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    print(f"readiness: {readiness.get('status', '-')}")  # noqa: T201
    print(f"report: {payload.get('report_path')}")  # noqa: T201


def _emit_field_deployed_smoke_payload(payload: dict[str, Any]) -> None:
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    print(f"field-deployed-smoke: {payload.get('status')} server={payload.get('server')}")  # noqa: T201
    for name, passed in checks.items():
        print(f"- {name}: {'passed' if passed else 'failed'}")  # noqa: T201
    print(f"report: {payload.get('report_path')}")  # noqa: T201


def _build_field_voice_smoke_handler(*, live_tts: bool) -> Any:
    if not live_tts:
        return _RecordingVoiceHandler()
    from askme.config import get_config
    from askme.voice.tts import TTSEngine

    cfg = get_config()
    voice_cfg = dict(cfg.get("voice", {}) if isinstance(cfg.get("voice"), dict) else {})
    return TTSEngine(voice_cfg)


def _field_smoke_run_id() -> str:
    import time

    return f"smoke-{int(time.time() * 1000)}"


def _make_field_voice_smoke_event_unique(payload: dict[str, Any]) -> None:
    """Avoid smoke-test events being deduped by earlier ingest-smoke events."""

    run_id = _field_smoke_run_id()
    payload["smoke_run_id"] = run_id
    for key in ("location", "plate_number", "zone_id"):
        value = payload.get(key)
        if value:
            payload[key] = f"{value}-{run_id}"


def _field_voice_smoke_event(scenario: str) -> dict[str, Any]:
    import time

    if scenario == "joint_fault":
        return {
            "scenario_id": "robot_abnormal_incident",
            "source": "robot",
            "observed_at": time.time(),
            "robot": {"fault_type": "joint_motor_fault", "joint_id": "hip-left"},
            "fault_type": "joint_motor_fault",
            "joint_id": "hip-left",
            "fault_code": "MOTOR_OVER_CURRENT",
            "location": "A区东侧",
            "image_path": "artifacts/evidence/joint-fault.jpg",
        }
    if scenario == "illegal_parking":
        return {
            "scenario_id": "illegal_parking",
            "source": "camera",
            "observed_at": time.time(),
            "location": "B区主通道",
            "zone_name": "主通道",
            "plate_number": "沪A12345",
            "duration_s": 180,
            "image_path": "artifacts/evidence/car.jpg",
        }
    return {
        "scenario_id": "fire_or_smoke",
        "source": "sensor",
        "observed_at": time.time(),
        "location": "配电间门口",
        "temperature_c": 68,
        "smoke_level": 0.82,
        "image_path": "artifacts/evidence/smoke.jpg",
    }


def _run_field_readiness(
    *,
    server: str,
    archive_path: str,
    scenario_report: str,
    smoke_report: str,
    voice_smoke_report: str,
    notification_smoke_report: str,
    site_profile: str = "",
    check_site_env: bool = False,
    audit_hmac_secret: str = "",
) -> dict[str, Any]:
    if server:
        return _get_json(f"{_normalise_server_url(server)}/api/field/readiness")
    from askme.pipeline.field_operations import FieldOperationsService

    action_audit_path = Path(archive_path).with_name("field-action-audit.jsonl")
    service = FieldOperationsService(
        config={
            "archive_path": archive_path,
            "scenario_report_path": scenario_report,
            "smoke_report_path": smoke_report,
            "voice_smoke_report_path": voice_smoke_report,
            "notification_smoke_report_path": notification_smoke_report,
            "site_profile_path": site_profile,
            "site_profile_check_env": check_site_env,
            "action_audit": _field_action_audit_config(
                action_audit_path,
                hmac_secret=audit_hmac_secret,
            ),
        }
    )
    return service.readiness_payload()


def _run_field_live_demo(
    *,
    output_dir: str,
    site_profile: str,
    server: str = "",
    timeout_s: float = 8.0,
    scenario_file: str = "",
    refresh_scenario_timestamps: bool = False,
) -> dict[str, Any]:
    from scripts.demo.live_field_operations_demo import run_live_demo

    return run_live_demo(
        output_dir=Path(output_dir),
        site_profile=Path(site_profile),
        server=server,
        timeout_s=timeout_s,
        scenario_file=Path(scenario_file) if scenario_file else None,
        refresh_scenario_timestamps=refresh_scenario_timestamps,
    )


def _run_field_audit_integrity(
    *,
    server: str,
    archive_path: str,
    audit_path: str,
    hmac_secret: str = "",
) -> dict[str, Any]:
    if server:
        return _get_json(f"{_normalise_server_url(server)}/api/field/audit/integrity")
    from askme.pipeline.field_operations import FieldOperationsService

    resolved_hmac_secret = _resolve_field_action_audit_hmac_secret(hmac_secret)
    action_audit: dict[str, Any] = {
        "enabled": True,
        "path": audit_path,
        "swallow_errors": False,
    }
    if resolved_hmac_secret:
        action_audit["hmac_secret"] = resolved_hmac_secret
    service = FieldOperationsService(
        config={
            "archive_path": archive_path,
            "action_audit": action_audit,
        }
    )
    return service.action_audit_integrity_payload()


def _run_field_audit_anchor(
    *,
    server: str,
    archive_path: str,
    audit_path: str,
    hmac_secret: str = "",
    output: str = "",
    webhook_url: str = "",
    webhook_retries: int = 3,
    retry_queue: str = "",
    require_valid: bool = True,
) -> dict[str, Any]:
    import time

    integrity = _run_field_audit_integrity(
        server=server,
        archive_path=archive_path,
        audit_path=audit_path,
        hmac_secret=hmac_secret,
    )
    valid = integrity.get("enabled") is False or integrity.get("valid") is True
    checkpoint = {
        "path": integrity.get("path") or audit_path,
        "latest_hash": integrity.get("latest_hash") or "",
        "hash_alg": integrity.get("hash_alg") or "",
        "checked_count": integrity.get("checked_count", 0),
        "expected_count": integrity.get("expected_count", 0),
        "signed": bool(integrity.get("signed")),
        "signature_alg": integrity.get("signature_alg") or "",
    }
    payload: dict[str, Any] = {
        "status": "blocked" if require_valid and not valid else "anchored",
        "target": "field-audit-anchor",
        "generated_at": round(time.time(), 3),
        "source": "server" if server else "local",
        "server": _normalise_server_url(server) if server else "",
        "archive_path": archive_path,
        "audit_path": audit_path,
        "checkpoint": checkpoint,
        "integrity": integrity,
        "output_path": output,
        "webhook_url": webhook_url,
        "webhook_delivery": None,
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if webhook_url and payload["status"] != "blocked":
        delivery = _post_json_with_retries(
            webhook_url,
            payload,
            attempts=max(1, int(webhook_retries or 1)),
        )
        payload["webhook_delivery"] = delivery
        if delivery.get("status") != "sent":
            payload["status"] = "delivery_failed"
            if retry_queue:
                _append_field_audit_retry_queue(retry_queue, payload)
        if output:
            path = Path(output)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _append_field_audit_retry_queue(queue: str, payload: dict[str, Any]) -> None:
    path = Path(queue)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "queued_at": payload.get("generated_at"),
        "webhook_url": payload.get("webhook_url") or "",
        "payload": payload,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_field_audit_delivery_retry(
    *,
    queue: str,
    webhook_retries: int = 3,
    lock_timeout_s: float = 300.0,
) -> dict[str, Any]:
    path = Path(queue)
    if not path.exists():
        return {
            "status": "empty",
            "target": "field-audit-retry-delivery",
            "queue": str(path),
            "attempted": 0,
            "sent": 0,
            "remaining": 0,
            "results": [],
        }
    lock = _acquire_field_audit_retry_lock(path, lock_timeout_s=lock_timeout_s)
    if not lock.get("acquired"):
        return {
            "status": "locked",
            "target": "field-audit-retry-delivery",
            "queue": str(path),
            "lock": lock,
            "attempted": 0,
            "sent": 0,
            "remaining": None,
            "results": [],
        }
    remaining: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                remaining.append({"line": line_number, "raw": line, "error": str(exc)})
                results.append({"line": line_number, "status": "invalid_json", "error": str(exc)})
                continue
            webhook_url = str(record.get("webhook_url") or "")
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
            if not webhook_url or not payload:
                remaining.append(record)
                results.append({"line": line_number, "status": "invalid_record"})
                continue
            delivery = _post_json_with_retries(
                webhook_url,
                payload,
                attempts=max(1, int(webhook_retries or 1)),
            )
            results.append({"line": line_number, "webhook_url": webhook_url, "delivery": delivery})
            if delivery.get("status") != "sent":
                remaining.append(record)
        if remaining:
            path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in remaining),
                encoding="utf-8",
            )
        else:
            path.unlink(missing_ok=True)
        sent = sum(1 for item in results if item.get("delivery", {}).get("status") == "sent")
        return {
            "status": "sent" if results and not remaining else "failed" if remaining else "empty",
            "target": "field-audit-retry-delivery",
            "queue": str(path),
            "lock": lock,
            "attempted": len(results),
            "sent": sent,
            "remaining": len(remaining),
            "results": results,
        }
    finally:
        lock_path = lock.get("path")
        if lock_path:
            Path(str(lock_path)).unlink(missing_ok=True)


def _acquire_field_audit_retry_lock(path: Path, *, lock_timeout_s: float) -> dict[str, Any]:
    import time

    lock_path = path.with_suffix(path.suffix + ".lock")
    now = time.time()
    lock_payload = {
        "pid": os.getpid(),
        "queue": str(path),
        "acquired_at": round(now, 3),
        "expires_at": round(now + max(1.0, float(lock_timeout_s or 1.0)), 3),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_path), flags)
    except FileExistsError:
        existing = _read_field_audit_retry_lock(lock_path)
        expires_at = float(existing.get("expires_at") or 0.0) if isinstance(existing, dict) else 0.0
        if expires_at and expires_at < now:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            return _acquire_field_audit_retry_lock(path, lock_timeout_s=lock_timeout_s)
        return {
            "acquired": False,
            "path": str(lock_path),
            "reason": "delivery_already_running",
            "existing": existing,
        }
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(lock_payload, ensure_ascii=False) + "\n")
    return {"acquired": True, "path": str(lock_path), **lock_payload}


def _read_field_audit_retry_lock(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "error": str(exc)}
    return payload if isinstance(payload, dict) else {"path": str(path), "error": "invalid_lock_payload"}


def _run_field_audit_retry_status(*, queue: str) -> dict[str, Any]:
    path = Path(queue)
    if not path.exists():
        return {
            "status": "empty",
            "target": "field-audit-retry-status",
            "queue": str(path),
            "pending": 0,
            "invalid": 0,
            "items": [],
        }
    items: list[dict[str, Any]] = []
    invalid = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid += 1
            items.append({"line": line_number, "status": "invalid_json", "error": str(exc)})
            continue
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        checkpoint = payload.get("checkpoint") if isinstance(payload.get("checkpoint"), dict) else {}
        items.append(
            {
                "line": line_number,
                "status": "pending",
                "webhook_url": record.get("webhook_url") or "",
                "queued_at": record.get("queued_at"),
                "latest_hash": checkpoint.get("latest_hash") or "",
                "checked_count": checkpoint.get("checked_count", 0),
            }
        )
    pending = sum(1 for item in items if item.get("status") == "pending")
    return {
        "status": "pending" if pending or invalid else "empty",
        "target": "field-audit-retry-status",
        "queue": str(path),
        "pending": pending,
        "invalid": invalid,
        "items": items,
    }


def _emit_field_readiness_payload(payload: dict[str, Any]) -> None:
    print(f"field-readiness: {payload.get('status')}")
    brief = payload.get("delivery_brief") if isinstance(payload.get("delivery_brief"), dict) else {}
    if brief:
        print(f"product-stage: {brief.get('stage_code') or '-'}")
        print(f"release-scope: {brief.get('release_scope') or '-'}")
    site_profile = payload.get("site_profile") if isinstance(payload.get("site_profile"), dict) else {}
    if site_profile:
        summary = site_profile.get("summary") if isinstance(site_profile.get("summary"), dict) else {}
        print(
            "site-profile: "
            f"configured={bool(site_profile.get('configured'))} "
            f"valid={bool(site_profile.get('valid'))} "
            f"site={summary.get('site_id') or '-'} "
            f"zones={summary.get('zone_count', 0)} "
            f"devices={summary.get('device_count', 0)}"
        )
    device_trust = payload.get("device_trust") if isinstance(payload.get("device_trust"), dict) else {}
    if device_trust:
        unsigned = device_trust.get("unsigned_device_ids")
        unsigned_ids = unsigned if isinstance(unsigned, list) else []
        unsigned_label = ",".join(str(item) for item in unsigned_ids[:5]) if unsigned_ids else "-"
        print(
            "device-trust: "
            f"registered={device_trust.get('registered_device_count', 0)} "
            f"signed={device_trust.get('signed_device_count', 0)} "
            f"unsigned={device_trust.get('unsigned_device_count', 0)} "
            f"all_ready={bool(device_trust.get('all_registered_devices_signature_ready'))} "
            f"unsigned_ids={unsigned_label}"
        )
    blockers = payload.get("blockers") or []
    warnings = payload.get("warnings") or []
    if blockers:
        print("blockers:")
        for item in blockers:
            print(f"- {item}")
    if warnings:
        print("warnings:")
        for item in warnings:
            print(f"- {item}")
    actions = payload.get("next_actions") or []
    if actions:
        print("next actions:")
        for item in actions:
            print(f"- {item}")


def _emit_field_live_demo_payload(payload: dict[str, Any]) -> None:
    print(
        "field-live-demo: "
        f"{payload.get('status')} "
        f"accepted={payload.get('accepted', 0)}/{payload.get('scenario_count', 0)} "
        f"mode={payload.get('mode') or '-'}"
    )
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    if readiness:
        print(f"readiness: {readiness.get('status') or '-'}")
    if payload.get("report_path"):
        print(f"report: {payload.get('report_path')}")
    if payload.get("guide_path"):
        print(f"guide: {payload.get('guide_path')}")
    if payload.get("html_report_path"):
        print(f"html: {payload.get('html_report_path')}")
    scenarios = payload.get("scenarios") if isinstance(payload.get("scenarios"), list) else []
    if scenarios:
        print("scenarios:")
        for item in scenarios[:10]:
            if not isinstance(item, dict):
                continue
            print(
                "- "
                f"{item.get('scenario_id') or '-'}: "
                f"http={item.get('http_status') or '-'} "
                f"accepted={bool(item.get('accepted'))} "
                f"event={item.get('event_id') or '-'}"
            )


def _emit_field_audit_integrity_payload(payload: dict[str, Any]) -> None:
    status = "valid" if payload.get("valid") else "invalid"
    print(f"field-audit-integrity: {status}")
    print(f"path: {payload.get('path') or '-'}")
    print(f"checked: {payload.get('checked_count', 0)} / expected: {payload.get('expected_count', '-')}")
    print(f"latest_hash: {payload.get('latest_hash') or '-'}")
    print(f"signed: {bool(payload.get('signed'))}")
    failures = payload.get("failures") or []
    if failures:
        print("failures:")
        for item in failures[:10]:
            line = item.get("line", 0)
            reason = item.get("reason") or "unknown"
            detail = item.get("detail")
            suffix = f" ({detail})" if detail else ""
            print(f"- line {line}: {reason}{suffix}")


def _emit_field_audit_anchor_payload(payload: dict[str, Any]) -> None:
    checkpoint = payload.get("checkpoint") if isinstance(payload.get("checkpoint"), dict) else {}
    print(f"field-audit-anchor: {payload.get('status')}")
    print(f"latest_hash: {checkpoint.get('latest_hash') or '-'}")
    print(f"checked: {checkpoint.get('checked_count', 0)} / expected: {checkpoint.get('expected_count', '-')}")
    print(f"signed: {bool(checkpoint.get('signed'))}")
    if payload.get("output_path"):
        print(f"output: {payload.get('output_path')}")
    if payload.get("webhook_url"):
        delivery = payload.get("webhook_delivery")
        print(f"webhook: {'sent' if delivery else 'not_sent'}")


def _emit_field_audit_delivery_retry_payload(payload: dict[str, Any]) -> None:
    print(f"field-audit-retry-delivery: {payload.get('status')}")
    print(f"attempted: {payload.get('attempted', 0)}")
    print(f"sent: {payload.get('sent', 0)}")
    print(f"remaining: {payload.get('remaining', 0)}")
    print(f"queue: {payload.get('queue') or '-'}")
    lock = payload.get("lock") if isinstance(payload.get("lock"), dict) else {}
    if lock:
        print(f"lock: {lock.get('path') or '-'}")
        if lock.get("reason"):
            print(f"lock_reason: {lock.get('reason')}")


def _emit_field_audit_retry_status_payload(payload: dict[str, Any]) -> None:
    print(f"field-audit-retry-status: {payload.get('status')}")
    print(f"pending: {payload.get('pending', 0)}")
    print(f"invalid: {payload.get('invalid', 0)}")
    print(f"queue: {payload.get('queue') or '-'}")
    for item in (payload.get("items") or [])[:10]:
        print(f"- line {item.get('line')}: {item.get('status')} {item.get('latest_hash') or item.get('error') or ''}")


def _field_ingest_smoke_device_secrets() -> dict[str, str]:
    return {
        "camera-main-road-1": "smoke-camera-main-road",
        "cam-main-road-01": "smoke-anpr-main-road",
        "camera-plaza-1": "smoke-camera-plaza",
        "smoke-power-room-1": "smoke-sensor-power-room",
        "robot-thunder-1": "smoke-robot-thunder",
        "bin-17": "smoke-bin-17",
    }


def _field_ingest_smoke_trusted_device_config() -> dict[str, Any]:
    return {
        "require_trusted_devices": True,
        "device_registry": {
            "camera-main-road-1": {
                "allowed_sources": ["camera"],
                "hmac_secret": "smoke-camera-main-road",
                "require_signature": True,
            },
            "cam-main-road-01": {
                "allowed_sources": ["camera"],
                "hmac_secret": "smoke-anpr-main-road",
                "require_signature": True,
            },
            "camera-plaza-1": {
                "allowed_sources": ["camera"],
                "hmac_secret": "smoke-camera-plaza",
                "require_signature": True,
            },
            "smoke-power-room-1": {
                "allowed_sources": ["sensor"],
                "hmac_secret": "smoke-sensor-power-room",
                "require_signature": True,
            },
            "robot-thunder-1": {
                "allowed_sources": ["robot"],
                "hmac_secret": "smoke-robot-thunder",
                "require_signature": True,
            },
            "bin-17": {
                "allowed_sources": ["camera"],
                "hmac_secret": "smoke-bin-17",
                "require_signature": True,
            },
        },
    }


def _write_field_smoke_events(path: Path) -> None:
    import time

    events = [
        {
            "device_id": "camera-main-road-1",
            "frame": {
                "timestamp": time.time(),
                "boxes": [{"cls": 2, "conf": 0.94, "xyxy": [12, 20, 120, 160]}],
            },
            "zone": {
                "id": "main-road-1",
                "name": "B区主通道",
                "type": "main_channel",
                "parking_allowed": False,
            },
            "duration_s": 180,
            "image_path": "artifacts/evidence/smoke-car.jpg",
        },
        {
            "device_id": "smoke-power-room-1",
            "timestamp": time.time(),
            "telemetry": {"temperature": 68, "smoke": 0.82},
            "location": "配电间门口",
            "image_path": "artifacts/evidence/smoke.jpg",
        },
        {
            "device_id": "robot-thunder-1",
            "timestamp": time.time(),
            "topic": "/diagnostics",
            "status": [
                {
                    "name": "left_hip_motor",
                    "level": 2,
                    "message": "motor overcurrent fault",
                    "values": [
                        {"key": "joint_id", "value": "hip-left"},
                        {"key": "fault_code", "value": "MOTOR_OVERCURRENT"},
                    ],
                }
            ],
            "location": "A区东侧",
        },
        {
            "timestamp": time.time(),
            "device_id": "bin-17",
            "telemetry": {"fill_percent": 91},
            "detections": [{"label": "trash_bin", "confidence": 0.88}],
            "bin_id": "bin-17",
            "location": "游客中心门口",
            "image_path": "artifacts/evidence/bin.jpg",
        },
        {
            "device_id": "camera-plaza-1",
            "timestamp": time.time(),
            "predictions": [{"class": "person", "confidence": 0.82} for _ in range(6)],
            "duration_min": 35,
            "location": "北广场",
            "image_path": "artifacts/evidence/crowd.jpg",
        },
        {
            "eventType": "ANPR",
            "dateTime": time.time(),
            "cameraIndexCode": "cam-main-road-01",
            "ANPR": {"plateNo": "沪A12345"},
            "zone_id": "main-road-1",
            "zone_name": "B区主通道",
            "location": "B区主通道",
            "duration_s": 180,
            "pictureUrl": "artifacts/evidence/anpr-car.jpg",
        },
        {
            "device_id": "smoke-power-room-1",
            "topic": "site/A/power-room/smoke-01",
            "payload": {
                "timestamp": time.time(),
                "temperatureC": 72,
                "smokeAlarm": True,
                "location": "配电间门口",
                "imageUrl": "artifacts/evidence/smoke-mqtt.jpg",
            },
        },
        {
            "device_id": "robot-thunder-1",
            "topic": "/thunder/status",
            "timestamp": time.time(),
            "robot": {"nav_state": "stuck", "recoverable": False},
            "location": "A区东侧",
        },
    ]
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in events),
        encoding="utf-8",
    )


def _start_field_smoke_server(
    *,
    archive_path: Path,
    voice_handler: Any | None = None,
    voice_enabled: bool = False,
    field_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import socket
    import threading
    import time

    import uvicorn

    from askme.health_server import build_health_snapshot, create_health_app
    from askme.pipeline.field_operations import FieldOperationsService

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()

    service_config = {"archive_path": str(archive_path)}
    if field_config:
        service_config.update(field_config)
    service = FieldOperationsService(config=service_config)

    def health() -> dict[str, Any]:
        return build_health_snapshot(
            app_name="askme",
            app_version="smoke",
            model_name="field-smoke",
            metrics_snapshot={"uptime_seconds": 1.0, "conversation_count": 0},
            active_skills=[],
            voice_status={"enabled": voice_enabled, "pipeline_ok": True},
        )

    app = create_health_app(
        health,
        field_operations_handler=service,
        voice_handler=voice_handler,
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://{host}:{port}"
    deadline = time.time() + 5
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            requests.get(f"{base_url}/health", timeout=0.5).raise_for_status()
            return {"server": server, "thread": thread, "base_url": base_url}
        except Exception as exc:
            last_error = exc
            time.sleep(0.05)
    server.should_exit = True
    thread.join(timeout=5)
    raise RuntimeError(f"field smoke server did not start: {last_error}")


def _start_local_webhook_collector() -> dict[str, Any]:
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    requests_seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body: Any = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError:
                body = {"raw": raw.decode("utf-8", errors="replace")}
            requests_seen.append({
                "path": self.path,
                "headers": {key: value for key, value in self.headers.items()},
                "body": body,
            })
            response = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()

    server = HTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return {
        "server": server,
        "thread": thread,
        "url": f"http://{host}:{port}/dingtalk",
        "requests": requests_seen,
    }


def _load_field_ingest_events(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        events = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    else:
        loaded = json.loads(raw)
        events = loaded if isinstance(loaded, list) else [loaded]
    result = [event for event in events if isinstance(event, dict)]
    if len(result) != len(events):
        raise SystemExit(f"Field ingest file must contain JSON objects: {path}")
    return result


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


def _run_local_agent_turn_sync(
    message: str,
    *,
    robot_mode: bool,
    speak: bool = False,
) -> dict[str, Any]:
    return asyncio.run(_run_local_agent_turn(message, robot_mode=robot_mode, speak=speak))


def _run_local_agent_turn_for_cli(
    message: str,
    *,
    robot_mode: bool,
    speak: bool,
) -> dict[str, Any]:
    if speak:
        return _run_local_agent_turn_sync(message, robot_mode=robot_mode, speak=True)
    return _run_local_agent_turn_sync(message, robot_mode=robot_mode)


async def _run_local_agent_turn(
    message: str,
    *,
    robot_mode: bool,
    speak: bool = False,
) -> dict[str, Any]:
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
    if not enabled:
        return
    reply = payload.get("reply", "")
    if not isinstance(reply, str) or not reply.strip():
        return
    try:
        _speak_agent_reply(reply.strip())
    except Exception as exc:
        _report_speak_error(exc)


def _speak_agent_reply(reply: str) -> None:
    """Play a one-shot agent reply using the local configured TTS output."""
    from askme.config import get_config
    from askme.voice.audio_agent import AudioAgent

    audio = AudioAgent(get_config(), voice_mode=False)
    audio.speak(reply)
    audio.start_playback()
    try:
        done = audio.wait_speaking_done()
        if done is False:
            raise TimeoutError("TTS playback did not finish within timeout")
    finally:
        audio.stop_playback()


def _report_speak_error(exc: Exception) -> None:
    print(f"[askme] speak failed: {exc}", file=sys.stderr)


def _send_agent_message_via_server(
    message: str,
    server: str,
    *,
    speak: bool = False,
) -> dict[str, Any]:
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
    response = requests.post(f"{base_url}/api/chat", **kwargs)
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


def _draft_mission_sync(
    text: str,
    *,
    operator_id: str = "",
    robot_id: str = "",
    site_id: str = "",
    server: str = "",
) -> dict[str, Any]:
    payload = _mission_context_payload(
        {"text": text, "channel": "cli"},
        operator_id=operator_id,
        robot_id=robot_id,
        site_id=site_id,
    )
    if server:
        return _post_json(f"{_normalise_server_url(server)}/api/missions/draft", payload)

    service = _load_local_mission_service()
    return service.draft_from_payload(payload)


def _run_mission_sync(
    source: str,
    *,
    dry_run: bool,
    confirmed: bool,
    operator_id: str = "",
    robot_id: str = "",
    site_id: str = "",
    server: str = "",
) -> dict[str, Any]:
    payload = _load_mission_source(source)
    payload = _mission_context_payload(
        payload,
        operator_id=operator_id,
        robot_id=robot_id,
        site_id=site_id,
    )
    payload.setdefault("channel", "cli")
    payload["dry_run"] = dry_run
    payload["confirmed"] = confirmed

    if server:
        return _post_json(f"{_normalise_server_url(server)}/api/missions", payload)

    service = _load_local_mission_service()
    return service.submit_from_payload(payload, trusted_confirmation=True)


def _mission_report_sync(mission_id: str, *, server: str = "") -> dict[str, Any]:
    if server:
        return _get_json(
            f"{_normalise_server_url(server)}/api/missions/{mission_id}/report"
        )

    service = _load_local_mission_service()
    return service.report_payload(mission_id)


def _mission_context_payload(
    payload: dict[str, Any],
    *,
    operator_id: str = "",
    robot_id: str = "",
    site_id: str = "",
) -> dict[str, Any]:
    result = dict(payload)
    if operator_id:
        result["operator_id"] = operator_id
    if robot_id:
        result["robot_id"] = robot_id
    if site_id:
        result["site_id"] = site_id
    return result


def _load_local_mission_service():
    from askme.config import get_config
    from askme.runtime.mission import MissionService

    return MissionService(get_config())


def _load_mission_source(source: str) -> dict[str, Any]:
    path = Path(source)
    if not path.exists():
        return {"text": source}

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(raw)
    else:
        return {"text": raw}

    if not isinstance(payload, dict):
        raise SystemExit(f"Mission source must be a JSON/YAML object: {path}")
    return payload


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"json": payload, "timeout": 5}
    headers = _server_auth_headers()
    if headers:
        kwargs["headers"] = headers
    response = requests.post(url, **kwargs)
    response.raise_for_status()
    return response.json()


def _post_json_with_retries(url: str, payload: dict[str, Any], *, attempts: int) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            response = _post_json(url, payload)
            return {
                "status": "sent",
                "attempts": attempt,
                "response": response,
            }
        except requests.RequestException as exc:
            last_error = str(exc)
    return {
        "status": "failed",
        "attempts": max(1, attempts),
        "error": last_error or "webhook_delivery_failed",
    }


def _get_json(url: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"timeout": 5}
    headers = _server_auth_headers()
    if headers:
        kwargs["headers"] = headers
    response = requests.get(url, **kwargs)
    response.raise_for_status()
    return response.json()


def _normalise_server_url(server: str) -> str:
    return server.rstrip("/")


def _server_auth_headers() -> dict[str, str] | None:
    token = (
        os.environ.get("ASKME_CONTROL_API_KEY")
        or os.environ.get("ASKME_HEALTH_API_KEY")
        or _configured_control_api_key()
    )
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def _configured_control_api_key() -> str:
    try:
        from askme.config import get_config

        raw = get_config().get("health_server", {}).get("control_api_key", "")
    except Exception:
        return ""
    return str(raw).strip()


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
    return json.dumps(payload, indent=2, ensure_ascii=not _stdout_supports_unicode())


def _stdout_supports_unicode() -> bool:
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("_", "-")
    return encoding in {"utf-8", "utf8"} or "65001" in encoding


if __name__ == "__main__":
    main(sys.argv[1:])
