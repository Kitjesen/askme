"""CLI argument parser — extracted from askme.cli for the cli/ subpackage."""

from __future__ import annotations

import argparse

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
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport mode (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP transport (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for HTTP transport (default: 8080)",
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

    runtime_blueprints = runtime_subparsers.add_parser(
        "blueprints",
        help="List local runtime blueprint product contracts",
    )
    runtime_blueprints.add_argument(
        "--name",
        default="",
        help="Inspect one blueprint by name or alias",
    )
    runtime_blueprints.add_argument(
        "--customer-visible",
        action="store_true",
        help="Show only customer-visible blueprints",
    )
    runtime_blueprints.add_argument(
        "--delivery-package",
        action="store_true",
        help=(
            "Include delivery package details; with --name, emit that single customer handoff package"
        ),
    )
    runtime_blueprints.add_argument(
        "--output",
        default="",
        help="Write the blueprint catalog or delivery package JSON to this path",
    )
    runtime_blueprints.add_argument("--json", action="store_true", help="Print raw JSON")

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

    runtime_dialogue_smoke = runtime_subparsers.add_parser(
        "dialogue-smoke",
        help="Run a real text dialogue turn with isolated memory retrieval evidence",
    )
    runtime_dialogue_smoke.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_dialogue_smoke.add_argument(
        "--message",
        default="What is Thunder's current test identifier? Answer only the identifier.",
        help="User message sent through ConversationService/TextLoop",
    )
    runtime_dialogue_smoke.add_argument(
        "--memory-text",
        default="",
        help="Temporary knowledge record. Defaults to a token-bearing test fact.",
    )
    runtime_dialogue_smoke.add_argument(
        "--memory-query",
        default="",
        help="Direct retrieval query before the chat turn. Defaults to --message.",
    )
    runtime_dialogue_smoke.add_argument(
        "--output-dir",
        default="",
        help="Artifact directory for seed data and report.json",
    )
    runtime_dialogue_smoke.add_argument(
        "--data-dir",
        default="",
        help="Isolated runtime data directory. Defaults under --output-dir.",
    )
    runtime_dialogue_smoke.add_argument(
        "--token",
        default="",
        help="Expected token for memory/reply checks. Defaults to a generated token.",
    )
    runtime_dialogue_smoke.add_argument("--chat-timeout", type=float, default=90.0)
    runtime_dialogue_smoke.add_argument("--memory-timeout", type=float, default=30.0)
    runtime_dialogue_smoke.add_argument("--vector-min-similarity", type=float, default=0.1)
    runtime_dialogue_smoke.add_argument(
        "--fake-llm",
        action="store_true",
        help="Use the fake LLM provider. Off by default so the smoke exercises configured LLM.",
    )
    runtime_dialogue_smoke.add_argument(
        "--allow-reply-without-token",
        action="store_true",
        help="Pass when retrieval evidence is correct but the LLM paraphrases the reply.",
    )
    runtime_dialogue_burst = runtime_subparsers.add_parser(
        "dialogue-burst",
        help="Run repeated real-machine text dialogue smokes with memory evidence",
    )
    runtime_dialogue_burst.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_dialogue_burst.add_argument("--fake-runs", type=int, default=5)
    runtime_dialogue_burst.add_argument("--real-runs", type=int, default=1)
    runtime_dialogue_burst.add_argument(
        "--output-dir",
        default="",
        help="Artifact directory for burst-report.json and per-run reports",
    )
    runtime_dialogue_burst.add_argument(
        "--token-prefix",
        default="",
        help="Token prefix for generated per-run identifiers",
    )
    runtime_dialogue_burst.add_argument("--chat-timeout", type=float, default=90.0)
    runtime_dialogue_burst.add_argument("--memory-timeout", type=float, default=30.0)
    runtime_dialogue_burst.add_argument("--vector-min-similarity", type=float, default=0.1)
    runtime_dialogue_burst.add_argument(
        "--allow-reply-without-token",
        action="store_true",
        help="Pass when retrieval evidence is correct but the LLM paraphrases the reply.",
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
    runtime_audio_devices = runtime_subparsers.add_parser(
        "audio-devices",
        help="List local microphone/speaker devices and recommended config",
    )
    runtime_audio_devices.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_audio_loopback = runtime_subparsers.add_parser(
        "audio-loopback",
        help="Play a short tone while recording the microphone",
    )
    runtime_audio_loopback.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_audio_loopback.add_argument("--input-device", default=None)
    runtime_audio_loopback.add_argument("--output-device", default=None)
    runtime_audio_loopback.add_argument("--sample-rate", type=int, default=None)
    runtime_audio_loopback.add_argument("--record-seconds", type=float, default=2.0)
    runtime_audio_loopback.add_argument("--tone-seconds", type=float, default=0.8)
    runtime_audio_loopback.add_argument("--frequency-hz", type=float, default=880.0)
    runtime_audio_loopback.add_argument("--output-gain", type=float, default=0.25)
    runtime_audio_loopback.add_argument("--min-capture-peak", type=int, default=300)
    runtime_audio_loopback.add_argument(
        "--wav-out",
        default="artifacts/audio/windows-audio-loopback.wav",
        help="Write captured microphone audio to this WAV file",
    )
    runtime_audio_loopback.add_argument(
        "--play-recording",
        action="store_true",
        help="Replay the captured microphone audio after the loopback test",
    )
    runtime_audio_beep_loopback = runtime_subparsers.add_parser(
        "audio-beep-loopback",
        help="Play a Windows system beep while recording the microphone",
    )
    runtime_audio_beep_loopback.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_audio_beep_loopback.add_argument("--input-device", default=None)
    runtime_audio_beep_loopback.add_argument("--sample-rate", type=int, default=None)
    runtime_audio_beep_loopback.add_argument("--record-seconds", type=float, default=3.0)
    runtime_audio_beep_loopback.add_argument("--tone-seconds", type=float, default=1.0)
    runtime_audio_beep_loopback.add_argument("--frequency-hz", type=float, default=880.0)
    runtime_audio_beep_loopback.add_argument("--min-capture-peak", type=int, default=300)
    runtime_audio_beep_loopback.add_argument(
        "--wav-out",
        default="artifacts/audio/windows-beep-loopback.wav",
        help="Write captured microphone audio to this WAV file",
    )
    runtime_audio_beep_loopback.add_argument(
        "--play-recording",
        action="store_true",
        help="Replay the captured microphone audio after the beep test",
    )
    runtime_audio_route_scan = runtime_subparsers.add_parser(
        "audio-route-scan",
        help="Scan microphone/speaker routes and rank real capture evidence",
    )
    runtime_audio_route_scan.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_audio_route_scan.add_argument(
        "--input-devices",
        default="",
        help="Comma-separated input device indexes. Empty scans same-hostapi routes.",
    )
    runtime_audio_route_scan.add_argument(
        "--output-devices",
        default="",
        help="Comma-separated output device indexes. Empty scans same-hostapi routes.",
    )
    runtime_audio_route_scan.add_argument(
        "--sample-rates",
        default="48000,44100",
        help="Comma-separated sample rates to try",
    )
    runtime_audio_route_scan.add_argument("--record-seconds", type=float, default=1.0)
    runtime_audio_route_scan.add_argument("--tone-seconds", type=float, default=0.35)
    runtime_audio_route_scan.add_argument("--frequency-hz", type=float, default=880.0)
    runtime_audio_route_scan.add_argument("--output-gain", type=float, default=0.35)
    runtime_audio_route_scan.add_argument("--min-capture-peak", type=int, default=300)
    runtime_audio_route_scan.add_argument("--max-routes", type=int, default=24)
    runtime_audio_route_scan.add_argument(
        "--all-pairs",
        action="store_true",
        help="Also try cross-hostapi pairs, which may report illegal combinations",
    )
    runtime_voice_online_smoke = runtime_subparsers.add_parser(
        "voice-online-smoke",
        help="Check the configured LLM, MiniMax TTS, and cloud ASR connectivity",
    )
    runtime_voice_online_smoke.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_voice_online_smoke.add_argument(
        "--text",
        default="你好，请问需要指路吗？服务中心在前方右转。",
        help="Text used for the MiniMax TTS online synthesis probe",
    )
    runtime_voice_online_smoke.add_argument(
        "--silence-seconds",
        type=float,
        default=0.2,
        help="Seconds of silence sent to the configured cloud ASR",
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
    runtime_field_ingest_file.add_argument(
        "--device-secret",
        action="append",
        default=[],
        metavar="DEVICE_ID=SECRET",
        help="Sign normalized events for a registered device id, source, or * wildcard",
    )
    runtime_field_ingest_file.add_argument(
        "--site-profile",
        default="",
        help="Load device HMAC secrets from this site profile's device secret_env entries",
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
    runtime_field_ingest_bridge.add_argument(
        "--site-profile",
        default="",
        help="Load device HMAC secrets from this site profile's device secret_env entries",
    )
    runtime_field_ingest_bridge.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_sign_device_payload = runtime_subparsers.add_parser(
        "field-sign-device-payload",
        help="Sign camera, sensor, or robot ingest JSON/JSONL with a field-device HMAC",
    )
    runtime_field_sign_device_payload.add_argument("source", help="JSON object/array or JSONL/NDJSON file")
    runtime_field_sign_device_payload.add_argument(
        "--output",
        default="",
        help="Write signed JSON/JSONL to this path. Omit to print the signed payload.",
    )
    runtime_field_sign_device_payload.add_argument(
        "--device-id",
        default="",
        help="Override or add device_id before signing",
    )
    runtime_field_sign_device_payload.add_argument(
        "--secret",
        default="",
        help="HMAC secret value. Prefer --secret-env for shared scripts.",
    )
    runtime_field_sign_device_payload.add_argument(
        "--secret-env",
        default="",
        help="Environment variable containing the HMAC secret",
    )
    runtime_field_sign_device_payload.add_argument(
        "--timestamp",
        type=float,
        default=0.0,
        help="Unix signature timestamp. Defaults to now.",
    )
    runtime_field_sign_device_payload.add_argument("--json", action="store_true", help="Print raw JSON")
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
    runtime_field_ingest_smoke.add_argument("--json", action="store_true", help="Print raw JSON")
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
    runtime_field_readiness.add_argument(
        "--review-path",
        default="",
        help="Unified audit review JSONL path used to clear readiness review blockers",
    )
    runtime_field_readiness.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_device_trust = runtime_subparsers.add_parser(
        "field-device-trust",
        help="Inspect registered field-device HMAC secret readiness from a site profile",
    )
    runtime_field_device_trust.add_argument(
        "--site-profile",
        default="deploy/site-profiles/park-demo.yaml",
        help="Field site profile YAML containing devices and secret_env values",
    )
    runtime_field_device_trust.add_argument(
        "--show-commands",
        action="store_true",
        help="Print signing commands for every registered device",
    )
    runtime_field_device_trust.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_field_site_env_template = runtime_subparsers.add_parser(
        "field-site-env-template",
        help="Generate a .env template from a field site profile",
    )
    runtime_field_site_env_template.add_argument(
        "--site-profile",
        default="deploy/site-profiles/park-demo.yaml",
        help="Field site profile YAML containing responder and device *_env values",
    )
    runtime_field_site_env_template.add_argument(
        "--output",
        default="",
        help="Write the generated .env template to this path. Prints it when omitted.",
    )
    runtime_field_site_env_template.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_audit_events = runtime_subparsers.add_parser(
        "audit-events",
        help="Inspect the unified product audit timeline and review queue",
    )
    runtime_audit_events.add_argument("--limit", type=int, default=100)
    runtime_audit_events.add_argument("--source", default="")
    runtime_audit_events.add_argument("--operator-id", default="")
    runtime_audit_events.add_argument("--action", default="")
    runtime_audit_events.add_argument("--outcome", default="")
    runtime_audit_events.add_argument("--q", default="")
    runtime_audit_events.add_argument("--since", default="")
    runtime_audit_events.add_argument("--until", default="")
    _add_unified_audit_path_args(runtime_audit_events)
    runtime_audit_events.add_argument(
        "--review-queue-only",
        action="store_true",
        help="Print only records requiring supervisor review in text mode",
    )
    runtime_audit_events.add_argument("--json", action="store_true", help="Print raw JSON")
    runtime_audit_review = runtime_subparsers.add_parser(
        "audit-review",
        help="Append a supervisor review decision for one unified audit record",
    )
    runtime_audit_review.add_argument("record_id", help="Unified audit record id, e.g. field:2")
    runtime_audit_review.add_argument(
        "decision",
        choices=["accepted", "resolved", "waived", "false_positive", "escalated", "rejected"],
        help="Supervisor decision to append to the audit review log",
    )
    runtime_audit_review.add_argument(
        "--reviewer-id",
        default="supervisor-1",
        help="Reviewer/operator id written to the append-only review log",
    )
    runtime_audit_review.add_argument("--note", default="", help="Short review note")
    _add_unified_audit_path_args(runtime_audit_review)
    runtime_audit_review.add_argument("--json", action="store_true", help="Print raw JSON")
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

    agent_parser = subparsers.add_parser(
        "agent",
        help="Send runtime chat messages",
        description=(
            "Send a one-shot chat message to a running Askme runtime, or to the "
            "local text runtime for development compatibility."
        ),
    )
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command")

    agent_send = agent_subparsers.add_parser(
        "send",
        help="Send a single message to askme",
    )
    agent_send.add_argument("message", help="Message to send")
    agent_send.add_argument(
        "--server",
        default="",
        help="Use a running runtime HTTP /api/chat endpoint",
    )
    agent_send.add_argument(
        "--local",
        action="store_true",
        help="Force a local text-runtime turn even if a runtime is already running",
    )
    agent_send.add_argument(
        "--robot",
        action="store_true",
        help="Select the robot runtime profile for the local text-runtime turn",
    )
    agent_send.add_argument(
        "--speak",
        action="store_true",
        help="Request or play the assistant reply through configured TTS output",
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
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport mode (default: stdio)",
    )
    mcp_serve.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP transport (default: localhost)",
    )
    mcp_serve.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for HTTP transport (default: 8080)",
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


def _add_unified_audit_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skill-audit", default="", help="Override skill audit JSONL path")
    parser.add_argument("--field-action-audit", default="", help="Override field action audit JSONL path")
    parser.add_argument("--field-event-archive", default="", help="Override field event archive JSONL path")
    parser.add_argument("--runtime-audit", default="", help="Override runtime handoff audit JSONL path")
    parser.add_argument("--review-path", default="", help="Override unified audit review JSONL path")


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
