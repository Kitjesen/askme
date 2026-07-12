"""Extract cli.py functions into askme/cli/ package.

Usage: python _extract_cli.py
"""
import ast
import textwrap
from pathlib import Path

CLI_SOURCE = Path("askme/cli.py")
source = CLI_SOURCE.read_text(encoding="utf-8")
lines = source.splitlines()
tree = ast.parse(source)

# Build function map: name -> (start_line, end_line) (0-indexed)
# Only top-level definitions — skip nested ones
func_map: dict[str, tuple[int, int]] = {}
for node in ast.iter_child_nodes(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        func_map[node.name] = (node.lineno - 1, node.end_lineno)


def extract_source(name: str) -> str:
    """Extract function source by name, preserving original indentation."""
    start, end = func_map[name]
    return "\n".join(lines[start:end])


# ─── Function-to-module mapping ────────────────────────────────────────────

UTILS = [
    "_post_json", "_post_json_with_retries", "_get_json",
    "_normalise_server_url", "_parse_csv_ints",
    "_server_auth_headers", "_configured_control_api_key",
    "_emit_agent_payload", "_emit_payload", "_json",
    "_stdout_supports_unicode", "_stdout_can_encode",
    "_stdout_should_emit_human_text",
    "_send_agent_message_via_server",
]

APP = [
    "build_parser", "_add_runtime_selection_args",
    "_add_unified_audit_path_args", "_add_mission_context_args",
]

SKILLS = ["_handle_skills_command", "_load_skill_manager"]
AGENT = ["_handle_agent_command"]
MISSION = [
    "_handle_mission_command",
    "_draft_mission_sync", "_load_local_mission_service",
    "_load_mission_source", "_mission_context_payload",
    "_mission_report_sync", "_run_mission_sync",
]
MEMORY = ["_handle_memory_command"]

RUNTIME = [
    "_handle_runtime_command", "_handle_mcp_command",
    "_resolve_runtime_flags", "_looks_like_mcp_request",
    "_run_interactive_runtime", "_run_terminal_tui", "_run_mcp_server",
    "_load_local_capabilities", "_load_local_capabilities_async",
    "_runtime_blueprints_payload", "_emit_runtime_blueprints_summary",
    "_run_local_agent_turn_sync", "_run_local_agent_turn_for_cli",
    "_run_local_agent_turn", "_speak_local_text_reply",
    "_speak_agent_payload", "_speak_agent_reply", "_report_speak_error",
]

VOICE = [
    "_run_voice_health_check", "_emit_voice_health_payload",
    "_run_mic_calibration", "_emit_mic_calibration_payload",
    "_run_sunrise_audio_doctor", "_emit_sunrise_audio_doctor_payload",
    "_run_sunrise_voice_readiness", "_emit_sunrise_voice_readiness_payload",
    "_run_s100p_readiness_bundle", "_emit_s100p_readiness_bundle_payload",
]

FIELD = [
    "_RecordingVoiceHandler",
    "_run_field_operations_eval", "_emit_field_operations_eval_payload",
    "_run_field_ingest_file", "_emit_field_ingest_file_payload",
    "_run_field_ingest_bridge", "_watch_field_ingest_bridge",
    "_parse_device_secret_args", "_resolve_field_device_secrets",
    "_device_secrets_from_site_profile", "_emit_field_ingest_bridge_payload",
    "_run_field_sign_device_payload", "_resolve_field_device_signing_secret",
    "_field_signed_payload_text", "_single_device_id",
    "_emit_field_sign_device_payload", "_resolve_field_action_audit_hmac_secret",
    "_field_action_audit_config",
    "_run_field_ingest_smoke", "_emit_field_ingest_smoke_payload",
    "_run_field_voice_smoke", "_run_field_notification_smoke",
    "_run_field_notification_preflight", "_run_field_disposition_smoke",
    "_run_field_smoke_suite", "_run_field_deployed_smoke",
    "_field_smoke_customer_summary", "_field_smoke_suite_html",
    "_html_escape",
    "_emit_field_voice_smoke_payload", "_emit_field_notification_smoke_payload",
    "_emit_field_notification_preflight_payload",
    "_emit_field_disposition_smoke_payload",
    "_emit_field_smoke_suite_payload", "_emit_field_deployed_smoke_payload",
    "_build_field_voice_smoke_handler", "_field_smoke_run_id",
    "_make_field_voice_smoke_event_unique", "_field_voice_smoke_event",
    "_run_field_readiness", "_run_field_device_trust",
    "_run_field_site_env_template", "_field_site_env_template_next_actions",
    "_field_device_signing_command", "_field_device_trust_next_actions",
    "_emit_field_readiness_payload", "_emit_field_device_trust_payload",
    "_emit_field_site_env_template_payload",
    "_run_field_live_demo", "_emit_field_live_demo_payload",
    "_field_ingest_smoke_device_secrets",
    "_field_ingest_smoke_trusted_device_config",
    "_write_field_smoke_events", "_start_field_smoke_server",
    "_start_local_webhook_collector", "_load_field_ingest_events",
]

AUDIT_CMD = [
    "_run_unified_audit_events", "_run_unified_audit_review",
    "_unified_audit_paths_from_cli",
    "_emit_unified_audit_events_payload", "_emit_unified_audit_review_payload",
]

FIELD_AUDIT = [
    "_run_field_audit_integrity", "_run_field_audit_anchor",
    "_append_field_audit_retry_queue", "_run_field_audit_delivery_retry",
    "_acquire_field_audit_retry_lock", "_read_field_audit_retry_lock",
    "_run_field_audit_retry_status",
    "_emit_field_audit_integrity_payload", "_emit_field_audit_anchor_payload",
    "_emit_field_audit_delivery_retry_payload",
    "_emit_field_audit_retry_status_payload",
]

INIT = ["main", "_apply_common_options", "_dispatch_compat_mode"]

# Validate: every function is assigned exactly once
all_assigned: set[str] = set()
for group in [UTILS, APP, SKILLS, AGENT, MISSION, MEMORY, RUNTIME, VOICE,
              FIELD, AUDIT_CMD, FIELD_AUDIT, INIT]:
    for fn in group:
        if fn in all_assigned:
            print(f"ERROR: Duplicate assignment: {fn}")
        all_assigned.add(fn)

available = set(func_map.keys())
missing_fn = available - all_assigned
extra_fn = all_assigned - available
if missing_fn:
    print(f"WARNING: Functions not assigned to any module: {sorted(missing_fn)}")
if extra_fn:
    print(f"WARNING: Assigned but not found in source: {sorted(extra_fn)}")


# ─── Cross-module import definitions ──────────────────────────────────────

DEFAULT_RUNTIME_URL = "http://127.0.0.1:8765"

BASE_IMPORTS = '''"""Structured CLI for askme with dimos-style command groups."""

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


'''

# Extra imports for each module (importing from other cli/ sub-modules)
MODULE_EXTRA_IMPORTS = {
    "utils.py": f"DEFAULT_RUNTIME_URL = {DEFAULT_RUNTIME_URL!r}\n\n",
    "_app.py": "DEFAULT_RUNTIME_URL = \"http://127.0.0.1:8765\"\n\n",
    "skills.py": "",
    "agent.py": (
        "from askme.cli.runtime import (\n"
        "    _run_local_agent_turn_for_cli,\n"
        "    _speak_agent_payload,\n"
        ")\n"
        "from askme.cli.utils import (\n"
        "    _emit_agent_payload,\n"
        "    _send_agent_message_via_server,\n"
        ")\n\n"
    ),
    "mission.py": (
        "from askme.cli.utils import (\n"
        "    _emit_payload,\n"
        "    _get_json,\n"
        "    _normalise_server_url,\n"
        "    _post_json,\n"
        ")\n\n"
    ),
    "memory.py": (
        "from askme.cli.utils import (\n"
        "    _emit_payload,\n"
        ")\n\n"
    ),
    "voice.py": (
        "from askme.cli.utils import (\n"
        "    _emit_payload,\n"
        ")\n\n"
    ),
    "field.py": (
        "from askme.cli.utils import (\n"
        "    _emit_payload,\n"
        "    _get_json,\n"
        "    _normalise_server_url,\n"
        "    _parse_csv_ints,\n"
        "    _post_json,\n"
        "    _post_json_with_retries,\n"
        "    _server_auth_headers,\n"
        ")\n\n"
    ),
    "audit_cmd.py": (
        "from askme.cli.utils import (\n"
        "    _emit_payload,\n"
        "    _get_json,\n"
        "    _server_auth_headers,\n"
        ")\n\n"
    ),
    "field_audit.py": (
        "from askme.cli.utils import (\n"
        "    _emit_payload,\n"
        "    _get_json,\n"
        "    _normalise_server_url,\n"
        "    _post_json,\n"
        "    _post_json_with_retries,\n"
        "    _server_auth_headers,\n"
        ")\n\n"
    ),
    "runtime.py": (
        "from askme.cli.utils import (\n"
        "    _emit_payload,\n"
        "    _get_json,\n"
        "    _normalise_server_url,\n"
        "    _parse_csv_ints,\n"
        "    _post_json,\n"
        "    _post_json_with_retries,\n"
        ")\n"
        "from askme.cli.voice import (\n"
        "    _emit_mic_calibration_payload,\n"
        "    _emit_s100p_readiness_bundle_payload,\n"
        "    _emit_sunrise_audio_doctor_payload,\n"
        "    _emit_sunrise_voice_readiness_payload,\n"
        "    _emit_voice_health_payload,\n"
        "    _run_mic_calibration,\n"
        "    _run_s100p_readiness_bundle,\n"
        "    _run_sunrise_audio_doctor,\n"
        "    _run_sunrise_voice_readiness,\n"
        "    _run_voice_health_check,\n"
        ")\n"
        "from askme.cli.field import (\n"
        "    _emit_field_deployed_smoke_payload,\n"
        "    _emit_field_device_trust_payload,\n"
        "    _emit_field_ingest_bridge_payload,\n"
        "    _emit_field_ingest_file_payload,\n"
        "    _emit_field_ingest_smoke_payload,\n"
        "    _emit_field_live_demo_payload,\n"
        "    _emit_field_notification_preflight_payload,\n"
        "    _emit_field_notification_smoke_payload,\n"
        "    _emit_field_operations_eval_payload,\n"
        "    _emit_field_readiness_payload,\n"
        "    _emit_field_sign_device_payload,\n"
        "    _emit_field_site_env_template_payload,\n"
        "    _emit_field_smoke_suite_payload,\n"
        "    _emit_field_voice_smoke_payload,\n"
        "    _resolve_field_device_secrets,\n"
        "    _run_field_deployed_smoke,\n"
        "    _run_field_device_trust,\n"
        "    _run_field_disposition_smoke,\n"
        "    _run_field_ingest_bridge,\n"
        "    _run_field_ingest_file,\n"
        "    _run_field_ingest_smoke,\n"
        "    _run_field_live_demo,\n"
        "    _run_field_notification_preflight,\n"
        "    _run_field_notification_smoke,\n"
        "    _run_field_operations_eval,\n"
        "    _run_field_readiness,\n"
        "    _run_field_sign_device_payload,\n"
        "    _run_field_site_env_template,\n"
        "    _run_field_smoke_suite,\n"
        "    _run_field_voice_smoke,\n"
        "    _watch_field_ingest_bridge,\n"
        ")\n"
        "from askme.cli.audit_cmd import (\n"
        "    _emit_unified_audit_events_payload,\n"
        "    _emit_unified_audit_review_payload,\n"
        "    _run_unified_audit_events,\n"
        "    _run_unified_audit_review,\n"
        ")\n"
        "from askme.cli.field_audit import (\n"
        "    _emit_field_audit_anchor_payload,\n"
        "    _emit_field_audit_delivery_retry_payload,\n"
        "    _emit_field_audit_integrity_payload,\n"
        "    _emit_field_audit_retry_status_payload,\n"
        "    _run_field_audit_anchor,\n"
        "    _run_field_audit_delivery_retry,\n"
        "    _run_field_audit_integrity,\n"
        "    _run_field_audit_retry_status,\n"
        ")\n\n"
    ),
    "__init__.py": (
        "from askme.cli._app import (\n"
        "    _add_mission_context_args,\n"
        "    _add_runtime_selection_args,\n"
        "    _add_unified_audit_path_args,\n"
        "    build_parser,\n"
        ")\n"
        "from askme.cli.agent import _handle_agent_command\n"
        "from askme.cli.audit_cmd import (\n"
        "    _emit_unified_audit_events_payload,\n"
        "    _emit_unified_audit_review_payload,\n"
        "    _run_unified_audit_events,\n"
        "    _run_unified_audit_review,\n"
        "    _unified_audit_paths_from_cli,\n"
        ")\n"
        "from askme.cli.field import (\n"
        "    _RecordingVoiceHandler,\n"
        "    _build_field_voice_smoke_handler,\n"
        "    _device_secrets_from_site_profile,\n"
        "    _emit_field_deployed_smoke_payload,\n"
        "    _emit_field_device_trust_payload,\n"
        "    _emit_field_disposition_smoke_payload,\n"
        "    _emit_field_ingest_bridge_payload,\n"
        "    _emit_field_ingest_file_payload,\n"
        "    _emit_field_ingest_smoke_payload,\n"
        "    _emit_field_live_demo_payload,\n"
        "    _emit_field_notification_preflight_payload,\n"
        "    _emit_field_notification_smoke_payload,\n"
        "    _emit_field_operations_eval_payload,\n"
        "    _emit_field_readiness_payload,\n"
        "    _emit_field_sign_device_payload,\n"
        "    _emit_field_site_env_template_payload,\n"
        "    _emit_field_smoke_suite_payload,\n"
        "    _emit_field_voice_smoke_payload,\n"
        "    _field_action_audit_config,\n"
        "    _field_device_signing_command,\n"
        "    _field_device_trust_next_actions,\n"
        "    _field_ingest_smoke_device_secrets,\n"
        "    _field_ingest_smoke_trusted_device_config,\n"
        "    _field_signed_payload_text,\n"
        "    _field_smoke_customer_summary,\n"
        "    _field_smoke_run_id,\n"
        "    _field_smoke_suite_html,\n"
        "    _field_site_env_template_next_actions,\n"
        "    _field_voice_smoke_event,\n"
        "    _html_escape,\n"
        "    _load_field_ingest_events,\n"
        "    _make_field_voice_smoke_event_unique,\n"
        "    _parse_device_secret_args,\n"
        "    _resolve_field_action_audit_hmac_secret,\n"
        "    _resolve_field_device_secrets,\n"
        "    _resolve_field_device_signing_secret,\n"
        "    _run_field_deployed_smoke,\n"
        "    _run_field_device_trust,\n"
        "    _run_field_disposition_smoke,\n"
        "    _run_field_ingest_bridge,\n"
        "    _run_field_ingest_file,\n"
        "    _run_field_ingest_smoke,\n"
        "    _run_field_live_demo,\n"
        "    _run_field_notification_preflight,\n"
        "    _run_field_notification_smoke,\n"
        "    _run_field_operations_eval,\n"
        "    _run_field_readiness,\n"
        "    _run_field_sign_device_payload,\n"
        "    _run_field_site_env_template,\n"
        "    _run_field_smoke_suite,\n"
        "    _run_field_voice_smoke,\n"
        "    _single_device_id,\n"
        "    _start_field_smoke_server,\n"
        "    _start_local_webhook_collector,\n"
        "    _watch_field_ingest_bridge,\n"
        "    _write_field_smoke_events,\n"
        ")\n"
        "from askme.cli.field_audit import (\n"
        "    _acquire_field_audit_retry_lock,\n"
        "    _append_field_audit_retry_queue,\n"
        "    _emit_field_audit_anchor_payload,\n"
        "    _emit_field_audit_delivery_retry_payload,\n"
        "    _emit_field_audit_integrity_payload,\n"
        "    _emit_field_audit_retry_status_payload,\n"
        "    _read_field_audit_retry_lock,\n"
        "    _run_field_audit_anchor,\n"
        "    _run_field_audit_delivery_retry,\n"
        "    _run_field_audit_integrity,\n"
        "    _run_field_audit_retry_status,\n"
        ")\n"
        "from askme.cli.memory import _handle_memory_command\n"
        "from askme.cli.mission import (\n"
        "    _draft_mission_sync,\n"
        "    _handle_mission_command,\n"
        "    _load_local_mission_service,\n"
        "    _load_mission_source,\n"
        "    _mission_context_payload,\n"
        "    _mission_report_sync,\n"
        "    _run_mission_sync,\n"
        ")\n"
        "from askme.cli.runtime import (\n"
        "    _emit_runtime_blueprints_summary,\n"
        "    _handle_mcp_command,\n"
        "    _handle_runtime_command,\n"
        "    _load_local_capabilities,\n"
        "    _load_local_capabilities_async,\n"
        "    _looks_like_mcp_request,\n"
        "    _resolve_runtime_flags,\n"
        "    _run_interactive_runtime,\n"
        "    _run_local_agent_turn,\n"
        "    _run_local_agent_turn_for_cli,\n"
        "    _run_local_agent_turn_sync,\n"
        "    _run_mcp_server,\n"
        "    _run_terminal_tui,\n"
        "    _runtime_blueprints_payload,\n"
        "    _speak_agent_payload,\n"
        "    _speak_agent_reply,\n"
        "    _speak_local_text_reply,\n"
        "    _report_speak_error,\n"
        ")\n"
        "from askme.cli.skills import (\n"
        "    _handle_skills_command,\n"
        "    _load_skill_manager,\n"
        ")\n"
        "from askme.cli.utils import (\n"
        "    DEFAULT_RUNTIME_URL,\n"
        "    _configured_control_api_key,\n"
        "    _emit_agent_payload,\n"
        "    _emit_payload,\n"
        "    _get_json,\n"
        "    _json,\n"
        "    _normalise_server_url,\n"
        "    _parse_csv_ints,\n"
        "    _post_json,\n"
        "    _post_json_with_retries,\n"
        "    _send_agent_message_via_server,\n"
        "    _server_auth_headers,\n"
        "    _stdout_can_encode,\n"
        "    _stdout_should_emit_human_text,\n"
        "    _stdout_supports_unicode,\n"
        ")\n"
        "from askme.cli.voice import (\n"
        "    _emit_mic_calibration_payload,\n"
        "    _emit_s100p_readiness_bundle_payload,\n"
        "    _emit_sunrise_audio_doctor_payload,\n"
        "    _emit_sunrise_voice_readiness_payload,\n"
        "    _emit_voice_health_payload,\n"
        "    _run_mic_calibration,\n"
        "    _run_s100p_readiness_bundle,\n"
        "    _run_sunrise_audio_doctor,\n"
        "    _run_sunrise_voice_readiness,\n"
        "    _run_voice_health_check,\n"
        ")\n\n"
    ),
}


# ─── Write module file helper ─────────────────────────────────────────────

def write_module(filename: str, func_names: list[str]) -> None:
    """Write a module file containing the given functions."""
    out = Path("askme/cli") / filename
    extra = MODULE_EXTRA_IMPORTS.get(filename, "")

    parts = [BASE_IMPORTS]
    if extra:
        parts.append(extra)

    for name in func_names:
        src = extract_source(name)
        # Dedent the body so it sits cleanly at column 0
        src = textwrap.dedent(src)
        parts.append(src)
        if not src.endswith("\n"):
            parts.append("\n")
        parts.append("\n")

    content = "".join(parts).rstrip("\n") + "\n"
    out.write_text(content, encoding="utf-8")
    # Validate syntax
    try:
        compile(content, str(out), "exec")
        print(f"  {filename}: {len(func_names)} functions, {len(content)} bytes [OK]")
    except SyntaxError as e:
        print(f"  {filename}: {len(func_names)} functions, {len(content)} bytes [SYNTAX ERROR: {e}]")


# ─── Write all modules ────────────────────────────────────────────────────

print("Creating askme/cli/ modules...\n")

write_module("utils.py", UTILS)
write_module("_app.py", APP)
write_module("skills.py", SKILLS)
write_module("agent.py", AGENT)
write_module("mission.py", MISSION)
write_module("memory.py", MEMORY)
write_module("voice.py", VOICE)
write_module("field.py", FIELD)
write_module("audit_cmd.py", AUDIT_CMD)
write_module("field_audit.py", FIELD_AUDIT)
write_module("runtime.py", RUNTIME)
write_module("__init__.py", INIT)

print("\nDone!")
