"""
Askme configuration loader.

Loads config.yaml, merges with .env environment variables,
and resolves ${VAR} references. Singleton access via get_config().
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")

# Project root: directory containing config.yaml (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ``${VAR}`` placeholders in strings from env."""
    if isinstance(value, str):

        def _replacer(match: re.Match) -> str:
            var_name = match.group(1)
            env_val = os.environ.get(var_name, "")
            return env_val

        return _ENV_VAR_PATTERN.sub(_replacer, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _coerce_numeric(value: Any) -> Any:
    """Coerce string values that look like numbers back into int/float.

    This is needed because ``${TTS_SPEED}`` resolves to the string ``"1"``
    even though it should be a float.  We walk the entire config tree once
    after env-var resolution.
    """
    if isinstance(value, str):
        # Try int first, then float
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
    if isinstance(value, dict):
        return {k: _coerce_numeric(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric(item) for item in value]
    return value


_FEATURE_FLAGS = {
    "ASKME_FEATURE_ROBOT": ("robot", "enabled"),
    "ASKME_FEATURE_VOICE": ("voice", "_enabled"),
    "ASKME_FEATURE_MEMORY": ("memory", "enabled"),
}

_TRUTHY = {"1", "true", "yes"}
_FALSY = {"0", "false", "no"}
_ALSA_DEVICE_PATTERN = re.compile(
    r"^(?:plug)?hw:|^sysdefault:|^dmix:|^dsnoop:|^front:|^surround",
    flags=re.IGNORECASE,
)
_PROJECT_RELATIVE_CONFIG_PATHS = (
    ("app", "data_dir"),
    ("app", "log_file"),
    ("brain", "soul_file"),
    ("memory", "mempalace_palace_path"),
    ("memory", "knowledge_index_jobs", "path"),
    ("conversation", "history_file"),
    ("voice", "asr", "model_dir"),
    ("voice", "vad", "model"),
    ("voice", "vad", "model_path"),
    ("voice", "punctuation", "model_path"),
    ("voice", "kws", "model_dir"),
    ("voice", "tts", "model_dir"),
    ("voice", "tts", "phrase_cache_dir"),
    ("voice", "tts", "voice_profile_state_path"),
    ("voice", "control_state_path"),
    ("vision", "model_path"),
    ("robot", "policy_model_path"),
    ("proactive", "alerts", "incident_archive_path"),
    ("ota", "state_file"),
    ("field_operations", "archive_path"),
    ("field_operations", "scenario_report_path"),
    ("field_operations", "site_profile_path"),
    (
        "field_operations",
        "delivery_resource_governance",
        "delivery_owner_notifications",
        "incident_archive_path",
    ),
    ("field_operations", "action_audit", "path"),
    ("field_operations", "action_audit", "retry_queue_path"),
    ("space_cognition", "store_path"),
    ("perception", "interaction_provider", "paths", "pose_gaze"),
    ("perception", "interaction_provider", "paths", "gesture"),
    ("perception", "interaction_provider", "paths", "sound_source"),
    (
        "perception",
        "interaction_provider",
        "paths",
        "audio_visual_association",
    ),
    ("perception", "interaction_provider", "paths", "approach_dwell"),
    (
        "perception",
        "interaction_provider",
        "paths",
        "multi_person_arbitration",
    ),
    ("audit", "export", "output_dir"),
    ("audit", "export", "retry_queue_path"),
    ("runtime_handoff", "audit", "path"),
    ("runtime_handoff", "store", "path"),
)


def _apply_feature_flags(config: dict) -> None:
    """Override config sections based on ``ASKME_FEATURE_*`` env vars."""
    for env_var, (section, key) in _FEATURE_FLAGS.items():
        val = os.environ.get(env_var, "").strip().lower()
        if not val:
            continue
        if section == "voice" and key == "_enabled":
            # Special: voice toggle removes/adds entire section
            if val in _FALSY:
                config.pop("voice", None)
            elif val in _TRUTHY:
                config.setdefault("voice", {})
        else:
            if val in _TRUTHY:
                config.setdefault(section, {})[key] = True
            elif val in _FALSY:
                config.setdefault(section, {})[key] = False


def _looks_like_alsa_device(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(_ALSA_DEVICE_PATTERN.match(value.strip()))


def _apply_project_relative_paths(config: dict) -> None:
    """Anchor configured local files to the project, not the process cwd."""
    for config_path in _PROJECT_RELATIVE_CONFIG_PATHS:
        section: Any = config
        for part in config_path[:-1]:
            if not isinstance(section, dict):
                break
            section = section.get(part)
        if not isinstance(section, dict):
            continue

        key = config_path[-1]
        value = section.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        section[key] = str(path.resolve())


def _apply_platform_audio_overrides(config: dict) -> None:
    """Normalize hardware-specific audio settings for the current OS.

    The field robot profile uses ALSA/MCP01 values such as ``plughw:1,0`` and
    ``usb_direct``.  Those are valid on the Linux robot, but on Windows they
    make PortAudio try an impossible device and break real microphone/speaker
    tests.  Keep the config file deployable to the robot while resolving the
    runtime config to native Windows sounddevice endpoints during local demos.
    """
    if os.environ.get("ASKME_DISABLE_PLATFORM_AUDIO_OVERRIDES", "").strip().lower() in _TRUTHY:
        return
    if os.name != "nt":
        return

    voice_cfg = config.get("voice")
    if not isinstance(voice_cfg, dict):
        return

    overrides: list[str] = []

    platform_overrides = voice_cfg.get("platform_overrides", {})
    windows_overrides = (
        platform_overrides.get("windows", {}) if isinstance(platform_overrides, dict) else {}
    )
    if isinstance(windows_overrides, dict):
        for key in (
            "input_device",
            "input_transport",
            "mic_native_rate",
            "mic_channels",
            "mic_channel_select",
        ):
            if key in windows_overrides:
                voice_cfg[key] = windows_overrides[key]
                overrides.append(f"voice.{key}: Windows platform override")

        tts_cfg = voice_cfg.get("tts")
        windows_tts = windows_overrides.get("tts", {})
        if isinstance(tts_cfg, dict) and isinstance(windows_tts, dict):
            for key in ("output_device", "output_transport"):
                if key in windows_tts:
                    tts_cfg[key] = windows_tts[key]
                    overrides.append(f"voice.tts.{key}: Windows platform override")

    if _looks_like_alsa_device(voice_cfg.get("input_device")):
        voice_cfg["input_device"] = None
        overrides.append("voice.input_device: ALSA name -> Windows default input")

    input_transport = str(voice_cfg.get("input_transport", "auto")).strip().lower()
    if input_transport == "usb_direct":
        voice_cfg["input_transport"] = "sounddevice"
        overrides.append("voice.input_transport: usb_direct -> sounddevice")

    tts_cfg = voice_cfg.get("tts")
    if isinstance(tts_cfg, dict):
        if _looks_like_alsa_device(tts_cfg.get("output_device")):
            tts_cfg["output_device"] = None
            overrides.append("voice.tts.output_device: ALSA name -> Windows default output")

        output_transport = str(tts_cfg.get("output_transport", "auto")).strip().lower()
        if output_transport in {"usb_direct", "aplay"}:
            tts_cfg["output_transport"] = "sounddevice"
            overrides.append(f"voice.tts.output_transport: {output_transport} -> sounddevice")

    if overrides:
        config["_platform_audio_overrides"] = overrides


def _load_config_from_disk() -> dict:
    """Load config.yaml, merge .env, resolve placeholders, return dict."""
    # 1. Load .env into os.environ
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    # 2. Load YAML (support ASKME_CONFIG_PATH override)
    config_path_env = os.environ.get("ASKME_CONFIG_PATH")
    config_path = Path(config_path_env) if config_path_env else _PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    # 3. Resolve ${VAR} references from environment
    resolved = _resolve_env_vars(raw)

    # 4. Coerce numeric strings produced by env-var substitution
    resolved = _coerce_numeric(resolved)

    # 5. Apply feature flag overrides
    _apply_feature_flags(resolved)

    # 6. Resolve project-owned files independently of process cwd
    _apply_project_relative_paths(resolved)

    # 7. Apply platform-specific safe defaults
    _apply_platform_audio_overrides(resolved)

    # 8. Inject convenience helpers
    resolved["_project_root"] = str(_PROJECT_ROOT)

    return resolved


# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------

_config_cache: dict | None = None


def get_config(*, reload: bool = False) -> dict:
    """Return the global config dict (cached singleton).

    Parameters
    ----------
    reload:
        If ``True``, re-read config.yaml and .env instead of returning cache.
    """
    global _config_cache
    if _config_cache is None or reload:
        _config_cache = _load_config_from_disk()
    return _config_cache


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def get_section(section: str) -> dict:
    """Shortcut to ``get_config()[section]``."""
    return get_config().get(section, {})


def project_root() -> Path:
    """Return the resolved project root path."""
    return _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(config: dict | None = None) -> list[str]:
    """Return a list of configuration problems (empty = all OK).

    Checks required fields and reports missing/invalid values.
    Warnings about type/range issues are also included — callers should
    log them but must not crash; only hard-missing fields are errors.
    """
    if config is None:
        config = get_config()

    errors: list[str] = []

    # Brain (required)
    brain = config.get("brain", {})
    provider = str(brain.get("provider") or "").strip().lower().replace("_", "-")
    if not provider:
        errors.append("brain.provider is required")
    uses_litellm = provider in {"litellm", "litellm-proxy", "llm-gateway"}
    api_key_env = "LITELLM_VIRTUAL_KEY" if uses_litellm else "DEEPSEEK_API_KEY"
    base_url_env = "LITELLM_BASE_URL" if uses_litellm else "DEEPSEEK_BASE_URL"
    if not brain.get("api_key"):
        errors.append(f"brain.api_key ({api_key_env}) is required")
    if not brain.get("base_url"):
        errors.append(f"brain.base_url ({base_url_env}) is required")

    # brain.timeout — must be a number > 0
    timeout_val = brain.get("timeout")
    if timeout_val is not None:
        try:
            timeout_f = float(timeout_val)
            if timeout_f <= 0:
                errors.append(f"brain.timeout must be > 0, got {timeout_val!r}")
        except (TypeError, ValueError):
            errors.append(f"brain.timeout must be a number, got {timeout_val!r}")

    # brain.max_retries — must be integer 0-10
    max_retries_val = brain.get("max_retries")
    if max_retries_val is not None:
        try:
            max_retries_i = int(max_retries_val)
            if not (0 <= max_retries_i <= 10):
                errors.append(f"brain.max_retries must be 0-10, got {max_retries_val!r}")
        except (TypeError, ValueError):
            errors.append(f"brain.max_retries must be an integer, got {max_retries_val!r}")

    if uses_litellm:
        if str(brain.get("health_model") or "").strip() != "health-probe":
            errors.append("brain.health_model must be 'health-probe' when provider=litellm")
        try:
            resolved_retries = 0 if max_retries_val is None else int(max_retries_val)
        except (TypeError, ValueError):
            resolved_retries = None
        if resolved_retries is not None and resolved_retries != 0:
            errors.append("brain.max_retries must be 0 when provider=litellm")
        if brain.get("fallback_models"):
            errors.append("brain.fallback_models must be empty when provider=litellm")
        if brain.get("minimax_api_key"):
            errors.append("brain.minimax_api_key must be empty when provider=litellm")

    # brain.model — must not be empty when present
    model_val = brain.get("model")
    if model_val is not None and not str(model_val).strip():
        errors.append("brain.model must not be an empty string")

    # conversation.max_history — must be integer 10-200
    conv = config.get("conversation", {})
    max_history_val = conv.get("max_history")
    if max_history_val is not None:
        try:
            max_history_i = int(max_history_val)
            if not (10 <= max_history_i <= 200):
                errors.append(f"conversation.max_history must be 10-200, got {max_history_val!r}")
        except (TypeError, ValueError):
            errors.append(f"conversation.max_history must be an integer, got {max_history_val!r}")

    for timeout_key in (
        "chat_timeout_s",
        "runtime_voice_turn_timeout_s",
        "chat_slow_threshold_ms",
    ):
        timeout_val = conv.get(timeout_key)
        if timeout_val is None:
            continue
        try:
            timeout_f = float(timeout_val)
            if timeout_f < 0:
                errors.append(f"conversation.{timeout_key} must be >= 0, got {timeout_val!r}")
        except (TypeError, ValueError):
            errors.append(f"conversation.{timeout_key} must be a number, got {timeout_val!r}")

    history_limit_val = conv.get("chat_diagnostics_history_limit")
    if history_limit_val is not None:
        try:
            history_limit_i = int(history_limit_val)
            if not (1 <= history_limit_i <= 1000):
                errors.append(
                    "conversation.chat_diagnostics_history_limit must be 1-1000, "
                    f"got {history_limit_val!r}"
                )
        except (TypeError, ValueError):
            errors.append(
                "conversation.chat_diagnostics_history_limit must be an integer, "
                f"got {history_limit_val!r}"
            )

    concurrency_val = conv.get("chat_max_concurrency")
    if concurrency_val is not None:
        try:
            concurrency_i = int(concurrency_val)
            if not (1 <= concurrency_i <= 256):
                errors.append(
                    f"conversation.chat_max_concurrency must be 1-256, got {concurrency_val!r}"
                )
        except (TypeError, ValueError):
            errors.append(
                f"conversation.chat_max_concurrency must be an integer, got {concurrency_val!r}"
            )

    # health_server.port — must be integer 1024-65535
    memory_cfg = config.get("memory", {})
    if memory_cfg is None:
        memory_cfg = {}
    if not isinstance(memory_cfg, dict):
        errors.append("memory must be a mapping")
        memory_cfg = {}

    backend_val = memory_cfg.get("backend")
    if backend_val is not None:
        allowed_memory_backends = {"auto", "mem0", "robotmem", "mempalace", "vector"}
        if str(backend_val).strip().lower() not in allowed_memory_backends:
            errors.append(
                "memory.backend must be one of "
                f"{sorted(allowed_memory_backends)}, got {backend_val!r}"
            )

    auto_backend_order_val = memory_cfg.get("auto_backend_order")
    if auto_backend_order_val is not None:
        allowed_auto_backends = {"mem0", "robotmem", "mempalace", "vector"}
        if not isinstance(auto_backend_order_val, list):
            errors.append("memory.auto_backend_order must be a list")
        else:
            invalid = [
                item
                for item in auto_backend_order_val
                if str(item).strip().lower() not in allowed_auto_backends
            ]
            if invalid:
                errors.append(
                    "memory.auto_backend_order contains unsupported backend(s): "
                    + ", ".join(str(item) for item in invalid)
                )

    retrieve_cache_ttl_val = memory_cfg.get("retrieve_cache_ttl_s")
    if retrieve_cache_ttl_val is not None:
        try:
            retrieve_cache_ttl_f = float(retrieve_cache_ttl_val)
            if retrieve_cache_ttl_f < 0:
                errors.append(
                    f"memory.retrieve_cache_ttl_s must be >= 0, got {retrieve_cache_ttl_val!r}"
                )
        except (TypeError, ValueError):
            errors.append(
                f"memory.retrieve_cache_ttl_s must be a number, got {retrieve_cache_ttl_val!r}"
            )

    retrieve_cache_max_entries_val = memory_cfg.get("retrieve_cache_max_entries")
    if retrieve_cache_max_entries_val is not None:
        try:
            retrieve_cache_max_entries_i = int(retrieve_cache_max_entries_val)
            if not (1 <= retrieve_cache_max_entries_i <= 10000):
                errors.append(
                    "memory.retrieve_cache_max_entries must be 1-10000, "
                    f"got {retrieve_cache_max_entries_val!r}"
                )
        except (TypeError, ValueError):
            errors.append(
                "memory.retrieve_cache_max_entries must be an integer, "
                f"got {retrieve_cache_max_entries_val!r}"
            )

    health_cfg = config.get("health_server", {})
    port_val = health_cfg.get("port")
    if port_val is not None:
        try:
            port_i = int(port_val)
            if not (1024 <= port_i <= 65535):
                errors.append(f"health_server.port must be 1024-65535, got {port_val!r}")
        except (TypeError, ValueError):
            errors.append(f"health_server.port must be an integer, got {port_val!r}")

    # tools.general_chat_max_safety_level — must be one of the allowed values
    _ALLOWED_SAFETY_LEVELS = {"normal", "dangerous", "critical"}
    tools_cfg = config.get("tools", {})
    if tools_cfg is None:
        tools_cfg = {}
    if not isinstance(tools_cfg, dict):
        errors.append("tools must be a mapping")
        tools_cfg = {}
    safety_level_val = tools_cfg.get("general_chat_max_safety_level")
    if safety_level_val is not None:
        if str(safety_level_val) not in _ALLOWED_SAFETY_LEVELS:
            errors.append(
                f"tools.general_chat_max_safety_level must be one of "
                f"{sorted(_ALLOWED_SAFETY_LEVELS)}, got {safety_level_val!r}"
            )

    for key, min_value, max_value in (
        ("executor_max_workers", 1, 128),
        ("queue_max_size", 1, 100000),
        ("job_history_limit", 1, 100000),
        ("circuit_failure_threshold", 0, 1000),
    ):
        val = tools_cfg.get(key)
        if val is None:
            continue
        try:
            int_val = int(val)
            if not (min_value <= int_val <= max_value):
                errors.append(f"tools.{key} must be {min_value}-{max_value}, got {val!r}")
        except (TypeError, ValueError):
            errors.append(f"tools.{key} must be an integer, got {val!r}")

    for key in ("rate_limit_per_minute", "circuit_cooldown_seconds"):
        val = tools_cfg.get(key)
        if val is None:
            continue
        try:
            float_val = float(val)
            if float_val < 0:
                errors.append(f"tools.{key} must be >= 0, got {val!r}")
        except (TypeError, ValueError):
            errors.append(f"tools.{key} must be a number, got {val!r}")

    priority_by_safety = tools_cfg.get("priority_by_safety")
    if priority_by_safety is not None:
        if not isinstance(priority_by_safety, dict):
            errors.append("tools.priority_by_safety must be a mapping")
        else:
            invalid_levels = [
                level for level in priority_by_safety if str(level) not in _ALLOWED_SAFETY_LEVELS
            ]
            if invalid_levels:
                errors.append(
                    "tools.priority_by_safety contains unsupported level(s): "
                    + ", ".join(str(level) for level in invalid_levels)
                )
            for level, priority in priority_by_safety.items():
                try:
                    int(priority)
                except (TypeError, ValueError):
                    errors.append(
                        f"tools.priority_by_safety.{level} must be an integer, got {priority!r}"
                    )

    # Optional realtime speech-to-speech lane.  Validation is offline and
    # never attempts to use credentials or open a provider connection.
    try:
        from askme.voice.realtime.config import resolve_realtime_voice_config

        errors.extend(resolve_realtime_voice_config(config).validation_errors())
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    # Voice TTS (no validation needed -- edge-tts requires no API key)

    ota = config.get("ota", {})
    if ota.get("enabled"):
        if not ota.get("server_url"):
            errors.append("ota.server_url (OTA_SERVER_URL) is required when ota.enabled=true")

    return errors
