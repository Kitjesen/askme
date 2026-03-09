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

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    # 3. Resolve ${VAR} references from environment
    resolved = _resolve_env_vars(raw)

    # 4. Coerce numeric strings produced by env-var substitution
    resolved = _coerce_numeric(resolved)

    # 5. Apply feature flag overrides
    _apply_feature_flags(resolved)

    # 6. Inject convenience helpers
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
    """
    if config is None:
        config = get_config()

    errors: list[str] = []

    # Brain (required)
    brain = config.get("brain", {})
    if not brain.get("api_key"):
        errors.append("brain.api_key (DEEPSEEK_API_KEY) is required")
    if not brain.get("base_url"):
        errors.append("brain.base_url (DEEPSEEK_BASE_URL) is required")

    # Voice TTS (no validation needed -- edge-tts requires no API key)

    ota = config.get("ota", {})
    if ota.get("enabled"):
        if not ota.get("server_url"):
            errors.append("ota.server_url (OTA_SERVER_URL) is required when ota.enabled=true")

    return errors
