"""HTTP runtime configuration helpers for the FastAPI app."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_TRUTHY_CONFIG_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSY_CONFIG_VALUES = {"0", "false", "no", "off", "disabled"}


@dataclass(frozen=True)
class ConversationRuntimeSettings:
    """Parsed runtime controls for chat and voice-turn endpoints."""

    chat_timeout_s: float | None
    chat_max_concurrency: int
    chat_slow_threshold_ms: float | None
    chat_diagnostics_history_limit: int
    runtime_voice_turn_timeout_s: float | None


def bool_config(value: Any, *, default: bool = False) -> bool:
    """Parse an operator-facing boolean config value."""

    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUTHY_CONFIG_VALUES:
        return True
    if text in _FALSY_CONFIG_VALUES:
        return False
    return default


def path_prefix_config(value: Any, *, default: str) -> str:
    """Normalize URL path prefixes used by product API docs."""

    text = str(value or default).strip() or default
    if not text.startswith("/"):
        text = f"/{text}"
    return "" if text == "/" else text.rstrip("/")


def api_documentation_urls(
    app_config: dict[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Return FastAPI docs/OpenAPI URLs, disabled unless explicitly enabled."""

    env = os.environ if env is None else env
    api_cfg = app_config.get("api", {}) if isinstance(app_config, dict) else {}
    if not isinstance(api_cfg, dict):
        api_cfg = {}
    enabled_value: Any = env.get("ASKME_OPENAPI_ENABLED")
    if enabled_value in (None, ""):
        enabled_value = api_cfg.get("openapi_enabled", api_cfg.get("docs_enabled"))
    if not bool_config(enabled_value, default=False):
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}

    prefix = path_prefix_config(
        env.get("ASKME_API_DOCS_PREFIX") or api_cfg.get("docs_prefix"),
        default="/api",
    )
    return {
        "docs_url": f"{prefix}/docs" if prefix else "/docs",
        "redoc_url": f"{prefix}/redoc" if prefix else "/redoc",
        "openapi_url": f"{prefix}/openapi.json" if prefix else "/openapi.json",
    }


def optional_positive_float_config(value: Any, *, default: float) -> float | None:
    """Parse positive float config values; non-positive values disable the timeout."""

    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return None
    return parsed


def positive_int_config(value: Any, *, default: int) -> int:
    """Parse positive integer config values and clamp at one."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def conversation_runtime_settings(app_config: dict[str, Any]) -> ConversationRuntimeSettings:
    """Parse conversation runtime settings from app config."""

    conversation_cfg = app_config.get("conversation", {}) if isinstance(app_config, dict) else {}
    if not isinstance(conversation_cfg, dict):
        conversation_cfg = {}
    return ConversationRuntimeSettings(
        chat_timeout_s=optional_positive_float_config(
            conversation_cfg.get("chat_timeout_s", 30.0),
            default=30.0,
        ),
        chat_max_concurrency=positive_int_config(
            conversation_cfg.get("chat_max_concurrency", 8),
            default=8,
        ),
        chat_slow_threshold_ms=optional_positive_float_config(
            conversation_cfg.get("chat_slow_threshold_ms", 2000.0),
            default=2000.0,
        ),
        chat_diagnostics_history_limit=positive_int_config(
            conversation_cfg.get("chat_diagnostics_history_limit", 20),
            default=20,
        ),
        runtime_voice_turn_timeout_s=optional_positive_float_config(
            conversation_cfg.get("runtime_voice_turn_timeout_s", 30.0),
            default=30.0,
        ),
    )


__all__ = [
    "ConversationRuntimeSettings",
    "api_documentation_urls",
    "bool_config",
    "conversation_runtime_settings",
    "optional_positive_float_config",
    "path_prefix_config",
    "positive_int_config",
]
