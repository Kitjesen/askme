"""Runtime lifecycle module for continuous provider-session warming."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from askme.runtime.core.module import Module, ModuleRegistry
from askme.runtime.warm_session_targets import (
    LLMWarmSessionTarget,
    TTSWarmSessionTarget,
)
from askme.runtime.warm_sessions import (
    WarmSessionBinding,
    WarmSessionManager,
    WarmSessionPolicy,
)

_POLICY_KEYS = frozenset(
    {
        "enabled",
        "startup_delay_seconds",
        "refresh_interval_seconds",
        "timeout_seconds",
        "initial_backoff_seconds",
        "max_backoff_seconds",
        "busy_retry_seconds",
        "jitter_ratio",
        "max_attempts_per_hour",
    }
)
_ROOT_KEYS = frozenset(
    {
        "enabled",
        "shutdown_timeout_seconds",
        "llm",
        "tts",
    }
)

_LLM_POLICY_DEFAULTS: dict[str, float | int] = {
    "startup_delay_seconds": 0.0,
    "refresh_interval_seconds": 45.0,
    "timeout_seconds": 20.0,
    "initial_backoff_seconds": 2.0,
    "max_backoff_seconds": 120.0,
    "busy_retry_seconds": 2.0,
    "jitter_ratio": 0.1,
    "max_attempts_per_hour": 80,
}
_TTS_POLICY_DEFAULTS: dict[str, float | int] = {
    "startup_delay_seconds": 0.5,
    "refresh_interval_seconds": 75.0,
    "timeout_seconds": 10.0,
    "initial_backoff_seconds": 2.0,
    "max_backoff_seconds": 60.0,
    "busy_retry_seconds": 1.0,
    "jitter_ratio": 0.1,
    "max_attempts_per_hour": 60,
}


def _unavailable_target_snapshot(reason: str) -> dict[str, Any]:
    """Return a schema-compatible target record for a missing runtime module."""

    return {
        "status": "unavailable",
        "attempts": 0,
        "successes": 0,
        "failures": 0,
        "skips": 0,
        "consecutive_failures": 0,
        "last_result": "unavailable",
        "last_status": "unavailable",
        "last_reason": reason,
        "last_latency_ms": None,
        "provider_session_key": None,
        "active_worker_age_seconds": None,
        "stuck_busy_threshold_seconds": None,
        "attempt_budget_remaining": 0,
        "last_success_age_seconds": None,
        "next_attempt_in_seconds": None,
    }


def _as_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    path: str,
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown key(s): {', '.join(unknown)}")


def _as_bool(value: Any, *, default: bool, path: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{path} must be a boolean")


def _as_float(value: Any, *, default: float, path: str) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a finite number")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite number") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{path} must be a finite number")
    return resolved


def _as_int(value: Any, *, default: int, path: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{path} must be an integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be an integer") from exc
    if isinstance(value, str) and str(resolved) != value.strip():
        raise ValueError(f"{path} must be an integer")
    return resolved


def _build_policy(
    value: Any,
    *,
    target: str,
    defaults: Mapping[str, float | int],
) -> tuple[bool, WarmSessionPolicy]:
    path = f"warm_sessions.{target}"
    section = _as_mapping(value, path=path)
    _reject_unknown_keys(section, allowed=_POLICY_KEYS, path=path)
    enabled = _as_bool(
        section.get("enabled"),
        default=True,
        path=f"{path}.enabled",
    )
    policy = WarmSessionPolicy(
        startup_delay_seconds=_as_float(
            section.get("startup_delay_seconds"),
            default=float(defaults["startup_delay_seconds"]),
            path=f"{path}.startup_delay_seconds",
        ),
        refresh_interval_seconds=_as_float(
            section.get("refresh_interval_seconds"),
            default=float(defaults["refresh_interval_seconds"]),
            path=f"{path}.refresh_interval_seconds",
        ),
        timeout_seconds=_as_float(
            section.get("timeout_seconds"),
            default=float(defaults["timeout_seconds"]),
            path=f"{path}.timeout_seconds",
        ),
        initial_backoff_seconds=_as_float(
            section.get("initial_backoff_seconds"),
            default=float(defaults["initial_backoff_seconds"]),
            path=f"{path}.initial_backoff_seconds",
        ),
        max_backoff_seconds=_as_float(
            section.get("max_backoff_seconds"),
            default=float(defaults["max_backoff_seconds"]),
            path=f"{path}.max_backoff_seconds",
        ),
        busy_retry_seconds=_as_float(
            section.get("busy_retry_seconds"),
            default=float(defaults["busy_retry_seconds"]),
            path=f"{path}.busy_retry_seconds",
        ),
        jitter_ratio=_as_float(
            section.get("jitter_ratio"),
            default=float(defaults["jitter_ratio"]),
            path=f"{path}.jitter_ratio",
        ),
        max_attempts_per_hour=_as_int(
            section.get("max_attempts_per_hour"),
            default=int(defaults["max_attempts_per_hour"]),
            path=f"{path}.max_attempts_per_hour",
        ),
    )
    return enabled, policy


class WarmSessionModule(Module):
    """Keep provider sessions ready from runtime start until runtime stop.

    The module owns the maintenance loops, not an immortal physical socket.
    Provider connections may rotate after TTL expiry, network failure, or a
    runtime hot switch while the product-level readiness contract stays alive.
    """

    name = "warm_sessions"
    depends_on = ("llm", "voice")
    provides = ("warm_sessions",)

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        section = _as_mapping(cfg.get("warm_sessions"), path="warm_sessions")
        _reject_unknown_keys(section, allowed=_ROOT_KEYS, path="warm_sessions")
        self._enabled = _as_bool(
            section.get("enabled"),
            default=False,
            path="warm_sessions.enabled",
        )
        self._registry = registry

        llm_enabled, llm_policy = _build_policy(
            section.get("llm"),
            target="llm",
            defaults=_LLM_POLICY_DEFAULTS,
        )
        tts_enabled, tts_policy = _build_policy(
            section.get("tts"),
            target="tts",
            defaults=_TTS_POLICY_DEFAULTS,
        )

        configured_targets: list[str] = []
        unavailable_targets: dict[str, str] = {}
        bindings: list[WarmSessionBinding] = []
        llm_module = registry.get("llm")
        voice_module = registry.get("voice")
        if self._enabled and llm_enabled:
            configured_targets.append("llm")
            if llm_module is None:
                unavailable_targets["llm"] = "llm_module_not_wired"
            else:
                bindings.append(
                    WarmSessionBinding(
                        name="llm",
                        target=LLMWarmSessionTarget(self._resolve_llm_target),
                        policy=llm_policy,
                    )
                )
        if self._enabled and tts_enabled:
            configured_targets.append("tts")
            if voice_module is None:
                unavailable_targets["tts"] = "voice_module_not_wired"
            else:
                bindings.append(
                    WarmSessionBinding(
                        name="tts",
                        target=TTSWarmSessionTarget(self._resolve_tts_target),
                        policy=tts_policy,
                    )
                )
        self._configured_targets = tuple(configured_targets)
        self._active_targets = tuple(binding.name for binding in bindings)
        self._unavailable_targets = unavailable_targets
        self._voice_module = voice_module

        shutdown_timeout = _as_float(
            section.get("shutdown_timeout_seconds"),
            default=0.5,
            path="warm_sessions.shutdown_timeout_seconds",
        )
        self._manager = WarmSessionManager(
            bindings,
            shutdown_timeout_seconds=shutdown_timeout,
        )

    async def start(self) -> None:
        if self._enabled:
            await self._manager.start()
            if "tts" in self._active_targets:
                self._set_tts_activation_callback(self._on_tts_activated)

    async def stop(self) -> None:
        self._set_tts_activation_callback(None)
        await self._manager.stop()

    def request_refresh(self, target: str) -> bool:
        """Request an immediate manager-owned refresh for a live target."""

        return self._manager.request_refresh(str(target).strip().lower())

    def health(self) -> dict[str, Any]:
        snapshot = self._manager.snapshot()
        running = bool(snapshot["running"])
        targets = dict(snapshot["targets"])
        for name, reason in self._unavailable_targets.items():
            targets[name] = _unavailable_target_snapshot(reason)
        degraded_targets = [
            name
            for name, target in targets.items()
            if target.get("status") in {"degraded", "error", "throttled", "unavailable"}
        ]
        status = "ok" if not self._enabled or running else "degraded"
        latency_warm = bool(
            self._enabled
            and running
            and self._configured_targets
            and not self._unavailable_targets
            and all(
                targets.get(name, {}).get("status") == "warm" for name in self._configured_targets
            )
        )
        return {
            "status": status,
            "enabled": self._enabled,
            "running": running,
            "latency_warm": latency_warm,
            "manager_status": snapshot["status"],
            "configured_targets": list(self._configured_targets),
            "active_targets": list(self._active_targets),
            "unavailable_targets": sorted(self._unavailable_targets),
            "degraded_targets": degraded_targets,
            "targets": targets,
        }

    def _on_tts_activated(self, _engine: Any) -> None:
        self.request_refresh("tts")

    def _set_tts_activation_callback(self, callback: Any | None) -> None:
        voice_module = self._voice_module
        if voice_module is None:
            return
        setter = getattr(voice_module, "set_tts_activation_callback", None)
        if callable(setter):
            setter(callback)

    def _resolve_llm_target(self) -> tuple[Any, str] | None:
        llm_module = self._registry.get("llm")
        if llm_module is None:
            return None
        resolver = getattr(llm_module, "resolve_live_warm_target", None) or getattr(
            llm_module, "resolve_warm_target", None
        )
        if callable(resolver):
            return resolver()

        client = getattr(llm_module, "client", None)
        if client is None:
            return None
        model = str(
            getattr(llm_module, "_warmup_model", None) or getattr(client, "model", "")
        ).strip()
        return (client, model) if model else None

    def _resolve_tts_target(self) -> Any | None:
        voice_module = self._registry.get("voice")
        if voice_module is None:
            return None
        return getattr(voice_module, "tts_provider", None)
