"""Fail-closed readiness decision for local full-duplex voice."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from askme.voice.input.aec_processor import AecStats


@dataclass(frozen=True, slots=True)
class FullDuplexDecision:
    requested: bool
    enabled: bool
    echo_control: str
    reason: str
    aec_backend: str


def decide_full_duplex(
    config: Mapping[str, Any] | None,
    *,
    aec_status: AecStats,
) -> FullDuplexDecision:
    """Enable concurrency only when a real echo-control path is known.

    ``hardware`` and ``system`` require a separate verification flag so a
    config typo cannot silently assert acoustic readiness. ``auto`` requires
    an active native AEC adapter; a noise/peak gate never qualifies.
    """

    cfg = config if isinstance(config, Mapping) else {}
    requested = bool(cfg.get("enabled", False))
    requested_echo_control = str(cfg.get("echo_control", "auto") or "auto").lower()

    if not requested:
        return FullDuplexDecision(
            requested=False,
            enabled=False,
            echo_control="none",
            reason="not_requested",
            aec_backend=aec_status.backend,
        )

    if requested_echo_control in {"hardware", "system"}:
        if not bool(cfg.get("echo_control_verified", False)):
            return FullDuplexDecision(
                requested=True,
                enabled=False,
                echo_control="none",
                reason="echo_control_unverified",
                aec_backend=aec_status.backend,
            )
        return FullDuplexDecision(
            requested=True,
            enabled=True,
            echo_control=requested_echo_control,
            reason="verified_echo_control",
            aec_backend=requested_echo_control,
        )

    if requested_echo_control not in {"auto", "native"}:
        return FullDuplexDecision(
            requested=True,
            enabled=False,
            echo_control="none",
            reason="invalid_echo_control",
            aec_backend=aec_status.backend,
        )

    if aec_status.available and aec_status.active and not aec_status.degraded:
        return FullDuplexDecision(
            requested=True,
            enabled=True,
            echo_control="native",
            reason="native_aec_ready",
            aec_backend=aec_status.backend,
        )

    return FullDuplexDecision(
        requested=True,
        enabled=False,
        echo_control="none",
        reason="aec_unavailable",
        aec_backend=aec_status.backend,
    )
