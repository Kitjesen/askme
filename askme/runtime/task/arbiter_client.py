"""Disabled-by-default readiness contract for external runtime transports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EXTERNAL_RUNTIME_PROFILES = frozenset({"external", "lab"})


@dataclass(frozen=True)
class RuntimeArbiterClientError:
    """Structured runtime-client configuration error."""

    code: str
    message: str
    remediation: str
    profile: str
    endpoint_configured: bool
    enable_external_runtime: bool
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "remediation": self.remediation,
            "profile": self.profile,
            "endpoint_configured": self.endpoint_configured,
            "enable_external_runtime": self.enable_external_runtime,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class RuntimeArbiterClient:
    """Local contract object for lab/external runtime submission gates.

    This class deliberately does not perform network I/O. It validates explicit
    enablement and builds a secret-free diagnostic envelope.  The actual HTTP
    transport is owned by :class:`ExternalRuntimeArbiter`.
    """

    profile: str
    endpoint: str = ""
    enable_external_runtime: bool = False
    timeout_s: float = 5.0

    @classmethod
    def from_config(
        cls,
        profile: str,
        config: dict[str, Any] | None,
    ) -> RuntimeArbiterClient:
        cfg = config or {}
        endpoint = cfg.get("endpoint") or cfg.get("runtime_endpoint") or ""
        return cls(
            profile=_normalize_external_profile(profile),
            endpoint=str(endpoint or "").strip(),
            enable_external_runtime=bool(cfg.get("enable_external_runtime", False)),
            timeout_s=max(0.1, float(cfg.get("timeout_seconds", cfg.get("timeout_s", 5.0)))),
        )

    @property
    def endpoint_configured(self) -> bool:
        return bool(self.endpoint)

    @property
    def hardware_dispatch(self) -> bool:
        return False

    def validate_submit_ready(self) -> RuntimeArbiterClientError | None:
        if self.profile not in EXTERNAL_RUNTIME_PROFILES:
            return RuntimeArbiterClientError(
                code="external_runtime_profile_required",
                message="RuntimeArbiterClient only accepts external or lab profiles.",
                remediation="Use fake, shadow, or sim for local execution, or select external/lab.",
                profile=self.profile,
                endpoint_configured=self.endpoint_configured,
                enable_external_runtime=self.enable_external_runtime,
            )
        if not self.enable_external_runtime:
            return RuntimeArbiterClientError(
                code="external_runtime_disabled",
                message="External runtime submission is disabled by default.",
                remediation="Set runtime_handoff.enable_external_runtime=true for lab/external profiles.",
                profile=self.profile,
                endpoint_configured=self.endpoint_configured,
                enable_external_runtime=False,
            )
        if not self.endpoint_configured:
            return RuntimeArbiterClientError(
                code="external_runtime_endpoint_required",
                message="External runtime submission requires an endpoint.",
                remediation="Set runtime_handoff.endpoint before using lab/external profiles.",
                profile=self.profile,
                endpoint_configured=False,
                enable_external_runtime=True,
            )
        return None

    def submission_envelope(self, handoff_payload: dict[str, Any]) -> dict[str, Any]:
        error = self.validate_submit_ready()
        if error is not None:
            return {
                "accepted": False,
                "reason": error.code,
                "error": error.to_dict(),
                "hardware_dispatch": False,
            }
        return {
            "accepted": True,
            "profile": self.profile,
            "endpoint": self.endpoint,
            "handoff_id": str(handoff_payload.get("handoff_id") or ""),
            "plan_id": str(handoff_payload.get("plan_id") or ""),
            "dispatch_mode": "transport_managed",
            "hardware_dispatch": False,
        }

    def safe_config(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "endpoint_configured": self.endpoint_configured,
            "enable_external_runtime": self.enable_external_runtime,
            "timeout_s": self.timeout_s,
            "hardware_dispatch": False,
        }


def _normalize_external_profile(value: str) -> str:
    profile = str(value or "").strip().lower()
    return profile if profile in EXTERNAL_RUNTIME_PROFILES else profile
