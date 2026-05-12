"""RuntimeHandoffModule - fake arbiter bridge for confirmed cognition plans."""

from __future__ import annotations

import logging
from typing import Any

from askme.runtime.handoff import RuntimeHandoffService
from askme.runtime.module import Module, ModuleRegistry, Out

logger = logging.getLogger(__name__)


class RuntimeHandoffModule(Module):
    """Own confirmed-plan handoff without owning physical robot execution."""

    name = "runtime_handoff"
    depends_on = ("cognition",)
    provides = ("runtime_handoff_service",)

    runtime_handoff_service: Out[RuntimeHandoffService]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        handoff_cfg = cfg.get("runtime_handoff", {}) if isinstance(cfg, dict) else {}
        runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
        voice_bridge = runtime_cfg.get("voice_bridge", {}) if isinstance(runtime_cfg, dict) else {}
        cognition_mod = registry.get("cognition")
        safety_mod = registry.get("safety")
        world_state = getattr(cognition_mod, "world_state", None) if cognition_mod else None
        dog_safety_client = _dog_safety_client_from(safety_mod)
        self.enabled = bool(handoff_cfg.get("enabled", True))
        profile = str(handoff_cfg.get("profile", "fake")).strip().lower()
        auto_complete = bool(
            handoff_cfg.get(
                f"{profile}_auto_complete",
                handoff_cfg.get("fake_auto_complete", True if profile == "fake" else False),
            )
        )
        self._service = RuntimeHandoffService(
            world_state=world_state,
            default_operator_id=str(
                handoff_cfg.get("operator_id")
                or voice_bridge.get("operator_id")
                or "askme.operator"
            ),
            planner_version=str(handoff_cfg.get("planner_version", "askme-cognition-v1")),
            profile=profile,
            auto_complete=auto_complete,
            max_world_state_age_s=float(handoff_cfg.get("max_world_state_age_seconds", 30.0)),
            max_runs=int(handoff_cfg.get("max_runs", 50)),
            audit_config=_audit_config_from(handoff_cfg),
            store_config=_store_config_from(handoff_cfg),
            dog_safety_client=dog_safety_client,
            require_dog_safety=bool(handoff_cfg.get("require_dog_safety", False)),
            external_runtime_config=_external_runtime_config_from(handoff_cfg),
        )
        logger.info(
            (
                "RuntimeHandoffModule: built "
                "(enabled=%s, profile=%s, auto_complete=%s, dog_safety=%s)"
            ),
            self.enabled,
            profile,
            auto_complete,
            dog_safety_client is not None,
        )

    @property
    def runtime_handoff_service(self) -> RuntimeHandoffService:  # type: ignore[override]
        return self._service

    def submit_plan_payload(self, plan: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("runtime handoff module is disabled")
        return self._service.submit_plan_payload(plan)

    def context_payload(self) -> dict[str, Any]:
        return self._service.context_payload()

    def list_payload(self) -> dict[str, Any]:
        return self._service.list_payload()

    def events_payload(
        self,
        *,
        after: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        return self._service.events_payload(after=after, limit=limit)

    def get_payload(self, run_id: str) -> dict[str, Any]:
        return self._service.get_payload(run_id)

    def report_payload(self, run_id: str) -> dict[str, Any]:
        return self._service.report_payload(run_id)

    def profiles_payload(self) -> dict[str, Any]:
        return self._service.profiles_payload()

    def pause_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any]:
        return self._service.pause_payload(
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
        )

    def resume_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any]:
        return self._service.resume_payload(
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
        )

    def cancel_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any]:
        return self._service.cancel_payload(
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
        )

    def advance_payload(
        self,
        run_id: str,
        *,
        operator_id: str = "askme.operator",
        reason: str = "",
        risk_acknowledgement: bool = False,
    ) -> dict[str, Any]:
        return self._service.advance_payload(
            run_id,
            operator_id=operator_id,
            reason=reason,
            risk_acknowledgement=risk_acknowledgement,
        )

    def handle_chat_control(self, text: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return self._service.handle_chat_control(text)

    def voice_turn_payload(
        self,
        text: str,
        *,
        transcript_id: str = "",
        confidence: float | None = None,
        is_final: bool = True,
        channel: str = "voice",
        speak: bool = False,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "handled": False,
                "reason": "runtime_handoff_disabled",
                "voice_turn": {
                    "recognized_text": str(text or ""),
                    "transcript_id": transcript_id,
                    "confidence": confidence,
                    "is_final": is_final,
                    "channel": channel,
                    "safety_bypass_allowed": False,
                },
            }
        return self._service.voice_turn_payload(
            text,
            transcript_id=transcript_id,
            confidence=confidence,
            is_final=is_final,
            channel=channel,
            speak=speak,
        )

    def health(self) -> dict[str, Any]:
        payload = self._service.health()
        payload["status"] = "ok" if self.enabled else "disabled"
        return payload

    def capabilities(self) -> dict[str, Any]:
        payload = self._service.capabilities()
        payload["enabled"] = self.enabled
        return payload


def _dog_safety_client_from(module: Any | None) -> Any | None:
    if module is None:
        return None
    for attr in ("safety_client", "client", "dog_safety"):
        client = getattr(module, attr, None)
        if client is not None:
            return client
    return None


def _audit_config_from(handoff_cfg: dict[str, Any]) -> dict[str, Any]:
    audit_cfg = handoff_cfg.get("audit", {})
    if not isinstance(audit_cfg, dict):
        audit_cfg = {}
    return {
        "enabled": bool(
            audit_cfg.get("enabled", handoff_cfg.get("audit_log_enabled", False))
        ),
        "path": (
            audit_cfg.get("path")
            or audit_cfg.get("jsonl_path")
            or handoff_cfg.get("audit_log_path")
        ),
        "swallow_errors": bool(audit_cfg.get("swallow_errors", True)),
    }


def _store_config_from(handoff_cfg: dict[str, Any]) -> dict[str, Any]:
    store_cfg = handoff_cfg.get("store", {})
    if not isinstance(store_cfg, dict):
        store_cfg = {}
    return {
        "enabled": bool(store_cfg.get("enabled", handoff_cfg.get("store_enabled", False))),
        "path": (
            store_cfg.get("path")
            or store_cfg.get("json_path")
            or handoff_cfg.get("store_path")
        ),
        "swallow_errors": bool(store_cfg.get("swallow_errors", True)),
    }


def _external_runtime_config_from(handoff_cfg: dict[str, Any]) -> dict[str, Any]:
    external_cfg = handoff_cfg.get("external_runtime", {})
    if not isinstance(external_cfg, dict):
        external_cfg = {}
    return {
        **external_cfg,
        "endpoint": (
            external_cfg.get("endpoint")
            or handoff_cfg.get("endpoint")
            or handoff_cfg.get("runtime_endpoint")
        ),
        "enable_external_runtime": bool(
            external_cfg.get(
                "enable_external_runtime",
                handoff_cfg.get("enable_external_runtime", False),
            )
        ),
        "timeout_seconds": external_cfg.get(
            "timeout_seconds",
            handoff_cfg.get("runtime_timeout_seconds", handoff_cfg.get("timeout_seconds", 5.0)),
        ),
    }
