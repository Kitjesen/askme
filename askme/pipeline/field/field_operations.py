"""Executable field-operation event workflow.

This module turns product scenarios into a working event loop:
camera/sensor/robot payload -> rule validation -> alert dispatch -> event archive.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from askme.llm.core.contracts import LLMCallContext
from askme.pipeline.field.alert_dispatcher import AlertDispatcher
from askme.pipeline.field.customer_projects import build_site_profile_report
from askme.pipeline.field.field_action_audit import (
    FIELD_ACTION_AUDIT_GENESIS as _FIELD_ACTION_AUDIT_GENESIS,
)
from askme.pipeline.field.field_action_audit import (
    FIELD_ACTION_AUDIT_HASH_ALG as _FIELD_ACTION_AUDIT_HASH_ALG,
)
from askme.pipeline.field.field_action_audit import (
    FIELD_ACTION_AUDIT_SIGNATURE_ALG as _FIELD_ACTION_AUDIT_SIGNATURE_ALG,
)
from askme.pipeline.field.field_action_audit import (
    FieldActionAuditIntegrityError,
)
from askme.pipeline.field.field_action_audit import (
    field_action_audit_counts_by_event as _field_action_audit_counts_by_event,
)
from askme.pipeline.field.field_action_audit import (
    field_action_audit_hash as _field_action_audit_hash,
)
from askme.pipeline.field.field_action_audit import (
    field_action_audit_signature as _field_action_audit_signature,
)
from askme.pipeline.field.field_action_audit import (
    strict_field_action_audit_checkpoint as _strict_field_action_audit_checkpoint,
)
from askme.pipeline.field.field_deployment_readiness import build_field_deployment_readiness
from askme.pipeline.field.field_device_signature import (
    FIELD_DEVICE_SIGNATURE_ALG as _FIELD_DEVICE_SIGNATURE_ALG,
)
from askme.pipeline.field.field_device_signature import (
    FIELD_DEVICE_SIGNATURE_FIELDS as _FIELD_DEVICE_SIGNATURE_FIELDS,
)
from askme.pipeline.field.field_device_signature import (
    field_device_id as _field_device_id,
)
from askme.pipeline.field.field_device_signature import (
    field_device_signature_timestamp as _field_device_signature_timestamp,
)
from askme.pipeline.field.field_device_signature import (
    field_device_signature_value as _field_device_signature_value,
)
from askme.pipeline.field.field_device_signature import (
    sign_field_device_payload,
)
from askme.pipeline.field.field_ingest_adapters import normalize_field_ingest_payload
from askme.pipeline.field.field_scenarios import FIELD_SCENARIOS, FieldScenario
from askme.pipeline.field.incident_alerts import format_incident_alert, format_incident_playbook
from askme.providers import resolve_voice_profile_id
from askme.robot_interaction.scenario_intents import SCENARIO_INTENT_RULES
from askme.schemas.field import FieldEventDetail
from askme.skills.contracts.field_capability_contracts import field_capability_routes

_SCENARIO_TO_INCIDENT_TOPIC = {
    "night_stranger_photo": "security.night_stranger_photo",
    "illegal_parking": "traffic.illegal_parking",
    "fire_or_smoke": "safety.fire_or_smoke",
    "trash_bin_full": "sanitation.trash_bin_full",
    "urgent_patrol_dispatch": "patrol.urgent_dispatch",
    "crowd_gathering": "security.crowd_gathering",
}

_ROBOT_FAULT_TOPICS = {
    "fall_unrecoverable": "robot.fall_unrecoverable",
    "immobilized": "navigation.immobilized",
    "malicious_blocking": "security.malicious_blocking",
    "joint_motor_fault": "actuator.joint_motor_fault",
}

_REQUIRED_PAYLOAD_KEYS = {
    "robot_abnormal_incident": ("location", "fault_type"),
    "night_stranger_photo": ("location", "zone_name", "image_path"),
    "illegal_parking": ("location", "zone_name", "image_path"),
    "fire_or_smoke": ("location", "image_path"),
    "trash_bin_full": ("location", "bin_id", "image_path"),
    "urgent_patrol_dispatch": ("target_location", "operator_id"),
    "crowd_gathering": ("location", "person_count", "duration_min", "image_path"),
    "wayfinding_help_point": ("help_point_id", "location"),
    "visitor_escort": ("destination", "location"),
}

_SCENARIO_INTENT_ALIASES = {
    "wayfinding": "wayfinding_help_point",
}

_SCENARIO_DEVICE_ENTRYPOINTS = {
    "robot_abnormal_incident": ("robot_status_event", "robot_fault"),
    "night_stranger_photo": ("domestic_night_photo",),
    "illegal_parking": ("camera_vehicle", "domestic_camera_parking"),
    "fire_or_smoke": ("smoke_sensor", "mqtt_smoke_alarm"),
    "trash_bin_full": ("trash_bin_alarm",),
    "urgent_patrol_dispatch": ("operator_dispatch",),
    "crowd_gathering": ("crowd_alarm",),
    "wayfinding_help_point": ("service_point_dwell", "voice_question"),
    "visitor_escort": ("voice_confirmed_destination", "runtime_handoff"),
}

_SCENARIO_ONSITE_DEPENDENCIES = {
    "robot_abnormal_incident": (
        "robot diagnostics",
        "robot runtime callback",
        "onsite voice playback",
    ),
    "night_stranger_photo": (
        "camera/VMS event stream",
        "sensitive-zone map",
        "security notification webhook",
    ),
    "illegal_parking": (
        "camera/VMS vehicle detection",
        "parking-zone map",
        "security notification webhook",
    ),
    "fire_or_smoke": (
        "smoke/temperature sensor",
        "camera evidence",
        "security notification webhook",
    ),
    "trash_bin_full": (
        "trash-bin point catalog",
        "camera evidence",
        "cleaning notification webhook",
    ),
    "urgent_patrol_dispatch": (
        "enterprise operator identity",
        "runtime arbiter",
        "robot navigation gateway",
    ),
    "crowd_gathering": (
        "people-count vision model",
        "dwell-time tracker",
        "security notification webhook",
    ),
    "wayfinding_help_point": (
        "service-point catalog",
        "approved space knowledge",
        "interaction gate",
    ),
    "visitor_escort": ("approved map route", "runtime arbiter", "robot navigation gateway"),
}

_FIELD_EVENT_SLA_SECONDS = {
    "P0": 300.0,
    "P1": 900.0,
    "P2": 1800.0,
    "P3": 3600.0,
}

_DEFAULT_OPERATOR_ROLES = {
    "askme.operator": ("operator",),
    "dashboard.operator": ("operator",),
    "guard-1": ("operator",),
    "security-1": ("operator",),
    "cleaning-1": ("operator",),
    "ops-1": ("operator",),
    "supervisor-1": ("supervisor",),
    "admin-1": ("admin",),
}

_FIELD_ACTION_ROLES = ("operator", "supervisor", "admin")
_SUPERVISOR_ROLES = ("supervisor", "admin")


def _audit_review_path_from_field_config(config: dict[str, Any]) -> Path | None:
    audit_cfg = config.get("audit") if isinstance(config.get("audit"), dict) else {}
    review_cfg = audit_cfg.get("review") if isinstance(audit_cfg.get("review"), dict) else {}
    raw = (
        review_cfg.get("path")
        or review_cfg.get("jsonl_path")
        or audit_cfg.get("review_path")
        or config.get("audit_review_path")
    )
    text = str(raw or "artifacts/audit/reviews.jsonl").strip()
    return Path(text) if text else None


_FIELD_RUNTIME_DELIVERY_STATUSES = frozenset(
    {
        "policy_ready",
        "submitted",
        "created",
        "validating",
        "preflight",
        "queued",
        "executing",
        "paused",
        "resuming",
        "blocked",
        "completed",
        "shadowed",
        "failed",
        "cancelling",
        "cancelled",
        "rejected",
        "skipped",
        "submission_failed",
    }
)


@dataclass
class FieldEventRecord:
    """One auditable field event."""

    event_id: str
    scenario_id: str
    scenario_name: str
    category: str
    priority: str
    status: str
    severity: str
    location: str
    created_at: float
    customer_id: str = ""
    project_id: str = ""
    site_id: str = ""
    site_name: str = ""
    industry: str = ""
    managed_object_id: str = ""
    managed_object_display: str = ""
    managed_object_category: str = ""
    managed_object_bindings: dict[str, Any] = field(default_factory=dict)
    resource_execution_context: dict[str, Any] = field(default_factory=dict)
    project_scope: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    incident_topic: str | None = None
    notification_group: str = "none"
    voice: str = ""
    dingtalk: str = ""
    operator_action: str = ""
    missing_evidence: list[str] = field(default_factory=list)
    evidence_media: list[dict[str, Any]] = field(default_factory=list)
    sent_channels: list[str] = field(default_factory=list)
    delivery_report: list[dict[str, Any]] = field(default_factory=list)
    notification_resends: list[dict[str, Any]] = field(default_factory=list)
    archive_required: bool = True
    freshness_status: str = "not_applicable"
    freshness_age_s: float | None = None
    confidence: float | None = None
    dedupe_key: str = ""
    duplicate_of: str | None = None
    llm_narrative_used: bool = False
    llm_narrative_status: str = "not_requested"
    llm_narrative_reason: str = ""
    playbook: dict[str, Any] = field(default_factory=dict)
    voice_directive: dict[str, Any] = field(default_factory=dict)
    voice_delivery: dict[str, Any] = field(default_factory=dict)
    runtime_delivery: dict[str, Any] = field(default_factory=dict)
    runtime_delivery_receipts: list[dict[str, Any]] = field(default_factory=list)
    memory_delivery: dict[str, Any] = field(default_factory=dict)
    incident_state: str = "active"
    incident_stage: str = "received"
    incident_workflow: dict[str, Any] = field(default_factory=dict)
    action_audit: list[dict[str, Any]] = field(default_factory=list)
    acknowledged_at: float | None = None
    acknowledged_by: str | None = None
    acknowledge_note: str = ""
    close_requested_at: float | None = None
    close_requested_by: str | None = None
    close_request_note: str = ""
    closed_at: float | None = None
    closed_by: str | None = None
    close_note: str = ""
    close_approval: dict[str, Any] = field(default_factory=dict)


class FieldOperationsService:
    """Rules, LLM-safe wording, and persistence for field operations."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        alert_dispatcher_factory: Any | None = None,
        llm_client: Any | None = None,
    ) -> None:
        cfg = _config_with_site_profile(config or {})
        self._config = cfg
        self._archive_path = Path(
            str(
                cfg.get("archive_path")
                or os.getenv("ASKME_FIELD_EVENTS_PATH")
                or "artifacts/field_ops/events.jsonl"
            )
        )
        audit_cfg = cfg.get("action_audit") if isinstance(cfg.get("action_audit"), dict) else {}
        audit_path = (
            audit_cfg.get("path")
            or cfg.get("action_audit_path")
            or self._archive_path.with_name("field-action-audit.jsonl")
        )
        self._action_audit_enabled = bool(
            audit_cfg.get("enabled", cfg.get("action_audit_enabled", True))
        )
        self._action_audit_path = Path(str(audit_path)) if audit_path else None
        self._action_audit_swallow_errors = bool(
            audit_cfg.get("swallow_errors", cfg.get("action_audit_swallow_errors", False))
        )
        self._action_audit_hmac_secret = _clean_secret(
            audit_cfg.get("hmac_secret")
            or cfg.get("action_audit_hmac_secret")
            or os.getenv("ASKME_FIELD_ACTION_AUDIT_HMAC_SECRET")
        )
        self._action_audit_signature_key_id = str(
            audit_cfg.get("signature_key_id")
            or cfg.get("action_audit_signature_key_id")
            or os.getenv("ASKME_FIELD_ACTION_AUDIT_SIGNATURE_KEY_ID")
            or "local-field-action-audit"
        )
        self._robot_id = str(cfg.get("robot_id") or os.getenv("ASKME_ROBOT_ID") or "robot-1")
        self._robot_name = str(
            cfg.get("robot_name") or os.getenv("ASKME_ROBOT_NAME") or "现场机器人"
        )
        self._alert_factory = alert_dispatcher_factory or AlertDispatcher
        self._llm = llm_client
        self._llm_narrative_enabled = bool(
            cfg.get("llm_narrative_enabled")
            or os.getenv("ASKME_FIELD_LLM_NARRATIVE") in {"1", "true", "yes"}
        )
        self._site_map = cfg.get("site_map") if isinstance(cfg.get("site_map"), dict) else {}
        self._project_scope = _field_project_scope_from_config(cfg)
        self._managed_objects = (
            cfg.get("managed_objects") if isinstance(cfg.get("managed_objects"), dict) else {}
        )
        self._thresholds = {
            "temperature_c": float(cfg.get("fire_temperature_c", 60.0)),
            "smoke_level": float(cfg.get("smoke_level", 0.7)),
            "trash_fill_ratio": float(cfg.get("trash_fill_ratio", 0.8)),
            "crowd_person_count": int(cfg.get("crowd_person_count", 5)),
            "crowd_duration_min": float(cfg.get("crowd_duration_min", 30.0)),
            "parking_duration_s": float(cfg.get("parking_duration_s", 120.0)),
            "night_stranger_dwell_s": float(cfg.get("night_stranger_dwell_s", 10.0)),
            "night_photo_dwell_s": float(cfg.get("night_photo_dwell_s", 3.0)),
        }
        self._max_input_age_s = float(cfg.get("max_input_age_s", 30.0))
        self._dedupe_window_s = float(cfg.get("dedupe_window_s", 120.0))
        self._min_detection_confidence = float(cfg.get("min_detection_confidence", 0.55))
        self._device_registry = _resolve_field_device_registry(
            cfg.get("device_registry") or cfg.get("field_devices") or cfg.get("devices")
        )
        self._require_trusted_devices = bool(
            cfg.get("require_trusted_devices")
        ) or _is_production_mode(cfg)
        self._device_signature_max_age_s = float(cfg.get("device_signature_max_age_s", 300.0))
        self._device_offline_after_s = float(cfg.get("device_offline_after_s", 300.0))
        self._webhooks = self._resolve_group_webhooks(cfg)
        self._webhook_secrets = self._resolve_group_secrets(cfg)
        self._operator_roles = _resolve_operator_roles(cfg.get("operators"))
        memory_cfg = (
            cfg.get("incident_memory") if isinstance(cfg.get("incident_memory"), dict) else {}
        )
        self._incident_memory_enabled = bool(
            memory_cfg.get("enabled", cfg.get("incident_memory_enabled", False))
            or os.getenv("ASKME_FIELD_INCIDENT_MEMORY") in {"1", "true", "yes"}
        )
        self._incident_memory_config = (
            memory_cfg.get("config")
            if isinstance(memory_cfg.get("config"), dict)
            else cfg.get("incident_memory_config")
            if isinstance(cfg.get("incident_memory_config"), dict)
            else {}
        )
        self._incident_memory_service = cfg.get("incident_memory_service")

    @classmethod
    def from_env(cls) -> FieldOperationsService:
        """Create a service using environment-level notification settings.

        The field workflow never owns LLM provider construction. Runtime owners
        that want narrative enrichment must inject their already-managed client.
        """

        return cls()

    def scenarios_payload(self) -> dict[str, Any]:
        return {
            "scenarios": [
                self._scenario_payload(scenario)
                for scenario in sorted(
                    FIELD_SCENARIOS.values(),
                    key=lambda s: (s.priority, s.scenario_id),
                )
            ]
        }

    def scenario_acceptance_matrix_payload(self) -> dict[str, Any]:
        """Return a customer-readable acceptance matrix for every product scenario."""

        rows = [
            self._scenario_acceptance_row(scenario)
            for scenario in sorted(
                FIELD_SCENARIOS.values(),
                key=lambda s: (s.priority, s.scenario_id),
            )
        ]
        scenario_count = len(rows)
        demo_ready_count = sum(1 for row in rows if row["acceptance_status"] == "demo_ready")
        natural_language_count = sum(1 for row in rows if row["natural_language_routes"])
        device_entrypoint_count = sum(1 for row in rows if row["device_entrypoints"])
        return {
            "matrix_type": "askme.field_scenario_acceptance_matrix",
            "summary": {
                "scenario_count": scenario_count,
                "demo_ready_count": demo_ready_count,
                "natural_language_route_count": natural_language_count,
                "device_entrypoint_count": device_entrypoint_count,
                "production_ready": False,
            },
            "policy": {
                "does_not_execute_skill": True,
                "does_not_dispatch_hardware": True,
                "customer_claim": (
                    "This matrix proves demo and integration acceptance coverage. "
                    "Production claims still require onsite devices, credentials, runtime callbacks, "
                    "and customer signoff evidence."
                ),
                "customer_claim_zh": (
                    "该矩阵证明演示和集成验收覆盖情况；生产上线声明仍需要真实设备、"
                    "生产凭证、运行回调和客户签收证据。"
                ),
            },
            "rows": rows,
        }

    def notification_preflight_payload(
        self,
        *,
        groups: list[str] | None = None,
        require_secret: bool = True,
    ) -> dict[str, Any]:
        """Report whether real responder notifications are configured."""

        group_names = groups or ["security", "cleaning", "operations"]
        allowed = {"security", "cleaning", "operations"}
        results: dict[str, dict[str, Any]] = {}
        blockers: list[str] = []
        for group in group_names:
            if group not in allowed:
                blockers.append(f"unknown notification group: {group}")
                results[group] = {
                    "ready": False,
                    "webhook_configured": False,
                    "secret_configured": False,
                    "missing_env": [],
                    "reason": "unknown_group",
                }
                continue
            webhook_configured = bool(self._webhooks.get(group))
            secret_configured = bool(self._webhook_secrets.get(group))
            missing_env = _notification_group_missing_env(
                group,
                webhook_configured=webhook_configured,
                secret_configured=secret_configured,
                require_secret=require_secret,
            )
            ready = webhook_configured and (secret_configured or not require_secret)
            if not ready:
                blockers.append(f"{group} notification is not fully configured")
            results[group] = {
                "ready": ready,
                "webhook_configured": webhook_configured,
                "secret_configured": secret_configured,
                "missing_env": missing_env,
                "reason": "" if ready else "missing_webhook_or_secret",
            }
        ready = not blockers and all(item["ready"] for item in results.values())
        return {
            "status": "ready" if ready else "blocked",
            "ready": ready,
            "require_secret": bool(require_secret),
            "groups": results,
            "blockers": blockers,
            "next_actions": _notification_preflight_next_actions(results),
        }

    def device_status_payload(self) -> dict[str, Any]:
        """Return registered and observed field-device online/trust status."""

        now = time.time()
        latest: dict[str, dict[str, Any]] = {}
        for event in self._read_events():
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            trust = (
                payload.get("device_trust") if isinstance(payload.get("device_trust"), dict) else {}
            )
            device_id = str(
                trust.get("device_id") or _field_device_id(payload, payload) or ""
            ).strip()
            if not device_id:
                continue
            seen_at = (
                _float_or_none(payload.get("_ingest_received_at"))
                or _float_or_none(event.get("created_at"))
                or 0.0
            )
            previous = latest.get(device_id)
            if previous and float(previous.get("last_seen_at") or 0.0) >= seen_at:
                continue
            latest[device_id] = {
                "device_id": device_id,
                "source": str(trust.get("source") or payload.get("source") or ""),
                "trusted": bool(trust.get("trusted")),
                "trust_status": str(trust.get("status") or "unknown"),
                "trust_reason": str(trust.get("reason") or ""),
                "signature_verified": bool(trust.get("signature_verified")),
                "last_seen_at": seen_at,
                "last_event_id": str(event.get("event_id") or ""),
                "last_scenario_id": str(event.get("scenario_id") or ""),
                "last_event_status": str(event.get("status") or ""),
            }

        devices: list[dict[str, Any]] = []
        for device_id in sorted(set(self._device_registry) | set(latest)):
            registry = self._device_registry.get(device_id, {})
            last = latest.get(device_id, {})
            last_seen_at = _float_or_none(last.get("last_seen_at"))
            age_s = None if last_seen_at is None else max(0.0, now - last_seen_at)
            if last_seen_at is None:
                status = "never_seen"
            elif age_s is not None and age_s > self._device_offline_after_s:
                status = "stale"
            else:
                status = "online"
            devices.append(
                {
                    "device_id": device_id,
                    "registered": device_id in self._device_registry,
                    "status": status,
                    "age_s": None if age_s is None else round(age_s, 3),
                    "offline_after_s": self._device_offline_after_s,
                    "allowed_sources": list(registry.get("allowed_sources") or []),
                    "signature_required": bool(registry.get("require_signature", False)),
                    "secret_configured": bool(registry.get("hmac_secret")),
                    **last,
                }
            )

        summary = {
            "registered": len(self._device_registry),
            "observed": len(latest),
            "online": len([item for item in devices if item["status"] == "online"]),
            "stale": len([item for item in devices if item["status"] == "stale"]),
            "never_seen": len([item for item in devices if item["status"] == "never_seen"]),
            "unregistered_observed": len([item for item in devices if not item["registered"]]),
        }
        return {
            "status": "ok",
            "require_trusted_devices": self._require_trusted_devices,
            "offline_after_s": self._device_offline_after_s,
            "summary": summary,
            "devices": devices,
        }

    def device_onboarding_payload(self) -> dict[str, Any]:
        """Return a delivery-facing onboarding report for real field devices."""

        status_payload = self.device_status_payload()
        devices = []
        ready_count = 0
        blocked_count = 0
        manual_check_count = 0
        for item in status_payload.get("devices", []):
            device = dict(item if isinstance(item, dict) else {})
            candidates = self._managed_objects_for_device(device)
            gate = _field_device_onboarding_gate(
                device,
                candidates,
                require_trusted_devices=self._require_trusted_devices,
            )
            if gate["status"] == "ready":
                ready_count += 1
            elif gate["status"] == "blocked":
                blocked_count += 1
            else:
                manual_check_count += 1
            devices.append(
                {
                    **device,
                    "managed_object_candidates": candidates,
                    "onboarding_gate": gate,
                }
            )
        summary = dict(status_payload.get("summary") or {})
        summary.update(
            {
                "ready": ready_count,
                "blocked": blocked_count,
                "manual_check": manual_check_count,
                "managed_object_candidate_count": sum(
                    len(item.get("managed_object_candidates") or []) for item in devices
                ),
            }
        )
        return {
            "report_type": "askme.field.device_onboarding_report.v1",
            "status": "ready" if blocked_count == 0 and manual_check_count == 0 else "manual_check",
            "customer_message": (
                "设备接入报告用于交付验收前确认：设备已登记、签名策略正确、"
                "已产生真实回传，并能绑定到客户现场对象。"
            ),
            "policy": {
                "device_payload_endpoint": "/api/field/ingest",
                "device_status_endpoint": "/api/field/devices",
                "signing_cli": "python -m askme runtime field-sign-device-payload",
                "client_project_fields_are_not_trusted": True,
                "production_signoff_requires_onsite_payloads": True,
            },
            "require_trusted_devices": self._require_trusted_devices,
            "offline_after_s": self._device_offline_after_s,
            "summary": summary,
            "devices": devices,
            "next_actions": _field_device_onboarding_next_actions(devices),
        }

    def list_payload(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
        notification_group: str | None = None,
        needs_attention: bool = False,
        tenant_id: str | None = None,
        delivery_namespace: str | None = None,
        customer_id: str | None = None,
        project_id: str | None = None,
        site_id: str | None = None,
        managed_object_id: str | None = None,
        project_scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        events = _filter_field_events_by_project_scope(
            self._read_events(),
            tenant_id=tenant_id,
            delivery_namespace=delivery_namespace,
            customer_id=customer_id,
            project_id=project_id,
            site_id=site_id,
            project_scope=project_scope,
        )
        filtered_events = _filter_field_events(
            events,
            status=status,
            notification_group=notification_group,
            needs_attention=needs_attention,
            customer_id=customer_id,
            project_id=project_id,
            site_id=site_id,
            managed_object_id=managed_object_id,
        )
        visible_events = [_field_event_view(event) for event in filtered_events]
        filter_payload: dict[str, Any] = {
            "status": status or "",
            "notification_group": notification_group or "",
            "needs_attention": bool(needs_attention),
        }
        for key, value in (
            ("tenant_id", tenant_id),
            ("delivery_namespace", delivery_namespace),
            ("customer_id", customer_id),
            ("project_id", project_id),
            ("site_id", site_id),
            ("managed_object_id", managed_object_id),
        ):
            if value:
                filter_payload[key] = value
        if project_scope:
            filter_payload["project_scope"] = project_scope
        return {
            "events": visible_events[-max(1, min(limit, 500)) :][::-1],
            "total": len(events),
            "filtered_total": len(filtered_events),
            "summary": _field_event_summary(events),
            "filter": filter_payload,
            "archive_path": str(self._archive_path),
        }

    def detail_payload(self, event_id: str) -> dict[str, Any]:
        event_key = str(event_id or "").strip()
        for event in self._read_events():
            if str(event.get("event_id") or "") == event_key:
                event_view = FieldEventDetail.from_dict(_field_event_view(event)).to_dict()
                return {
                    "found": True,
                    "event_id": event_key,
                    "event": event_view,
                }
        return {"found": False, "reason": "event_not_found", "event_id": event_key}

    def ingest_help_payload(self) -> dict[str, Any]:
        return self._product_ingest_help_payload()

    def _product_ingest_help_payload(self) -> dict[str, Any]:
        """Return clean product examples for real device integrations."""
        return {
            "accepted_sources": ["camera", "sensor", "robot", "map", "operator"],
            "examples": {
                "camera_vehicle": {
                    "source": "camera",
                    "observed_at": "2026-05-11T10:30:00+08:00",
                    "detections": [{"class_id": "2", "label": "vehicle", "confidence": 0.92}],
                    "zone_id": "main-road-1",
                    "duration_s": 180,
                    "image_path": "artifacts/evidence/car.jpg",
                },
                "domestic_camera_parking": {
                    "eventType": "车辆违停",
                    "timestamp": "2026-05-11T10:30:00+08:00",
                    "cameraIndexCode": "cam-main-road-01",
                    "zone_id": "main-road-1",
                    "duration_s": 180,
                    "pictureUrl": "artifacts/evidence/domestic-parking.jpg",
                },
                "domestic_night_photo": {
                    "alarmType": "夜间陌生人拍照",
                    "timestamp": "2026-05-11T22:30:00+08:00",
                    "cameraIndexCode": "cam-window-01",
                    "zone_id": "window-corner-1",
                    "is_night": True,
                    "known_person": False,
                    "duration_s": 4,
                    "snapshotUrl": "artifacts/evidence/night-photo.jpg",
                },
                "smoke_sensor": {
                    "source": "sensor",
                    "observed_at": "2026-05-11T10:30:00+08:00",
                    "sensor": {"temperature_c": 68, "smoke_level": 0.8},
                    "location": "配电室",
                    "image_path": "artifacts/evidence/smoke.jpg",
                },
                "trash_bin_alarm": {
                    "eventType": "垃圾桶满溢",
                    "timestamp": "2026-05-11T10:30:00+08:00",
                    "binId": "bin-17",
                    "location": "花园出口",
                    "imageUrl": "artifacts/evidence/bin-full.jpg",
                },
                "crowd_alarm": {
                    "eventType": "人员聚集",
                    "timestamp": "2026-05-11T10:30:00+08:00",
                    "personCount": 8,
                    "duration_min": 35,
                    "location": "中心广场",
                    "imageUrl": "artifacts/evidence/crowd.jpg",
                },
                "robot_fault": {
                    "source": "robot",
                    "observed_at": "2026-05-11T10:30:00+08:00",
                    "robot": {"fault_type": "joint_motor_fault", "joint_id": "hip-left"},
                    "location": "A区东路",
                },
                "mqtt_smoke_alarm": {
                    "topic": "site/A/power-room/smoke-01",
                    "payload": {
                        "timestamp": "2026-05-11T10:30:00+08:00",
                        "temperatureC": 72,
                        "smokeAlarm": True,
                        "location": "配电室",
                        "imageUrl": "artifacts/evidence/smoke.jpg",
                    },
                },
            },
            "bridge_contract": {
                "sign_payload": (
                    "python -m askme runtime field-sign-device-payload camera-event.json "
                    "--secret-env ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET --output camera-event.signed.json"
                ),
                "dry_run": (
                    "python -m askme runtime field-ingest-bridge camera-events.jsonl "
                    "--dry-run --json"
                ),
                "watch": (
                    "python -m askme runtime field-ingest-bridge camera-events.jsonl "
                    "--watch --server http://127.0.0.1:8765"
                ),
                "jsonl_state": "JSONL uses an offset state file so processed device events are not resent.",
                "snapshot_state": "JSON snapshots use file fingerprinting and are ignored until changed.",
            },
            "freshness_contract": {
                "timestamp_fields": ["observed_at", "captured_at", "timestamp", "source_timestamp"],
                "max_input_age_s": self._max_input_age_s,
                "min_detection_confidence": self._min_detection_confidence,
                "dedupe_window_s": self._dedupe_window_s,
            },
            "device_trust_contract": {
                "required": self._require_trusted_devices,
                "signature_alg": _FIELD_DEVICE_SIGNATURE_ALG,
                "signature_fields": sorted(_FIELD_DEVICE_SIGNATURE_FIELDS),
                "timestamp_field": "device_signature_timestamp",
                "max_signature_age_s": self._device_signature_max_age_s,
                "registered_device_count": len(self._device_registry),
                "signing_cli": "python -m askme runtime field-sign-device-payload",
            },
        }

        return {
            "accepted_sources": ["camera", "sensor", "robot", "map", "operator"],
            "examples": {
                "camera_vehicle": {
                    "source": "camera",
                    "observed_at": "2026-05-11T10:30:00+08:00",
                    "detections": [{"class_id": "2", "label": "vehicle", "confidence": 0.92}],
                    "zone_id": "main-road-1",
                    "duration_s": 180,
                    "image_path": "artifacts/evidence/car.jpg",
                },
                "smoke_sensor": {
                    "source": "sensor",
                    "observed_at": "2026-05-11T10:30:00+08:00",
                    "sensor": {"temperature_c": 68, "smoke_level": 0.8},
                    "location": "配电间门口",
                    "image_path": "artifacts/evidence/smoke.jpg",
                },
                "robot_fault": {
                    "source": "robot",
                    "observed_at": "2026-05-11T10:30:00+08:00",
                    "robot": {"fault_type": "joint_motor_fault", "joint_id": "hip-left"},
                    "location": "A区东侧",
                },
                "hikvision_anpr_vehicle": {
                    "eventType": "ANPR",
                    "dateTime": "2026-05-11T10:30:00+08:00",
                    "cameraIndexCode": "cam-main-road-01",
                    "ANPR": {"plateNo": "沪A12345"},
                    "zone_id": "main-road-1",
                    "duration_s": 180,
                    "pictureUrl": "artifacts/evidence/anpr-car.jpg",
                },
                "mqtt_smoke_alarm": {
                    "topic": "site/A/power-room/smoke-01",
                    "payload": {
                        "timestamp": "2026-05-11T10:30:00+08:00",
                        "temperatureC": 72,
                        "smokeAlarm": True,
                        "location": "配电间门口",
                        "imageUrl": "artifacts/evidence/smoke.jpg",
                    },
                },
                "robot_status_event": {
                    "topic": "/thunder/status",
                    "timestamp": "2026-05-11T10:30:00+08:00",
                    "robot": {
                        "nav_state": "stuck",
                        "recoverable": False,
                    },
                    "location": "A区东侧",
                },
            },
            "bridge_contract": {
                "sign_payload": (
                    "python -m askme runtime field-sign-device-payload camera-event.json "
                    "--secret-env ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET --output camera-event.signed.json"
                ),
                "dry_run": (
                    "python -m askme runtime field-ingest-bridge camera-events.jsonl "
                    "--dry-run --json"
                ),
                "watch": (
                    "python -m askme runtime field-ingest-bridge camera-events.jsonl "
                    "--watch --server http://127.0.0.1:8765"
                ),
                "jsonl_state": "JSONL uses an offset state file so processed device events are not resent.",
                "snapshot_state": "JSON snapshots use file fingerprinting and are ignored until changed.",
            },
            "freshness_contract": {
                "timestamp_fields": ["observed_at", "captured_at", "timestamp", "source_timestamp"],
                "max_input_age_s": self._max_input_age_s,
                "min_detection_confidence": self._min_detection_confidence,
                "dedupe_window_s": self._dedupe_window_s,
            },
            "device_trust_contract": {
                "required": self._require_trusted_devices,
                "signature_alg": _FIELD_DEVICE_SIGNATURE_ALG,
                "signature_fields": sorted(_FIELD_DEVICE_SIGNATURE_FIELDS),
                "timestamp_field": "device_signature_timestamp",
                "max_signature_age_s": self._device_signature_max_age_s,
                "registered_device_count": len(self._device_registry),
                "signing_cli": "python -m askme runtime field-sign-device-payload",
            },
        }

    def readiness_payload(self) -> dict[str, Any]:
        """Return deployment readiness gates for customer field operations."""

        return build_field_deployment_readiness(
            config=self._config,
            archive_path=self._archive_path,
            webhooks=self._webhooks,
            webhook_secrets=self._webhook_secrets,
            action_audit_integrity=self.action_audit_integrity_payload(),
            unified_audit=self._unified_audit_payload(),
            device_onboarding=self.device_onboarding_payload(),
        )

    def _unified_audit_payload(self) -> dict[str, Any]:
        try:
            from askme.audit.query import AuditPaths, AuditQueryService

            skill_audit_path = Path(
                str(
                    self._config.get("skill_audit_path")
                    or self._archive_path.with_name("skill-audit-not-configured.jsonl")
                )
            )
            return AuditQueryService(
                paths=AuditPaths(
                    skill_audit=skill_audit_path,
                    field_action_audit=self._action_audit_path,
                    field_event_archive=self._archive_path,
                    audit_reviews=_audit_review_path_from_field_config(self._config),
                )
            ).query(limit=500)
        except Exception as exc:
            return {
                "product_summary": {
                    "status": "query_failed",
                    "customer_status": "audit review status could not be verified",
                    "record_count": 0,
                    "requires_review_count": 1,
                    "high_or_critical_count": 1,
                },
                "review_queue": [
                    {
                        "record_id": "audit:query_failed",
                        "customer_label": "Unified audit",
                        "severity": "high",
                        "action": "query",
                        "outcome": "failed",
                        "reason": str(exc),
                    }
                ],
                "source_health": {},
            }

    async def ingest_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        received_at = time.time()
        normalized = self._normalize_ingest(body)
        normalized["_ingested"] = True
        normalized["_ingest_received_at"] = received_at
        normalized.setdefault("source", body.get("source") or "unknown")
        device_trust = self._device_trust_assessment(body, normalized, received_at)
        normalized["device_trust"] = device_trust
        if not device_trust.get("trusted", False):
            return {
                "accepted": False,
                "status": "rejected",
                "reason": "device_not_trusted",
                "normalized": normalized,
                "ingest_scope_contract": _field_ingest_scope_contract(
                    normalized=normalized,
                    accepted=False,
                    status="rejected",
                    reason="device_not_trusted",
                ),
            }
        if not normalized.get("scenario_id"):
            return {
                "accepted": False,
                "status": "ignored",
                "reason": "no_matching_field_scenario",
                "normalized": normalized,
                "ingest_scope_contract": _field_ingest_scope_contract(
                    normalized=normalized,
                    accepted=False,
                    status="ignored",
                    reason="no_matching_field_scenario",
                ),
            }
        result = await self.trigger_payload(normalized)
        event = result.get("event") if isinstance(result.get("event"), dict) else {}
        if event.get("managed_object_id"):
            normalized.setdefault("managed_object_id", event.get("managed_object_id"))
            normalized.setdefault("managed_object_display", event.get("managed_object_display"))
            normalized.setdefault("managed_object_category", event.get("managed_object_category"))
        result["normalized"] = normalized
        result["ingest_scope_contract"] = _field_ingest_scope_contract(
            normalized=normalized,
            event=event,
            accepted=bool(result.get("accepted")),
            status=str(result.get("status") or event.get("status") or ""),
            reason=str(result.get("reason") or ""),
        )
        return result

    async def trigger_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        scenario_id = str(body.get("scenario_id") or "").strip()
        if not scenario_id:
            scenario_id = self._infer_scenario_id(body)
        scenario = FIELD_SCENARIOS.get(scenario_id)
        if scenario is None:
            return {
                "accepted": False,
                "status": "rejected",
                "reason": "unknown_scenario",
                "scenario_id": scenario_id,
            }

        payload = dict(body if body.get("_ingested") else body.get("payload") or body)
        payload.setdefault("scenario_id", scenario_id)
        payload = self._enrich_with_zone(payload)
        payload.setdefault("location", body.get("location") or payload.get("location") or "-")
        event = self._build_event(scenario, payload)

        sensor_block = self._sensor_acceptance_block(event, scenario)
        if sensor_block is not None:
            event.status = "needs_review"
            event.operator_action = sensor_block
            self._append_event(event)
            return {
                "accepted": False,
                "status": event.status,
                "reason": "sensor_input_not_trusted",
                "event": asdict(event),
            }

        if event.missing_evidence and scenario.priority == "P0":
            event.status = "needs_evidence"
            event.operator_action = (
                event.operator_action
                or "证据不足，不能升级为现场告警；请补充位置、照片、传感器或管理员身份。"
            )
            self._append_event(event)
            return {
                "accepted": False,
                "status": event.status,
                "reason": "missing_required_evidence",
                "missing_evidence": event.missing_evidence,
                "event": asdict(event),
            }

        duplicate = self._recent_duplicate_event(event)
        if duplicate is not None:
            event.status = "duplicate"
            event.duplicate_of = str(duplicate.get("event_id") or "")
            event.operator_action = (
                f"已合并到 {event.duplicate_of}，不重复通知；如现场状态变化请上传新的证据。"
            )
            self._append_event(event)
            return {
                "accepted": True,
                "status": event.status,
                "duplicate_of": event.duplicate_of,
                "event": asdict(event),
            }

        await self._maybe_enrich_voice_with_llm(event, scenario)
        if event.incident_topic:
            event.sent_channels = self._dispatch_incident(event)
        self._append_event(event)
        return {"accepted": True, "status": event.status, "event": asdict(event)}

    async def test_notification_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        """Send a low-risk notification smoke test to a configured responder group."""
        group = str(body.get("notification_group") or body.get("group") or "security").strip()
        if group not in {"security", "cleaning", "operations"}:
            return {
                "sent": False,
                "status": "invalid_group",
                "notification_group": group,
                "sent_channels": [],
                "delivery_report": [],
                "reason": "notification_group must be security, cleaning, or operations",
            }

        webhook = self._webhooks.get(group) or self._webhooks.get("security")
        secret = self._webhook_secrets.get(group) or self._webhook_secrets.get("security")
        severity = str(body.get("severity") or "warning").strip() or "warning"
        message = str(body.get("message") or f"{self._robot_name} 通知测试：{group} 群配置校验。")
        dispatcher = self._alert_factory(
            robot_id=self._robot_id,
            robot_name=self._robot_name,
            config={
                "dingtalk_webhook": webhook,
                "dingtalk_secret": secret,
                "incident_archive_path": str(self._archive_path.with_name("incident-alerts.jsonl")),
                "severity_routes": {
                    "info": ["dingtalk", "log"],
                    "warning": ["dingtalk", "log"],
                    "error": ["dingtalk", "log"],
                },
            },
        )
        sent_channels = dispatcher.dispatch(
            message,
            severity=severity,
            topic="field.notification_test",
            payload={
                "event_id": str(
                    body.get("event_id") or f"notification-test-{uuid.uuid4().hex[:8]}"
                ),
                "operator_id": str(body.get("operator_id") or "dashboard.operator"),
                "notification_group": group,
                "test": True,
            },
        )
        delivery_report = (
            list(dispatcher.last_delivery_report)
            if hasattr(dispatcher, "last_delivery_report")
            else [{"channel": channel, "status": "sent", "reason": ""} for channel in sent_channels]
        )
        return {
            "sent": "dingtalk" in sent_channels,
            "status": "sent" if "dingtalk" in sent_channels else "not_sent",
            "notification_group": group,
            "webhook_configured": bool(webhook),
            "secret_configured": bool(secret),
            "sent_channels": sent_channels,
            "delivery_report": delivery_report,
            "message": message,
        }

    def acknowledge_payload(self, event_id: str, body: dict[str, Any]) -> dict[str, Any]:
        events = self._read_events()
        updated: dict[str, Any] | None = None
        for event in events:
            if event.get("event_id") == event_id:
                actor = self._authorize_operator(body, required_roles=_FIELD_ACTION_ROLES)
                if actor["authorized"] is not True:
                    self._append_action_audit(
                        event,
                        action="acknowledge",
                        outcome="denied",
                        actor=actor,
                        reason="operator_not_authorized",
                        note=body.get("note") or body.get("acknowledge_note"),
                    )
                    self._write_events(events)
                    return {
                        "acknowledged": False,
                        "reason": "operator_not_authorized",
                        **actor,
                        "event": _field_event_view(event),
                    }
                if event.get("status") in {"closed", "duplicate"}:
                    self._append_action_audit(
                        event,
                        action="acknowledge",
                        outcome="denied",
                        actor=actor,
                        reason="event_already_closed",
                        note=body.get("note") or body.get("acknowledge_note"),
                    )
                    self._write_events(events)
                    return {
                        "acknowledged": False,
                        "reason": "event_already_closed",
                        "event": _field_event_view(event),
                    }
                if event.get("status") not in {
                    "needs_review",
                    "needs_evidence",
                    "pending_close_approval",
                }:
                    event["status"] = "acknowledged"
                event["acknowledged_at"] = time.time()
                event["acknowledged_by"] = actor["operator_id"]
                event["acknowledge_note"] = str(
                    body.get("note") or body.get("acknowledge_note") or ""
                )
                self._append_action_audit(
                    event,
                    action="acknowledge",
                    outcome="accepted",
                    actor=actor,
                    note=event["acknowledge_note"],
                )
                updated = event
                break
        if updated is None:
            return {"acknowledged": False, "reason": "event_not_found", "event_id": event_id}
        self._write_events(events)
        return {"acknowledged": True, "event": updated}

    def record_voice_delivery_payload(
        self,
        event_id: str,
        delivery: dict[str, Any],
    ) -> dict[str, Any]:
        events = self._read_events()
        updated: dict[str, Any] | None = None
        clean_delivery = dict(delivery)
        clean_delivery.setdefault("recorded_at", round(time.time(), 3))
        for event in events:
            if event.get("event_id") != event_id:
                continue
            event["voice_delivery"] = clean_delivery
            updated = event
            break
        if updated is None:
            return {"recorded": False, "reason": "event_not_found", "event_id": event_id}
        self._write_events(events)
        return {
            "recorded": True,
            "event": _field_event_view(updated),
            "voice_delivery": clean_delivery,
        }

    def record_runtime_delivery_payload(
        self,
        event_id: str,
        delivery: dict[str, Any],
    ) -> dict[str, Any]:
        events = self._read_events()
        updated: dict[str, Any] | None = None
        clean_delivery = dict(delivery)
        status = str(clean_delivery.get("status") or "").strip()
        if not status:
            return {
                "recorded": False,
                "reason": "runtime_delivery_status_required",
                "event_id": event_id,
            }
        if status not in _FIELD_RUNTIME_DELIVERY_STATUSES:
            return {
                "recorded": False,
                "reason": "invalid_runtime_delivery_status",
                "event_id": event_id,
                "status": status,
                "allowed_statuses": sorted(_FIELD_RUNTIME_DELIVERY_STATUSES),
            }
        clean_delivery["status"] = status
        clean_delivery.setdefault("recorded_at", round(time.time(), 3))
        clean_delivery.setdefault("hardware_dispatch", False)
        callback_id = str(clean_delivery.get("runtime_callback_id") or "").strip()
        for event in events:
            if event.get("event_id") != event_id:
                continue
            receipts = event.get("runtime_delivery_receipts")
            if not isinstance(receipts, list):
                receipts = []
                event["runtime_delivery_receipts"] = receipts
            if callback_id:
                for receipt in receipts:
                    if not isinstance(receipt, dict):
                        continue
                    if str(receipt.get("runtime_callback_id") or "") == callback_id:
                        existing_delivery = (
                            event.get("runtime_delivery")
                            if isinstance(event.get("runtime_delivery"), dict)
                            else {}
                        )
                        return {
                            "recorded": True,
                            "duplicate": True,
                            "reason": "runtime_callback_already_recorded",
                            "event": _field_event_view(event),
                            "runtime_delivery": existing_delivery,
                            "runtime_delivery_receipt": receipt,
                        }
            event["runtime_delivery"] = clean_delivery
            if callback_id:
                receipts.append(_field_runtime_delivery_receipt(clean_delivery))
                event["runtime_delivery_receipts"] = receipts[-100:]
            updated = event
            break
        if updated is None:
            return {"recorded": False, "reason": "event_not_found", "event_id": event_id}
        self._write_events(events)
        return {
            "recorded": True,
            "event": _field_event_view(updated),
            "runtime_delivery": clean_delivery,
        }

    def resend_notification_payload(self, event_id: str, body: dict[str, Any]) -> dict[str, Any]:
        events = self._read_events()
        updated: dict[str, Any] | None = None
        resend_report: dict[str, Any] | None = None
        for event in events:
            if event.get("event_id") != event_id:
                continue
            actor = self._authorize_operator(body, required_roles=_FIELD_ACTION_ROLES)
            if actor["authorized"] is not True:
                self._append_action_audit(
                    event,
                    action="resend_notification",
                    outcome="denied",
                    actor=actor,
                    reason="operator_not_authorized",
                    note=body.get("note") or body.get("resend_note"),
                )
                self._write_events(events)
                return {
                    "resent": False,
                    "reason": "operator_not_authorized",
                    **actor,
                    "event": _field_event_view(event),
                }
            if event.get("status") in {"closed", "duplicate", "needs_evidence", "needs_review"}:
                reason = f"event_status_{event.get('status')}"
                self._append_action_audit(
                    event,
                    action="resend_notification",
                    outcome="denied",
                    actor=actor,
                    reason=reason,
                    note=body.get("note") or body.get("resend_note"),
                )
                self._write_events(events)
                return {
                    "resent": False,
                    "reason": reason,
                    "event": _field_event_view(event),
                }
            if not event.get("incident_topic"):
                self._append_action_audit(
                    event,
                    action="resend_notification",
                    outcome="denied",
                    actor=actor,
                    reason="event_has_no_notification_topic",
                    note=body.get("note") or body.get("resend_note"),
                )
                self._write_events(events)
                return {
                    "resent": False,
                    "reason": "event_has_no_notification_topic",
                    "event": _field_event_view(event),
                }
            self._ensure_action_audit_appendable()
            record = _field_event_record_from_dict(event)
            sent_channels = self._dispatch_incident(record)
            event["sent_channels"] = sent_channels
            event["delivery_report"] = list(record.delivery_report)
            resend_report = {
                "resent_at": time.time(),
                "resent_by": actor["operator_id"],
                "note": str(body.get("note") or body.get("resend_note") or ""),
                "sent_channels": sent_channels,
                "delivery_report": list(record.delivery_report),
            }
            event.setdefault("notification_resends", [])
            if isinstance(event["notification_resends"], list):
                event["notification_resends"].append(resend_report)
            self._append_action_audit(
                event,
                action="resend_notification",
                outcome="accepted",
                actor=actor,
                note=resend_report["note"],
            )
            updated = event
            break
        if updated is None:
            return {"resent": False, "reason": "event_not_found", "event_id": event_id}
        self._write_events(events)
        return {
            "resent": "dingtalk" in (resend_report or {}).get("sent_channels", []),
            "event": updated,
            "delivery_report": (resend_report or {}).get("delivery_report", []),
            "sent_channels": (resend_report or {}).get("sent_channels", []),
        }

    def event_report_payload(self, event_id: str) -> dict[str, Any]:
        for event in self._read_events():
            if event.get("event_id") == event_id:
                view = _field_event_view(event)
                return {
                    "found": True,
                    "event_id": event_id,
                    "report": _field_event_report(view),
                    "markdown": _field_event_report_markdown(view),
                }
        return {"found": False, "reason": "event_not_found", "event_id": event_id}

    def action_audit_integrity_payload(self) -> dict[str, Any]:
        """Verify the append-only field action audit hash chain."""
        if not self._action_audit_enabled or self._action_audit_path is None:
            return {
                "enabled": False,
                "path": str(self._action_audit_path or ""),
                "exists": False,
                "valid": False,
                "checked_count": 0,
                "latest_hash": "",
                "failures": [{"line": 0, "reason": "action_audit_disabled"}],
            }
        path = self._action_audit_path
        if not path.exists():
            return {
                "enabled": True,
                "path": str(path),
                "exists": False,
                "valid": False,
                "checked_count": 0,
                "latest_hash": _FIELD_ACTION_AUDIT_GENESIS,
                "failures": [{"line": 0, "reason": "audit_file_missing"}],
            }

        failures: list[dict[str, Any]] = []
        previous_hash = _FIELD_ACTION_AUDIT_GENESIS
        latest_hash = previous_hash
        checked_count = 0
        expected_counts = _field_action_audit_counts_by_event(self._read_events())
        actual_counts: dict[str, int] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append({"line": line_number, "reason": "invalid_json", "detail": str(exc)})
                continue
            if not isinstance(record, dict):
                failures.append({"line": line_number, "reason": "record_not_object"})
                continue
            checked_count += 1
            if record.get("sequence") != checked_count:
                failures.append(
                    {
                        "line": line_number,
                        "reason": "sequence_mismatch",
                        "expected": checked_count,
                        "actual": record.get("sequence"),
                    }
                )
            event_id = str(record.get("event_id") or "")
            if event_id:
                actual_counts[event_id] = actual_counts.get(event_id, 0) + 1
            if record.get("prev_hash") != previous_hash:
                failures.append(
                    {
                        "line": line_number,
                        "reason": "prev_hash_mismatch",
                        "expected": previous_hash,
                        "actual": record.get("prev_hash"),
                    }
                )
            expected_hash = _field_action_audit_hash(record)
            actual_hash = str(record.get("record_hash") or "")
            if actual_hash != expected_hash:
                failures.append(
                    {
                        "line": line_number,
                        "reason": "record_hash_mismatch",
                        "expected": expected_hash,
                        "actual": actual_hash,
                    }
                )
            if self._action_audit_hmac_secret:
                if record.get("signature_alg") != _FIELD_ACTION_AUDIT_SIGNATURE_ALG:
                    failures.append(
                        {
                            "line": line_number,
                            "reason": "signature_alg_mismatch",
                            "expected": _FIELD_ACTION_AUDIT_SIGNATURE_ALG,
                            "actual": record.get("signature_alg"),
                        }
                    )
                expected_signature = _field_action_audit_signature(
                    record,
                    secret=self._action_audit_hmac_secret,
                )
                actual_signature = str(record.get("record_signature") or "")
                if not hmac.compare_digest(actual_signature, expected_signature):
                    failures.append(
                        {
                            "line": line_number,
                            "reason": "record_signature_mismatch",
                            "expected": expected_signature,
                            "actual": actual_signature,
                        }
                    )
            latest_hash = actual_hash or expected_hash
            previous_hash = latest_hash
        expected_count = sum(expected_counts.values())
        if checked_count != expected_count:
            failures.append(
                {
                    "line": 0,
                    "reason": "audit_count_mismatch",
                    "expected": expected_count,
                    "actual": checked_count,
                }
            )
        for event_id, expected in expected_counts.items():
            actual = actual_counts.get(event_id, 0)
            if actual != expected:
                failures.append(
                    {
                        "line": 0,
                        "reason": "event_audit_count_mismatch",
                        "event_id": event_id,
                        "expected": expected,
                        "actual": actual,
                    }
                )

        return {
            "enabled": True,
            "path": str(path),
            "exists": True,
            "valid": checked_count > 0 and expected_count == checked_count and not failures,
            "checked_count": checked_count,
            "expected_count": expected_count,
            "latest_hash": latest_hash,
            "hash_alg": _FIELD_ACTION_AUDIT_HASH_ALG,
            "signed": bool(self._action_audit_hmac_secret),
            "signature_alg": (
                _FIELD_ACTION_AUDIT_SIGNATURE_ALG if self._action_audit_hmac_secret else ""
            ),
            "failures": failures,
        }

    def request_close_payload(self, event_id: str, body: dict[str, Any]) -> dict[str, Any]:
        events = self._read_events()
        updated: dict[str, Any] | None = None
        for event in events:
            if event.get("event_id") != event_id:
                continue
            actor = self._authorize_operator(body, required_roles=_FIELD_ACTION_ROLES)
            if actor["authorized"] is not True:
                self._append_action_audit(
                    event,
                    action="request_close",
                    outcome="denied",
                    actor=actor,
                    reason="operator_not_authorized",
                    note=body.get("note") or body.get("close_request_note"),
                )
                self._write_events(events)
                return {
                    "requested": False,
                    "reason": "operator_not_authorized",
                    **actor,
                    "event": _field_event_view(event),
                }
            status = str(event.get("status") or "")
            if status == "closed":
                self._append_action_audit(
                    event,
                    action="request_close",
                    outcome="denied",
                    actor=actor,
                    reason="event_already_closed",
                    note=body.get("note") or body.get("close_request_note"),
                )
                self._write_events(events)
                return {
                    "requested": False,
                    "reason": "event_already_closed",
                    "event": _field_event_view(event),
                }
            if status in {"needs_review", "needs_evidence", "duplicate"}:
                self._append_action_audit(
                    event,
                    action="request_close",
                    outcome="denied",
                    actor=actor,
                    reason="event_not_closable",
                    note=body.get("note") or body.get("close_request_note"),
                )
                self._write_events(events)
                return {
                    "requested": False,
                    "reason": "event_not_closable",
                    "status": status,
                    "event": _field_event_view(event),
                }
            if not _field_event_requires_close_approval(event):
                closed = self.close_payload(event_id, body)
                return {
                    "requested": bool(closed.get("closed")),
                    "closed": bool(closed.get("closed")),
                    "reason": closed.get("reason", ""),
                    "event": closed.get("event"),
                }
            event["status"] = "pending_close_approval"
            event["close_requested_at"] = time.time()
            event["close_requested_by"] = actor["operator_id"]
            event["close_request_note"] = str(
                body.get("note") or body.get("close_request_note") or ""
            )
            self._append_action_audit(
                event,
                action="request_close",
                outcome="accepted",
                actor=actor,
                note=event["close_request_note"],
            )
            updated = event
            break
        if updated is None:
            return {"requested": False, "reason": "event_not_found", "event_id": event_id}
        self._write_events(events)
        return {
            "requested": True,
            "requires_approval": True,
            "event": _field_event_view(updated),
        }

    def close_payload(self, event_id: str, body: dict[str, Any]) -> dict[str, Any]:
        events = self._read_events()
        updated: dict[str, Any] | None = None
        for event in events:
            if event.get("event_id") == event_id:
                actor = self._authorize_operator(body, required_roles=_FIELD_ACTION_ROLES)
                if actor["authorized"] is not True:
                    self._append_action_audit(
                        event,
                        action="close",
                        outcome="denied",
                        actor=actor,
                        reason="operator_not_authorized",
                        note=body.get("note") or body.get("close_note"),
                    )
                    self._write_events(events)
                    return {
                        "closed": False,
                        "reason": "operator_not_authorized",
                        **actor,
                        "event": _field_event_view(event),
                    }
                status = str(event.get("status") or "")
                if status == "closed":
                    self._append_action_audit(
                        event,
                        action="close",
                        outcome="denied",
                        actor=actor,
                        reason="event_already_closed",
                        note=body.get("note") or body.get("close_note"),
                    )
                    self._write_events(events)
                    return {
                        "closed": False,
                        "reason": "event_already_closed",
                        "event": _field_event_view(event),
                    }
                if status in {"needs_review", "needs_evidence", "duplicate"}:
                    self._append_action_audit(
                        event,
                        action="close",
                        outcome="denied",
                        actor=actor,
                        reason="event_not_closable",
                        note=body.get("note") or body.get("close_note"),
                    )
                    self._write_events(events)
                    return {
                        "closed": False,
                        "reason": "event_not_closable",
                        "status": status,
                        "event": _field_event_view(event),
                    }
                if _field_event_requires_close_approval(event) and not (
                    body.get("supervisor_approved") and body.get("supervisor_id")
                ):
                    self._append_action_audit(
                        event,
                        action="close",
                        outcome="denied",
                        actor=actor,
                        reason="close_requires_supervisor_approval",
                        note=body.get("note") or body.get("close_note"),
                    )
                    self._write_events(events)
                    return {
                        "closed": False,
                        "reason": "close_requires_supervisor_approval",
                        "requires_approval": True,
                        "event": _field_event_view(event),
                    }
                if _field_event_requires_close_approval(event):
                    supervisor = self._authorize_operator(
                        {"operator_id": body.get("supervisor_id")},
                        required_roles=_SUPERVISOR_ROLES,
                    )
                    if supervisor["authorized"] is not True:
                        self._append_action_audit(
                            event,
                            action="close",
                            outcome="denied",
                            actor=actor,
                            reason="supervisor_not_authorized",
                            note=body.get("note") or body.get("close_note"),
                            supervisor=supervisor,
                        )
                        self._write_events(events)
                        return {
                            "closed": False,
                            "reason": "supervisor_not_authorized",
                            "requires_approval": True,
                            **supervisor,
                            "event": _field_event_view(event),
                        }
                event["status"] = "closed"
                event["closed_at"] = time.time()
                event["closed_by"] = actor["operator_id"]
                event["close_note"] = str(body.get("note") or body.get("close_note") or "")
                if body.get("supervisor_approved") and body.get("supervisor_id"):
                    event["close_approval"] = {
                        "approved": True,
                        "supervisor_id": str(body.get("supervisor_id")),
                        "approved_at": time.time(),
                        "approval_note": str(body.get("approval_note") or ""),
                    }
                self._append_action_audit(
                    event,
                    action="close",
                    outcome="accepted",
                    actor=actor,
                    note=event["close_note"],
                    supervisor=(
                        self._authorize_operator(
                            {"operator_id": body.get("supervisor_id")},
                            required_roles=_SUPERVISOR_ROLES,
                        )
                        if body.get("supervisor_id")
                        else None
                    ),
                )
                self._write_closed_event_memory(event)
                updated = event
                break
        if updated is None:
            return {"closed": False, "reason": "event_not_found", "event_id": event_id}
        self._write_events(events)
        return {"closed": True, "event": updated}

    def _authorize_operator(
        self,
        body: dict[str, Any],
        *,
        required_roles: tuple[str, ...],
    ) -> dict[str, Any]:
        raw_operator_id = body.get("operator_id")
        operator_id = str(raw_operator_id or "").strip()
        if not operator_id:
            return {
                "authorized": False,
                "operator_id": "anonymous",
                "operator_roles": [],
                "required_roles": list(required_roles),
                "authorization_reason": "operator_identity_required",
            }
        roles = self._operator_roles.get(operator_id, ())
        authorized = bool(set(roles).intersection(required_roles))
        return {
            "authorized": authorized,
            "operator_id": operator_id,
            "operator_roles": list(roles),
            "required_roles": list(required_roles),
            "authorization_reason": "authorized" if authorized else "operator_not_authorized",
        }

    def _append_action_audit(
        self,
        event: dict[str, Any],
        *,
        action: str,
        outcome: str,
        actor: dict[str, Any],
        reason: str = "",
        note: Any = None,
        supervisor: dict[str, Any] | None = None,
    ) -> None:
        audit = event.setdefault("action_audit", [])
        if not isinstance(audit, list):
            audit = []
            event["action_audit"] = audit
        record: dict[str, Any] = {
            "at": round(time.time(), 3),
            "action": action,
            "outcome": outcome,
            "operator_id": str(actor.get("operator_id") or ""),
            "operator_roles": list(actor.get("operator_roles") or []),
            "required_roles": list(actor.get("required_roles") or []),
        }
        clean_reason = str(reason or "").strip()
        if clean_reason:
            record["reason"] = clean_reason
        clean_note = str(note or "").strip()
        if clean_note:
            record["note"] = clean_note
        if supervisor is not None:
            record["supervisor_id"] = str(supervisor.get("operator_id") or "")
            record["supervisor_roles"] = list(supervisor.get("operator_roles") or [])
            record["supervisor_required_roles"] = list(supervisor.get("required_roles") or [])
        audit.append(record)
        self._append_action_audit_record(event, record)

    def _ensure_action_audit_appendable(self) -> None:
        if not self._action_audit_enabled or self._action_audit_path is None:
            return
        _strict_field_action_audit_checkpoint(
            self._action_audit_path,
            secret=self._action_audit_hmac_secret,
        )

    def _append_action_audit_record(self, event: dict[str, Any], record: dict[str, Any]) -> None:
        if not self._action_audit_enabled or self._action_audit_path is None:
            return
        audit_record = {
            "kind": "field_event_action",
            "robot_id": self._robot_id,
            "event_id": str(event.get("event_id") or ""),
            "scenario_id": str(event.get("scenario_id") or ""),
            "customer_id": str(event.get("customer_id") or ""),
            "project_id": str(event.get("project_id") or ""),
            "site_id": str(event.get("site_id") or ""),
            "managed_object_id": str(event.get("managed_object_id") or ""),
            "status": str(event.get("status") or ""),
            "priority": str(event.get("priority") or ""),
            "severity": str(event.get("severity") or ""),
            "location": str(event.get("location") or ""),
            "audit": dict(record),
        }
        try:
            self._action_audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_record["hash_alg"] = _FIELD_ACTION_AUDIT_HASH_ALG
            if self._action_audit_hmac_secret:
                audit_record["signature_alg"] = _FIELD_ACTION_AUDIT_SIGNATURE_ALG
                audit_record["signature_key_id"] = self._action_audit_signature_key_id
            sequence, previous_hash = _strict_field_action_audit_checkpoint(
                self._action_audit_path,
                secret=self._action_audit_hmac_secret,
            )
            audit_record["sequence"] = sequence
            audit_record["prev_hash"] = previous_hash
            audit_record["record_hash"] = _field_action_audit_hash(audit_record)
            if self._action_audit_hmac_secret:
                audit_record["record_signature"] = _field_action_audit_signature(
                    audit_record,
                    secret=self._action_audit_hmac_secret,
                )
            line = json.dumps(
                audit_record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            with self._action_audit_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.write("\n")
        except (OSError, FieldActionAuditIntegrityError):
            raise

    def _write_closed_event_memory(self, event: dict[str, Any]) -> None:
        if not self._incident_memory_enabled:
            return
        if (
            isinstance(event.get("memory_delivery"), dict)
            and event["memory_delivery"].get("status") == "written"
        ):
            return
        summary = _field_incident_memory_text(event)
        location = str(event.get("location") or "-")
        memory_kind = _field_incident_memory_kind(event)
        try:
            memory = self._incident_memory_service
            if memory is None:
                from askme.memory.core.service import get_memory_service

                memory = get_memory_service(self._incident_memory_config)
            if memory_kind == "anomaly":
                memory.record_anomaly(location, summary)
            else:
                memory.record_observation(location, summary)
            if hasattr(memory, "save"):
                memory.save()
            event["memory_delivery"] = {
                "status": "written",
                "target": "site_knowledge",
                "kind": memory_kind,
                "written_at": round(time.time(), 3),
                "location": location,
                "summary": summary,
            }
        except Exception as exc:
            event["memory_delivery"] = {
                "status": "failed",
                "target": "site_knowledge",
                "kind": memory_kind,
                "reason": str(exc),
            }

    def _normalize_ingest(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = normalize_field_ingest_payload(body)
        payload = self._enrich_with_zone(payload)
        detections = (
            payload.get("detections") if isinstance(payload.get("detections"), list) else []
        )
        labels = {
            str(item.get("label") or item.get("class") or "").lower()
            for item in detections
            if isinstance(item, dict)
        }
        sensor = payload.get("sensor") if isinstance(payload.get("sensor"), dict) else {}
        robot = payload.get("robot") if isinstance(payload.get("robot"), dict) else {}

        temperature = _float_or_none(payload.get("temperature_c", sensor.get("temperature_c")))
        smoke = _float_or_none(payload.get("smoke_level", sensor.get("smoke_level")))
        if (
            "fire" in labels
            or "smoke" in labels
            or (temperature is not None and temperature >= self._thresholds["temperature_c"])
            or (smoke is not None and smoke >= self._thresholds["smoke_level"])
        ):
            payload["scenario_id"] = "fire_or_smoke"
            if temperature is not None:
                payload["temperature_c"] = temperature
            if smoke is not None:
                payload["smoke_level"] = smoke
            return payload

        fault_type = str(robot.get("fault_type") or payload.get("fault_type") or "").strip()
        if fault_type in _ROBOT_FAULT_TOPICS:
            payload["scenario_id"] = "robot_abnormal_incident"
            payload["fault_type"] = fault_type
            return payload

        duration_s = _float_or_none(payload.get("duration_s")) or 0.0
        zone_type = str(payload.get("zone_type") or payload.get("area_type") or "").lower()
        if "vehicle" in labels or "car" in labels:
            parking_allowed = bool(payload.get("parking_allowed", False))
            if not parking_allowed and duration_s >= self._thresholds["parking_duration_s"]:
                payload["scenario_id"] = "illegal_parking"
                return payload

        detected_persons = len(
            [
                item
                for item in detections
                if isinstance(item, dict)
                and str(item.get("label") or item.get("class") or "").lower() == "person"
            ]
        )
        person_count = int(_float_or_none(payload.get("person_count")) or detected_persons)
        duration_min = _float_or_none(payload.get("duration_min")) or 0.0
        if (
            person_count > self._thresholds["crowd_person_count"]
            and duration_min >= self._thresholds["crowd_duration_min"]
        ):
            payload["scenario_id"] = "crowd_gathering"
            payload["person_count"] = person_count
            payload["duration_min"] = duration_min
            return payload

        fill_ratio = _ratio_or_none(payload.get("fill_ratio") or sensor.get("fill_ratio"))
        if ("trash_bin" in labels or "trash" in labels or payload.get("bin_id")) and (
            fill_ratio is not None and fill_ratio >= self._thresholds["trash_fill_ratio"]
        ):
            payload["scenario_id"] = "trash_bin_full"
            payload["fill_ratio"] = f"{round(fill_ratio * 100)}%"
            return payload

        photo_evidence = bool(
            payload.get("taking_photo")
            or payload.get("phone_detected")
            or payload.get("camera_detected")
            or labels.intersection({"phone", "camera", "taking_photo"})
        )
        if photo_evidence:
            payload["taking_photo"] = True

        if (
            "person" in labels
            and payload.get("is_night")
            and zone_type in {"window", "corner", "restricted", "blind_spot"}
        ):
            dwell_s = _float_or_none(payload.get("dwell_s", payload.get("duration_s"))) or 0.0
            known_person = bool(payload.get("known_person") or payload.get("authorized_person"))
            required_dwell_s = (
                self._thresholds["night_photo_dwell_s"]
                if photo_evidence
                else self._thresholds["night_stranger_dwell_s"]
            )
            if not known_person and dwell_s >= required_dwell_s:
                payload["scenario_id"] = "night_stranger_photo"
                payload["duration_s"] = dwell_s
                return payload

        if "person" in labels and zone_type == "help_point":
            dwell_s = _float_or_none(payload.get("dwell_s", payload.get("duration_s"))) or 0.0
            if dwell_s >= 5:
                payload["scenario_id"] = "wayfinding_help_point"
                payload.setdefault("help_point_id", payload.get("zone_id"))
                return payload

        return payload

    def _device_trust_assessment(
        self,
        body: dict[str, Any],
        normalized: dict[str, Any],
        received_at: float,
    ) -> dict[str, Any]:
        device_id = _field_device_id(body, normalized)
        source = str(normalized.get("source") or body.get("source") or "").strip()
        base = {
            "required": self._require_trusted_devices,
            "device_id": device_id,
            "source": source,
            "registered": False,
            "signature_verified": False,
            "trusted": True,
            "status": "not_required",
            "reason": "",
        }
        if not self._device_registry:
            if self._require_trusted_devices:
                return {
                    **base,
                    "trusted": False,
                    "status": "blocked",
                    "reason": "device_registry_not_configured",
                }
            return {**base, "status": "not_configured"}
        if not device_id:
            if self._require_trusted_devices:
                return {
                    **base,
                    "trusted": False,
                    "status": "blocked",
                    "reason": "missing_device_id",
                }
            return {**base, "status": "unidentified"}
        device = self._device_registry.get(device_id)
        if device is None:
            if self._require_trusted_devices:
                return {
                    **base,
                    "trusted": False,
                    "status": "blocked",
                    "reason": "unregistered_device",
                }
            return {**base, "status": "unregistered"}

        base["registered"] = True
        allowed_sources = device.get("allowed_sources")
        if isinstance(allowed_sources, list) and allowed_sources and source not in allowed_sources:
            return {
                **base,
                "trusted": False,
                "status": "blocked",
                "reason": "device_source_not_allowed",
            }

        secret = _clean_secret(device.get("hmac_secret") or device.get("secret"))
        require_signature = bool(
            device.get("require_signature", bool(secret) or self._require_trusted_devices)
        )
        if not require_signature:
            return {**base, "status": "registered_unsigned"}
        if not secret:
            return {
                **base,
                "trusted": False,
                "status": "blocked",
                "reason": "device_secret_not_configured",
            }

        signature = _field_device_signature_value(body)
        if not signature:
            return {
                **base,
                "trusted": False,
                "status": "blocked",
                "reason": "missing_device_signature",
            }
        signature_alg = str(
            body.get("device_signature_alg")
            or body.get("signature_alg")
            or _FIELD_DEVICE_SIGNATURE_ALG
        )
        if signature_alg != _FIELD_DEVICE_SIGNATURE_ALG:
            return {
                **base,
                "trusted": False,
                "status": "blocked",
                "reason": "unsupported_device_signature_alg",
            }
        expected = sign_field_device_payload(body, secret=secret)
        if not hmac.compare_digest(signature, expected):
            return {
                **base,
                "trusted": False,
                "status": "blocked",
                "reason": "device_signature_mismatch",
            }

        signature_timestamp = _field_device_signature_timestamp(body)
        if signature_timestamp is None:
            return {
                **base,
                "trusted": False,
                "status": "blocked",
                "reason": "missing_device_signature_timestamp",
            }
        age_s = max(0.0, received_at - signature_timestamp)
        if signature_timestamp - received_at > 2.0:
            return {
                **base,
                "trusted": False,
                "status": "blocked",
                "reason": "device_signature_from_future",
            }
        if age_s > self._device_signature_max_age_s:
            return {
                **base,
                "trusted": False,
                "status": "blocked",
                "reason": "device_signature_expired",
            }
        return {
            **base,
            "signature_verified": True,
            "trusted": True,
            "status": "trusted",
            "signature_age_s": round(age_s, 3),
        }

    async def _maybe_enrich_voice_with_llm(
        self,
        event: FieldEventRecord,
        scenario: FieldScenario,
    ) -> None:
        if not self._should_use_llm_narrative(event, scenario):
            event.llm_narrative_status = "skipped"
            event.llm_narrative_reason = self._llm_narrative_skip_reason(event, scenario)
            event.payload["llm_narrative_status"] = event.llm_narrative_status
            event.payload["llm_narrative_reason"] = event.llm_narrative_reason
            return
        if self._llm is None:
            event.llm_narrative_status = "unavailable"
            event.llm_narrative_reason = "llm_client_not_configured"
            event.payload["llm_narrative_status"] = event.llm_narrative_status
            event.payload["llm_narrative_reason"] = event.llm_narrative_reason
            return
        prompt = (
            "你是园区机器狗的现场播报文案助手。只根据给定事实写一句中文现场播报。"
            "不能新增事实，不能夸大风险，不能使用 Markdown，不能超过 35 个汉字。"
            f"\n场景：{scenario.name}\n地点：{event.location}\n事实：{json.dumps(event.payload, ensure_ascii=False)}"
            f"\n默认播报：{event.voice}"
        )
        try:
            narrative = await asyncio.wait_for(
                self._llm.chat(
                    [
                        {"role": "system", "content": "只输出一句可直接播报的中文。"},
                        {"role": "user", "content": prompt},
                    ],
                    model="robot-action",
                    context=LLMCallContext(
                        call_id=uuid.uuid4().hex,
                        purpose="assistant_response",
                        channel="background",
                        request_class="robot_action",
                        privacy_class="restricted",
                        allow_cache=False,
                    ),
                ),
                timeout=2.5,
            )
        except Exception:
            event.llm_narrative_status = "failed"
            event.llm_narrative_reason = "llm_call_failed_or_timeout"
            event.payload["llm_narrative_status"] = event.llm_narrative_status
            event.payload["llm_narrative_reason"] = event.llm_narrative_reason
            return
        cleaned = _clean_narrative(str(narrative or ""))
        safe, reason = _validate_llm_narrative(cleaned, event)
        if not safe:
            event.llm_narrative_status = "rejected"
            event.llm_narrative_reason = reason
            event.payload["llm_narrative_status"] = event.llm_narrative_status
            event.payload["llm_narrative_reason"] = event.llm_narrative_reason
            return
        event.voice = cleaned
        if event.voice_directive:
            event.voice_directive["text"] = cleaned
        event.llm_narrative_used = True
        event.llm_narrative_status = "used"
        event.llm_narrative_reason = "accepted_low_risk_narrative"
        event.payload["llm_narrative_used"] = True
        event.payload["llm_narrative_status"] = event.llm_narrative_status
        event.payload["llm_narrative_reason"] = event.llm_narrative_reason

    def _should_use_llm_narrative(self, event: FieldEventRecord, scenario: FieldScenario) -> bool:
        allowed_by_playbook = bool(event.playbook.get("allow_llm_narrative"))
        allowed_by_payload = bool(event.payload.get("allow_llm_narrative"))
        if not self._llm_narrative_enabled and not allowed_by_payload and not allowed_by_playbook:
            return False
        if scenario.priority == "P0" or event.severity == "error":
            return False
        return (
            scenario.category in {"visitor_service", "facility_service"}
            or allowed_by_payload
            or allowed_by_playbook
        )

    def _llm_narrative_skip_reason(self, event: FieldEventRecord, scenario: FieldScenario) -> str:
        if scenario.priority == "P0" or event.severity == "error":
            return "high_risk_event_uses_fixed_playbook"
        if (
            not self._llm_narrative_enabled
            and not event.payload.get("allow_llm_narrative")
            and not event.playbook.get("allow_llm_narrative")
        ):
            return "llm_narrative_disabled"
        return "scenario_not_allowed_for_llm_narrative"

    def _sensor_acceptance_block(
        self,
        event: FieldEventRecord,
        scenario: FieldScenario,
    ) -> str | None:
        if not event.payload.get("_ingested"):
            return None
        if scenario.priority not in {"P0", "P1"}:
            return None
        if event.freshness_status != "fresh":
            age = "-" if event.freshness_age_s is None else f"{event.freshness_age_s:.1f}s"
            return (
                f"传感器证据 freshness 不合格: {event.freshness_status}, age={age}. "
                "高优先级事件必须使用新鲜、可追溯的现场证据。"
            )
        if event.confidence is not None and event.confidence < self._min_detection_confidence:
            return (
                f"检测置信度 {event.confidence:.2f} 低于阈值 {self._min_detection_confidence:.2f}. "
                "需要人工确认或补充证据后再通知和联动机器人。"
            )
        if event.freshness_status != "fresh":
            age = "-" if event.freshness_age_s is None else f"{event.freshness_age_s:.1f}s"
            return (
                f"传感器证据 freshness 不合格: {event.freshness_status}, age={age}. "
                "高优先级事件必须使用新鲜、可追溯的现场证据。"
            )
        if event.confidence is not None and event.confidence < self._min_detection_confidence:
            return (
                f"检测置信度 {event.confidence:.2f} 低于阈值 "
                f"{self._min_detection_confidence:.2f}，需要人工确认或补充证据。"
            )
        if event.freshness_status != "fresh":
            age = "-" if event.freshness_age_s is None else f"{event.freshness_age_s:.1f}s"
            return (
                f"感知数据未通过 freshness 校验（{event.freshness_status}, age={age}），"
                "已归档为待复核事件，未通知现场处置群。"
            )
        if event.confidence is not None and event.confidence < self._min_detection_confidence:
            return (
                f"识别置信度 {event.confidence:.2f} 低于阈值 "
                f"{self._min_detection_confidence:.2f}，需要人工复核后再升级。"
            )
        return None

    def _recent_duplicate_event(self, event: FieldEventRecord) -> dict[str, Any] | None:
        if not event.dedupe_key or event.status != "triggered":
            return None
        cutoff = time.time() - self._dedupe_window_s
        for item in reversed(self._read_events()):
            if item.get("status") not in {"triggered", "needs_review", "needs_evidence"}:
                continue
            if item.get("dedupe_key") != event.dedupe_key:
                continue
            created_at = _float_or_none(item.get("created_at")) or 0.0
            if created_at >= cutoff:
                return item
        return None

    def _build_event(self, scenario: FieldScenario, payload: dict[str, Any]) -> FieldEventRecord:
        scope = self._event_project_scope(payload)
        managed_object = self._managed_object_for_payload(
            scenario.scenario_id,
            payload,
            scope=scope,
        )
        if managed_object:
            payload.setdefault("managed_object_id", managed_object.get("object_id"))
            payload.setdefault("managed_object_category", managed_object.get("category"))
        for key in ("customer_id", "project_id", "site_id", "site_name", "industry"):
            if not scope.get(key):
                continue
            if payload.get("_ingested"):
                payload[key] = scope.get(key)
            else:
                payload.setdefault(key, scope.get(key))
        incident_topic = self._incident_topic_for(scenario.scenario_id, payload)
        alert = format_incident_alert(incident_topic, payload) if incident_topic else None
        playbook = (
            format_incident_playbook(incident_topic, payload)
            if incident_topic
            else self._service_playbook_for(scenario, payload)
        ) or {}
        freshness_status, freshness_age_s = self._freshness_status(payload)
        confidence = self._payload_confidence(payload)
        dedupe_key = self._dedupe_key(scenario.scenario_id, incident_topic, payload)
        missing = [
            key
            for key in _REQUIRED_PAYLOAD_KEYS.get(scenario.scenario_id, ())
            if payload.get(key) in (None, "")
        ]
        status = "triggered"
        severity = "info"
        voice = ""
        dingtalk = ""
        operator_action = ""
        notification_group = scenario.notification_group
        if alert:
            severity = str(alert["severity"])
            voice = str(alert["voice"])
            dingtalk = str(alert["dingtalk"])
            operator_action = str(alert["operator_action"])
            notification_group = str(alert.get("notification_group") or notification_group)
        elif scenario.scenario_id == "wayfinding_help_point":
            voice = "你好，请问有什么需要指路的吗？"
            operator_action = "仅在固定路引点主动询问；未知地点必须拒绝编造路线。"
        elif scenario.scenario_id == "visitor_escort":
            voice = f"我可以带你去{payload.get('destination', '目的地')}，请跟在我侧后方。"
            operator_action = "确认目的地存在于地图数据库，路线安全可达后再开始带路。"
        if scenario.scenario_id == "wayfinding_help_point":
            voice = "你好，请问需要指路吗？我可以告诉你园区内的路线。"
            operator_action = "确认访客问题只涉及指路，不触发巡检、拍摄或带路任务。"
        elif scenario.scenario_id == "visitor_escort":
            voice = (
                f"收到，我可以低速带你前往{payload.get('destination', '目标地点')}，"
                "请跟我保持安全距离。"
            )
            operator_action = "确认目的地存在于地图数据库，路线安全且不会进入禁行区域。"
        voice_directive = self._voice_directive_for(
            voice=voice,
            playbook=playbook,
            severity=severity,
            scenario=scenario,
        )
        return FieldEventRecord(
            event_id=str(payload.get("event_id") or f"field-{uuid.uuid4().hex[:12]}"),
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            category=scenario.category,
            priority=scenario.priority,
            status=status,
            severity=severity,
            location=str(payload.get("location") or "-"),
            created_at=_timestamp_or_now(payload.get("created_at")),
            customer_id=str(scope.get("customer_id") or ""),
            project_id=str(scope.get("project_id") or ""),
            site_id=str(scope.get("site_id") or ""),
            site_name=str(scope.get("site_name") or ""),
            industry=str(scope.get("industry") or ""),
            managed_object_id=str(managed_object.get("object_id") or "") if managed_object else "",
            managed_object_display=str(managed_object.get("display_name") or "")
            if managed_object
            else "",
            managed_object_category=str(managed_object.get("category") or "")
            if managed_object
            else "",
            managed_object_bindings=dict(managed_object.get("bindings") or {})
            if managed_object
            else {},
            resource_execution_context=_field_resource_execution_context(
                scenario,
                payload,
                managed_object,
            ),
            project_scope=scope,
            payload=payload,
            incident_topic=incident_topic,
            notification_group=notification_group,
            voice=voice,
            dingtalk=dingtalk,
            operator_action=operator_action,
            missing_evidence=missing,
            evidence_media=_field_evidence_media(payload),
            archive_required=scenario.archive_required,
            freshness_status=freshness_status,
            freshness_age_s=freshness_age_s,
            confidence=confidence,
            dedupe_key=dedupe_key,
            playbook=playbook,
            voice_directive=voice_directive,
        )

    def _event_project_scope(self, payload: dict[str, Any]) -> dict[str, Any]:
        scope = dict(self._project_scope)
        if payload.get("_ingested"):
            return scope
        explicit_scope = payload.get("project_scope")
        if isinstance(explicit_scope, dict):
            for key in (
                "tenant_id",
                "delivery_namespace",
                "customer_id",
                "project_id",
                "site_id",
                "site_name",
                "industry",
            ):
                if explicit_scope.get(key):
                    scope[key] = str(explicit_scope.get(key))
        for key in (
            "tenant_id",
            "delivery_namespace",
            "customer_id",
            "project_id",
            "site_id",
            "site_name",
            "industry",
        ):
            if payload.get(key):
                scope[key] = str(payload.get(key))
        return scope

    def _managed_object_for_payload(
        self,
        scenario_id: str,
        payload: dict[str, Any],
        *,
        scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        scope = scope or self._event_project_scope(payload)
        explicit = str(payload.get("managed_object_id") or "").strip()
        if explicit and explicit in self._managed_objects:
            item = self._managed_objects.get(explicit)
            if not isinstance(item, dict):
                return {}
            scope_match, _scope_score = _managed_object_scope_score(item, scope)
            if not scope_match:
                return {}
            resolved = dict(item)
            resolved.setdefault("object_id", explicit)
            return resolved
        candidates: list[tuple[int, str, dict[str, Any]]] = []
        labels = _payload_detection_labels(payload)
        zone_type = str(payload.get("zone_type") or "").strip()
        source = str(payload.get("source") or "").strip()
        for object_id, item in self._managed_objects.items():
            if not isinstance(item, dict):
                continue
            scope_match, scope_score = _managed_object_scope_score(item, scope)
            if not scope_match:
                continue
            score = scope_score
            scenario_ids = {str(value) for value in _as_list(item.get("scenario_ids"))}
            if scenario_id in scenario_ids:
                score += 100
            object_labels = {str(value) for value in _as_list(item.get("object_labels"))}
            if object_labels.intersection(labels):
                score += 20
            zone_types = {str(value) for value in _as_list(item.get("zone_types"))}
            if zone_type and zone_type in zone_types:
                score += 10
            device_sources = {str(value) for value in _as_list(item.get("device_sources"))}
            if source and source in device_sources:
                score += 5
            if score > 0:
                resolved = dict(item)
                resolved.setdefault("object_id", object_id)
                candidates.append((score, str(object_id), resolved))
        if not candidates:
            return {}
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]

    def _managed_objects_for_device(self, device: dict[str, Any]) -> list[dict[str, Any]]:
        device_id = str(device.get("device_id") or "").strip()
        source = str(device.get("source") or "").strip()
        allowed_sources = {
            str(item).strip()
            for item in _as_list(device.get("allowed_sources"))
            if str(item).strip()
        }
        if source:
            allowed_sources.add(source)
        rows: list[dict[str, Any]] = []
        for object_id, item in self._managed_objects.items():
            if not isinstance(item, dict):
                continue
            object_device_ids = {
                str(value).strip()
                for key in ("device_ids", "devices", "field_devices")
                for value in _as_list(item.get(key))
                if str(value).strip()
            }
            object_sources = {
                str(value).strip()
                for value in _as_list(item.get("device_sources"))
                if str(value).strip()
            }
            exact_device = bool(device_id and device_id in object_device_ids)
            source_match = bool(allowed_sources and object_sources.intersection(allowed_sources))
            if not exact_device and not source_match:
                continue
            binding = _field_managed_object_binding_gate(item)
            rows.append(
                {
                    "object_id": str(item.get("object_id") or object_id),
                    "display_name": str(item.get("display_name") or item.get("name") or object_id),
                    "category": str(item.get("category") or ""),
                    "match_type": "device_id" if exact_device else "source",
                    "device_sources": sorted(object_sources),
                    "binding_status": binding["status"],
                    "blockers": binding["blockers"],
                    "manual_checks": binding["manual_checks"],
                }
            )
        rows.sort(key=lambda row: (0 if row["match_type"] == "device_id" else 1, row["object_id"]))
        return rows

    def _voice_directive_for(
        self,
        *,
        voice: str,
        playbook: dict[str, Any],
        severity: str,
        scenario: FieldScenario,
    ) -> dict[str, Any]:
        text = str(voice or "").strip()
        if not text:
            return {}
        requested = str(playbook.get("tts_profile") or "").strip()
        if not requested:
            if scenario.category == "visitor_service":
                requested = "visitor_service"
            elif scenario.priority == "P0" or severity == "error":
                requested = "emergency_alert"
            else:
                requested = "patrol_notice"
        resolved = resolve_voice_profile_id(requested)
        urgent = severity == "error" or scenario.priority == "P0"
        return {
            "text": text,
            "requested_profile": requested,
            "resolved_profile": resolved,
            "interrupt_current_speech": urgent,
            "playback_mode": "immediate" if urgent else "queued",
            "source": "field_event_playbook",
        }

    def _service_playbook_for(
        self,
        scenario: FieldScenario,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if scenario.scenario_id == "wayfinding_help_point":
            return {
                "topic": "visitor.wayfinding_help_point",
                "customer_status": "访客在路引点发起问路，系统只提供路线帮助。",
                "robot_motion_policy": "hold_position",
                "tts_profile": "visitor_service",
                "responder_group": "none",
                "operator_checklist": [
                    "确认访客问题是路线问询",
                    "只使用已审批地图知识回答",
                    "无法确认路线时转人工或要求补充目的地",
                ],
                "evidence_policy": ["help_point_id", "location", "destination", "map_version"],
                "escalation_after_s": 0,
                "allow_llm_narrative": True,
            }
        if scenario.scenario_id == "visitor_escort":
            return {
                "topic": "visitor.escort",
                "customer_status": f"访客请求带路到 {payload.get('destination', '目标地点')}。",
                "robot_motion_policy": "low_speed_escort",
                "tts_profile": "visitor_service",
                "responder_group": "none",
                "operator_checklist": [
                    "确认目的地在地图数据库中",
                    "确认路线不经过禁行或危险区域",
                    "到达后结束带路并记录服务结果",
                ],
                "evidence_policy": ["destination", "location", "route_id", "map_version"],
                "escalation_after_s": 0,
                "allow_llm_narrative": True,
            }
        if scenario.scenario_id == "wayfinding_help_point":
            return {
                "topic": "visitor.wayfinding_help_point",
                "customer_status": "固定路引点检测到访客停留，允许主动询问一次。",
                "robot_motion_policy": "hold_position",
                "tts_profile": "visitor_service",
                "responder_group": "none",
                "operator_checklist": [
                    "只回答地图数据库中存在的地点",
                    "目的地未知时拒绝编造路线",
                    "访客离开或无回应时结束交互",
                ],
                "evidence_policy": ["help_point_id", "location", "destination", "map_version"],
                "escalation_after_s": 0,
                "allow_llm_narrative": True,
            }
        if scenario.scenario_id == "visitor_escort":
            return {
                "topic": "visitor.escort",
                "customer_status": f"访客请求带路到 {payload.get('destination', '目的地')}。",
                "robot_motion_policy": "low_speed_escort",
                "tts_profile": "visitor_service",
                "responder_group": "none",
                "operator_checklist": [
                    "确认目的地存在于园区地图",
                    "确认路线不穿越禁行区",
                    "跟随者丢失时停下并询问",
                ],
                "evidence_policy": ["destination", "location", "route_id", "map_version"],
                "escalation_after_s": 0,
                "allow_llm_narrative": True,
            }
        return {}

    def _freshness_status(self, payload: dict[str, Any]) -> tuple[str, float | None]:
        if not payload.get("_ingested"):
            return "not_applicable", None
        observed_at = self._payload_observed_at(payload)
        if observed_at is None:
            return "missing_timestamp", None
        received_at = _float_or_none(payload.get("_ingest_received_at")) or time.time()
        age_s = max(0.0, received_at - observed_at)
        if observed_at - received_at > 2.0:
            return "future_timestamp", round(age_s, 3)
        if age_s > self._max_input_age_s:
            return "stale", round(age_s, 3)
        return "fresh", round(age_s, 3)

    @staticmethod
    def _payload_observed_at(payload: dict[str, Any]) -> float | None:
        for key in ("observed_at", "captured_at", "timestamp", "source_timestamp", "created_at"):
            parsed = _parse_timestamp(payload.get(key))
            if parsed is not None:
                return parsed
        sensor = payload.get("sensor") if isinstance(payload.get("sensor"), dict) else {}
        robot = payload.get("robot") if isinstance(payload.get("robot"), dict) else {}
        for source in (sensor, robot):
            for key in ("observed_at", "captured_at", "timestamp", "source_timestamp"):
                parsed = _parse_timestamp(source.get(key))
                if parsed is not None:
                    return parsed
        return None

    @staticmethod
    def _payload_confidence(payload: dict[str, Any]) -> float | None:
        candidates: list[float] = []
        for key in ("confidence", "score", "probability"):
            parsed = _float_or_none(payload.get(key))
            if parsed is not None:
                candidates.append(parsed / 100.0 if parsed > 1.0 else parsed)
        detections = (
            payload.get("detections") if isinstance(payload.get("detections"), list) else []
        )
        for item in detections:
            if not isinstance(item, dict):
                continue
            for key in ("confidence", "score", "probability"):
                parsed = _float_or_none(item.get(key))
                if parsed is not None:
                    candidates.append(parsed / 100.0 if parsed > 1.0 else parsed)
                    break
        return max(candidates) if candidates else None

    @staticmethod
    def _dedupe_key(
        scenario_id: str,
        incident_topic: str | None,
        payload: dict[str, Any],
    ) -> str:
        parts = [
            scenario_id,
            incident_topic or "",
            str(payload.get("zone_id") or payload.get("map_zone_id") or ""),
            str(payload.get("location") or ""),
            str(payload.get("fault_type") or ""),
            str(payload.get("bin_id") or ""),
            str(payload.get("plate_number") or ""),
        ]
        return "|".join(part.strip().lower() for part in parts if part is not None)

    def _dispatch_incident(self, event: FieldEventRecord) -> list[str]:
        webhook = self._webhooks.get(event.notification_group) or self._webhooks.get("security")
        secret = self._webhook_secrets.get(event.notification_group) or self._webhook_secrets.get(
            "security"
        )
        dispatcher = self._alert_factory(
            robot_id=self._robot_id,
            robot_name=self._robot_name,
            config={
                "dingtalk_webhook": webhook,
                "dingtalk_secret": secret,
                "incident_archive_path": str(self._archive_path.with_name("incident-alerts.jsonl")),
                "severity_routes": {
                    "info": ["log"],
                    "warning": ["dingtalk", "log"],
                    "error": ["dingtalk", "log"],
                },
            },
        )
        payload = dict(event.payload)
        payload.update(
            {
                "event_id": event.event_id,
                "dingtalk_message": event.dingtalk,
                "operator_action": event.operator_action,
                "archive_required": event.archive_required,
                "notification_group": event.notification_group,
            }
        )
        sent = dispatcher.dispatch(
            event.voice or event.scenario_name,
            severity=event.severity,
            topic=event.incident_topic or event.scenario_id,
            payload=payload,
        )
        if hasattr(dispatcher, "last_delivery_report"):
            event.delivery_report = list(dispatcher.last_delivery_report)
        else:
            event.delivery_report = [
                {"channel": channel, "status": "sent", "reason": ""} for channel in sent
            ]
        return sent

    @staticmethod
    def _finalize_event_workflow(event: FieldEventRecord) -> None:
        data = asdict(event)
        _refresh_field_incident_workflow(data)
        event.incident_state = str(data.get("incident_state") or "active")
        event.incident_stage = str(data.get("incident_stage") or "received")
        workflow = data.get("incident_workflow")
        event.incident_workflow = workflow if isinstance(workflow, dict) else {}

    def _append_event(self, event: FieldEventRecord) -> None:
        self._finalize_event_workflow(event)
        self._archive_path.parent.mkdir(parents=True, exist_ok=True)
        with self._archive_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")) + "\n")

    def _read_events(self) -> list[dict[str, Any]]:
        if not self._archive_path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self._archive_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    events.append(item)
        return events

    def _write_events(self, events: list[dict[str, Any]]) -> None:
        self._archive_path.parent.mkdir(parents=True, exist_ok=True)
        with self._archive_path.open("w", encoding="utf-8") as fh:
            for event in events:
                _refresh_field_incident_workflow(event)
                fh.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _enrich_with_zone(self, payload: dict[str, Any]) -> dict[str, Any]:
        zone_id = str(payload.get("zone_id") or payload.get("map_zone_id") or "").strip()
        zones = self._site_map.get("zones") if isinstance(self._site_map, dict) else None
        if not zone_id or not isinstance(zones, dict):
            return payload
        zone = zones.get(zone_id)
        if not isinstance(zone, dict):
            return payload
        enriched = dict(payload)
        enriched.setdefault("zone_name", zone.get("name") or zone_id)
        enriched.setdefault("location", zone.get("location") or zone.get("name") or zone_id)
        enriched.setdefault("zone_type", zone.get("type") or zone.get("zone_type"))
        if "parking_allowed" in zone and "parking_allowed" not in enriched:
            enriched["parking_allowed"] = bool(zone.get("parking_allowed"))
        if zone.get("help_point_id") and "help_point_id" not in enriched:
            enriched["help_point_id"] = zone.get("help_point_id")
        return enriched

    @staticmethod
    def _scenario_payload(scenario: FieldScenario) -> dict[str, Any]:
        return {
            "scenario_id": scenario.scenario_id,
            "name": scenario.name,
            "category": scenario.category,
            "priority": scenario.priority,
            "trigger_rule": scenario.trigger_rule,
            "required_evidence": list(scenario.required_evidence),
            "robot_behavior": list(scenario.robot_behavior),
            "notification_group": scenario.notification_group,
            "archive_required": scenario.archive_required,
            "interrupts_current_task": scenario.interrupts_current_task,
            "requires_operator_approval": scenario.requires_operator_approval,
            "acceptance_criteria": list(scenario.acceptance_criteria),
            "required_payload_keys": list(_REQUIRED_PAYLOAD_KEYS.get(scenario.scenario_id, ())),
        }

    def _scenario_acceptance_row(self, scenario: FieldScenario) -> dict[str, Any]:
        routes = _scenario_intent_routes_for(scenario.scenario_id)
        return {
            "scenario_id": scenario.scenario_id,
            "name": scenario.name,
            "category": scenario.category,
            "priority": scenario.priority,
            "acceptance_status": "demo_ready",
            "production_status": "onsite_evidence_required",
            "customer_visible": True,
            "manual_entrypoint": {
                "endpoint": "/api/field/events",
                "method": "POST",
                "required_payload_keys": list(_REQUIRED_PAYLOAD_KEYS.get(scenario.scenario_id, ())),
            },
            "device_entrypoints": [
                {
                    "entrypoint_id": entrypoint,
                    "endpoint": "/api/field/ingest",
                    "method": "POST",
                }
                for entrypoint in _SCENARIO_DEVICE_ENTRYPOINTS.get(scenario.scenario_id, ())
            ],
            "natural_language_routes": routes,
            "notification_group": scenario.notification_group,
            "archive_required": scenario.archive_required,
            "interrupts_current_task": scenario.interrupts_current_task,
            "requires_operator_approval": scenario.requires_operator_approval,
            "trigger_rule": scenario.trigger_rule,
            "required_evidence": list(scenario.required_evidence),
            "robot_behavior": list(scenario.robot_behavior),
            "acceptance_criteria": list(scenario.acceptance_criteria),
            "onsite_dependencies": list(
                _SCENARIO_ONSITE_DEPENDENCIES.get(scenario.scenario_id, ())
            ),
            "verification_artifacts": [
                "scripts/eval/check_dashboard_visual.py::field_scenario_matrix",
                "tests/scenario_tests/test_field_operations_evaluation.py",
                "tests/test_scenario_intent_router.py",
                "tests/test_capability_scenario_intent_routes.py",
            ],
            "customer_next_step": (
                "Run the Dashboard scenario matrix, then attach onsite device, notification, "
                "and robot runtime evidence before production signoff."
            ),
        }

    @staticmethod
    def _incident_topic_for(scenario_id: str, payload: dict[str, Any]) -> str | None:
        if scenario_id == "robot_abnormal_incident":
            return _ROBOT_FAULT_TOPICS.get(str(payload.get("fault_type") or ""))
        return _SCENARIO_TO_INCIDENT_TOPIC.get(scenario_id)

    @staticmethod
    def _infer_scenario_id(body: dict[str, Any]) -> str:
        text = " ".join(
            str(body.get(key) or "") for key in ("type", "event_type", "label", "topic")
        )
        lowered = text.lower()
        if "fire" in lowered or "smoke" in lowered or "火" in text or "烟" in text:
            return "fire_or_smoke"
        if "parking" in lowered or "vehicle" in lowered or "违停" in text:
            return "illegal_parking"
        if "trash" in lowered or "垃圾" in text:
            return "trash_bin_full"
        if "crowd" in lowered or "聚集" in text:
            return "crowd_gathering"
        if "stranger" in lowered or "陌生" in text:
            return "night_stranger_photo"
        return str(body.get("scenario_id") or "")

    @staticmethod
    def _resolve_group_webhooks(cfg: dict[str, Any]) -> dict[str, str]:
        raw = cfg.get("dingtalk_webhooks") if isinstance(cfg.get("dingtalk_webhooks"), dict) else {}
        return {
            "security": _clean_secret(
                raw.get("security")
                or cfg.get("dingtalk_security_webhook")
                or cfg.get("dingtalk_webhook")
                or os.getenv("ASKME_DINGTALK_SECURITY_WEBHOOK")
                or os.getenv("ASKME_DINGTALK_WEBHOOK")
                or ""
            ),
            "cleaning": _clean_secret(
                raw.get("cleaning")
                or cfg.get("dingtalk_cleaning_webhook")
                or os.getenv("ASKME_DINGTALK_CLEANING_WEBHOOK")
                or ""
            ),
            "operations": _clean_secret(
                raw.get("operations")
                or cfg.get("dingtalk_operations_webhook")
                or os.getenv("ASKME_DINGTALK_OPERATIONS_WEBHOOK")
                or ""
            ),
        }

    @staticmethod
    def _resolve_group_secrets(cfg: dict[str, Any]) -> dict[str, str]:
        raw = cfg.get("dingtalk_secrets") if isinstance(cfg.get("dingtalk_secrets"), dict) else {}
        return {
            "security": _clean_secret(
                raw.get("security")
                or cfg.get("dingtalk_security_secret")
                or cfg.get("dingtalk_secret")
                or os.getenv("ASKME_DINGTALK_SECURITY_SECRET")
                or os.getenv("ASKME_DINGTALK_SECRET")
                or ""
            ),
            "cleaning": _clean_secret(
                raw.get("cleaning")
                or cfg.get("dingtalk_cleaning_secret")
                or os.getenv("ASKME_DINGTALK_CLEANING_SECRET")
                or ""
            ),
            "operations": _clean_secret(
                raw.get("operations")
                or cfg.get("dingtalk_operations_secret")
                or os.getenv("ASKME_DINGTALK_OPERATIONS_SECRET")
                or ""
            ),
        }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_secret(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"none", "null", "false", "0"}:
        return ""
    if raw.startswith("${") and raw.endswith("}"):
        return os.getenv(raw[2:-1].strip(), "").strip()
    return raw


def _resolve_field_device_registry(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    registry: dict[str, dict[str, Any]] = {}
    for device_id, value in raw.items():
        key = str(device_id or "").strip()
        if not key:
            continue
        item = value if isinstance(value, dict) else {}
        allowed = item.get("allowed_sources") or item.get("sources") or item.get("source")
        if isinstance(allowed, str):
            allowed_sources = [allowed]
        elif isinstance(allowed, list):
            allowed_sources = [str(source).strip() for source in allowed if str(source).strip()]
        else:
            allowed_sources = []
        registry[key] = {
            "allowed_sources": allowed_sources,
            "hmac_secret": _clean_secret(item.get("hmac_secret") or item.get("secret")),
            "require_signature": bool(item.get("require_signature", True)),
        }
    return registry


def _config_with_site_profile(config: dict[str, Any]) -> dict[str, Any]:
    profile_path = config.get("site_profile_path") or os.getenv("ASKME_FIELD_SITE_PROFILE")
    if not profile_path:
        return dict(config)
    path = Path(str(profile_path))
    report = build_site_profile_report(path, check_env=_site_profile_env_check_requested(config))
    if report.get("status") != "passed":
        errors = ", ".join(str(item) for item in report.get("errors", []))
        raise ValueError(f"field site profile validation failed: {errors}")
    profile_config = report.get("field_operations_config")
    merged = dict(profile_config if isinstance(profile_config, dict) else {})
    merged.update(config)
    merged["site_profile_path"] = str(path)
    merged["site_profile"] = {
        "status": report.get("status"),
        "summary": report.get("summary", {}),
        "readiness": report.get("readiness", {}),
        "warnings": report.get("warnings", []),
    }
    return merged


def _field_project_scope_from_config(config: dict[str, Any]) -> dict[str, str]:
    scope = (
        config.get("customer_project") if isinstance(config.get("customer_project"), dict) else {}
    )
    return {
        "tenant_id": str(scope.get("tenant_id") or config.get("tenant_id") or ""),
        "delivery_namespace": str(
            scope.get("delivery_namespace") or config.get("delivery_namespace") or ""
        ),
        "customer_id": str(scope.get("customer_id") or config.get("customer_id") or ""),
        "project_id": str(scope.get("project_id") or config.get("project_id") or ""),
        "site_id": str(scope.get("site_id") or config.get("site_id") or ""),
        "site_name": str(scope.get("site_name") or config.get("site_name") or ""),
        "industry": str(scope.get("industry") or config.get("industry") or ""),
    }


def _site_profile_env_check_requested(config: dict[str, Any]) -> bool:
    raw = (
        config.get("site_profile_check_env")
        if "site_profile_check_env" in config
        else config.get("check_site_profile_env")
    )
    if raw is None:
        raw = os.getenv("ASKME_FIELD_SITE_PROFILE_CHECK_ENV")
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "prod", "production"}


def _is_production_mode(config: dict[str, Any]) -> bool:
    mode = (
        str(
            config.get("deployment_mode")
            or config.get("readiness_mode")
            or config.get("field_deployment_mode")
            or ""
        )
        .strip()
        .lower()
    )
    return mode in {"prod", "production"}


def _notification_group_env_names(group: str) -> tuple[list[str], list[str]]:
    if group == "security":
        return (
            ["ASKME_DINGTALK_SECURITY_WEBHOOK", "ASKME_DINGTALK_WEBHOOK"],
            ["ASKME_DINGTALK_SECURITY_SECRET", "ASKME_DINGTALK_SECRET"],
        )
    if group == "cleaning":
        return (
            ["ASKME_DINGTALK_CLEANING_WEBHOOK"],
            ["ASKME_DINGTALK_CLEANING_SECRET"],
        )
    if group == "operations":
        return (
            ["ASKME_DINGTALK_OPERATIONS_WEBHOOK"],
            ["ASKME_DINGTALK_OPERATIONS_SECRET"],
        )
    return ([], [])


def _notification_group_missing_env(
    group: str,
    *,
    webhook_configured: bool,
    secret_configured: bool,
    require_secret: bool,
) -> list[str]:
    webhook_env, secret_env = _notification_group_env_names(group)
    missing: list[str] = []
    if not webhook_configured:
        missing.extend(webhook_env)
    if require_secret and not secret_configured:
        missing.extend(secret_env)
    return missing


def _notification_preflight_next_actions(results: dict[str, dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for group, result in results.items():
        missing = result.get("missing_env") if isinstance(result.get("missing_env"), list) else []
        if missing:
            actions.append(f"Configure {group}: " + " or ".join(str(item) for item in missing))
    if not actions:
        actions.append(
            "Run: python -m askme runtime field-notification-smoke --server <deployed-url> --json"
        )
    return actions


def _resolve_operator_roles(raw: Any) -> dict[str, tuple[str, ...]]:
    """Normalize the demo/operator directory used by field-operation RBAC."""
    roles: dict[str, tuple[str, ...]] = dict(_DEFAULT_OPERATOR_ROLES)
    if not isinstance(raw, dict):
        return roles
    for operator_id, value in raw.items():
        key = str(operator_id or "").strip()
        if not key:
            continue
        parsed: list[str] = []
        if isinstance(value, dict):
            candidate = value.get("roles", value.get("role"))
        else:
            candidate = value
        if isinstance(candidate, str):
            parsed = [candidate]
        elif isinstance(candidate, (list, tuple, set)):
            parsed = [str(item) for item in candidate]
        clean = tuple(
            role.strip().lower() for role in parsed if isinstance(role, str) and role.strip()
        )
        if clean:
            roles[key] = clean
    return roles


def _ratio_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.endswith("%"):
        parsed = _float_or_none(value[:-1])
        return parsed / 100.0 if parsed is not None else None
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return parsed / 100.0 if parsed > 1.0 else parsed


def _timestamp_or_now(value: Any) -> float:
    parsed = _parse_timestamp(value)
    return parsed if parsed is not None else time.time()


def _parse_timestamp(value: Any) -> float | None:
    parsed = _float_or_none(value)
    if parsed is not None:
        return parsed
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # noqa: UP017 - keep Python 3.10 compatible.
    return dt.timestamp()


def _clean_narrative(text: str) -> str:
    cleaned = text.strip().strip("`").replace("\n", "")
    for prefix in ("播报：", "回答：", "文案："):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    if len(cleaned) > 45:
        cleaned = cleaned[:45].rstrip("，。；、") + "。"
    return cleaned


def _validate_llm_narrative(text: str, event: FieldEventRecord) -> tuple[bool, str]:
    if not text:
        return False, "empty_narrative"
    if any(token in text for token in ("http://", "https://", "#", "*", "`")):
        return False, "contains_markup_or_link"
    if len(text) > 45:
        return False, "too_long"
    high_risk_tokens = (
        "火灾",
        "爆炸",
        "死亡",
        "报警",
        "撤离",
        "危险",
        "警察",
        "罚款",
        "强制",
    )
    fixed_voice = str(event.voice or "")
    for token in high_risk_tokens:
        if token in text and token not in fixed_voice:
            return False, "adds_high_risk_claim"
    return True, "ok"


def _field_event_record_from_dict(event: dict[str, Any]) -> FieldEventRecord:
    allowed = {item.name for item in fields(FieldEventRecord)}
    data = {key: value for key, value in event.items() if key in allowed}
    return FieldEventRecord(**data)


def _field_resource_execution_context(
    scenario: FieldScenario,
    payload: dict[str, Any],
    managed_object: dict[str, Any],
) -> dict[str, Any]:
    if not managed_object:
        return {
            "overall_status": "manual_check",
            "reason": "no_managed_object_matched",
            "scenario_id": scenario.scenario_id,
            "ingest_endpoint": "/api/field/ingest",
        }
    bindings = (
        managed_object.get("bindings") if isinstance(managed_object.get("bindings"), dict) else {}
    )
    source = str(payload.get("source") or "").strip()
    device_sources = [
        str(item) for item in _as_list(managed_object.get("device_sources")) if str(item).strip()
    ]
    vision_models = [
        str(item) for item in _as_list(bindings.get("vision_models")) if str(item).strip()
    ]
    sensor_protocols = [
        str(item) for item in _as_list(bindings.get("sensor_protocols")) if str(item).strip()
    ]
    skill_packages = [
        str(item) for item in _as_list(bindings.get("skill_packages")) if str(item).strip()
    ]
    acceptance_tests = [
        str(item) for item in _as_list(bindings.get("acceptance_tests")) if str(item).strip()
    ]
    capability_routes = field_capability_routes(
        skill_packages,
        scenario_id=scenario.scenario_id,
        required_inputs=_REQUIRED_PAYLOAD_KEYS.get(scenario.scenario_id, ()),
    )
    blockers: list[str] = []
    manual_checks: list[str] = []
    for key, values in (
        ("vision_models", vision_models),
        ("sensor_protocols", sensor_protocols),
        ("skill_packages", skill_packages),
        ("acceptance_tests", acceptance_tests),
    ):
        if not values:
            blockers.append(f"{key} binding missing")
    if source and device_sources and source not in device_sources:
        blockers.append(f"payload source {source} is outside managed object device_sources")
    protocol_sources = sorted(
        {
            protocol_source
            for protocol in sensor_protocols
            for protocol_source in _field_sensor_protocol_sources(protocol)
        }
    )
    source_covered = (
        not source or source in protocol_sources or (source == "camera" and bool(vision_models))
    )
    if not source_covered:
        manual_checks.append(f"payload source {source} has no explicit protocol coverage")
    missing_contracts = [
        str(item.get("package_id") or item.get("capability") or "")
        for item in capability_routes
        if not item.get("installed_contract")
    ]
    for package_id in missing_contracts:
        manual_checks.append(f"skill package {package_id} has no installed executable contract")
    selected_route = capability_routes[0] if capability_routes else {}
    approval_required = any(bool(item.get("approval_required")) for item in capability_routes)
    overall_status = "blocked" if blockers else "manual_check" if manual_checks else "ready"
    return {
        "overall_status": overall_status,
        "scenario_id": scenario.scenario_id,
        "managed_object_id": str(managed_object.get("object_id") or ""),
        "source": source,
        "device_sources": device_sources,
        "vision_models": vision_models,
        "sensor_protocols": sensor_protocols,
        "protocol_sources": protocol_sources,
        "skill_packages": skill_packages,
        "selected_skill_package": skill_packages[0] if skill_packages else "",
        "selected_capability": str(selected_route.get("capability") or ""),
        "capability_routes": capability_routes,
        "approval_required": approval_required,
        "action_boundary": str(selected_route.get("hardware_boundary") or ""),
        "output_contract": str(selected_route.get("output_contract") or ""),
        "acceptance_tests": acceptance_tests,
        "ingest_endpoint": "/api/field/ingest",
        "runtime_callback_endpoint": "/api/field/events/{event_id}/runtime-delivery",
        "runtime_boundary": "Skill packages can shape the field action, but hardware execution must still pass runtime arbiter and safety preflight.",
        "blockers": blockers,
        "manual_checks": manual_checks,
    }


def _field_ingest_scope_contract(
    *,
    normalized: dict[str, Any],
    event: dict[str, Any] | None = None,
    accepted: bool,
    status: str,
    reason: str = "",
) -> dict[str, Any]:
    """Return the customer/project/object binding evidence for one device ingest."""
    event = event if isinstance(event, dict) else {}
    has_event = bool(event)
    trust = (
        normalized.get("device_trust") if isinstance(normalized.get("device_trust"), dict) else {}
    )
    resource = (
        event.get("resource_execution_context")
        if isinstance(event.get("resource_execution_context"), dict)
        else {}
    )
    scope = event.get("project_scope") if isinstance(event.get("project_scope"), dict) else {}
    managed_object_id = str(
        event.get("managed_object_id")
        or resource.get("managed_object_id")
        or (normalized.get("managed_object_id") if not has_event else "")
        or ""
    )
    managed_object_display = str(
        event.get("managed_object_display")
        or (normalized.get("managed_object_display") if not has_event else "")
        or ""
    )
    managed_object_category = str(
        event.get("managed_object_category")
        or (normalized.get("managed_object_category") if not has_event else "")
        or ""
    )
    binding_status = _field_ingest_binding_status(
        accepted=accepted,
        status=status,
        reason=reason,
        managed_object_id=managed_object_id,
        resource_status=str(resource.get("overall_status") or ""),
        trust=trust,
    )
    gate = _field_ingest_production_gate(
        binding_status=binding_status,
        resource=resource,
        reason=reason,
        scenario_id=str(event.get("scenario_id") or normalized.get("scenario_id") or ""),
    )
    return {
        "contract_type": "askme.field.ingest_scope_contract.v1",
        "endpoint": "/api/field/ingest",
        "accepted": accepted,
        "status": status,
        "reason": reason,
        "scenario": {
            "scenario_id": str(event.get("scenario_id") or normalized.get("scenario_id") or ""),
            "scenario_name": str(event.get("scenario_name") or ""),
            "source": "normalized_device_payload",
        },
        "device": {
            "device_id": str(trust.get("device_id") or normalized.get("device_id") or ""),
            "source": str(normalized.get("source") or ""),
            "trusted": bool(trust.get("trusted", False)),
            "trust_status": str(trust.get("status") or ""),
            "trust_reason": str(trust.get("reason") or ""),
        },
        "customer_project": {
            "scope_source": "server_customer_project",
            "tenant_id": str(scope.get("tenant_id") or event.get("tenant_id") or ""),
            "delivery_namespace": str(scope.get("delivery_namespace") or ""),
            "customer_id": str(scope.get("customer_id") or event.get("customer_id") or ""),
            "project_id": str(scope.get("project_id") or event.get("project_id") or ""),
            "site_id": str(scope.get("site_id") or event.get("site_id") or ""),
            "site_name": str(scope.get("site_name") or event.get("site_name") or ""),
            "industry": str(scope.get("industry") or event.get("industry") or ""),
            "client_scope_ignored": bool(normalized.get("_ingested")),
        },
        "managed_object": {
            "bound": bool(managed_object_id),
            "object_id": managed_object_id,
            "display_name": managed_object_display,
            "category": managed_object_category,
            "binding_status": binding_status,
        },
        "resource_execution": {
            "overall_status": str(resource.get("overall_status") or ""),
            "reason": str(resource.get("reason") or ""),
            "selected_skill_package": str(resource.get("selected_skill_package") or ""),
            "selected_capability": str(resource.get("selected_capability") or ""),
            "approval_required": bool(resource.get("approval_required", False)),
            "blockers": [
                str(item) for item in _as_list(resource.get("blockers")) if str(item).strip()
            ],
            "manual_checks": [
                str(item) for item in _as_list(resource.get("manual_checks")) if str(item).strip()
            ],
        },
        "production_gate": gate,
        "audit": {
            "event_id": str(event.get("event_id") or ""),
            "evidence_count": len(
                event.get("evidence_media") if isinstance(event.get("evidence_media"), list) else []
            ),
            "freshness_status": str(event.get("freshness_status") or ""),
            "confidence": event.get("confidence"),
        },
    }


def _field_managed_object_binding_gate(managed_object: dict[str, Any]) -> dict[str, Any]:
    bindings = (
        managed_object.get("bindings") if isinstance(managed_object.get("bindings"), dict) else {}
    )
    blockers: list[str] = []
    manual_checks: list[str] = []
    required = (
        ("vision_models", "vision model binding missing"),
        ("sensor_protocols", "sensor protocol binding missing"),
        ("skill_packages", "skill package binding missing"),
        ("acceptance_tests", "acceptance test binding missing"),
    )
    for key, message in required:
        if not [item for item in _as_list(bindings.get(key)) if str(item).strip()]:
            blockers.append(message)
    if not [item for item in _as_list(managed_object.get("scenario_ids")) if str(item).strip()]:
        manual_checks.append("scenario binding not declared")
    if not [item for item in _as_list(managed_object.get("device_sources")) if str(item).strip()]:
        manual_checks.append("device source binding not declared")
    status = "blocked" if blockers else "manual_check" if manual_checks else "ready"
    return {"status": status, "blockers": blockers, "manual_checks": manual_checks}


def _field_device_onboarding_gate(
    device: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    require_trusted_devices: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    manual_checks: list[str] = []
    if not device.get("registered"):
        blockers.append("device is not registered")
    if require_trusted_devices and not device.get("signature_required"):
        blockers.append("trusted device mode requires device signatures")
    if device.get("signature_required") and not device.get("secret_configured"):
        blockers.append("device signature secret is not configured")
    if device.get("status") == "never_seen":
        manual_checks.append("device has not sent an onsite payload yet")
    elif device.get("status") == "stale":
        manual_checks.append("device last payload is stale")
    if not candidates:
        manual_checks.append("device is not linked to a managed object")
    elif not any(candidate.get("binding_status") == "ready" for candidate in candidates):
        manual_checks.append("linked managed object still has binding gaps")
    if (
        device.get("trust_status") == "blocked"
        or device.get("trusted") is False
        and device.get("last_seen_at")
    ):
        blockers.append(str(device.get("trust_reason") or "latest device payload was not trusted"))
    status = "blocked" if blockers else "manual_check" if manual_checks else "ready"
    if status == "ready":
        message = "设备已登记、签名策略可用、已回传，并能绑定到客户现场对象。"
        required_action = "把最近一次真实回传加入客户验收证据。"
    elif status == "blocked":
        message = "设备接入存在阻断项，不能作为现场验收证据。"
        required_action = "; ".join(blockers)
    else:
        message = "设备接入需要交付复核，当前只能作为试点或演示证据。"
        required_action = "; ".join(manual_checks)
    return {
        "status": status,
        "ready": status == "ready",
        "customer_message": message,
        "required_action": required_action,
        "blockers": blockers,
        "manual_checks": manual_checks,
    }


def _field_device_onboarding_next_actions(devices: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    if any(not item.get("registered") for item in devices):
        actions.append("Register every observed device in field_operations.device_registry.")
    if any(
        item.get("signature_required") and not item.get("secret_configured") for item in devices
    ):
        actions.append("Configure HMAC secrets for devices that require signed payloads.")
    if any(item.get("status") == "never_seen" for item in devices):
        actions.append(
            "Send one real or lab-signed payload from each registered device to /api/field/ingest."
        )
    if any(not item.get("managed_object_candidates") for item in devices):
        actions.append("Bind device sources or device IDs to customer managed objects.")
    if any(
        candidate.get("binding_status") != "ready"
        for item in devices
        for candidate in item.get("managed_object_candidates") or []
    ):
        actions.append(
            "Complete managed-object bindings: vision model, sensor protocol, skill package, and acceptance test."
        )
    return actions or ["Archive the latest signed device payloads as onsite acceptance evidence."]


def _field_ingest_binding_status(
    *,
    accepted: bool,
    status: str,
    reason: str,
    managed_object_id: str,
    resource_status: str,
    trust: dict[str, Any],
) -> str:
    if not bool(trust.get("trusted", False)):
        return "blocked_device_trust"
    if not accepted and reason == "no_matching_field_scenario":
        return "no_matching_scenario"
    if not managed_object_id:
        return "unbound_managed_object"
    if resource_status == "ready":
        return "bound_ready"
    if resource_status == "blocked":
        return "bound_blocked"
    if resource_status == "manual_check":
        return "bound_manual_check"
    return status or "unknown"


def _field_ingest_production_gate(
    *,
    binding_status: str,
    resource: dict[str, Any],
    reason: str,
    scenario_id: str,
) -> dict[str, Any]:
    if binding_status == "bound_ready":
        return {
            "ready": True,
            "status": "ready",
            "reason": "device_project_object_resource_binding_ready",
            "customer_message": "该设备事件已绑定到客户项目、现场对象和可执行能力，可进入现场验收记录。",
            "required_action": "归档事件证据，并按场景验收用例复核结果。",
        }
    if binding_status == "blocked_device_trust":
        trust_reason = reason or "device_not_trusted"
        return {
            "ready": False,
            "status": "blocked",
            "reason": trust_reason,
            "customer_message": "设备身份未通过信任校验，不能作为现场事件证据。",
            "required_action": "登记设备、配置签名密钥，并重新发送带签名的设备 payload。",
        }
    if binding_status == "no_matching_scenario":
        return {
            "ready": False,
            "status": "ignored",
            "reason": "no_matching_field_scenario",
            "customer_message": "设备数据已收到，但没有匹配到可处置的业务场景。",
            "required_action": "补充适配器规则，或确认该设备数据不应触发现场事件。",
        }
    if binding_status == "unbound_managed_object":
        return {
            "ready": False,
            "status": "manual_check",
            "reason": "managed_object_binding_required",
            "customer_message": "事件可以记录，但还没有绑定到客户现场对象，不能作为生产验收闭环。",
            "required_action": "把设备来源、视觉模型、传感器协议、技能包和验收用例绑定到客户项目对象。",
        }
    resource_reason = str(resource.get("reason") or "")
    blockers = [str(item) for item in _as_list(resource.get("blockers")) if str(item).strip()]
    manual_checks = [
        str(item) for item in _as_list(resource.get("manual_checks")) if str(item).strip()
    ]
    return {
        "ready": False,
        "status": "manual_check",
        "reason": resource_reason or "resource_binding_requires_review",
        "customer_message": "事件已绑定现场对象，但资源绑定仍需交付复核。",
        "required_action": (
            "补齐阻断项：" + "; ".join(blockers or manual_checks or [f"scenario={scenario_id}"])
        ),
    }


def _field_sensor_protocol_sources(protocol_id: str) -> list[str]:
    text = str(protocol_id or "").lower()
    sources: list[str] = []
    if any(token in text for token in ("camera", "vision", "detection", "video")):
        sources.append("camera")
    if any(token in text for token in ("sensor", "smoke", "temperature", "mqtt", "iot")):
        sources.append("sensor")
    if any(token in text for token in ("robot", "route", "runtime", "status", "voice")):
        sources.append("robot")
    return sorted(set(sources)) or ["custom"]


def _field_event_view(event: dict[str, Any]) -> dict[str, Any]:
    view = dict(event)
    if not isinstance(view.get("evidence_media"), list):
        payload = view.get("payload") if isinstance(view.get("payload"), dict) else {}
        view["evidence_media"] = _field_evidence_media(payload)
    _refresh_field_incident_workflow(view)
    payload = view.get("payload") if isinstance(view.get("payload"), dict) else {}
    if payload.get("_ingested"):
        view["ingest_scope_contract"] = _field_ingest_scope_contract(
            normalized=payload,
            event=view,
            accepted=str(view.get("status") or "") not in {"rejected", "ignored"},
            status=str(view.get("status") or ""),
            reason=str(view.get("reason") or ""),
        )
    view["admission_decision"] = _field_admission_decision(view)
    view["sla"] = _field_event_sla(view)
    view["close_approval_required"] = _field_event_requires_close_approval(view)
    return view


def _field_admission_decision(event: dict[str, Any]) -> dict[str, Any]:
    """Return customer-readable trigger, block, and escalation reasoning."""

    status = str(event.get("status") or "")
    operator_action = str(event.get("operator_action") or event.get("narrative") or "")
    missing_evidence = (
        [str(item) for item in event.get("missing_evidence", []) if str(item).strip()]
        if isinstance(event.get("missing_evidence"), list)
        else []
    )
    freshness = str(event.get("freshness_status") or "")
    freshness_age_s = _float_or_none(event.get("freshness_age_s"))
    confidence = _float_or_none(event.get("confidence"))
    duplicate_of = str(event.get("duplicate_of") or "")
    resource_context = (
        event.get("resource_execution_context")
        if isinstance(event.get("resource_execution_context"), dict)
        else {}
    )
    resource_status = str(resource_context.get("overall_status") or "")
    resource_reason = str(resource_context.get("reason") or "")
    technical_reasons: list[str] = []
    evidence_facts: list[dict[str, Any]] = []

    if freshness and freshness not in {"fresh", "not_applicable"}:
        technical_reasons.append("stale" if freshness == "stale" else freshness)
        evidence_facts.append(
            {
                "label": "freshness",
                "value": freshness,
                "age_s": freshness_age_s,
            }
        )
    if confidence is not None:
        evidence_facts.append({"label": "confidence", "value": round(confidence, 3)})
        if status == "needs_review" and "置信度" in operator_action:
            technical_reasons.append("low_confidence")
    if missing_evidence:
        technical_reasons.append("missing_evidence")
        evidence_facts.append({"label": "missing_evidence", "value": missing_evidence})
    if duplicate_of:
        technical_reasons.append("duplicate")
        evidence_facts.append({"label": "duplicate_of", "value": duplicate_of})
    if resource_status and resource_status != "ready":
        if resource_reason == "no_managed_object_matched":
            technical_reasons.append("unbound_managed_object")
        elif resource_status in {"manual_check", "blocked"}:
            technical_reasons.append("resource_binding_check")
        evidence_facts.append(
            {
                "label": "resource_binding",
                "value": resource_reason or resource_status,
                "status": resource_status,
            }
        )

    blocked = status in {"needs_review", "needs_evidence", "duplicate", "rejected", "ignored"}
    title = {
        "needs_review": "未升级告警：需要人工复核",
        "needs_evidence": "未通知处置群：缺少必需证据",
        "duplicate": "未重复通知：重复事件已合并",
        "rejected": "未接收：来源或权限不可信",
        "ignored": "未触发：没有匹配到现场场景",
        "triggered": "已触发现场处置",
        "acknowledged": "已确认，等待后续处置",
        "pending_close_approval": "等待主管关闭审批",
        "closed": "事件已关闭",
    }.get(status, "触发准入判定")
    customer_status = "blocked_or_review" if blocked else "accepted"
    if status == "duplicate":
        customer_status = "deduped"
    return {
        "status": status or "unknown",
        "customer_status": customer_status,
        "blocked": blocked,
        "title": title,
        "reason": operator_action,
        "technical_reasons": sorted(set(technical_reasons)),
        "evidence_facts": evidence_facts,
        "next_step": _field_admission_next_step(status, technical_reasons),
    }


def _field_admission_next_step(status: str, technical_reasons: list[str]) -> str:
    reasons = set(technical_reasons)
    if "stale" in reasons or "missing_timestamp" in reasons:
        return "重新采集新鲜现场证据后再触发。"
    if "low_confidence" in reasons:
        return "人工确认或补充更高置信度识别证据。"
    if "missing_evidence" in reasons:
        return "补齐照片、位置、传感器或管理员身份。"
    if status == "duplicate":
        return "如现场状态变化，请上传新的证据。"
    if "unbound_managed_object" in reasons or "resource_binding_check" in reasons:
        return "当前事件可继续处置；交付验收前需要将设备、视觉模型、传感器协议和验收用例绑定到客户现场对象。"
    if status in {"triggered", "acknowledged"}:
        return "继续跟踪通知、机器人动作和现场处置结果。"
    return "由现场操作员复核后决定是否升级。"


def _scenario_intent_routes_for(scenario_id: str) -> list[dict[str, Any]]:
    """Return deterministic natural-language preview routes for a field scenario."""

    target_id = str(scenario_id or "")
    routes: list[dict[str, Any]] = []
    for rule in SCENARIO_INTENT_RULES:
        rule_scenario_id = _SCENARIO_INTENT_ALIASES.get(rule.scenario_id, rule.scenario_id)
        if rule_scenario_id != target_id:
            continue
        routes.append(
            {
                "rule_id": rule.rule_id,
                "skill_name": rule.skill_name,
                "scenario_id": rule.scenario_id,
                "preview_endpoint": "/api/scenario-intents/preview",
                "risk_level": rule.risk_level,
                "confidence": rule.confidence,
                "example_terms": list(rule.any_terms[:6]),
                "exclude_terms": list(rule.exclude_terms),
                "does_not_execute_skill": True,
            }
        )
    return routes


def _refresh_field_incident_workflow(event: dict[str, Any]) -> None:
    workflow = _field_incident_workflow(event)
    event["incident_state"] = workflow["state"]
    event["incident_stage"] = workflow["stage"]
    event["incident_workflow"] = workflow


def _field_incident_workflow(event: dict[str, Any]) -> dict[str, Any]:
    stages = [
        _field_workflow_stage(
            "admission",
            _field_admission_status(event),
            owner="device-gate",
            detail=_field_admission_detail(event),
        ),
        _field_workflow_stage(
            "assessment",
            _field_assessment_status(event),
            owner="scenario-engine",
            detail=str(event.get("status") or ""),
        ),
        _field_workflow_stage(
            "notification",
            _field_notification_status(event),
            owner=str(event.get("notification_group") or "none"),
            detail=_field_notification_detail(event),
        ),
        _field_workflow_stage(
            "voice",
            _field_voice_status(event),
            owner="voice-runtime",
            detail=_field_voice_detail(event),
        ),
        _field_workflow_stage(
            "robot_motion",
            _field_robot_motion_status(event),
            owner="runtime-arbiter",
            detail=_field_robot_motion_detail(event),
        ),
        _field_workflow_stage(
            "operator",
            _field_operator_status(event),
            owner="field-ops",
            detail=str(event.get("acknowledged_by") or event.get("closed_by") or ""),
        ),
        _field_workflow_stage(
            "archive",
            "written",
            owner="field-archive",
            detail="event_jsonl",
        ),
        _field_workflow_stage(
            "memory",
            _field_memory_status(event),
            owner="memory",
            detail=_field_memory_detail(event),
        ),
    ]
    state, stage = _field_workflow_state(stages, str(event.get("status") or ""))
    return {
        "state": state,
        "stage": stage,
        "stages": stages,
        "timeline": _field_workflow_timeline(event, stages),
        "open_gaps": [
            item["stage"]
            for item in stages
            if item["status"] in {"blocked", "failed", "pending", "not_connected"}
        ],
    }


def _field_workflow_stage(
    stage: str,
    status: str,
    *,
    owner: str,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": status,
        "owner": owner,
        "detail": detail,
    }


def _field_workflow_state(stages: list[dict[str, Any]], status: str) -> tuple[str, str]:
    if status == "closed":
        return "closed", "closed"
    if status == "duplicate":
        return "duplicate", "duplicate"
    if status == "pending_close_approval":
        return "pending_operator_approval", "operator"
    for item in stages:
        if item["status"] in {"blocked", "failed"}:
            return "blocked", str(item["stage"])
    for item in stages:
        if item["status"] in {"pending", "not_connected"}:
            return "active", str(item["stage"])
    if status == "acknowledged":
        return "acknowledged", "operator"
    return "active", "monitoring"


def _field_admission_status(event: dict[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    trust = payload.get("device_trust") if isinstance(payload.get("device_trust"), dict) else {}
    if trust:
        return "accepted" if trust.get("trusted") else "blocked"
    if payload.get("trigger_source") == "operator_manual":
        return "operator_manual"
    if payload.get("_ingested"):
        return "accepted"
    return "not_required"


def _field_admission_detail(event: dict[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    trust = payload.get("device_trust") if isinstance(payload.get("device_trust"), dict) else {}
    if trust:
        return str(trust.get("reason") or trust.get("status") or trust.get("device_id") or "")
    return str(payload.get("operator_id") or payload.get("source") or "")


def _field_assessment_status(event: dict[str, Any]) -> str:
    status = str(event.get("status") or "")
    if status == "needs_evidence":
        return "blocked"
    if status == "needs_review":
        return "blocked"
    if status == "duplicate":
        return "duplicate"
    return "accepted"


def _field_notification_status(event: dict[str, Any]) -> str:
    if not event.get("incident_topic") or event.get("notification_group") == "none":
        return "not_required"
    delivery = (
        event.get("delivery_report") if isinstance(event.get("delivery_report"), list) else []
    )
    sent_channels = (
        event.get("sent_channels") if isinstance(event.get("sent_channels"), list) else []
    )
    if not delivery and not sent_channels:
        return "pending"
    if any(isinstance(item, dict) and item.get("status") == "sent" for item in delivery):
        if any(
            isinstance(item, dict) and item.get("status") not in {"sent", ""} for item in delivery
        ):
            return "partial"
        return "sent"
    if sent_channels:
        return "sent"
    return "failed"


def _field_notification_detail(event: dict[str, Any]) -> str:
    delivery = (
        event.get("delivery_report") if isinstance(event.get("delivery_report"), list) else []
    )
    if delivery:
        return ",".join(
            f"{item.get('channel')}:{item.get('status')}"
            for item in delivery
            if isinstance(item, dict)
        )
    return ",".join(str(item) for item in event.get("sent_channels") or [])


def _field_voice_status(event: dict[str, Any]) -> str:
    directive = (
        event.get("voice_directive") if isinstance(event.get("voice_directive"), dict) else {}
    )
    if not directive:
        return "not_required"
    delivery = event.get("voice_delivery") if isinstance(event.get("voice_delivery"), dict) else {}
    if delivery:
        return str(delivery.get("status") or "queued")
    return "queued"


def _field_voice_detail(event: dict[str, Any]) -> str:
    directive = (
        event.get("voice_directive") if isinstance(event.get("voice_directive"), dict) else {}
    )
    delivery = event.get("voice_delivery") if isinstance(event.get("voice_delivery"), dict) else {}
    return str(
        delivery.get("status")
        or directive.get("resolved_profile")
        or directive.get("requested_profile")
        or ""
    )


def _field_robot_motion_status(event: dict[str, Any]) -> str:
    playbook = event.get("playbook") if isinstance(event.get("playbook"), dict) else {}
    policy = str(playbook.get("robot_motion_policy") or "").strip()
    if not policy:
        return "not_required"
    runtime = (
        event.get("runtime_delivery") if isinstance(event.get("runtime_delivery"), dict) else {}
    )
    if runtime:
        return str(runtime.get("status") or "submitted")
    return "policy_ready"


def _field_robot_motion_detail(event: dict[str, Any]) -> str:
    playbook = event.get("playbook") if isinstance(event.get("playbook"), dict) else {}
    runtime = (
        event.get("runtime_delivery") if isinstance(event.get("runtime_delivery"), dict) else {}
    )
    return str(runtime.get("status") or playbook.get("robot_motion_policy") or "")


def _field_operator_status(event: dict[str, Any]) -> str:
    status = str(event.get("status") or "")
    if status == "closed":
        return "closed"
    if status == "pending_close_approval":
        return "pending_approval"
    if event.get("acknowledged_at"):
        return "acknowledged"
    return "pending"


def _field_memory_status(event: dict[str, Any]) -> str:
    memory = event.get("memory_delivery") if isinstance(event.get("memory_delivery"), dict) else {}
    if memory:
        return str(memory.get("status") or "written")
    if event.get("status") == "closed":
        return "pending"
    return "not_connected"


def _field_memory_detail(event: dict[str, Any]) -> str:
    memory = event.get("memory_delivery") if isinstance(event.get("memory_delivery"), dict) else {}
    if memory:
        return str(memory.get("target") or memory.get("status") or "")
    return "incident_summary_not_written"


def _field_incident_memory_kind(event: dict[str, Any]) -> str:
    if event.get("severity") == "error" or event.get("priority") == "P0":
        return "anomaly"
    category = str(event.get("category") or "").lower()
    topic = str(event.get("incident_topic") or "").lower()
    if any(token in category or token in topic for token in ("security", "safety", "robot")):
        return "anomaly"
    return "observation"


def _field_incident_memory_text(event: dict[str, Any]) -> str:
    parts = [
        f"field_event={event.get('event_id') or '-'}",
        f"scenario={event.get('scenario_id') or '-'}",
        f"topic={event.get('incident_topic') or '-'}",
        f"status={event.get('status') or '-'}",
        f"severity={event.get('severity') or '-'}",
        f"priority={event.get('priority') or '-'}",
        f"location={event.get('location') or '-'}",
        f"closed_by={event.get('closed_by') or '-'}",
        f"close_note={event.get('close_note') or '-'}",
    ]
    media = event.get("evidence_media") if isinstance(event.get("evidence_media"), list) else []
    if media:
        parts.append(
            "evidence="
            + ",".join(str(item.get("path") or item.get("preview_url") or "") for item in media[:3])
        )
    return "; ".join(parts)


def _field_workflow_timeline(
    event: dict[str, Any],
    stages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    created_at = _float_or_none(event.get("created_at")) or time.time()
    timeline = [
        {
            "type": "workflow_stage",
            "stage": item["stage"],
            "status": item["status"],
            "owner": item["owner"],
            "detail": item["detail"],
            "at": round(created_at, 3),
        }
        for item in stages
        if item["status"] not in {"not_required"}
    ]
    timeline.extend(_field_event_timeline(event))
    return sorted(timeline, key=lambda item: float(item.get("at") or 0.0))


def _field_evidence_media(payload: dict[str, Any]) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    image_keys = (
        "image_url",
        "image_path",
        "photo_url",
        "photo_path",
        "snapshot_url",
        "snapshot_path",
        "frame_url",
        "frame_path",
    )
    video_keys = ("video_url", "video_path", "clip_url", "clip_path")
    for key in image_keys:
        _append_evidence_media(media, payload.get(key), media_type="image", source_key=key)
    for key in video_keys:
        _append_evidence_media(media, payload.get(key), media_type="video", source_key=key)

    detections = payload.get("detections") if isinstance(payload.get("detections"), list) else []
    for index, item in enumerate(detections):
        if not isinstance(item, dict):
            continue
        for key in image_keys:
            _append_evidence_media(
                media,
                item.get(key),
                media_type="image",
                source_key=f"detections[{index}].{key}",
            )
        for key in video_keys:
            _append_evidence_media(
                media,
                item.get(key),
                media_type="video",
                source_key=f"detections[{index}].{key}",
            )
    return media[:8]


def _field_event_sla(event: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    created_at = _float_or_none(event.get("created_at")) or time.time()
    playbook = event.get("playbook") if isinstance(event.get("playbook"), dict) else {}
    seconds = _float_or_none(playbook.get("escalation_after_s"))
    if seconds is None or seconds <= 0:
        seconds = _FIELD_EVENT_SLA_SECONDS.get(str(event.get("priority") or ""), 1800.0)
    current = time.time() if now is None else now
    due_at = created_at + seconds
    closed = event.get("status") == "closed"
    remaining_s = due_at - current
    if closed:
        state = "closed"
    elif remaining_s < 0:
        state = "overdue"
    elif remaining_s <= min(300.0, seconds * 0.25):
        state = "due_soon"
    else:
        state = "active"
    return {
        "state": state,
        "due_at": round(due_at, 3),
        "remaining_s": round(remaining_s, 1),
        "target_s": round(seconds, 1),
    }


def _append_evidence_media(
    media: list[dict[str, Any]],
    value: Any,
    *,
    media_type: str,
    source_key: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        return
    path = value.strip()
    if any(item.get("path") == path for item in media):
        return
    preview_url = path if path.startswith(("http://", "https://", "data:", "/api/")) else ""
    if not preview_url and _is_local_evidence_path(path):
        preview_url = f"/api/field/evidence?path={quote(path, safe='')}"
    media.append(
        {
            "type": media_type,
            "source_key": source_key,
            "path": path,
            "preview_url": preview_url,
            "label": "现场照片" if media_type == "image" else "现场视频",
        }
    )


def _is_local_evidence_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("/")
    return normalized.startswith(("artifacts/", "output/", "data/field_evidence/"))


def _field_event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Return product-facing field-event counters for the operations UI."""
    open_events = [event for event in events if event.get("status") != "closed"]
    needs_attention = [event for event in open_events if _field_event_needs_attention(event)]
    now = time.time()
    overdue = [
        event for event in open_events if _field_event_sla(event, now=now)["state"] == "overdue"
    ]
    due_soon = [
        event for event in open_events if _field_event_sla(event, now=now)["state"] == "due_soon"
    ]
    by_status: dict[str, int] = {}
    by_group: dict[str, int] = {}
    by_project: dict[str, int] = {}
    by_managed_object: dict[str, int] = {}
    for event in events:
        status = str(event.get("status") or "unknown")
        group = str(event.get("notification_group") or "none")
        project = str(event.get("project_id") or "unscoped")
        managed_object = str(event.get("managed_object_id") or "unbound")
        by_status[status] = by_status.get(status, 0) + 1
        by_group[group] = by_group.get(group, 0) + 1
        by_project[project] = by_project.get(project, 0) + 1
        by_managed_object[managed_object] = by_managed_object.get(managed_object, 0) + 1
    latest = events[-1] if events else {}
    return {
        "total": len(events),
        "open": len(open_events),
        "needs_attention": len(needs_attention),
        "overdue": len(overdue),
        "due_soon": len(due_soon),
        "closed": by_status.get("closed", 0),
        "acknowledged": by_status.get("acknowledged", 0),
        "by_status": by_status,
        "by_notification_group": by_group,
        "by_project": by_project,
        "by_managed_object": by_managed_object,
        "latest_event_id": latest.get("event_id"),
        "latest_scenario": latest.get("scenario_id"),
    }


def _filter_field_events_by_project_scope(
    events: list[dict[str, Any]],
    *,
    tenant_id: str | None = None,
    delivery_namespace: str | None = None,
    customer_id: str | None = None,
    project_id: str | None = None,
    site_id: str | None = None,
    project_scope: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    exact_filters = {
        "tenant_id": str(tenant_id or "").strip(),
        "delivery_namespace": str(delivery_namespace or "").strip(),
        "customer_id": str(customer_id or "").strip(),
        "project_id": str(project_id or "").strip(),
        "site_id": str(site_id or "").strip(),
    }
    scope = project_scope if isinstance(project_scope, dict) else {}
    filtered: list[dict[str, Any]] = []
    for event in events:
        item = _field_event_project_scope_item(event)
        if any(value and item.get(key) != value for key, value in exact_filters.items()):
            continue
        if not _project_scope_allows_field_event(scope, item):
            continue
        filtered.append(event)
    return filtered


def _field_event_project_scope_item(event: dict[str, Any]) -> dict[str, str]:
    scope = event.get("project_scope") if isinstance(event.get("project_scope"), dict) else {}
    return {
        "tenant_id": str(event.get("tenant_id") or scope.get("tenant_id") or "").strip(),
        "delivery_namespace": str(
            event.get("delivery_namespace") or scope.get("delivery_namespace") or ""
        ).strip(),
        "customer_id": str(event.get("customer_id") or scope.get("customer_id") or "").strip(),
        "project_id": str(event.get("project_id") or scope.get("project_id") or "").strip(),
        "site_id": str(event.get("site_id") or scope.get("site_id") or "").strip(),
    }


def _project_scope_allows_field_event(
    project_scope: dict[str, Any],
    item: dict[str, str],
) -> bool:
    if not project_scope or not any(project_scope.values()):
        return True
    for scope_key, item_key in (
        ("tenant_ids", "tenant_id"),
        ("delivery_namespaces", "delivery_namespace"),
        ("customer_ids", "customer_id"),
        ("project_ids", "project_id"),
        ("site_ids", "site_id"),
    ):
        allowed = [
            str(value).strip()
            for value in _as_list(project_scope.get(scope_key))
            if str(value).strip()
        ]
        if not allowed or "*" in allowed:
            continue
        if item.get(item_key) not in allowed:
            return False
    return True


def _filter_field_events(
    events: list[dict[str, Any]],
    *,
    status: str | None = None,
    notification_group: str | None = None,
    needs_attention: bool = False,
    customer_id: str | None = None,
    project_id: str | None = None,
    site_id: str | None = None,
    managed_object_id: str | None = None,
) -> list[dict[str, Any]]:
    status_filter = str(status or "").strip()
    group_filter = str(notification_group or "").strip()
    customer_filter = str(customer_id or "").strip()
    project_filter = str(project_id or "").strip()
    site_filter = str(site_id or "").strip()
    object_filter = str(managed_object_id or "").strip()
    filtered: list[dict[str, Any]] = []
    for event in events:
        if status_filter and event.get("status") != status_filter:
            continue
        if group_filter and event.get("notification_group") != group_filter:
            continue
        if customer_filter and event.get("customer_id") != customer_filter:
            continue
        if project_filter and event.get("project_id") != project_filter:
            continue
        if site_filter and event.get("site_id") != site_filter:
            continue
        if object_filter and event.get("managed_object_id") != object_filter:
            continue
        if needs_attention and not _field_event_needs_attention(event):
            continue
        filtered.append(event)
    return filtered


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    return [value]


def _managed_object_scope_score(
    managed_object: dict[str, Any],
    scope: dict[str, Any],
) -> tuple[bool, int]:
    score = 0
    for list_key, scope_key in (
        ("tenant_ids", "tenant_id"),
        ("delivery_namespaces", "delivery_namespace"),
        ("customer_ids", "customer_id"),
        ("project_ids", "project_id"),
        ("site_ids", "site_id"),
    ):
        allowed = [
            str(item).strip()
            for item in _as_list(managed_object.get(list_key) or managed_object.get(scope_key))
            if str(item).strip()
        ]
        if not allowed:
            continue
        value = str(scope.get(scope_key) or "").strip()
        if "*" in allowed:
            score += 1
            continue
        if not value or value not in allowed:
            return False, 0
        score += 50
    return True, score


def _payload_detection_labels(payload: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for item in _as_list(payload.get("object_labels")):
        if str(item).strip():
            labels.add(str(item).strip())
    detections = payload.get("detections") if isinstance(payload.get("detections"), list) else []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        label = str(
            detection.get("label")
            or detection.get("class")
            or detection.get("class_name")
            or detection.get("name")
            or ""
        ).strip()
        if label:
            labels.add(label)
    return labels


def _field_event_needs_attention(event: dict[str, Any]) -> bool:
    if event.get("status") in {"closed", "duplicate"}:
        return False
    delivery = (
        event.get("delivery_report") if isinstance(event.get("delivery_report"), list) else []
    )
    if any(
        item.get("status") and item.get("status") != "sent"
        for item in delivery
        if isinstance(item, dict)
    ):
        return True
    if event.get("status") == "acknowledged":
        return False
    if event.get("status") == "pending_close_approval":
        return True
    return event.get("status") != "closed"


def _round_duration(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, value), 1)


def _field_event_timeline(event: dict[str, Any]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    created_at = _float_or_none(event.get("created_at"))
    if created_at is not None:
        timeline.append(
            {
                "type": "created",
                "at": round(created_at, 3),
                "actor": "system",
                "note": str(event.get("scenario_name") or event.get("scenario_id") or ""),
            }
        )
    if event.get("sent_channels"):
        timeline.append(
            {
                "type": "notification_sent",
                "at": round(created_at or time.time(), 3),
                "actor": "system",
                "note": ",".join(str(item) for item in event.get("sent_channels") or []),
            }
        )
    for resend in event.get("notification_resends") or []:
        if not isinstance(resend, dict):
            continue
        timeline.append(
            {
                "type": "notification_resent",
                "at": round(_float_or_none(resend.get("resent_at")) or time.time(), 3),
                "actor": str(resend.get("resent_by") or "askme.operator"),
                "note": str(resend.get("note") or ""),
            }
        )
    acknowledged_at = _float_or_none(event.get("acknowledged_at"))
    if acknowledged_at is not None:
        timeline.append(
            {
                "type": "acknowledged",
                "at": round(acknowledged_at, 3),
                "actor": str(event.get("acknowledged_by") or "askme.operator"),
                "note": str(event.get("acknowledge_note") or ""),
            }
        )
    close_requested_at = _float_or_none(event.get("close_requested_at"))
    if close_requested_at is not None:
        timeline.append(
            {
                "type": "close_requested",
                "at": round(close_requested_at, 3),
                "actor": str(event.get("close_requested_by") or "askme.operator"),
                "note": str(event.get("close_request_note") or ""),
            }
        )
    approval = event.get("close_approval") if isinstance(event.get("close_approval"), dict) else {}
    approved_at = _float_or_none(approval.get("approved_at")) if approval else None
    if approved_at is not None:
        timeline.append(
            {
                "type": "close_approved",
                "at": round(approved_at, 3),
                "actor": str(approval.get("supervisor_id") or "supervisor"),
                "note": str(approval.get("approval_note") or ""),
            }
        )
    closed_at = _float_or_none(event.get("closed_at"))
    if closed_at is not None:
        timeline.append(
            {
                "type": "closed",
                "at": round(closed_at, 3),
                "actor": str(event.get("closed_by") or "askme.operator"),
                "note": str(event.get("close_note") or ""),
            }
        )
    return sorted(timeline, key=lambda item: float(item.get("at") or 0.0))


def _field_notification_attempts(event: dict[str, Any]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    delivery = (
        event.get("delivery_report") if isinstance(event.get("delivery_report"), list) else []
    )
    if delivery or event.get("sent_channels"):
        attempts.append(
            {
                "kind": "initial",
                "at": round(_float_or_none(event.get("created_at")) or time.time(), 3),
                "by": "system",
                "sent_channels": list(event.get("sent_channels") or []),
                "delivery_report": delivery,
                "note": "",
            }
        )
    for item in event.get("notification_resends") or []:
        if not isinstance(item, dict):
            continue
        attempts.append(
            {
                "kind": "resend",
                "at": round(_float_or_none(item.get("resent_at")) or time.time(), 3),
                "by": str(item.get("resent_by") or "askme.operator"),
                "sent_channels": list(item.get("sent_channels") or []),
                "delivery_report": list(item.get("delivery_report") or []),
                "note": str(item.get("note") or ""),
            }
        )
    return attempts


def _field_runtime_delivery_receipt(delivery: dict[str, Any]) -> dict[str, Any]:
    trust = (
        delivery.get("runtime_callback_trust")
        if isinstance(delivery.get("runtime_callback_trust"), dict)
        else {}
    )
    return {
        "runtime_callback_id": str(delivery.get("runtime_callback_id") or ""),
        "status": str(delivery.get("status") or ""),
        "run_id": str(delivery.get("run_id") or ""),
        "handoff_id": str(delivery.get("handoff_id") or ""),
        "hardware_dispatch": bool(delivery.get("hardware_dispatch", False)),
        "trusted": bool(trust.get("trusted", False)),
        "trust_status": str(trust.get("status") or ""),
        "recorded_at": _float_or_none(delivery.get("recorded_at")) or round(time.time(), 3),
    }


def _field_event_report(event: dict[str, Any]) -> dict[str, Any]:
    delivery = (
        event.get("delivery_report") if isinstance(event.get("delivery_report"), list) else []
    )
    evidence_media = (
        event.get("evidence_media") if isinstance(event.get("evidence_media"), list) else []
    )
    action_audit = event.get("action_audit") if isinstance(event.get("action_audit"), list) else []
    sla = event.get("sla") if isinstance(event.get("sla"), dict) else {}
    created_at = _float_or_none(event.get("created_at"))
    acknowledged_at = _float_or_none(event.get("acknowledged_at"))
    close_requested_at = _float_or_none(event.get("close_requested_at"))
    closed_at = _float_or_none(event.get("closed_at"))
    due_at = _float_or_none(sla.get("due_at"))
    resolution_latency_s = (
        _round_duration(closed_at - created_at)
        if closed_at is not None and created_at is not None
        else None
    )
    ack_latency_s = (
        _round_duration(acknowledged_at - created_at)
        if acknowledged_at is not None and created_at is not None
        else None
    )
    sla_met = closed_at <= due_at if closed_at is not None and due_at is not None else None
    return {
        "title": f"{event.get('scenario_name') or event.get('scenario_id')} 处置报告",
        "event_id": event.get("event_id"),
        "scenario_id": event.get("scenario_id"),
        "customer_id": event.get("customer_id") or "",
        "project_id": event.get("project_id") or "",
        "site_id": event.get("site_id") or "",
        "site_name": event.get("site_name") or "",
        "industry": event.get("industry") or "",
        "managed_object_id": event.get("managed_object_id") or "",
        "managed_object_display": event.get("managed_object_display") or "",
        "managed_object_category": event.get("managed_object_category") or "",
        "managed_object_bindings": (
            event.get("managed_object_bindings")
            if isinstance(event.get("managed_object_bindings"), dict)
            else {}
        ),
        "resource_execution_context": (
            event.get("resource_execution_context")
            if isinstance(event.get("resource_execution_context"), dict)
            else {}
        ),
        "status": event.get("status"),
        "incident_state": event.get("incident_state") or "",
        "incident_stage": event.get("incident_stage") or "",
        "incident_workflow": (
            event.get("incident_workflow")
            if isinstance(event.get("incident_workflow"), dict)
            else {}
        ),
        "priority": event.get("priority"),
        "severity": event.get("severity"),
        "location": event.get("location"),
        "notification_group": event.get("notification_group"),
        "voice": event.get("voice") or event.get("operator_action") or "",
        "operator_action": event.get("operator_action") or "",
        "sla_state": sla.get("state"),
        "sla_remaining_s": sla.get("remaining_s"),
        "sla_target_s": sla.get("target_s"),
        "sla_due_at": due_at,
        "sla_met": sla_met,
        "created_at": created_at,
        "acknowledged_at": acknowledged_at,
        "close_requested_at": close_requested_at,
        "closed_at": closed_at,
        "ack_latency_s": ack_latency_s,
        "resolution_latency_s": resolution_latency_s,
        "evidence_count": len(evidence_media),
        "delivery_statuses": [
            {
                "channel": item.get("channel"),
                "status": item.get("status"),
                "reason": item.get("reason") or item.get("error") or "",
            }
            for item in delivery
            if isinstance(item, dict)
        ],
        "acknowledged_by": event.get("acknowledged_by") or "",
        "acknowledge_note": event.get("acknowledge_note") or "",
        "close_requested_by": event.get("close_requested_by") or "",
        "close_request_note": event.get("close_request_note") or "",
        "closed_by": event.get("closed_by") or "",
        "close_note": event.get("close_note") or "",
        "close_approval": event.get("close_approval")
        if isinstance(event.get("close_approval"), dict)
        else {},
        "memory_delivery": (
            event.get("memory_delivery") if isinstance(event.get("memory_delivery"), dict) else {}
        ),
        "runtime_delivery": (
            event.get("runtime_delivery") if isinstance(event.get("runtime_delivery"), dict) else {}
        ),
        "runtime_delivery_receipts": (
            event.get("runtime_delivery_receipts")
            if isinstance(event.get("runtime_delivery_receipts"), list)
            else []
        ),
        "evidence_media": evidence_media,
        "action_audit": action_audit,
        "notification_attempts": _field_notification_attempts(event),
        "timeline": _field_event_timeline(event),
    }


def _field_event_report_markdown(event: dict[str, Any]) -> str:
    report = _field_event_report(event)
    delivery = report["delivery_statuses"] or []
    media = report["evidence_media"] or []
    lines = [
        f"# {report['title']}",
        "",
        f"- 事件号：{report['event_id']}",
        f"- 状态：{report['status']}",
        f"- 位置：{report['location']}",
        f"- 优先级：{report['priority']} / {report['severity']}",
        f"- 响应组：{report['notification_group']}",
        f"- SLA：{report['sla_state']} / {report['sla_remaining_s']}s",
        f"- 现场播报：{report['voice']}",
        f"- 处理建议：{report['operator_action']}",
        f"- 确认：{report['acknowledged_by'] or '-'} {report['acknowledge_note'] or ''}",
        f"- 关闭：{report['closed_by'] or '-'} {report['close_note'] or ''}",
        f"- 关闭审批：{(report['close_approval'] or {}).get('supervisor_id') or '-'}",
        "",
        "## 通知送达",
    ]
    if delivery:
        lines.extend(
            f"- {item.get('channel')}: {item.get('status')} {item.get('reason') or ''}"
            for item in delivery
        )
    else:
        lines.append("- 无通知记录")
    lines.extend(["", "## 证据"])
    if media:
        lines.extend(f"- {item.get('label')}: {item.get('path')}" for item in media)
    else:
        lines.append("- 无证据媒体")
    lines.extend(["", "## 处置时间线"])
    for item in report["timeline"] or []:
        lines.append(
            f"- {item.get('type')}: {item.get('actor')} {item.get('note') or ''} @ {item.get('at')}"
        )
    lines.extend(["", "## 通知尝试"])
    for item in report["notification_attempts"] or []:
        lines.append(
            f"- {item.get('kind')}: {item.get('by')} "
            f"{','.join(str(channel) for channel in item.get('sent_channels') or [])}"
        )
    if not report["notification_attempts"]:
        lines.append("- none")
    lines.extend(["", "## 操作审计"])
    for item in report["action_audit"] or []:
        reason = f" / {item.get('reason')}" if item.get("reason") else ""
        lines.append(
            f"- {item.get('action')}: {item.get('outcome')} by "
            f"{item.get('operator_id')}{reason} @ {item.get('at')}"
        )
    if not report["action_audit"]:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"


def _field_event_requires_close_approval(event: dict[str, Any]) -> bool:
    if event.get("status") == "closed":
        return False
    return event.get("priority") == "P0" or event.get("severity") == "error"
