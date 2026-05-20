"""Delivery-resource governance notification delivery helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from askme.pipeline.field.alert_dispatcher import AlertDispatcher

DispatcherFactory = Callable[..., Any]


def delivery_owner_notification_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract delivery-owner notification settings from the app config."""
    field_cfg = config.get("field_operations") if isinstance(config.get("field_operations"), dict) else {}
    governance_cfg = (
        field_cfg.get("delivery_resource_governance")
        if isinstance(field_cfg.get("delivery_resource_governance"), dict)
        else {}
    )
    notify_cfg = (
        governance_cfg.get("delivery_owner_notifications")
        if isinstance(governance_cfg.get("delivery_owner_notifications"), dict)
        else {}
    )
    return dict(notify_cfg)


def delivery_owner_notification_channels(notify_cfg: dict[str, Any]) -> list[str]:
    """Resolve notification channels for warning-level governance escalations."""
    routes = notify_cfg.get("severity_routes")
    if isinstance(routes, dict) and isinstance(routes.get("warning"), list):
        return [
            str(item).strip()
            for item in routes.get("warning", [])
            if str(item).strip()
        ]
    channels: list[str] = []
    if notify_cfg.get("webhook_url"):
        channels.append("webhook")
    if notify_cfg.get("dingtalk_webhook"):
        channels.append("dingtalk")
    if notify_cfg.get("wecom_webhook"):
        channels.append("wecom")
    if notify_cfg.get("feishu_webhook"):
        channels.append("feishu")
    channels.append("log")
    return channels


def deliver_resource_governance_notification(
    escalation: dict[str, Any],
    *,
    config: dict[str, Any],
    dispatcher_factory: DispatcherFactory = AlertDispatcher,
) -> dict[str, Any]:
    """Deliver an overdue governance escalation to configured delivery owners."""
    notify_cfg = delivery_owner_notification_config(config)
    notification = (
        escalation.get("notification")
        if isinstance(escalation.get("notification"), dict)
        else {}
    )
    message = str(notification.get("message") or "")
    if not bool(notify_cfg.get("enabled")):
        return {
            "status": "queued",
            "delivery_mode": "local_queue",
            "reason": "delivery_owner_notification_not_enabled",
            "sent_channels": [],
            "delivery_report": [
                {
                    "channel": "delivery_owner_queue",
                    "status": "queued",
                    "reason": "local_delivery_owner_queue",
                }
            ],
        }

    channels = delivery_owner_notification_channels(notify_cfg)
    dispatcher = dispatcher_factory(
        config={
            "webhook_url": notify_cfg.get("webhook_url") or "",
            "wecom_webhook": notify_cfg.get("wecom_webhook") or "",
            "dingtalk_webhook": notify_cfg.get("dingtalk_webhook") or "",
            "dingtalk_secret": notify_cfg.get("dingtalk_secret") or "",
            "feishu_webhook": notify_cfg.get("feishu_webhook") or "",
            "severity_routes": {
                "warning": channels,
                "info": ["log"],
                "error": channels,
            },
            "incident_archive_path": notify_cfg.get("incident_archive_path") or "",
        },
        robot_id=str(notify_cfg.get("robot_id") or "askme-delivery"),
        robot_name=str(notify_cfg.get("robot_name") or "AskMe Delivery"),
    )
    sent_channels = dispatcher.dispatch(
        message,
        severity="warning",
        topic="delivery_resource_governance.overdue",
        payload={
            "escalation": escalation,
            "dingtalk_message": message,
        },
    )
    return {
        "status": "sent" if sent_channels else "not_sent",
        "delivery_mode": "configured_channels",
        "sent_channels": sent_channels,
        "delivery_report": dispatcher.last_delivery_report,
    }


__all__ = [
    "deliver_resource_governance_notification",
    "delivery_owner_notification_channels",
    "delivery_owner_notification_config",
]
