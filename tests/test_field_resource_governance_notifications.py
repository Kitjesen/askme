from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from askme.api.services.field_resource_governance_notifications import (
    deliver_resource_governance_notification,
    delivery_owner_notification_channels,
    delivery_owner_notification_config,
)


def test_delivery_owner_notification_config_extracts_nested_settings() -> None:
    config = {
        "field_operations": {
            "delivery_resource_governance": {
                "delivery_owner_notifications": {
                    "enabled": True,
                    "dingtalk_webhook": "https://example.invalid/dingtalk",
                }
            }
        }
    }

    assert delivery_owner_notification_config(config) == {
        "enabled": True,
        "dingtalk_webhook": "https://example.invalid/dingtalk",
    }
    assert delivery_owner_notification_config({}) == {}
    assert delivery_owner_notification_config({"field_operations": "bad"}) == {}


def test_delivery_owner_notification_channels_prefers_explicit_warning_route() -> None:
    assert delivery_owner_notification_channels(
        {"severity_routes": {"warning": [" dingtalk ", "", " log "]}}
    ) == ["dingtalk", "log"]


def test_delivery_owner_notification_channels_derives_channels_from_webhooks() -> None:
    assert delivery_owner_notification_channels(
        {
            "webhook_url": "https://example.invalid/webhook",
            "dingtalk_webhook": "https://example.invalid/dingtalk",
            "wecom_webhook": "https://example.invalid/wecom",
            "feishu_webhook": "https://example.invalid/feishu",
        }
    ) == ["webhook", "dingtalk", "wecom", "feishu", "log"]
    assert delivery_owner_notification_channels({}) == ["log"]


def test_deliver_resource_governance_notification_queues_when_disabled() -> None:
    result = deliver_resource_governance_notification(
        {"notification": {"message": "overdue"}},
        config={},
        dispatcher_factory=_UnexpectedDispatcher,
    )

    assert result == {
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


def test_deliver_resource_governance_notification_uses_configured_dispatcher() -> None:
    FakeDispatcher.instances.clear()
    escalation = {"notification": {"message": "治理请求已超期"}}
    result = deliver_resource_governance_notification(
        escalation,
        config={
            "field_operations": {
                "delivery_resource_governance": {
                    "delivery_owner_notifications": {
                        "enabled": True,
                        "robot_id": "delivery-01",
                        "robot_name": "Delivery Owner Bot",
                        "severity_routes": {"warning": ["dingtalk", "log"]},
                        "dingtalk_webhook": "https://example.invalid/dingtalk",
                        "incident_archive_path": "artifacts/incidents.jsonl",
                    }
                }
            }
        },
        dispatcher_factory=FakeDispatcher,
    )

    assert result == {
        "status": "sent",
        "delivery_mode": "configured_channels",
        "sent_channels": ["dingtalk", "log"],
        "delivery_report": [
            {"channel": "dingtalk", "status": "sent"},
            {"channel": "log", "status": "sent"},
        ],
    }
    dispatcher = FakeDispatcher.instances[0]
    assert dispatcher.robot_id == "delivery-01"
    assert dispatcher.robot_name == "Delivery Owner Bot"
    assert dispatcher.config["severity_routes"]["warning"] == ["dingtalk", "log"]
    assert dispatcher.config["incident_archive_path"] == "artifacts/incidents.jsonl"
    assert dispatcher.dispatch_calls == [
        {
            "message": "治理请求已超期",
            "severity": "warning",
            "topic": "delivery_resource_governance.overdue",
            "payload": {
                "escalation": escalation,
                "dingtalk_message": "治理请求已超期",
            },
        }
    ]


def test_resource_governance_notification_service_is_leaf_and_route_imports_service() -> None:
    service_path = Path("askme/api/services/field_resource_governance_notifications.py")
    route_path = Path("askme/api/routes/field.py")
    service_tree = ast.parse(service_path.read_text(encoding="utf-8"))
    route_tree = ast.parse(route_path.read_text(encoding="utf-8"))

    service_imports = {
        node.module
        for node in ast.walk(service_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(service_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    route_imports = {
        (node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(route_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    route_functions = {
        node.name
        for node in ast.walk(route_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "fastapi" not in service_imports
    assert "askme.health_server" not in service_imports
    assert "askme.pipeline.field.alert_dispatcher" not in {
        module for module, _aliases in route_imports
    }
    assert (
        "askme.api.services.field_resource_governance_notifications",
        ("deliver_resource_governance_notification",),
    ) in route_imports
    assert "_resource_governance_delivery_config" not in route_functions
    assert "_resource_governance_delivery_channels" not in route_functions


class FakeDispatcher:
    instances: list["FakeDispatcher"] = []

    def __init__(
        self,
        *,
        config: dict[str, Any],
        robot_id: str,
        robot_name: str,
    ) -> None:
        self.config = config
        self.robot_id = robot_id
        self.robot_name = robot_name
        self.dispatch_calls: list[dict[str, Any]] = []
        self.last_delivery_report = [
            {"channel": "dingtalk", "status": "sent"},
            {"channel": "log", "status": "sent"},
        ]
        self.instances.append(self)

    def dispatch(
        self,
        message: str,
        *,
        severity: str,
        topic: str,
        payload: dict[str, Any],
    ) -> list[str]:
        self.dispatch_calls.append(
            {
                "message": message,
                "severity": severity,
                "topic": topic,
                "payload": payload,
            }
        )
        return ["dingtalk", "log"]


class _UnexpectedDispatcher:
    def __init__(self, **_kwargs: Any) -> None:
        raise AssertionError("dispatcher should not be created when notifications are disabled")
