"""Shared utility functions extracted from askme.cli for CLI subcommands."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)


def _cli_root_override(name: str, default: Any) -> Any:
    """Return explicit monkeypatch/compat overrides set on ``askme.cli``."""
    cli_root = sys.modules.get("askme.cli")
    if cli_root is None:
        return default
    return vars(cli_root).get(name, default)


# ---------------------------------------------------------------------------
# HTTP 工具函数
# ---------------------------------------------------------------------------


def _normalise_server_url(server: str) -> str:
    """Remove trailing slashes from a server URL for consistent concatenation."""
    return server.rstrip("/")


def _loopback_proxy_kwargs(url: str) -> dict[str, Any]:
    """Keep in-process control traffic off developer and CI proxies."""
    hostname = (urlsplit(url).hostname or "").rstrip(".").casefold()
    if not hostname:
        return {}

    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ip_address(hostname).is_loopback
        except ValueError:
            return {}

    if not is_loopback:
        return {}
    return {"proxies": {"http": None, "https": None, "all": None}}


@contextmanager
def _loopback_proxy_environment() -> Iterator[None]:
    """Temporarily add loopback hosts to urllib's proxy bypass list."""
    names = ("NO_PROXY", "no_proxy")
    previous = {name: os.environ.get(name) for name in names}

    try:
        for name in names:
            raw = str(previous[name] or "")
            values = [item.strip() for item in raw.split(",") if item.strip()]
            known = {item.casefold() for item in values}
            for entry in ("localhost", "127.0.0.1", "::1"):
                if entry.casefold() not in known:
                    values.append(entry)
                    known.add(entry.casefold())
            os.environ[name] = ",".join(values)
        yield
    finally:
        for name, prior in previous.items():
            if prior is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prior


def _server_auth_headers() -> dict[str, str] | None:
    """Build Bearer auth headers from environment or config for server requests."""
    token = (
        os.environ.get("ASKME_CONTROL_API_KEY")
        or os.environ.get("ASKME_HEALTH_API_KEY")
        or _configured_control_api_key()
    )
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def _configured_control_api_key() -> str:
    """Read the control API key from the health_server config section."""
    try:
        from askme.config import get_config

        raw = get_config().get("health_server", {}).get("control_api_key", "")
    except Exception:
        return ""
    return str(raw).strip()


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST a JSON payload to *url* and return the decoded response."""
    kwargs: dict[str, Any] = {"json": payload, "timeout": 5}
    kwargs.update(_loopback_proxy_kwargs(url))
    headers = _server_auth_headers()
    if headers:
        kwargs["headers"] = headers
    http_requests = _cli_root_override("requests", requests)
    response = http_requests.post(url, **kwargs)
    response.raise_for_status()
    return response.json()


def _post_json_with_retries(url: str, payload: dict[str, Any], *, attempts: int) -> dict[str, Any]:
    """POST a JSON payload with retry logic for transient failures."""
    last_error = ""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            post_json = _cli_root_override("_post_json", _post_json)
            response = post_json(url, payload)
            return {
                "status": "sent",
                "attempts": attempt,
                "response": response,
            }
        except requests.RequestException as exc:
            last_error = str(exc)
    return {
        "status": "failed",
        "attempts": max(1, attempts),
        "error": last_error or "webhook_delivery_failed",
    }


def _get_json(url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """GET a JSON resource from *url* and return the decoded response."""
    kwargs: dict[str, Any] = {"timeout": 5}
    kwargs.update(_loopback_proxy_kwargs(url))
    request_headers = _server_auth_headers() or {}
    if headers:
        request_headers.update(headers)
    if request_headers:
        kwargs["headers"] = request_headers
    http_requests = _cli_root_override("requests", requests)
    response = http_requests.get(url, **kwargs)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# 输出工具函数
# ---------------------------------------------------------------------------


def _emit_payload(payload: dict[str, Any], *, json_output: bool) -> None:
    """Print a command result payload to stdout (human-readable or JSON)."""
    if json_output:
        print(_json(payload))  # noqa: T201
        return

    if "profile" in payload and "components" in payload:
        profile = payload["profile"]
        print(f"profile: {profile.get('name')} ({profile.get('primary_loop')})")  # noqa: T201
        for name, component in payload.get("components", {}).items():
            health = component.get("health", {})
            print(f"{name}: {health.get('status', 'unknown')}")  # noqa: T201
        return

    if "skills" in payload and isinstance(payload["skills"], list):
        for skill in payload["skills"]:
            state = "enabled" if skill.get("enabled", False) else "disabled"
            execution = skill.get("execution", "?")
            name = skill.get("name", "?")
            description = skill.get("description", "")
            print(f"{name:20} {state:8} {execution:14} {description}")  # noqa: T201
        return

    print(_json(payload))  # noqa: T201


def _emit_agent_payload(payload: dict[str, Any], *, json_output: bool) -> None:
    """Print an agent-turn result (reply text or full JSON)."""
    if json_output:
        print(_json(payload))  # noqa: T201
        return
    print(payload.get("reply", ""))  # noqa: T201


def _json(payload: dict[str, Any]) -> str:
    """Pretty-print a dict as JSON, falling back to ASCII-safe output when stdout cannot encode Unicode."""
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if _stdout_should_emit_human_text(text):
        return text
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _stdout_supports_unicode() -> bool:
    """Check whether stdout encoding supports Unicode characters."""
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("_", "-")
    return encoding in {"utf-8", "utf8"} or "65001" in encoding


def _stdout_can_encode(text: str) -> bool:
    """Check whether *text* can be encoded with the current stdout encoding."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _stdout_should_emit_human_text(text: str) -> bool:
    """Decide whether it is safe to emit human-readable (non-ASCII-escaped) text to stdout."""
    if not _stdout_can_encode(text):
        return False
    if _stdout_supports_unicode():
        return True
    isatty = getattr(sys.stdout, "isatty", None)
    if not callable(isatty):
        return False
    return bool(isatty())


# ---------------------------------------------------------------------------
# 解析/配置工具函数
# ---------------------------------------------------------------------------


def _parse_csv_ints(value: str | None) -> list[int]:
    """Parse a comma-separated string of integers into a list of unique ints."""
    result: list[int] = []
    for part in str(value or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            parsed = int(token)
        except ValueError:
            continue
        if parsed not in result:
            result.append(parsed)
    return result


def _parse_device_secret_args(values: list[str] | tuple[str, ...] | None) -> dict[str, str]:
    """Parse --device-secret CLI args (DEVICE_ID=SECRET) into a dict."""
    secrets: dict[str, str] = {}
    for item in values or []:
        raw = str(item or "")
        if "=" not in raw:
            raise SystemExit("--device-secret must use DEVICE_ID=SECRET")
        key, secret = raw.split("=", 1)
        key = key.strip()
        if not key or not secret:
            raise SystemExit("--device-secret must include a non-empty device id and secret")
        secrets[key] = secret
    return secrets


def _resolve_runtime_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    """Resolve voice/robot mode flags from CLI args and profile."""
    voice_mode = True
    robot_mode = bool(getattr(args, "robot", False))
    profile = getattr(args, "profile", "") or ""

    if profile == "text":
        voice_mode = False
    elif profile in {"voice", "edge_robot"}:
        voice_mode = True

    if getattr(args, "text", False):
        voice_mode = False
    if getattr(args, "voice", False):
        voice_mode = True

    if profile == "edge_robot":
        robot_mode = True

    return voice_mode, robot_mode


def _resolve_field_device_secrets(
    values: list[str] | tuple[str, ...] | None,
    *,
    site_profile: str = "",
) -> dict[str, str]:
    """Merge device secrets from site profile with explicit CLI --device-secret args."""
    secrets = _device_secrets_from_site_profile(site_profile)
    secrets.update(_parse_device_secret_args(values))
    return secrets


def _device_secrets_from_site_profile(site_profile: str) -> dict[str, str]:
    """Read device HMAC secrets from a site profile's device registry + env vars."""
    profile_path = str(site_profile or "").strip()
    if not profile_path:
        return {}
    from askme.pipeline.field.customer_project_template_support import load_field_site_profile

    profile = load_field_site_profile(Path(profile_path))
    devices = profile.get("devices") if isinstance(profile.get("devices"), dict) else {}
    secrets: dict[str, str] = {}
    for device_id, device in devices.items():
        if not isinstance(device, dict):
            continue
        secret_env = str(device.get("secret_env") or "").strip()
        if not secret_env:
            continue
        secret = str(os.getenv(secret_env) or "").strip()
        if secret:
            secrets[str(device_id)] = secret
    return secrets


def _resolve_field_device_signing_secret(*, secret: str, secret_env: str) -> str:
    """Resolve a device signing HMAC secret from a literal value or env var."""
    if secret:
        return str(secret).strip()
    if not secret_env:
        return ""
    return str(os.getenv(str(secret_env).strip()) or "").strip()


def _field_signed_payload_text(
    signed_events: list[dict[str, Any]],
    *,
    source_path: Path,
    output_path: Path,
) -> str:
    """Format signed events as JSON text (JSONL for .jsonl/.ndjson, otherwise pretty-printed)."""
    if output_path.suffix.lower() in {".jsonl", ".ndjson"} or source_path.suffix.lower() in {
        ".jsonl",
        ".ndjson",
    }:
        return (
            "\n".join(
                json.dumps(event, ensure_ascii=False, sort_keys=True) for event in signed_events
            )
            + "\n"
        )
    payload: dict[str, Any] | list[dict[str, Any]]
    payload = signed_events[0] if len(signed_events) == 1 else signed_events
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _single_device_id(events: list[dict[str, Any]]) -> str:
    """Return the single common device_id across events, or '' if there are zero or multiple."""
    ids = {str(event.get("device_id") or "").strip() for event in events if event.get("device_id")}
    return next(iter(ids)) if len(ids) == 1 else ""


def _resolve_field_action_audit_hmac_secret(hmac_secret: str = "") -> str:
    """Resolve the field action audit HMAC secret from arg or env var."""
    return str(hmac_secret or os.getenv("ASKME_FIELD_ACTION_AUDIT_HMAC_SECRET") or "").strip()


# ---------------------------------------------------------------------------
# 加载器函数
# ---------------------------------------------------------------------------


def _load_skill_manager():
    """Load and return the SkillManager singleton (lazy import)."""
    from askme.skills.core.skill_manager import SkillManager

    manager = SkillManager()
    manager.load()
    return manager


def _load_local_mission_service():
    """Build a local MissionService from config (no server round-trip)."""
    from askme.config import get_config
    from askme.runtime.task.mission import MissionService

    return MissionService(get_config())


def _load_mission_source(source: str) -> dict[str, Any]:
    """Load a mission definition from a JSON/YAML file or plain text string."""
    path = Path(source)
    if not path.exists():
        return {"text": source}

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(raw)
    else:
        return {"text": raw}

    if not isinstance(payload, dict):
        raise SystemExit(f"Mission source must be a JSON/YAML object: {path}")
    return payload


def _load_local_capabilities(*, voice_mode: bool, robot_mode: bool) -> dict[str, Any]:
    """Load capabilities from the local runtime synchronously."""
    return asyncio.run(_load_local_capabilities_async(voice_mode=voice_mode, robot_mode=robot_mode))


async def _load_local_capabilities_async(
    *,
    voice_mode: bool,
    robot_mode: bool,
) -> dict[str, Any]:
    """Async helper that builds a blueprint and collects module capabilities."""
    from askme.config import get_config
    from askme.main import _select_blueprint
    from askme.runtime.core.profiles import legacy_profile_for

    cfg = get_config()
    blueprint = _select_blueprint(voice_mode=voice_mode, robot_mode=robot_mode)
    app = await blueprint.build(cfg)
    profile = legacy_profile_for(voice_mode=voice_mode, robot_mode=robot_mode)

    skill_mod = app.modules.get("skill")
    sm = getattr(skill_mod, "skill_manager", None) if skill_mod else None
    contracts = sm.get_contracts() if sm else []
    openapi_doc = (
        sm.openapi_document() if sm else {"info": {"title": "", "version": ""}, "paths": {}}
    )

    from askme import __version__ as ASKME_VERSION

    app_name = cfg.get("app", {}).get("name", "askme")
    app_version = cfg.get("app", {}).get("version") or ASKME_VERSION

    components: dict[str, dict[str, Any]] = {}
    for name, mod in app.modules.items():
        components[name] = {
            "health": mod.health(),
            "capabilities": mod.capabilities(),
        }

    return {
        "app": {
            "name": app_name,
            "version": app_version,
            "voice_mode": voice_mode,
            "robot_mode": robot_mode,
        },
        "profile": profile.snapshot(),
        "components": components,
        "skills": {
            "count": len(sm.get_all()) if sm else 0,
            "enabled_count": len(sm.get_enabled()) if sm else 0,
            "contract_count": len(contracts),
            "code_contract_count": sum(1 for c in contracts if c.source == "code"),
            "legacy_contract_count": sum(1 for c in contracts if c.source != "code"),
            "catalog": sm.get_contract_catalog() if sm else [],
        },
        "openapi": {
            "title": openapi_doc["info"]["title"],
            "version": openapi_doc["info"]["version"],
            "path_count": len(openapi_doc["paths"]),
        },
    }


def _runtime_blueprints_payload(
    *,
    name: str = "",
    customer_visible: bool | None = None,
    delivery_package: bool = False,
) -> dict[str, Any]:
    """Build a structured payload describing one or all blueprints with inspection + readiness."""
    from askme.blueprints import (
        blueprint_delivery_package,
        blueprint_readiness,
        catalog_payload,
        get_blueprint_spec,
        inspect_blueprint,
        list_blueprints,
    )
    from askme.config import get_config

    config = get_config()

    if name:
        spec = get_blueprint_spec(name)
        if delivery_package:
            return blueprint_delivery_package(spec.name, config=config)
        items = [
            {
                **spec.to_dict(),
                "inspection": inspect_blueprint(spec.name),
                "readiness": blueprint_readiness(spec.name, config=config),
            }
        ]
    else:
        specs = list_blueprints(customer_visible=customer_visible)
        if customer_visible is None:
            payload = catalog_payload(config=config)
            if not delivery_package:
                for item in payload.get("items", []):
                    if isinstance(item, dict):
                        item.pop("delivery_package", None)
            return payload
        items = [
            {
                **spec.to_dict(),
                "inspection": inspect_blueprint(spec.name),
                "readiness": blueprint_readiness(spec.name, config=config),
                **(
                    {"delivery_package": blueprint_delivery_package(spec.name, config=config)}
                    if delivery_package
                    else {}
                ),
            }
            for spec in specs
        ]
    return {
        "summary": {
            "blueprint_count": len(items),
            "customer_visible_count": sum(1 for item in items if item["customer_visible"]),
            "valid_count": sum(1 for item in items if item["inspection"]["valid"]),
            "ready_for_validation_count": sum(
                1 for item in items if item["readiness"]["status"] == "ready_for_validation"
            ),
            "configuration_incomplete_count": sum(
                1 for item in items if item["readiness"]["status"] == "configuration_incomplete"
            ),
            "pilot_blueprints": [
                item["name"] for item in items if item["product_stage"] in {"pilot", "lab"}
            ],
        },
        "items": items,
    }


def _emit_runtime_blueprints_summary(payload: dict[str, Any]) -> None:
    """Print a human-readable summary of a blueprints payload."""
    if payload.get("package_id") and payload.get("blueprint"):
        print(  # noqa: T201
            "delivery-package: {package_id} status={status} claim={claim}".format(
                package_id=payload.get("package_id"),
                status=payload.get("status"),
                claim=payload.get("customer_claim"),
            )
        )
        print(  # noqa: T201
            "blueprint={blueprint} run={run}".format(
                blueprint=payload.get("blueprint"),
                run=payload.get("startup_command"),
            )
        )
        stop_conditions = payload.get("stop_conditions")
        if isinstance(stop_conditions, list) and stop_conditions:
            print(f"stop: {stop_conditions[0]}")  # noqa: T201
        return

    summary = payload.get("summary", {})
    print(  # noqa: T201
        "blueprints={blueprint_count} customer_visible={customer_visible_count} valid={valid_count}".format(
            **summary
        )
    )
    for item in payload.get("items", []):
        inspection = item.get("inspection", {})
        print(  # noqa: T201
            "{name}: {title} | stage={stage} | modules={modules} | valid={valid} | readiness={readiness} | run={run}".format(
                name=item.get("name"),
                title=item.get("title"),
                stage=item.get("product_stage"),
                modules=inspection.get("module_count"),
                valid=inspection.get("valid"),
                readiness=item.get("readiness", {}).get("status", "unknown"),
                run=item.get("startup_command"),
            )
        )
        package = (
            item.get("delivery_package") if isinstance(item.get("delivery_package"), dict) else {}
        )
        if package:
            print(  # noqa: T201
                "  delivery-package: {package_id} status={status} claim={claim}".format(
                    package_id=package.get("package_id"),
                    status=package.get("status"),
                    claim=package.get("customer_claim"),
                )
            )
            stop_conditions = package.get("stop_conditions")
            if isinstance(stop_conditions, list) and stop_conditions:
                print(f"  stop: {stop_conditions[0]}")  # noqa: T201


# ---------------------------------------------------------------------------
# Field 内部 helper
# ---------------------------------------------------------------------------


def _write_field_smoke_events(path: Path) -> None:
    """Write a set of synthetic field events (JSONL) for smoke-test ingestion."""

    events = [
        {
            "device_id": "camera-main-road-1",
            "frame": {
                "timestamp": time.time(),
                "boxes": [{"cls": 2, "conf": 0.94, "xyxy": [12, 20, 120, 160]}],
            },
            "zone": {
                "id": "main-road-1",
                "name": "B区主通道",
                "type": "main_channel",
                "parking_allowed": False,
            },
            "duration_s": 180,
            "image_path": "artifacts/evidence/smoke-car.jpg",
        },
        {
            "device_id": "smoke-power-room-1",
            "timestamp": time.time(),
            "telemetry": {"temperature": 68, "smoke": 0.82},
            "location": "配电间门口",
            "image_path": "artifacts/evidence/smoke.jpg",
        },
        {
            "device_id": "robot-thunder-1",
            "timestamp": time.time(),
            "topic": "/diagnostics",
            "status": [
                {
                    "name": "left_hip_motor",
                    "level": 2,
                    "message": "motor overcurrent fault",
                    "values": [
                        {"key": "joint_id", "value": "hip-left"},
                        {"key": "fault_code", "value": "MOTOR_OVERCURRENT"},
                    ],
                }
            ],
            "location": "A区东侧",
        },
        {
            "timestamp": time.time(),
            "device_id": "bin-17",
            "telemetry": {"fill_percent": 91},
            "detections": [{"label": "trash_bin", "confidence": 0.88}],
            "bin_id": "bin-17",
            "location": "游客中心门口",
            "image_path": "artifacts/evidence/bin.jpg",
        },
        {
            "device_id": "camera-plaza-1",
            "timestamp": time.time(),
            "predictions": [{"class": "person", "confidence": 0.82} for _ in range(6)],
            "duration_min": 35,
            "location": "北广场",
            "image_path": "artifacts/evidence/crowd.jpg",
        },
        {
            "eventType": "ANPR",
            "dateTime": time.time(),
            "cameraIndexCode": "cam-main-road-01",
            "ANPR": {"plateNo": "沪A12345"},
            "zone_id": "main-road-1",
            "zone_name": "B区主通道",
            "location": "B区主通道",
            "duration_s": 180,
            "pictureUrl": "artifacts/evidence/anpr-car.jpg",
        },
        {
            "device_id": "smoke-power-room-1",
            "topic": "site/A/power-room/smoke-01",
            "payload": {
                "timestamp": time.time(),
                "temperatureC": 72,
                "smokeAlarm": True,
                "location": "配电间门口",
                "imageUrl": "artifacts/evidence/smoke-mqtt.jpg",
            },
        },
        {
            "device_id": "robot-thunder-1",
            "topic": "/thunder/status",
            "timestamp": time.time(),
            "robot": {"nav_state": "stuck", "recoverable": False},
            "location": "A区东侧",
        },
    ]
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in events),
        encoding="utf-8",
    )


def _start_field_smoke_server(
    *,
    archive_path: Path,
    voice_handler: Any | None = None,
    voice_enabled: bool = False,
    field_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a local uvicorn server for field smoke tests and return its metadata."""
    import socket
    import threading

    import uvicorn

    from askme.health_server import build_health_snapshot, create_health_app
    from askme.pipeline.field.field_operations import FieldOperationsService

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()

    service_config = {"archive_path": str(archive_path)}
    if field_config:
        service_config.update(field_config)
    service = FieldOperationsService(config=service_config)

    def health() -> dict[str, Any]:
        return build_health_snapshot(
            app_name="askme",
            app_version="smoke",
            model_name="field-smoke",
            metrics_snapshot={"uptime_seconds": 1.0, "conversation_count": 0},
            active_skills=[],
            voice_status={"enabled": voice_enabled, "pipeline_ok": True},
        )

    app = create_health_app(
        health,
        field_operations_handler=service,
        voice_handler=voice_handler,
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://{host}:{port}"
    health_url = f"{base_url}/health"
    deadline = time.time() + 5
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            requests.get(
                health_url,
                timeout=0.5,
                **_loopback_proxy_kwargs(health_url),
            ).raise_for_status()
            return {"server": server, "thread": thread, "base_url": base_url}
        except Exception as exc:
            last_error = exc
            time.sleep(0.05)
    server.should_exit = True
    thread.join(timeout=5)
    raise RuntimeError(f"field smoke server did not start: {last_error}")


def _start_local_webhook_collector() -> dict[str, Any]:
    """Start a local HTTP server that collects incoming webhook POSTs for smoke-test verification."""
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    requests_seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body: Any = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError:
                body = {"raw": raw.decode("utf-8", errors="replace")}
            requests_seen.append(
                {
                    "path": self.path,
                    "headers": {key: value for key, value in self.headers.items()},
                    "body": body,
                }
            )
            response = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()

    server = HTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return {
        "server": server,
        "thread": thread,
        "url": f"http://{host}:{port}/dingtalk",
        "requests": requests_seen,
    }


def _load_field_ingest_events(path: Path) -> list[dict[str, Any]]:
    """Load field events from a JSON or JSONL file, returning a list of event dicts."""
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        loaded = json.loads(raw)
        events = loaded if isinstance(loaded, list) else [loaded]
    result = [event for event in events if isinstance(event, dict)]
    if len(result) != len(events):
        raise SystemExit(f"Field ingest file must contain JSON objects: {path}")
    return result


def _field_action_audit_config(path: Path, *, hmac_secret: str = "") -> dict[str, Any]:
    """Build a field action audit configuration dict with optional HMAC secret."""
    resolved_secret = _resolve_field_action_audit_hmac_secret(hmac_secret)
    config: dict[str, Any] = {
        "enabled": True,
        "path": str(path),
        "swallow_errors": False,
    }
    if resolved_secret:
        config["hmac_secret"] = resolved_secret
    return config


def _acquire_field_audit_retry_lock(path: Path, *, lock_timeout_s: float) -> dict[str, Any]:
    """Acquire an exclusive filesystem lock for audit-webhook retry delivery, with stale-lock timeout."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    now = time.time()
    lock_payload = {
        "pid": os.getpid(),
        "queue": str(path),
        "acquired_at": round(now, 3),
        "expires_at": round(now + max(1.0, float(lock_timeout_s or 1.0)), 3),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_path), flags)
    except FileExistsError:
        existing = _read_field_audit_retry_lock(lock_path)
        expires_at = float(existing.get("expires_at") or 0.0) if isinstance(existing, dict) else 0.0
        if expires_at and expires_at < now:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            return _acquire_field_audit_retry_lock(path, lock_timeout_s=lock_timeout_s)
        return {
            "acquired": False,
            "path": str(lock_path),
            "reason": "delivery_already_running",
            "existing": existing,
        }
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(lock_payload, ensure_ascii=False) + "\n")
    return {"acquired": True, "path": str(lock_path), **lock_payload}


def _append_field_audit_retry_queue(queue: str, payload: dict[str, Any]) -> None:
    """Append a webhook delivery failure to the retry queue file for later retry."""
    path = Path(queue)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "queued_at": payload.get("generated_at"),
        "webhook_url": payload.get("webhook_url") or "",
        "payload": payload,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_field_audit_retry_lock(path: Path) -> dict[str, Any]:
    """Read an existing audit retry lock file (used internally by _acquire_field_audit_retry_lock)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "error": str(exc)}
    return (
        payload
        if isinstance(payload, dict)
        else {"path": str(path), "error": "invalid_lock_payload"}
    )
