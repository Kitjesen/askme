"""Navigation and spatial-memory provider adapters."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from uuid import uuid4

from askme.ports import NavigationPort, TemporalMemoryPort

_DISPATCH_PATH = "/api/v1/navigation/dispatch"
_STATUS_PATH = "/api/v1/navigation/status"
_TEMPORAL_MEMORY_PATH = "/api/v1/memory/temporal"


class NavGatewayClient:
    """HTTP client for the nav-gateway capability router."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._base_url = (
            cfg.get("base_url")
            or os.environ.get("NAV_GATEWAY_URL")
            or os.environ.get("DOG_NAV_SERVICE_URL")
            or ""
        ).rstrip("/")

    def is_configured(self) -> bool:
        return bool(self._base_url)

    def dispatch_navigation(
        self,
        capability: str,
        params: dict[str, Any] | None = None,
        *,
        mission_type: str = "voice_command",
        mission_id: str = "",
    ) -> dict[str, Any]:
        if not self._base_url:
            return {"error": "NAV_GATEWAY_URL not configured"}
        body = {
            "mission_id": mission_id or uuid4().hex[:16],
            "mission_type": mission_type,
            "requested_capability": capability,
            "parameters": params or {},
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}{_DISPATCH_PATH}",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read(4096).decode("utf-8", errors="replace")
                if not raw:
                    return {"status": "ok", "http_status": getattr(resp, "status", 200)}
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {
                        "status": "ok",
                        "http_status": getattr(resp, "status", 200),
                        "raw": raw,
                    }
        except urllib.error.HTTPError as exc:
            body_text = exc.read(256).decode("utf-8", errors="replace")
            return {"error": body_text or exc.reason, "http_status": exc.code}
        except urllib.error.URLError as exc:
            return {"error": f"Navigation service unreachable: {exc.reason}"}
        except (TimeoutError, OSError):
            return {"error": "Navigation request timed out"}
        except Exception as exc:
            return {"error": f"Navigation request failed: {exc}"}

    def status(self) -> dict[str, Any]:
        if not self._base_url:
            return {"error": "NAV_GATEWAY_URL not configured"}
        try:
            with urllib.request.urlopen(f"{self._base_url}{_STATUS_PATH}", timeout=3) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            return {"error": str(exc)}

    def query_temporal_observations(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._base_url:
            return {"error": "NAV_GATEWAY_URL not configured"}
        query = urllib.parse.urlencode(params)
        url = f"{self._base_url}{_TEMPORAL_MEMORY_PATH}?{query}"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except urllib.error.URLError as exc:
            return {"error": f"LingTu temporal memory unreachable: {exc.reason}"}
        except Exception as exc:
            return {"error": f"LingTu temporal memory query failed: {exc}"}


def build_navigation(config: dict[str, Any] | None = None) -> NavigationPort:
    """Build the configured navigation implementation."""
    return NavGatewayClient(config)


def build_temporal_memory(config: dict[str, Any] | None = None) -> TemporalMemoryPort:
    """Build the configured temporal memory implementation."""
    return NavGatewayClient(config)


__all__ = ["NavGatewayClient", "build_navigation", "build_temporal_memory"]
