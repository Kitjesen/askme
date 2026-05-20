"""Legacy HTTP helper for standalone robot tool fallbacks.

Runtime-built tools should receive a robot-control port from the module graph.
This helper remains for old direct tool usage and for navigation endpoints that
do not yet have a dedicated port.
"""

from __future__ import annotations

import json
from typing import Any


def call_runtime_api(
    service: str,
    method: str,
    path: str,
    body: dict | None = None,
) -> dict[str, Any]:
    """Call a runtime service via HTTP and return parsed JSON or an error dict."""
    import os
    import urllib.error
    import urllib.request

    port_map = {
        "control": 5080,
        "nav": 8088,
        "safety": 5070,
    }
    port = port_map.get(service)
    if not port:
        return {"error": f"unknown service: {service}"}

    env_keys = {
        "control": ("DOG_CONTROL_SERVICE_URL",),
        "nav": ("NAV_GATEWAY_URL", "DOG_NAV_SERVICE_URL"),
        "safety": ("DOG_SAFETY_SERVICE_URL",),
    }
    base_url = next(
        (os.environ[key].rstrip("/") for key in env_keys.get(service, ()) if os.environ.get(key)),
        f"http://localhost:{port}",
    )

    url = f"{base_url}{path}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {"error": f"服务不可达 ({service}:{port}): {exc.reason}"}
    except Exception as exc:
        return {"error": f"请求失败: {exc}"}


__all__ = ["call_runtime_api"]
