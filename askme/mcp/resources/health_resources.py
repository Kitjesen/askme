"""MCP health-check resource for askme."""

from __future__ import annotations

import json
import sys
import time

from askme.config import get_config, get_section
from askme.mcp.server import mcp

_START_TIME = time.monotonic()


@mcp.resource("askme://health")
def health_check() -> str:
    """Server health: version, active subsystems, uptime."""
    from askme import __version__

    cfg = get_config()
    robot_cfg = get_section("robot")
    voice_cfg = get_section("voice")

    return json.dumps({
        "status": "ok",
        "version": __version__,
        "python": sys.version.split()[0],
        "subsystems": {
            "brain": True,
            "robot": robot_cfg.get("enabled", False),
            "voice": bool(voice_cfg),
            "memory": cfg.get("memory", {}).get("enabled", False),
        },
        "uptime_seconds": round(time.monotonic() - _START_TIME, 1),
    })
