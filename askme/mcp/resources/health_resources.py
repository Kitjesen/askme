"""MCP health-check resource for askme."""

from __future__ import annotations

import json
import sys
import time

from askme.mcp.registration import mcp
from askme.mcp.resource_surface import get_resource_surface

_START_TIME = time.monotonic()


@mcp.resource("askme://health")
def health_check() -> str:
    """Server health: version, active subsystems, uptime."""
    from askme import __version__

    return json.dumps(get_resource_surface().health_payload(
        version=__version__,
        python_version=sys.version.split()[0],
        uptime_seconds=round(time.monotonic() - _START_TIME, 1),
    ))
