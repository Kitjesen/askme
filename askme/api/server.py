"""Product-facing FastAPI app factory.

The legacy factory still lives in :mod:`askme.health_server` for compatibility
with existing runtime modules and tests. New HTTP work should import from this
module so API structure can continue moving toward ``askme.api.routes`` without
forcing all callers to know about the legacy health-server file.
"""

from __future__ import annotations

from askme.health_server import create_health_app as create_api_app

__all__ = ["create_api_app"]

