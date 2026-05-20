"""Internal FastAPI route modules.

New callers should register HTTP endpoints through the audience surfaces:

- ``askme.api.platform`` for health and monitoring.
- ``askme.api.product`` for Dashboard and customer/operator workflows.
- ``askme.api.admin`` for governance and delivery administration.
- ``askme.api.internal`` for runtime, cognition, vision, and device bridges.

This package intentionally exports no route registrars. Keeping the route
modules behind the surface packages prevents product code from coupling to
internal implementation files.
"""

from __future__ import annotations

__all__: list[str] = []

