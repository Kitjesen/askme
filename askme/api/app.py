"""Canonical FastAPI application factory for askme HTTP surfaces.

The runtime still builds the app through :mod:`askme.health_server` because
that file owns legacy dependency wiring. New imports should use this module so
callers see a product API boundary instead of a health-server implementation
detail.
"""

from __future__ import annotations

from typing import Any


def create_api_app(*args: Any, **kwargs: Any) -> Any:
    from askme.health_server import create_health_app as _create_health_app

    return _create_health_app(*args, **kwargs)


create_health_app = create_api_app
create_product_app = create_api_app

__all__ = ["create_api_app", "create_health_app", "create_product_app"]
