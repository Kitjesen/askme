"""HTTP API package for askme product endpoints.

Use this package-level factory for new product integrations. The historical
implementation still lives in :mod:`askme.health_server`, but external callers
should not need to know that file boundary.
"""

from __future__ import annotations

from askme.api.app import create_api_app, create_health_app, create_product_app

__all__ = ["create_api_app", "create_health_app", "create_product_app"]
