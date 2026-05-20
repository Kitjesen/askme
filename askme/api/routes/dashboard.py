"""Dashboard FastAPI routes."""

from __future__ import annotations

import mimetypes
from collections.abc import Callable, Container, Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response

from askme.api.schemas.monitor import DashboardPageRegistryResponse
from askme.api.services.dashboard_pages import dashboard_pages_payload

JsonError = Callable[..., JSONResponse]


def register_dashboard_routes(
    app: FastAPI,
    *,
    dashboard_html: str,
    dashboard_asset_dir: Path,
    dashboard_pages: Container[str],
    json_error: JsonError,
    route_inventory_provider: Callable[[], Mapping[str, Any]] | None = None,
) -> None:
    """Register dashboard shell and asset routes."""

    @app.get("/dashboard", tags=["Monitor"])
    async def dashboard() -> Response:
        """Serve the product dashboard shell."""
        return Response(content=dashboard_html, media_type="text/html")

    @app.get(
        "/api/dashboard/pages",
        tags=["Monitor"],
        response_model=DashboardPageRegistryResponse,
        response_model_exclude_none=True,
    )
    async def dashboard_page_manifest() -> JSONResponse:
        """Return the product page map for the Dashboard shell."""
        route_inventory = route_inventory_provider() if route_inventory_provider else None
        payload = dashboard_pages_payload(route_inventory=route_inventory)
        response = DashboardPageRegistryResponse.model_validate(payload)
        return JSONResponse(
            response.model_dump(mode="python", exclude_unset=True),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/dashboard/{asset_path:path}", tags=["Monitor"])
    async def dashboard_asset(asset_path: str) -> Response:
        """Serve dashboard pages and assets without mixing them into one HTML file."""
        clean_path = asset_path.strip("/")
        if not clean_path or clean_path in dashboard_pages:
            return Response(content=dashboard_html, media_type="text/html")
        resolved = (dashboard_asset_dir / clean_path).resolve()
        try:
            resolved.relative_to(dashboard_asset_dir.resolve())
        except ValueError:
            return json_error("dashboard asset path is outside static directory", status_code=404)
        if not resolved.is_file():
            return json_error("dashboard asset not found", status_code=404)
        return FileResponse(
            resolved,
            media_type=mimetypes.guess_type(str(resolved))[0] or "application/octet-stream",
            headers={"Cache-Control": "private, max-age=30"},
        )
