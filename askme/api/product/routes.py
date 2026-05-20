"""Customer-facing route registration.

This surface is for Dashboard and customer-visible workflows. It may keep
legacy URLs for compatibility, but the modules registered here should speak in
product terms: conversation, knowledge, field events, space guidance,
capabilities, missions, and voice profiles.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from askme.api.field_surface import register_field_surface_routes
from askme.api.routes.capabilities import register_capability_routes
from askme.api.routes.conversation import register_conversation_routes
from askme.api.routes.dashboard import register_dashboard_routes
from askme.api.routes.memory import register_memory_routes
from askme.api.routes.mission import register_mission_routes
from askme.api.routes.space import register_space_routes
from askme.api.routes.voice import register_voice_routes


def register_product_routes(app: FastAPI, deps: Any) -> None:
    """Register routes used by customer-facing screens and workflows."""

    def route_inventory_provider() -> dict[str, Any]:
        from askme.api.composition import api_route_inventory

        return api_route_inventory(app)

    register_memory_routes(
        app,
        dispatch_memory=deps.dispatch_memory,
        mission_json=deps.mission_json,
        cors_options_response=deps.cors_options_response,
        logger=deps.logger,
        authorize=deps.authorize,
    )
    register_voice_routes(
        app,
        dispatch_voice=deps.dispatch_voice,
        mission_json=deps.mission_json,
        optional_json_body=deps.optional_json_body,
        cors_options_response=deps.cors_options_response,
        authorize=deps.authorize,
    )
    register_space_routes(
        app,
        dispatch_space=deps.dispatch_space,
        mission_json=deps.mission_json,
        optional_json_body=deps.optional_json_body,
        cors_options_response=deps.cors_options_response,
        logger=deps.logger,
        authorize=deps.authorize,
    )
    register_field_surface_routes(app, deps, surfaces={"product"})
    register_dashboard_routes(
        app,
        dashboard_html=deps.dashboard_html,
        dashboard_asset_dir=deps.dashboard_asset_dir,
        dashboard_pages=deps.dashboard_pages,
        json_error=deps.json_error,
        route_inventory_provider=route_inventory_provider,
    )
    register_capability_routes(
        app,
        capabilities_provider=deps.capabilities_provider,
        blueprints_provider=deps.blueprints_provider,
        space_dispatch=deps.dispatch_space,
        logger=deps.logger,
    )
    register_conversation_routes(
        app,
        conversation_service=deps.conversation_service,
        runtime_available=deps.runtime_available,
        dispatch_runtime=deps.dispatch_runtime,
        cors_options_response=deps.cors_options_response,
        logger=deps.logger,
        runtime_voice_turn_timeout_s=deps.runtime_voice_turn_timeout_s,
    )
    register_mission_routes(
        app,
        dispatch_mission=deps.dispatch_mission,
        json_error=deps.json_error,
        mission_json=deps.mission_json,
        cors_options_response=deps.cors_options_response,
        request_has_control_auth=deps.request_has_control_auth,
        logger=deps.logger,
    )


__all__ = ["register_product_routes"]
