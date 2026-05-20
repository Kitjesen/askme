"""Internal route registration.

These routes expose runtime, cognition, and raw perception capabilities. They
are not the vocabulary customers should see in the product UI.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from askme.api.field_surface import register_field_surface_routes
from askme.api.routes.cognition import register_cognition_routes
from askme.api.routes.runtime import register_runtime_routes
from askme.api.routes.vision import register_vision_routes


def register_internal_routes(app: FastAPI, deps: Any) -> None:
    """Register routes for runtime control and machine-facing integrations."""

    register_field_surface_routes(app, deps, surfaces={"internal"})
    register_cognition_routes(
        app,
        dispatch_cognition=deps.dispatch_cognition,
        json_error=deps.json_error,
        cors_options_response=deps.cors_options_response,
        cors_headers=deps.cors_headers,
    )
    register_runtime_routes(
        app,
        dispatch_runtime=deps.dispatch_runtime,
        json_error=deps.json_error,
        cors_options_response=deps.cors_options_response,
        optional_json_body=deps.optional_json_body,
        operator_action_kwargs=deps.operator_action_kwargs,
        authorize=deps.authorize,
        cors_headers=deps.cors_headers,
    )
    register_vision_routes(
        app,
        vision_snapshot_handler=deps.vision_snapshot_handler,
        vision_analyze_handler=deps.vision_analyze_handler,
        archive_snapshot_handler=deps.archive_snapshot_handler,
        archive_list_handler=deps.archive_list_handler,
        archive_get_handler=deps.archive_get_handler,
        archive_delete_handler=deps.archive_delete_handler,
        cors_headers=dict(deps.cors_headers),
        cors_options_response=deps.cors_options_response,
        logger=deps.logger,
    )


__all__ = ["register_internal_routes"]
