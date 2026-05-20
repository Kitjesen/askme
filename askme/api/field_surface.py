"""Shared registration helper for Field API surfaces."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from fastapi import FastAPI

from askme.api.routes.field import register_field_routes


def register_field_surface_routes(
    app: FastAPI,
    deps: Any,
    *,
    surfaces: Collection[str],
) -> None:
    """Register Field routes for one or more audience-specific surfaces."""

    register_field_routes(
        app,
        dispatch_field_operations=deps.dispatch_field_operations,
        mission_json=deps.mission_json,
        optional_json_body=deps.optional_json_body,
        cors_options_response=deps.cors_options_response,
        logger=deps.logger,
        authorize=deps.authorize,
        field_manual_trigger_body=deps.field_manual_trigger_body,
        looks_like_device_ingest_without_scenario=deps.looks_like_device_ingest_without_scenario,
        dispatch_field_voice_directive=deps.dispatch_field_voice_directive,
        dispatch_field_runtime_policy=deps.dispatch_field_runtime_policy,
        runtime_callback_trust=deps.runtime_callback_trust,
        runtime_callback_delivery_body=deps.runtime_callback_delivery_body,
        runtime_callback_secret=deps.runtime_callback_secret,
        runtime_callback_max_age_s=deps.runtime_callback_max_age_s,
        cors_headers=deps.cors_headers,
        identity_readiness_payload=deps.identity_readiness_payload,
        **dict(deps.field_path_roots),
        config_provider=deps.config_provider,
        surfaces=surfaces,
    )


__all__ = ["register_field_surface_routes"]
