"""Field operations FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Collection
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.routes.field_admin import register_field_admin_routes
from askme.api.routes.field_customer_project_acceptance import (
    register_customer_project_acceptance_routes,
)
from askme.api.routes.field_customer_project_artifacts import (
    register_customer_project_artifact_routes,
)
from askme.api.routes.field_customer_project_execution import (
    register_customer_project_execution_routes,
)
from askme.api.routes.field_customer_project_profiles import (
    register_customer_project_profile_routes,
)
from askme.api.routes.field_customer_project_templates import (
    register_customer_project_template_routes,
)
from askme.api.routes.field_delivery_resources import (
    register_delivery_resource_routes,
)
from askme.api.routes.field_events import register_field_event_routes
from askme.api.routes.field_internal import register_field_internal_routes
from askme.api.routes.field_product_catalog import register_field_product_catalog_routes
from askme.api.routes.field_project_scope import (
    apply_single_scope_defaults as _apply_single_scope_defaults,
)
from askme.api.routes.field_project_scope import (
    has_explicit_project_scope as _has_explicit_project_scope,
)
from askme.api.routes.field_project_scope import (
    operator_project_scope as _operator_project_scope,
)
from askme.api.routes.field_project_scope import (
    resource_has_explicit_scope as _resource_has_explicit_scope,
)
from askme.api.routes.field_project_scope import (
    scope_allows as _scope_allows,
)
from askme.api.routes.field_project_scope import (
    scope_item_from_create_body as _scope_item_from_create_body,
)
from askme.api.routes.field_project_scope import (
    scope_item_from_detail as _scope_item_from_detail,
)
from askme.api.routes.field_project_scope import (
    scope_item_from_dossier as _scope_item_from_dossier,
)
from askme.api.routes.field_project_scope import (
    scope_item_from_event_detail as _scope_item_from_event_detail,
)
from askme.api.routes.field_project_scope import (
    scope_item_from_event_payload as _scope_item_from_event_payload,
)
from askme.api.routes.field_project_scope import (
    scope_item_from_package as _scope_item_from_package,
)
from askme.api.routes.field_project_scope import (
    scope_item_from_profile as _scope_item_from_profile,
)
from askme.api.routes.field_project_scope import (
    scope_item_from_proposal as _scope_item_from_proposal,
)
from askme.api.routes.field_project_scope import (
    scope_item_from_resource as _scope_item_from_resource,
)
from askme.api.routes.field_project_scope import (
    scope_item_from_site as _scope_item_from_site,
)
from askme.api.routes.field_project_scope import (
    scoped_query_value as _scoped_query_value,
)
from askme.api.services.field_resource_governance_notifications import (
    deliver_resource_governance_notification,
)
from askme.api.services.field_route_roots import build_field_route_roots
from askme.api.services.field_scope_guards import (
    customer_project_scope_allows,
    field_event_scope_allows,
)

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
MissionJson = Callable[..., JSONResponse]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
CorsOptions = Callable[[str], Response]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
ManualTriggerBody = Callable[[Request, dict[str, Any]], dict[str, Any]]
LooksLikeDeviceIngest = Callable[[dict[str, Any]], bool]
FieldResultHook = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
FieldRuntimePolicy = Callable[..., Awaitable[dict[str, Any]]]
RuntimeCallbackTrust = Callable[..., dict[str, Any]]
RuntimeCallbackDeliveryBody = Callable[..., dict[str, Any]]
ConfigProvider = Callable[[], dict[str, Any]]
IdentityReadinessPayload = Callable[[], dict[str, Any]]

_ALL_FIELD_ROUTE_SURFACES = frozenset({"product", "admin", "internal"})


def register_field_routes(
    app: FastAPI,
    *,
    dispatch_field_operations: Dispatch,
    mission_json: MissionJson,
    optional_json_body: OptionalJsonBody,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    authorize: Authorize,
    field_manual_trigger_body: ManualTriggerBody,
    looks_like_device_ingest_without_scenario: LooksLikeDeviceIngest,
    dispatch_field_voice_directive: FieldResultHook,
    dispatch_field_runtime_policy: FieldRuntimePolicy,
    runtime_callback_trust: RuntimeCallbackTrust,
    runtime_callback_delivery_body: RuntimeCallbackDeliveryBody,
    runtime_callback_secret: str | None,
    runtime_callback_max_age_s: float,
    cors_headers: dict[str, str],
    identity_readiness_payload: IdentityReadinessPayload,
    site_profile_root: Path | None = None,
    customer_project_template_root: Path | None = None,
    delivery_resource_root: Path | None = None,
    customer_project_package_root: Path | None = None,
    customer_project_acceptance_dossier_root: Path | None = None,
    customer_project_proposal_root: Path | None = None,
    config_provider: ConfigProvider | None = None,
    surfaces: Collection[str] | None = None,
) -> None:
    """Register field routes for the requested API surface names."""

    enabled_surfaces = {str(surface) for surface in (surfaces or _ALL_FIELD_ROUTE_SURFACES)}
    route_roots = build_field_route_roots(
        site_profile_root=site_profile_root,
        customer_project_template_root=customer_project_template_root,
        delivery_resource_root=delivery_resource_root,
        customer_project_package_root=customer_project_package_root,
        customer_project_acceptance_dossier_root=customer_project_acceptance_dossier_root,
        customer_project_proposal_root=customer_project_proposal_root,
    )

    def _route_enabled(surface: str) -> bool:
        return surface in enabled_surfaces

    def _project_read_auth(request: Request) -> tuple[JSONResponse | None, dict[str, Any]]:
        body: dict[str, Any] = {}
        return authorize(request, body, "field:project:read"), body

    def _resource_governance_config() -> dict[str, Any]:
        config = config_provider() if config_provider is not None else {}
        if not isinstance(config, dict):
            return {}
        return config

    def _resource_governance_notification_delivery(
        escalation: dict[str, Any],
    ) -> dict[str, Any]:
        return deliver_resource_governance_notification(
            escalation,
            config=_resource_governance_config(),
        )

    def _project_scope_forbidden() -> JSONResponse:
        return mission_json(
            {
                "error": "operator not authorized for this customer project",
                "reason": "project_scope_not_allowed",
            },
            status_code=403,
        )

    def _customer_project_scope_failure(
        root: Path,
        identifier: str,
        scope: dict[str, list[str]],
    ) -> JSONResponse | None:
        if not customer_project_scope_allows(
            root,
            identifier,
            scope,
            scope_allows=_scope_allows,
            scope_item_from_detail=_scope_item_from_detail,
        ):
            return _project_scope_forbidden()
        return None

    async def _field_event_scope_failure(
        event_id: str,
        scope: dict[str, list[str]],
    ) -> JSONResponse | None:
        if not await field_event_scope_allows(
            event_id,
            scope,
            dispatch_field_operations=dispatch_field_operations,
            scope_allows=_scope_allows,
            scope_item_from_event_detail=_scope_item_from_event_detail,
        ):
            return _project_scope_forbidden()
        return None

    if _route_enabled("product"):
        register_field_event_routes(
            app,
            dispatch_field_operations=dispatch_field_operations,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            authorize=authorize,
            project_read_auth=_project_read_auth,
            operator_project_scope=_operator_project_scope,
            scoped_query_value=_scoped_query_value,
            scope_allows=_scope_allows,
            scope_item_from_event_detail=_scope_item_from_event_detail,
            scope_item_from_event_payload=_scope_item_from_event_payload,
            has_explicit_project_scope=_has_explicit_project_scope,
            apply_single_scope_defaults=_apply_single_scope_defaults,
            project_scope_forbidden=_project_scope_forbidden,
            field_event_scope_failure=_field_event_scope_failure,
            field_manual_trigger_body=field_manual_trigger_body,
            looks_like_device_ingest_without_scenario=looks_like_device_ingest_without_scenario,
            dispatch_field_voice_directive=dispatch_field_voice_directive,
            dispatch_field_runtime_policy=dispatch_field_runtime_policy,
            cors_headers=cors_headers,
            cors_options_response=cors_options_response,
            logger=logger,
        )
    if _route_enabled("internal"):
        register_field_internal_routes(
            app,
            dispatch_field_operations=dispatch_field_operations,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            cors_options_response=cors_options_response,
            logger=logger,
            dispatch_field_voice_directive=dispatch_field_voice_directive,
            dispatch_field_runtime_policy=dispatch_field_runtime_policy,
            runtime_callback_trust=runtime_callback_trust,
            runtime_callback_delivery_body=runtime_callback_delivery_body,
            runtime_callback_secret=runtime_callback_secret,
            runtime_callback_max_age_s=runtime_callback_max_age_s,
        )

    if _route_enabled("admin"):
        register_field_admin_routes(
            app,
            dispatch_field_operations=dispatch_field_operations,
            mission_json=mission_json,
            optional_json_body=optional_json_body,
            cors_options_response=cors_options_response,
            logger=logger,
            authorize=authorize,
            site_profile_root=route_roots.site_profile_root,
            template_root=route_roots.template_root,
            operator_project_scope=_operator_project_scope,
            scope_allows=_scope_allows,
            scope_item_from_create_body=_scope_item_from_create_body,
            project_scope_forbidden=_project_scope_forbidden,
        )

    if _route_enabled("product"):
        def _dashboard_pages_payload() -> dict[str, Any]:
            from askme.api.composition import api_route_inventory
            from askme.api.services.dashboard_pages import dashboard_pages_payload

            return dashboard_pages_payload(route_inventory=api_route_inventory(app))

        register_field_product_catalog_routes(
            app,
            dispatch_field_operations=dispatch_field_operations,
            mission_json=mission_json,
            project_read_auth=_project_read_auth,
            operator_project_scope=_operator_project_scope,
            scope_allows=_scope_allows,
            scope_item_from_site=_scope_item_from_site,
            scope_item_from_resource=_scope_item_from_resource,
            resource_has_explicit_scope=_resource_has_explicit_scope,
            site_profile_root=route_roots.site_profile_root,
            template_root=route_roots.template_root,
            delivery_resource_root=route_roots.delivery_resource_root,
            identity_readiness_payload=identity_readiness_payload,
            dashboard_pages_payload=_dashboard_pages_payload,
            cors_options_response=cors_options_response,
            logger=logger,
        )

    if _route_enabled("product"):
        register_customer_project_template_routes(
            app,
            template_root=route_roots.template_root,
            project_read_auth=_project_read_auth,
            optional_json_body=optional_json_body,
            authorize=authorize,
            operator_project_scope=_operator_project_scope,
            project_scope_forbidden=_project_scope_forbidden,
            mission_json=mission_json,
            cors_options_response=cors_options_response,
            logger=logger,
        )
    if _route_enabled("admin"):
        register_delivery_resource_routes(
            app,
            delivery_resource_root=route_roots.delivery_resource_root,
            project_read_auth=_project_read_auth,
            optional_json_body=optional_json_body,
            authorize=authorize,
            operator_project_scope=_operator_project_scope,
            apply_single_scope_defaults=_apply_single_scope_defaults,
            scope_allows=_scope_allows,
            scope_item_from_resource=_scope_item_from_resource,
            resource_has_explicit_scope=_resource_has_explicit_scope,
            project_scope_forbidden=_project_scope_forbidden,
            notification_delivery=_resource_governance_notification_delivery,
            mission_json=mission_json,
            cors_options_response=cors_options_response,
            logger=logger,
        )

    if _route_enabled("product"):
        register_customer_project_acceptance_routes(
            app,
            site_profile_root=route_roots.site_profile_root,
            project_read_auth=_project_read_auth,
            optional_json_body=optional_json_body,
            authorize=authorize,
            operator_project_scope=_operator_project_scope,
            scope_allows=_scope_allows,
            scope_item_from_detail=_scope_item_from_detail,
            project_scope_forbidden=_project_scope_forbidden,
            mission_json=mission_json,
            cors_options_response=cors_options_response,
            logger=logger,
        )

    if _route_enabled("product"):
        register_customer_project_execution_routes(
            app,
            site_profile_root=route_roots.site_profile_root,
            delivery_resource_root=route_roots.delivery_resource_root,
            project_read_auth=_project_read_auth,
            optional_json_body=optional_json_body,
            authorize=authorize,
            operator_project_scope=_operator_project_scope,
            scope_allows=_scope_allows,
            scope_item_from_detail=_scope_item_from_detail,
            project_scope_forbidden=_project_scope_forbidden,
            dispatch_field_operations=dispatch_field_operations,
            mission_json=mission_json,
            cors_options_response=cors_options_response,
            logger=logger,
        )

    if _route_enabled("product"):
        register_customer_project_artifact_routes(
            app,
            site_profile_root=route_roots.site_profile_root,
            template_root=route_roots.template_root,
            package_output_root=route_roots.package_output_root,
            acceptance_dossier_output_root=route_roots.acceptance_dossier_output_root,
            proposal_output_root=route_roots.proposal_output_root,
            project_read_auth=_project_read_auth,
            optional_json_body=optional_json_body,
            authorize=authorize,
            operator_project_scope=_operator_project_scope,
            scope_allows=_scope_allows,
            scope_item_from_package=_scope_item_from_package,
            scope_item_from_proposal=_scope_item_from_proposal,
            scope_item_from_dossier=_scope_item_from_dossier,
            scope_item_from_profile=_scope_item_from_profile,
            customer_project_scope_failure=_customer_project_scope_failure,
            project_scope_forbidden=_project_scope_forbidden,
            mission_json=mission_json,
            cors_options_response=cors_options_response,
            logger=logger,
        )

    if _route_enabled("product"):
        register_customer_project_profile_routes(
            app,
            site_profile_root=route_roots.site_profile_root,
            project_read_auth=_project_read_auth,
            optional_json_body=optional_json_body,
            authorize=authorize,
            operator_project_scope=_operator_project_scope,
            scope_allows=_scope_allows,
            scope_item_from_detail=_scope_item_from_detail,
            scope_item_from_profile=_scope_item_from_profile,
            project_scope_forbidden=_project_scope_forbidden,
            mission_json=mission_json,
            cors_options_response=cors_options_response,
            logger=logger,
        )
