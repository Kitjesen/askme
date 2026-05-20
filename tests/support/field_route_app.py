"""Shared field-route FastAPI app builders for HTTP route tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

import askme.health_server as health_server
from askme.api.routes.field import register_field_routes


def field_route_test_app(
    site_profile_root: Path,
    *,
    customer_project_template_root: Path | None = None,
    customer_project_package_root: Path | None = None,
    customer_project_acceptance_dossier_root: Path | None = None,
    customer_project_proposal_root: Path | None = None,
    authorize_callback=None,
) -> FastAPI:
    app = FastAPI()

    async def optional_json_body(request):
        return await request.json()

    def mission_json(payload, status_code=200):
        return JSONResponse(payload, status_code=status_code)

    async def passthrough_result(result):
        return result

    register_field_routes(
        app,
        dispatch_field_operations=lambda *_args, **_kwargs: {},
        mission_json=mission_json,
        optional_json_body=optional_json_body,
        cors_options_response=lambda methods: Response(
            headers={"Access-Control-Allow-Methods": methods}
        ),
        logger=health_server.logger,
        authorize=authorize_callback or (lambda _request, _body, _permission: None),
        field_manual_trigger_body=lambda _request, body: body,
        looks_like_device_ingest_without_scenario=lambda _body: False,
        dispatch_field_voice_directive=passthrough_result,
        dispatch_field_runtime_policy=lambda result, **_kwargs: passthrough_result(result),
        runtime_callback_trust=lambda _body, **_kwargs: {"trusted": True},
        runtime_callback_delivery_body=lambda body, **_kwargs: body,
        runtime_callback_secret=None,
        runtime_callback_max_age_s=60.0,
        cors_headers={},
        identity_readiness_payload=lambda: {},
        site_profile_root=site_profile_root,
        customer_project_template_root=customer_project_template_root,
        customer_project_package_root=customer_project_package_root,
        customer_project_acceptance_dossier_root=customer_project_acceptance_dossier_root,
        customer_project_proposal_root=customer_project_proposal_root,
        config_provider=lambda: {},
    )
    return app


def scoped_project_authorize(
    project_scope: dict[str, Any],
    operator_id: str = "scoped.operator",
):
    def authorize(request, body, permission):
        body["operator_id"] = request.headers.get("x-askme-operator-id", operator_id)
        body["operator_auth"] = {
            "allowed": True,
            "permission": permission,
            "operator": {
                "operator_id": body["operator_id"],
                "project_scope": project_scope,
            },
        }
        return None

    return authorize
