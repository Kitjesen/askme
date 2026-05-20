"""Administration route registration."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from askme.api.field_surface import register_field_surface_routes
from askme.api.routes.agent_profiles import register_agent_profile_routes
from askme.api.routes.audit import register_audit_routes
from askme.api.routes.governance import register_governance_routes
from askme.api.routes.skills import register_skill_routes


def register_admin_routes(app: FastAPI, deps: Any) -> None:
    """Register routes used by operators, supervisors, and delivery teams."""

    register_field_surface_routes(app, deps, surfaces={"admin"})
    register_governance_routes(
        app,
        governance_payload=deps.governance_payload,
        identity_readiness_payload=deps.identity_readiness_payload,
        current_operator_payload=deps.current_operator_payload,
        authorization_payload=deps.authorization_payload,
        mission_json=deps.mission_json,
        cors_options_response=deps.cors_options_response,
    )
    register_audit_routes(
        app,
        config_provider=deps.config_provider,
        optional_json_body=deps.optional_json_body,
        authorize=deps.authorize,
        operator_id_from_request=deps.operator_id_from_request,
        logger=deps.logger,
    )
    register_skill_routes(
        app,
        optional_json_body=deps.optional_json_body,
        authorize=deps.authorize,
        operator_id_from_request=deps.operator_id_from_request,
        skill_growth_candidate_prompt=deps.skill_growth_candidate_prompt,
        logger=deps.logger,
    )
    register_agent_profile_routes(
        app,
        optional_json_body=deps.optional_json_body,
        authorize=deps.authorize,
        operator_id_from_request=deps.operator_id_from_request,
        logger=deps.logger,
    )


__all__ = ["register_admin_routes"]
