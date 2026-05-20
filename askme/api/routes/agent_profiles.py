"""Agent Profile governance FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from askme.api.routes._request_validation import (
    RequestFieldError,
    field_error_response,
    optional_float_field,
    optional_int_field,
    route_error_response,
)
from askme.api.schemas.agent_profiles import (
    AgentProfileCatalogResponse,
    AgentProfilePreviewResponse,
    AgentProfileUpsertResponse,
)
from askme.api.services.agent_profile_tools import agent_profile_known_tools

OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
OperatorIdFromRequest = Callable[[Request, dict[str, Any]], str]
KnownToolsProvider = Callable[[], set[str]]
AgentProfileRegistryFactory = Callable[[], Any]

NO_STORE_HEADERS = {"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"}
CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}


def register_agent_profile_routes(
    app: FastAPI,
    *,
    optional_json_body: OptionalJsonBody,
    authorize: Authorize,
    operator_id_from_request: OperatorIdFromRequest,
    logger: logging.Logger,
    known_tools_provider: KnownToolsProvider | None = None,
    agent_profile_registry_factory: AgentProfileRegistryFactory | None = None,
) -> None:
    """Register product-reviewable Agent Profile catalog and write routes."""

    def _known_tools() -> set[str]:
        if known_tools_provider is not None:
            return known_tools_provider()
        return agent_profile_known_tools()

    def _registry() -> Any:
        if agent_profile_registry_factory is not None:
            return agent_profile_registry_factory()
        from askme.agent_shell.agent_profile import AgentProfileRegistry

        return AgentProfileRegistry()

    @app.get(
        "/api/agent-profiles",
        tags=["System"],
        response_model=AgentProfileCatalogResponse,
        response_model_exclude_none=True,
    )
    async def agent_profiles() -> JSONResponse:
        """Return product-reviewable agent profile catalog."""
        try:
            catalog = _registry().catalog()
            response = AgentProfileCatalogResponse.model_validate(catalog)
            return JSONResponse(
                response.model_dump(mode="python", exclude_none=True),
                headers=NO_STORE_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="agent profile route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.post(
        "/api/agent-profiles",
        tags=["System"],
        response_model=AgentProfileUpsertResponse,
        response_model_exclude_none=True,
    )
    async def upsert_agent_profile(request: Request) -> JSONResponse:
        """Create or update a project-level agent profile Markdown file."""
        try:
            body = await optional_json_body(request)
            denied = authorize(request, body, "skill:review")
            if denied is not None:
                return denied
            max_iterations = optional_int_field(
                body,
                "max_iterations",
                aliases=("maxTurns",),
                min_value=1,
            )
            timeout_seconds = optional_float_field(
                body,
                "timeout_seconds",
                aliases=("timeoutSeconds",),
                min_value=0,
            )
            registry = _registry()
            result = registry.write_project_profile(
                name=str(body.get("name") or ""),
                display_name=str(body.get("display_name") or ""),
                description=str(body.get("description") or ""),
                instructions=str(body.get("instructions") or ""),
                tools=body.get("tools"),
                disallowed_tools=body.get("disallowed_tools", body.get("disallowedTools")),
                spawnable_profiles=body.get("spawnable_profiles", body.get("spawnableProfiles")),
                skills=body.get("skills"),
                mcp_servers=body.get("mcp_servers", body.get("mcpServers")),
                hooks=body.get("hooks") if isinstance(body.get("hooks"), dict) else None,
                model=str(body.get("model") or "inherit"),
                permission_mode=str(
                    body.get("permission_mode") or body.get("permissionMode") or "default"
                ),
                risk_level=str(body.get("risk_level") or body.get("riskLevel") or "medium"),
                customer_visible=bool(
                    body.get("customer_visible", body.get("customerVisible", True))
                ),
                memory_scope=str(body.get("memory") or body.get("memory_scope") or ""),
                max_iterations=max_iterations,
                timeout_seconds=timeout_seconds,
                operator_id=operator_id_from_request(request, body),
                known_tools=_known_tools(),
                overwrite=bool(body.get("overwrite", True)),
            )
            response = AgentProfileUpsertResponse.model_validate(result)
            return JSONResponse(
                response.model_dump(mode="python", exclude_none=True),
                status_code=200 if result.get("ok") else 400,
                headers=NO_STORE_HEADERS,
            )
        except RequestFieldError as exc:
            return field_error_response(exc, headers=CORS_HEADERS)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400, headers=CORS_HEADERS)
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="agent profile route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )

    @app.get(
        "/api/agent-profiles/{profile_name}/preview",
        tags=["System"],
        response_model=AgentProfilePreviewResponse,
        response_model_exclude_none=True,
    )
    async def agent_profile_preview(profile_name: str) -> JSONResponse:
        """Return parsed profile policy plus raw Markdown when available."""
        try:
            payload = _registry().preview(profile_name)
            response = AgentProfilePreviewResponse.model_validate(payload)
            return JSONResponse(
                response.model_dump(mode="python", exclude_none=True),
                status_code=200 if payload.get("ok") else 404,
                headers=NO_STORE_HEADERS,
            )
        except Exception as exc:
            return route_error_response(
                logger,
                public_error="agent profile route failed",
                exc=exc,
                headers=CORS_HEADERS,
            )


__all__ = ["register_agent_profile_routes"]
