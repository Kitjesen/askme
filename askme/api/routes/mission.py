"""Mission draft and submission FastAPI routes."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from askme.api.schemas.mission import (
    MissionDetailResponse,
    MissionDraftResponse,
    MissionListResponse,
    MissionReportResponse,
    MissionSubmitResponse,
)
from askme.api.services.http_helpers import require_json_object

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
JsonError = Callable[..., JSONResponse]
MissionJson = Callable[..., JSONResponse]
CorsOptions = Callable[[str], Response]
RequestHasControlAuth = Callable[[Request], bool]


def register_mission_routes(
    app: FastAPI,
    *,
    dispatch_mission: Dispatch,
    json_error: JsonError,
    mission_json: MissionJson,
    cors_options_response: CorsOptions,
    request_has_control_auth: RequestHasControlAuth,
    logger: logging.Logger,
) -> None:
    """Register high-level mission draft/submission routes."""

    def mission_failure(message: str, exc: Exception) -> JSONResponse:
        _ = exc
        logger.exception(message)
        return json_error("mission request failed", status_code=500)

    @app.post("/api/missions/draft",
        tags=["Mission"],
        response_model=MissionDraftResponse,
        response_model_exclude_none=True,
    )
    async def mission_draft(request: Request) -> JSONResponse:
        """Draft a high-level mission without dispatching hardware."""
        try:
            body = require_json_object(await request.json())
            payload = await dispatch_mission("draft_from_payload", body)
            return mission_json(_mission_payload(payload, MissionDraftResponse))
        except ValueError as exc:
            return json_error(str(exc), status_code=400)
        except RuntimeError as exc:
            return json_error(str(exc), status_code=503)
        except Exception as exc:
            return mission_failure("Mission draft failed", exc)

    @app.post(
        "/api/missions",
        tags=["Mission"],
        response_model=MissionSubmitResponse,
        response_model_exclude_none=True,
    )
    async def mission_submit(request: Request) -> JSONResponse:
        """Dry-run or submit a mission through the configured runtime arbiter."""
        try:
            body = require_json_object(await request.json())
            payload = await dispatch_mission(
                "submit_from_payload",
                body,
                trusted_confirmation=request_has_control_auth(request),
            )
            return mission_json(_mission_payload(payload, MissionSubmitResponse))
        except ValueError as exc:
            return json_error(str(exc), status_code=400)
        except RuntimeError as exc:
            return json_error(str(exc), status_code=503)
        except Exception as exc:
            return mission_failure("Mission submit failed", exc)

    @app.get(
        "/api/missions",
        tags=["Mission"],
        response_model=MissionListResponse,
        response_model_exclude_none=True,
    )
    async def mission_list() -> JSONResponse:
        """Return locally drafted/submitted mission records."""
        try:
            payload = await dispatch_mission("list_payload")
            return mission_json(_mission_payload(payload, MissionListResponse))
        except RuntimeError as exc:
            return json_error(str(exc), status_code=503)
        except Exception as exc:
            return mission_failure("Mission list failed", exc)

    @app.get(
        "/api/missions/{mission_id}",
        tags=["Mission"],
        response_model=MissionDetailResponse,
        response_model_exclude_none=True,
    )
    async def mission_get(mission_id: str) -> JSONResponse:
        """Return a single mission plan and its latest submission state."""
        try:
            payload = await dispatch_mission("get_payload", mission_id)
            status_code = 404 if payload.get("error") else 200
            return mission_json(
                _mission_payload(payload, MissionDetailResponse),
                status_code=status_code,
            )
        except RuntimeError as exc:
            return json_error(str(exc), status_code=503)
        except Exception as exc:
            return mission_failure("Mission get failed", exc)

    @app.get("/api/missions/{mission_id}/report",
        tags=["Mission"],
        response_model=MissionReportResponse,
        response_model_exclude_none=True,
    )
    async def mission_report(mission_id: str) -> JSONResponse:
        """Build an inspection report shell from mission evidence."""
        try:
            payload = await dispatch_mission("report_payload", mission_id)
            status_code = 404 if payload.get("error") else 200
            return mission_json(
                _mission_payload(payload, MissionReportResponse),
                status_code=status_code,
            )
        except RuntimeError as exc:
            return json_error(str(exc), status_code=503)
        except Exception as exc:
            return mission_failure("Mission report failed", exc)

    @app.options("/api/missions", include_in_schema=False)
    @app.options("/api/missions/draft", include_in_schema=False)
    async def mission_collection_cors() -> Response:
        return cors_options_response("GET, POST, OPTIONS")

    @app.options("/api/missions/{mission_id}", include_in_schema=False)
    @app.options("/api/missions/{mission_id}/report", include_in_schema=False)
    async def mission_item_cors(mission_id: str) -> Response:
        _ = mission_id
        return cors_options_response("GET, OPTIONS")


def _mission_payload(payload: dict[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
    return schema.model_validate(payload).model_dump(mode="python")
