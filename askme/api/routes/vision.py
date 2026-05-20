"""Vision snapshot and image archive FastAPI routes."""

from __future__ import annotations

import base64
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from askme.api.schemas.vision import (
    VisionAnalyzeResponse,
    VisionCaptureDeleteResponse,
    VisionCaptureDetailResponse,
    VisionCaptureListResponse,
    VisionSnapshotResponse,
)
from askme.api.services.http_helpers import require_json_object

VisionSnapshotHandler = Callable[[], Awaitable[dict[str, Any] | None]]
VisionAnalyzeHandler = Callable[[str], Awaitable[str]]
ArchiveSnapshotHandler = Callable[[bytes, str, str, int, int], Awaitable[dict[str, Any]]]
ArchiveListHandler = Callable[[], Awaitable[list[dict[str, Any]]]]
ArchiveGetHandler = Callable[[str], Awaitable[dict[str, Any] | None]]
ArchiveDeleteHandler = Callable[[str], Awaitable[bool]]
CorsOptions = Callable[[str], Response]


def register_vision_routes(
    app: FastAPI,
    *,
    vision_snapshot_handler: VisionSnapshotHandler | None,
    vision_analyze_handler: VisionAnalyzeHandler | None,
    archive_snapshot_handler: ArchiveSnapshotHandler | None,
    archive_list_handler: ArchiveListHandler | None,
    archive_get_handler: ArchiveGetHandler | None,
    archive_delete_handler: ArchiveDeleteHandler | None,
    cors_headers: dict[str, str],
    cors_options_response: CorsOptions,
    logger: logging.Logger,
) -> None:
    """Register vision snapshot, analysis, and image archive routes."""

    @app.get(
        "/api/vision/snapshot",
        tags=["Vision"],
        response_model=VisionSnapshotResponse,
        response_model_exclude_none=True,
    )
    async def vision_snapshot() -> JSONResponse:
        """Capture a frame from the robot camera and return it as base64 JPEG."""
        if vision_snapshot_handler is None:
            return _vision_json(
                {"error": "vision not configured"},
                VisionSnapshotResponse,
                status_code=503,
                headers=cors_headers,
            )
        try:
            result = await vision_snapshot_handler()
            if result is None:
                return _vision_json(
                    {"error": "camera not available"},
                    VisionSnapshotResponse,
                    status_code=503,
                    headers=cors_headers,
                )
            if archive_snapshot_handler is not None:
                try:
                    image_bytes = base64.b64decode(result.get("image_base64", ""))
                    if image_bytes:
                        meta = await archive_snapshot_handler(
                            image_bytes,
                            "manual",
                            "",
                            result.get("width", 0),
                            result.get("height", 0),
                        )
                        result = dict(result)
                        result["capture_id"] = meta.get("id")
                except Exception as archive_exc:
                    logger.warning("[Vision] Auto-archive failed: %s", archive_exc)
            return _vision_json(result, VisionSnapshotResponse, headers=cors_headers)
        except Exception as exc:
            logger.error("Vision snapshot failed: %s", exc)
            return _vision_json(
                {"error": str(exc)},
                VisionSnapshotResponse,
                status_code=500,
                headers=cors_headers,
            )

    @app.options("/api/vision/snapshot", include_in_schema=False)
    async def vision_snapshot_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.post(
        "/api/vision/analyze",
        tags=["Vision"],
        response_model=VisionAnalyzeResponse,
        response_model_exclude_none=True,
    )
    async def vision_analyze(request: Request) -> JSONResponse:
        """Analyze an image (base64 JPEG) with the VLM and return a description."""
        if vision_analyze_handler is None:
            return _vision_json(
                {"error": "vision not configured"},
                VisionAnalyzeResponse,
                status_code=503,
                headers=cors_headers,
            )
        try:
            body = require_json_object(await request.json())
            image_b64: str = body.get("image_base64", "")
            if not image_b64:
                return _vision_json(
                    {"error": "image_base64 required"},
                    VisionAnalyzeResponse,
                    status_code=400,
                    headers=cors_headers,
                )
            description = await vision_analyze_handler(image_b64)
            return _vision_json(
                {"description": description},
                VisionAnalyzeResponse,
                headers=cors_headers,
            )
        except ValueError as exc:
            return _vision_json(
                {"error": str(exc)},
                VisionAnalyzeResponse,
                status_code=400,
                headers=cors_headers,
            )
        except Exception as exc:
            logger.error("Vision analyze failed: %s", exc)
            return _vision_json(
                {"error": str(exc)},
                VisionAnalyzeResponse,
                status_code=500,
                headers=cors_headers,
            )

    @app.options("/api/vision/analyze", include_in_schema=False)
    async def vision_analyze_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.get(
        "/api/vision/captures",
        tags=["Vision"],
        response_model=VisionCaptureListResponse,
        response_model_exclude_none=True,
    )
    async def vision_captures_list(
        limit: int = 50,
        label: str | None = None,
    ) -> JSONResponse:
        """List archived captures metadata without image_base64."""
        if archive_list_handler is None:
            return _vision_json(
                {"error": "image archive not configured"},
                VisionCaptureListResponse,
                status_code=503,
                headers=cors_headers,
            )
        try:
            captures = await archive_list_handler()
            if label is not None:
                captures = [capture for capture in captures if capture.get("label") == label]
            captures = captures[:limit]
            return _vision_json(
                {"captures": captures, "count": len(captures)},
                VisionCaptureListResponse,
                headers={"Cache-Control": "no-store", **cors_headers},
            )
        except Exception as exc:
            logger.error("Captures list failed: %s", exc)
            return _vision_json(
                {"error": str(exc)},
                VisionCaptureListResponse,
                status_code=500,
                headers=cors_headers,
            )

    @app.options("/api/vision/captures", include_in_schema=False)
    async def vision_captures_list_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.get(
        "/api/vision/captures/{capture_id}",
        tags=["Vision"],
        response_model=VisionCaptureDetailResponse,
        response_model_exclude_none=True,
    )
    async def vision_captures_get(capture_id: str) -> JSONResponse:
        """Return full metadata plus image_base64 for a capture."""
        if archive_get_handler is None:
            return _vision_json(
                {"error": "image archive not configured"},
                VisionCaptureDetailResponse,
                status_code=503,
                headers=cors_headers,
            )
        try:
            data = await archive_get_handler(capture_id)
            if data is None:
                return _vision_json(
                    {"error": "capture not found"},
                    VisionCaptureDetailResponse,
                    status_code=404,
                    headers=cors_headers,
                )
            return _vision_json(
                data,
                VisionCaptureDetailResponse,
                headers={"Cache-Control": "no-store", **cors_headers},
            )
        except Exception as exc:
            logger.error("Captures get failed: %s", exc)
            return _vision_json(
                {"error": str(exc)},
                VisionCaptureDetailResponse,
                status_code=500,
                headers=cors_headers,
            )

    @app.options("/api/vision/captures/{capture_id}", include_in_schema=False)
    async def vision_captures_item_cors(capture_id: str) -> Response:
        _ = capture_id
        return cors_options_response("GET, DELETE, OPTIONS")

    @app.delete(
        "/api/vision/captures/{capture_id}",
        tags=["Vision"],
        response_model=VisionCaptureDeleteResponse,
        response_model_exclude_none=True,
    )
    async def vision_captures_delete(capture_id: str) -> JSONResponse:
        """Delete a capture record and its image artifact."""
        if archive_delete_handler is None:
            return _vision_json(
                {"error": "image archive not configured"},
                VisionCaptureDeleteResponse,
                status_code=503,
                headers=cors_headers,
            )
        try:
            deleted = await archive_delete_handler(capture_id)
            if not deleted:
                return _vision_json(
                    {"error": "capture not found"},
                    VisionCaptureDeleteResponse,
                    status_code=404,
                    headers=cors_headers,
                )
            return _vision_json(
                {"deleted": True, "capture_id": capture_id},
                VisionCaptureDeleteResponse,
                headers=cors_headers,
            )
        except Exception as exc:
            logger.error("Captures delete failed: %s", exc)
            return _vision_json(
                {"error": str(exc)},
                VisionCaptureDeleteResponse,
                status_code=500,
                headers=cors_headers,
            )


__all__ = ["register_vision_routes"]


def _vision_json(
    payload: dict[str, Any],
    schema: type[BaseModel],
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Return a response validated against the public vision API contract."""

    return JSONResponse(
        schema.model_validate(payload).model_dump(mode="python", exclude_unset=True),
        status_code=status_code,
        headers=headers,
    )
