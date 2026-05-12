"""Runtime TaskRun FastAPI routes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

Dispatch = Callable[..., Awaitable[dict[str, Any]]]
JsonError = Callable[..., JSONResponse]
CorsOptions = Callable[[str], Response]
OptionalJsonBody = Callable[[Request], Awaitable[dict[str, Any]]]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]
OperatorActionKwargs = Callable[[dict[str, Any]], dict[str, Any]]


def register_runtime_routes(
    app: FastAPI,
    *,
    dispatch_runtime: Dispatch,
    json_error: JsonError,
    cors_options_response: CorsOptions,
    optional_json_body: OptionalJsonBody,
    operator_action_kwargs: OperatorActionKwargs,
    authorize: Authorize,
    cors_headers: dict[str, str],
) -> None:
    """Register runtime handoff and TaskRun control routes."""

    @app.get("/api/runtime/context", tags=["Runtime"])
    async def runtime_context() -> JSONResponse:
        try:
            payload = await dispatch_runtime("runtime_context_payload")
            return JSONResponse(payload, headers={"Cache-Control": "no-store", **cors_headers})
        except RuntimeError as exc:
            return json_error(str(exc), status_code=503)
        except Exception as exc:
            return json_error(str(exc), status_code=500)

    @app.get("/api/runtime/events", tags=["Runtime"], response_model=None)
    async def runtime_events(request: Request) -> Response:
        once = _truthy_query(request.query_params.get("once"))
        after = _float_query(request.query_params.get("after"))
        limit = _int_query(request.query_params.get("limit"), default=20, minimum=1, maximum=100)

        async def _stream() -> Any:
            cursor = after
            sent_initial = False
            while True:
                payload = await dispatch_runtime("runtime_events_payload", after=cursor, limit=limit)
                cursor = _float_query(payload.get("cursor"), default=cursor)
                if payload.get("events") or not sent_initial:
                    yield _sse_packet("runtime.events", payload)
                    sent_initial = True
                if once or await request.is_disconnected():
                    break
                yield ": keepalive\n\n"
                await asyncio.sleep(1.0)

        try:
            return StreamingResponse(
                _stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store", "Connection": "keep-alive", **cors_headers},
            )
        except RuntimeError as exc:
            return json_error(str(exc), status_code=503)
        except Exception as exc:
            return json_error(str(exc), status_code=500)

    @app.get("/api/runtime/profiles", tags=["Runtime"])
    async def runtime_profiles() -> JSONResponse:
        return await _runtime_get(dispatch_runtime, json_error, cors_headers, "runtime_profiles_payload")

    @app.get("/api/runtime/runs", tags=["Runtime"])
    async def runtime_runs() -> JSONResponse:
        return await _runtime_get(dispatch_runtime, json_error, cors_headers, "runtime_list_payload")

    @app.get("/api/runtime/runs/{run_id}", tags=["Runtime"])
    async def runtime_run_get(run_id: str) -> JSONResponse:
        return await _runtime_get(dispatch_runtime, json_error, cors_headers, "runtime_get_payload", run_id)

    @app.get("/api/runtime/runs/{run_id}/report", tags=["Runtime"])
    async def runtime_run_report(run_id: str) -> JSONResponse:
        return await _runtime_get(dispatch_runtime, json_error, cors_headers, "runtime_report_payload", run_id)

    @app.post("/api/runtime/runs/{run_id}/pause", tags=["Runtime"])
    async def runtime_run_pause(run_id: str, request: Request) -> JSONResponse:
        return await _runtime_action(
            request,
            run_id,
            permission="runtime:pause",
            method_name="runtime_pause_payload",
            dispatch_runtime=dispatch_runtime,
            json_error=json_error,
            optional_json_body=optional_json_body,
            operator_action_kwargs=operator_action_kwargs,
            authorize=authorize,
            cors_headers=cors_headers,
        )

    @app.post("/api/runtime/runs/{run_id}/resume", tags=["Runtime"])
    async def runtime_run_resume(run_id: str, request: Request) -> JSONResponse:
        return await _runtime_action(
            request,
            run_id,
            permission="runtime:resume",
            method_name="runtime_resume_payload",
            dispatch_runtime=dispatch_runtime,
            json_error=json_error,
            optional_json_body=optional_json_body,
            operator_action_kwargs=operator_action_kwargs,
            authorize=authorize,
            cors_headers=cors_headers,
        )

    @app.post("/api/runtime/runs/{run_id}/cancel", tags=["Runtime"])
    async def runtime_run_cancel(run_id: str, request: Request) -> JSONResponse:
        return await _runtime_action(
            request,
            run_id,
            permission="runtime:cancel",
            method_name="runtime_cancel_payload",
            dispatch_runtime=dispatch_runtime,
            json_error=json_error,
            optional_json_body=optional_json_body,
            operator_action_kwargs=operator_action_kwargs,
            authorize=authorize,
            cors_headers=cors_headers,
        )

    @app.post("/api/runtime/runs/{run_id}/advance", tags=["Runtime"])
    async def runtime_run_advance(run_id: str, request: Request) -> JSONResponse:
        return await _runtime_action(
            request,
            run_id,
            permission="runtime:advance",
            method_name="runtime_advance_payload",
            dispatch_runtime=dispatch_runtime,
            json_error=json_error,
            optional_json_body=optional_json_body,
            operator_action_kwargs=operator_action_kwargs,
            authorize=authorize,
            cors_headers=cors_headers,
        )

    @app.options("/api/runtime/context", include_in_schema=False)
    @app.options("/api/runtime/events", include_in_schema=False)
    @app.options("/api/runtime/profiles", include_in_schema=False)
    @app.options("/api/runtime/runs", include_in_schema=False)
    async def runtime_collection_cors() -> Response:
        return cors_options_response("GET, OPTIONS")

    @app.options("/api/runtime/runs/{run_id}", include_in_schema=False)
    @app.options("/api/runtime/runs/{run_id}/report", include_in_schema=False)
    @app.options("/api/runtime/runs/{run_id}/pause", include_in_schema=False)
    @app.options("/api/runtime/runs/{run_id}/resume", include_in_schema=False)
    @app.options("/api/runtime/runs/{run_id}/cancel", include_in_schema=False)
    @app.options("/api/runtime/runs/{run_id}/advance", include_in_schema=False)
    async def runtime_item_cors(run_id: str) -> Response:
        _ = run_id
        return cors_options_response("GET, POST, OPTIONS")


async def _runtime_get(
    dispatch_runtime: Dispatch,
    json_error: JsonError,
    cors_headers: dict[str, str],
    method_name: str,
    *args: Any,
) -> JSONResponse:
    try:
        payload = await dispatch_runtime(method_name, *args)
        status_code = 404 if payload.get("error") else 200
        return JSONResponse(payload, status_code=status_code, headers={"Cache-Control": "no-store", **cors_headers})
    except RuntimeError as exc:
        return json_error(str(exc), status_code=503)
    except Exception as exc:
        return json_error(str(exc), status_code=500)


async def _runtime_action(
    request: Request,
    run_id: str,
    *,
    permission: str,
    method_name: str,
    dispatch_runtime: Dispatch,
    json_error: JsonError,
    optional_json_body: OptionalJsonBody,
    operator_action_kwargs: OperatorActionKwargs,
    authorize: Authorize,
    cors_headers: dict[str, str],
) -> JSONResponse:
    try:
        body = await optional_json_body(request)
        failure = authorize(request, body, permission)
        if failure is not None:
            return failure
        payload = await dispatch_runtime(method_name, run_id, **operator_action_kwargs(body))
        status_code = 404 if payload.get("error") else 200
        return JSONResponse(payload, status_code=status_code, headers={"Cache-Control": "no-store", **cors_headers})
    except ValueError as exc:
        return json_error(str(exc), status_code=400)
    except RuntimeError as exc:
        return json_error(str(exc), status_code=503)
    except Exception as exc:
        return json_error(str(exc), status_code=500)


def _sse_packet(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _truthy_query(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _float_query(value: Any, *, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_query(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))
