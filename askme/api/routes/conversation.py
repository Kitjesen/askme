"""Conversation FastAPI routes."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.schemas.conversation import (
    ChatResponse,
    ConversationDiagnosticsResponse,
    RuntimeVoiceTurnResponse,
)
from askme.api.services.conversation_service import (
    ChatOverloaded,
    ChatTimeout,
    ChatUnavailable,
    ConversationService,
    EmptyChatText,
    authorized_runtime_context_from_body,
    runtime_control_permission_from_body,
)
from askme.api.services.http_helpers import require_json_object
from askme.conversation import TurnInProgress, canonical_thread_id
from askme.runtime.control_intent import runtime_control_permission

DispatchRuntime = Callable[..., Awaitable[dict[str, Any]]]
CorsOptions = Callable[[str], Response]
Authorize = Callable[[Request, dict[str, Any], str], JSONResponse | None]

_NO_STORE_HEADERS = {"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"}
_CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}


def register_conversation_routes(
    app: FastAPI,
    *,
    conversation_service: ConversationService,
    runtime_available: bool,
    dispatch_runtime: DispatchRuntime,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    authorize: Authorize,
    runtime_voice_turn_timeout_s: float | None = 30.0,
) -> None:
    """Register chat and voice-turn routes."""

    app.include_router(
        create_conversation_router(
            conversation_service=conversation_service,
            runtime_available=runtime_available,
            dispatch_runtime=dispatch_runtime,
            cors_options_response=cors_options_response,
            logger=logger,
            authorize=authorize,
            runtime_voice_turn_timeout_s=runtime_voice_turn_timeout_s,
        )
    )


def create_conversation_router(
    *,
    conversation_service: ConversationService,
    runtime_available: bool,
    dispatch_runtime: DispatchRuntime,
    cors_options_response: CorsOptions,
    logger: logging.Logger,
    authorize: Authorize,
    runtime_voice_turn_timeout_s: float | None = 30.0,
) -> APIRouter:
    """Create the conversation router without binding it to an app factory."""

    router = APIRouter()

    @router.post(
        "/api/chat",
        tags=["Monitor"],
        response_model=ChatResponse,
        response_model_exclude_none=True,
    )
    async def chat(request: Request) -> JSONResponse:
        """Send text to the brain pipeline and return the response."""
        trace_id = _request_trace_id(request)
        trace_headers = _with_trace(_CORS_HEADERS, trace_id)
        try:
            body = require_json_object(await request.json())
            body.pop("operator_auth", None)
            runtime_permission = runtime_control_permission_from_body(body)
            if runtime_permission is not None and not await _chat_runtime_target_available(
                configured=runtime_available,
                dispatch_runtime=dispatch_runtime,
            ):
                runtime_permission = None
            if runtime_permission is not None:
                failure = authorize(request, body, runtime_permission)
                if failure is not None:
                    failure.headers["X-Askme-Trace-Id"] = trace_id
                    return failure
            payload = await conversation_service.chat_payload_from_body(body, trace_id=trace_id)
            ChatResponse.model_validate(payload)
            return JSONResponse(payload, headers=_with_trace(_NO_STORE_HEADERS, trace_id))
        except EmptyChatText:
            return JSONResponse(
                {"error": "empty text"},
                status_code=400,
                headers=trace_headers,
            )
        except ChatOverloaded as exc:
            return JSONResponse(
                {"error": "chat overloaded", "max_concurrency": exc.max_concurrency},
                status_code=429,
                headers=trace_headers,
            )
        except ChatTimeout as exc:
            return JSONResponse(
                {"error": "chat timed out", "timeout_s": exc.timeout_s},
                status_code=504,
                headers=trace_headers,
            )
        except TurnInProgress as exc:
            return JSONResponse(
                {
                    "error": "conversation turn in progress",
                    "conversation_thread_id": exc.thread_id,
                    "blocking_turn_id": exc.blocking_turn_id,
                },
                status_code=409,
                headers=trace_headers,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400, headers=trace_headers)
        except ChatUnavailable as exc:
            return JSONResponse({"error": str(exc)}, status_code=503, headers=trace_headers)
        except Exception as exc:
            logger.exception("Chat endpoint failed")
            return JSONResponse({"error": str(exc)}, status_code=500, headers=trace_headers)

    @router.get(
        "/api/conversation/diagnostics",
        tags=["Monitor"],
        response_model=ConversationDiagnosticsResponse,
        response_model_exclude_none=True,
    )
    async def conversation_diagnostics() -> JSONResponse:
        """Return non-sensitive chat execution diagnostics."""
        payload = conversation_service.diagnostics_snapshot()
        ConversationDiagnosticsResponse.model_validate(payload)
        return JSONResponse(payload, headers=_NO_STORE_HEADERS)

    @router.post(
        "/api/runtime/voice-turn",
        tags=["Runtime"],
        response_model=RuntimeVoiceTurnResponse,
        response_model_exclude_none=True,
    )
    async def runtime_voice_turn(request: Request) -> JSONResponse:
        """Route a final voice transcript to runtime controls only."""
        try:
            body = require_json_object(await request.json())
            body.pop("operator_auth", None)
            header_operator_id = _clean_optional_text(
                request.headers.get("x-askme-operator-id") or request.headers.get("x-operator-id")
            )
            if header_operator_id:
                body["operator_id"] = header_operator_id
            raw_text = body.get("text") or body.get("message") or body.get("transcript") or ""
            text = str(raw_text).strip()
            runtime_permission = runtime_control_permission(text, default="runtime:submit")
            failure = authorize(request, body, str(runtime_permission))
            if failure is not None:
                return failure
            if not runtime_available:
                return JSONResponse(
                    {"error": "runtime handler not configured"},
                    status_code=503,
                    headers=_CORS_HEADERS,
                )
            if not text:
                return JSONResponse(
                    {
                        "handled": False,
                        "reason": "empty_transcript",
                        "voice_turn": conversation_service.voice_turn_payload_from_body(
                            body,
                            text=text,
                        ),
                    },
                    status_code=400,
                    headers=_CORS_HEADERS,
                )
            conversation_session_id = canonical_thread_id(
                thread_id=_clean_optional_text(body.get("thread_id")),
                conversation_thread_id=_clean_optional_text(body.get("conversation_thread_id")),
                conversation_session_id=_clean_optional_text(body.get("conversation_session_id")),
                conversation_id=_clean_optional_text(body.get("conversation_id")),
                chat_session_id=_clean_optional_text(body.get("chat_session_id")),
                session_id=_clean_optional_text(body.get("session_id")),
            )
            operator_context = authorized_runtime_context_from_body(
                body,
                conversation_session_id=conversation_session_id or "",
                permission=str(runtime_permission),
            )
            if operator_context is None:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "operator authorization provenance unavailable",
                        "reason": "runtime_control_provenance_mismatch",
                        "message": (
                            "Runtime voice control requires a trusted, action-scoped "
                            "operator identity."
                        ),
                    },
                    status_code=403,
                    headers=_CORS_HEADERS,
                )
            dispatch = dispatch_runtime(
                "voice_turn_payload",
                text,
                speak=bool(body.get("speak") or body.get("play_audio")),
                transcript_id=str(body.get("transcript_id") or ""),
                confidence=body.get("asr_confidence", body.get("confidence")),
                is_final=bool(body.get("is_final", True)),
                channel=str(body.get("channel") or "voice"),
                conversation_session_id=conversation_session_id,
                planning_session_id=_clean_optional_text(body.get("planning_session_id")),
                operator_id=operator_context.operator_id,
                operator_roles=list(operator_context.operator_roles),
                operator_authenticated=operator_context.operator_authenticated,
                operator_source=operator_context.operator_source,
                runtime_permission=operator_context.permission,
                reason=str(body.get("reason") or ""),
                risk_acknowledgement=bool(
                    body.get("risk_acknowledgement")
                    or body.get("risk_ack")
                    or body.get("acknowledged")
                ),
            )
            if runtime_voice_turn_timeout_s is not None and runtime_voice_turn_timeout_s > 0:
                payload = await asyncio.wait_for(dispatch, timeout=runtime_voice_turn_timeout_s)
            else:
                payload = await dispatch
            RuntimeVoiceTurnResponse.model_validate(payload)
            return JSONResponse(payload, headers=_NO_STORE_HEADERS)
        except TimeoutError:
            return JSONResponse(
                {
                    "error": "runtime voice-turn timed out",
                    "timeout_s": runtime_voice_turn_timeout_s,
                },
                status_code=504,
                headers=_CORS_HEADERS,
            )
        except TurnInProgress as exc:
            return JSONResponse(
                {
                    "error": "conversation turn in progress",
                    "conversation_thread_id": exc.thread_id,
                    "blocking_turn_id": exc.blocking_turn_id,
                },
                status_code=409,
                headers=_CORS_HEADERS,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400, headers=_CORS_HEADERS)
        except Exception as exc:
            logger.exception("Runtime voice-turn endpoint failed")
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @router.options("/api/chat", include_in_schema=False)
    async def chat_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @router.options("/api/runtime/voice-turn", include_in_schema=False)
    async def runtime_voice_turn_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    return router


async def _chat_runtime_target_available(
    *,
    configured: bool,
    dispatch_runtime: DispatchRuntime,
) -> bool:
    if not configured:
        return False
    try:
        context = dispatch_runtime("runtime_context_payload")
        if isawaitable(context):
            await context
    except RuntimeError as exc:
        if str(exc).strip().lower() == "runtime handler not configured":
            return False
        return True
    except Exception:
        # An unhealthy configured runtime still requires authorization.  The
        # HealthModule control path will deny the mutation if provenance or
        # target validation cannot be completed.
        return True
    return True


def _request_trace_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id") or request.headers.get("x-askme-trace-id")
    cleaned = "".join(
        char for char in str(supplied or "").strip()[:128] if char.isalnum() or char in "-_.:"
    )
    return cleaned or f"chat-{secrets.token_hex(8)}"


def _with_trace(headers: dict[str, str], trace_id: str) -> dict[str, str]:
    return {**headers, "X-Askme-Trace-Id": trace_id}


def _clean_optional_text(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None
