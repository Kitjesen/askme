"""Conversation FastAPI routes."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from askme.api.services.conversation_service import (
    ChatOverloaded,
    ChatTimeout,
    ChatUnavailable,
    ConversationService,
    EmptyChatText,
)

DispatchRuntime = Callable[..., Awaitable[dict[str, Any]]]
CorsOptions = Callable[[str], Response]

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
    runtime_voice_turn_timeout_s: float | None = 30.0,
) -> None:
    """Register chat and voice-turn routes."""

    @app.post("/api/chat", tags=["Monitor"])
    async def chat(request: Request) -> JSONResponse:
        """Send text to the brain pipeline and return the response."""
        trace_id = _request_trace_id(request)
        trace_headers = _with_trace(_CORS_HEADERS, trace_id)
        if not conversation_service.chat_available:
            return JSONResponse(
                {"error": "chat not available"},
                status_code=503,
                headers=trace_headers,
            )
        try:
            body = await request.json()
            payload = await conversation_service.chat_payload_from_body(body, trace_id=trace_id)
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
        except ChatUnavailable as exc:
            return JSONResponse({"error": str(exc)}, status_code=503, headers=trace_headers)
        except Exception as exc:
            logger.error("Chat endpoint failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=trace_headers)

    @app.get("/api/conversation/diagnostics", tags=["Monitor"])
    async def conversation_diagnostics() -> JSONResponse:
        """Return non-sensitive chat execution diagnostics."""
        return JSONResponse(conversation_service.diagnostics_snapshot(), headers=_NO_STORE_HEADERS)

    @app.post("/api/runtime/voice-turn", tags=["Runtime"])
    async def runtime_voice_turn(request: Request) -> JSONResponse:
        """Route a final voice transcript to runtime controls only."""
        if not runtime_available:
            return JSONResponse(
                {"error": "runtime handler not configured"},
                status_code=503,
                headers=_CORS_HEADERS,
            )
        try:
            body = await request.json()
            raw_text = body.get("text") or body.get("message") or body.get("transcript") or ""
            text = str(raw_text).strip()
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
            dispatch = dispatch_runtime(
                "voice_turn_payload",
                text,
                speak=bool(body.get("speak") or body.get("play_audio")),
                transcript_id=str(body.get("transcript_id") or ""),
                confidence=body.get("asr_confidence", body.get("confidence")),
                is_final=bool(body.get("is_final", True)),
                channel=str(body.get("channel") or "voice"),
            )
            if runtime_voice_turn_timeout_s is not None and runtime_voice_turn_timeout_s > 0:
                payload = await asyncio.wait_for(dispatch, timeout=runtime_voice_turn_timeout_s)
            else:
                payload = await dispatch
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
        except Exception as exc:
            logger.error("Runtime voice-turn endpoint failed: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500, headers=_CORS_HEADERS)

    @app.options("/api/chat", include_in_schema=False)
    async def chat_cors() -> Response:
        return cors_options_response("POST, OPTIONS")

    @app.options("/api/runtime/voice-turn", include_in_schema=False)
    async def runtime_voice_turn_cors() -> Response:
        return cors_options_response("POST, OPTIONS")


def _request_trace_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id") or request.headers.get("x-askme-trace-id")
    cleaned = "".join(
        char for char in str(supplied or "").strip()[:128] if char.isalnum() or char in "-_.:"
    )
    return cleaned or f"chat-{secrets.token_hex(8)}"


def _with_trace(headers: dict[str, str], trace_id: str) -> dict[str, str]:
    return {**headers, "X-Askme-Trace-Id": trace_id}
