"""FastAPI global exception handlers for structured error responses.

Registers two handlers on a FastAPI application:

* ``AppError`` — mapped to a JSON body with ``error.code``, ``error.message``
  and the HTTP status carried by the exception.
* ``Exception`` (catch-all) — logs the full traceback and returns a generic 500.

Usage::

    from askme.api.middleware.error_handler import register_error_handlers
    from fastapi import FastAPI

    app = FastAPI(...)
    register_error_handlers(app)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from askme.errors.taxonomy import AppError

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Install global exception handlers on *app*.

    Call once during application setup, after the ``FastAPI(...)`` constructor
    and before route registration.
    """

    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "AppError %s [%s/%s]: %s",
            exc.code,
            exc.category.value if exc.category else "unknown",
            exc.severity.value if exc.severity else "unknown",
            exc.message,
        )
        payload: dict[str, Any] = {
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        }
        if exc.details is not None:
            payload["error"]["details"] = exc.details
        return JSONResponse(payload, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "Unhandled exception handling %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            {"error": {"code": "internal_error", "message": "Internal server error"}},
            status_code=500,
        )


__all__ = ["register_error_handlers"]
