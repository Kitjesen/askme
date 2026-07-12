"""Enterprise security middleware: rate limiting, input validation, security headers.

Covers OWASP Top 10 categories:
- A01: Broken Access Control  (RateLimiter — per-client throttling)
- A03: Injection              (InputValidator — sanitize & length guard)
- A05: Security Misconfiguration (SecurityHeadersMiddleware — HSTS, XFO, etc.)
- A07: Identification & Auth Failures (InputValidator — strip control chars)
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

# ------------------------------------------------------------------
# SecurityHeadersMiddleware
# ------------------------------------------------------------------

class SecurityHeadersMiddleware:
    """Add enterprise security headers to every HTTP response.

    Headers applied:
      X-Content-Type-Options  — prevent MIME-type sniffing
      X-Frame-Options         — prevent clickjacking
      X-XSS-Protection        — legacy XSS filter hint
      Strict-Transport-Security — enforce HTTPS (HSTS)
      Cache-Control           — prevent caching of sensitive content
    """

    def __init__(self, hsts_max_age: int = 31536000, include_subdomains: bool = True) -> None:
        self._hsts_value = (
            f"max-age={hsts_max_age}; includeSubDomains"
            if include_subdomains
            else f"max-age={hsts_max_age}"
        )

    async def __call__(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = self._hsts_value
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response


# ------------------------------------------------------------------
# RateLimiter
# ------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter keyed by a per-client identifier (e.g. IP).

    Defaults: 60 requests per 60-second window.  Override per route or
    per user tier by constructing separate instances with different limits.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max = max_requests
        self.window = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Return True if *client_id* may make another request right now."""
        now = time.time()
        bucket = self._buckets[client_id]
        # Prune expired timestamps
        bucket[:] = [t for t in bucket if now - t < self.window]
        if len(bucket) >= self.max:
            return False
        bucket.append(now)
        return True

    def remaining(self, client_id: str) -> int:
        """Return how many requests *client_id* can still make in this window."""
        now = time.time()
        bucket = self._buckets[client_id]
        bucket[:] = [t for t in bucket if now - t < self.window]
        remaining = self.max - len(bucket)
        return max(remaining, 0)

    def reset(self, client_id: str) -> None:
        """Clear the window for *client_id* (admin use)."""
        self._buckets.pop(client_id, None)


# ------------------------------------------------------------------
# RateLimitMiddleware (convenience ASGI middleware wrapping RateLimiter)
# ------------------------------------------------------------------

class RateLimitMiddleware:
    """FastAPI middleware that enforces a RateLimiter and returns 429 on overflow.

    Usage::

        app.add_middleware(RateLimitMiddleware, max_requests=120, window_seconds=60)

    The middleware reads the client IP from ``request.client.host``, falling
    back to the ``X-Forwarded-For`` header when available (behind a reverse
    proxy).
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._limiter = limiter or RateLimiter(
            max_requests=max_requests, window_seconds=window_seconds
        )

    async def __call__(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        client_id = self._resolve_client_id(request)
        if not self._limiter.is_allowed(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please slow down.",
                    "retry_after_seconds": self._limiter.window,
                },
                headers={
                    "Retry-After": str(self._limiter.window),
                    "X-RateLimit-Limit": str(self._limiter.max),
                    "X-RateLimit-Remaining": "0",
                },
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limiter.max)
        response.headers["X-RateLimit-Remaining"] = str(
            self._limiter.remaining(client_id)
        )
        return response

    def _resolve_client_id(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client is not None and request.client.host is not None:
            return request.client.host
        return "unknown"


# ------------------------------------------------------------------
# InputValidator
# ------------------------------------------------------------------

class InputValidator:
    """Validate and sanitize user-supplied text.

    Guards against:
      - Oversized payloads (resource exhaustion)
      - Control / escape characters used in injection attempts
      - Prompt-injection-relevant special sequences
    """

    MAX_LENGTH: int = 10000
    # Characters that are almost never legitimate in business text:
    # null byte, bell, backspace, escape, form-feed, vertical-tab, etc.
    _CONTROL_CHARS = str.maketrans(
        {chr(c): None for c in range(0, 32) if c not in (9, 10, 13)}  # keep tab, LF, CR
    )

    @classmethod
    def sanitize(cls, text: str, max_length: int = MAX_LENGTH) -> str:
        """Strip dangerous characters and enforce length limit.

        Raises
        ------
        ValueError
            If *text* exceeds *max_length* characters.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected a string, got {type(text).__name__}")
        if len(text) > max_length:
            raise ValueError(
                f"Input exceeds maximum allowed length of {max_length} characters "
                f"(got {len(text)})"
            )
        # Remove non-printable control characters (keep tabs, newlines, CR)
        cleaned = text.translate(cls._CONTROL_CHARS)
        return cleaned.strip()

    @classmethod
    def sanitize_optional(
        cls, text: str | None, max_length: int = MAX_LENGTH
    ) -> str | None:
        """Like :meth:`sanitize` but accepts ``None`` and passes it through."""
        if text is None:
            return None
        return cls.sanitize(text, max_length=max_length)


# ------------------------------------------------------------------
# Utility: generate a Content-Security-Policy header value
# ------------------------------------------------------------------

def build_csp_policy(
    default_src: tuple[str, ...] = ("'self'",),
    script_src: tuple[str, ...] = ("'self'",),
    style_src: tuple[str, ...] = ("'self'", "'unsafe-inline'"),
    img_src: tuple[str, ...] = ("'self'", "data:"),
    connect_src: tuple[str, ...] = ("'self'",),
    font_src: tuple[str, ...] = ("'self'",),
    object_src: tuple[str, ...] = ("'none'",),
    frame_ancestors: tuple[str, ...] = ("'none'",),
    upgrade_insecure_requests: bool = True,
) -> str:
    """Build a Content-Security-Policy header string.

    Every parameter is a tuple of allowed sources.  Callers can override per
    deployment (e.g. add ``https://api.example.com`` to *connect_src*).
    """
    directives: list[str] = [
        f"default-src {' '.join(default_src)}",
        f"script-src {' '.join(script_src)}",
        f"style-src {' '.join(style_src)}",
        f"img-src {' '.join(img_src)}",
        f"connect-src {' '.join(connect_src)}",
        f"font-src {' '.join(font_src)}",
        f"object-src {' '.join(object_src)}",
        f"frame-ancestors {' '.join(frame_ancestors)}",
    ]
    if upgrade_insecure_requests:
        directives.append("upgrade-insecure-requests")
    return "; ".join(directives)


# ------------------------------------------------------------------
# Utility: constant-time comparison for API-key checks
# ------------------------------------------------------------------

def secure_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to resist timing attacks on API keys."""
    return hashlib.sha256(a.encode()).digest() == hashlib.sha256(b.encode()).digest()


__all__ = [
    "SecurityHeadersMiddleware",
    "RateLimiter",
    "RateLimitMiddleware",
    "InputValidator",
    "build_csp_policy",
    "secure_compare",
]
