"""Fail-closed HTTP adapter for an external runtime task executor."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote, urlencode, urljoin, urlsplit

import requests

from askme.ports.runtime_executor import (
    AmbiguousRuntimeSubmissionError,
    RuntimeExecutorCancelRequest,
    RuntimeExecutorCancelResult,
    RuntimeExecutorStatusRequest,
    RuntimeExecutorStatusUpdate,
    RuntimeExecutorSubmitRequest,
    RuntimeExecutorSubmitResult,
    RuntimeExecutorTransportError,
    RuntimeExecutorUpdate,
)

_ALLOWED_STATUSES = frozenset(
    {
        "submitted",
        "created",
        "validating",
        "preflight",
        "queued",
        "executing",
        "paused",
        "resuming",
        "input_required",
        "auth_required",
        "blocked",
        "cancelling",
        "cancelled",
        "completed",
        "failed",
        "rejected",
        "shadowed",
    }
)
_STATUS_ALIASES = {"working": "executing", "canceled": "cancelled"}
_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_REDIRECT_HTTP_STATUSES = frozenset({301, 302, 303, 307, 308})


def build_runtime_executor_transport(
    profile: str,
    config: Mapping[str, Any],
) -> HttpRuntimeExecutorTransport | None:
    """Build the external executor adapter from one runtime profile config."""
    normalized_profile = str(profile or "").strip().lower()
    if normalized_profile not in {"external", "lab"}:
        return None
    if not bool(config.get("enable_external_runtime", False)):
        return None
    endpoint = str(config.get("endpoint") or "").strip()
    if not endpoint:
        return None

    parsed_endpoint = urlsplit(endpoint)
    if normalized_profile == "external" and parsed_endpoint.scheme.lower() != "https":
        raise ValueError("external runtime endpoint must use HTTPS")

    credential_env_var = str(
        config.get("credential_env_var") or config.get("auth_token_env") or ""
    ).strip()
    hostname = (parsed_endpoint.hostname or "").lower()
    authless_lab_loopback = normalized_profile == "lab" and hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
    if not credential_env_var and not authless_lab_loopback:
        raise ValueError(
            "external runtime credential_env_var is required outside authless lab loopback"
        )

    timeout_s = float(config.get("timeout_seconds", 5.0))
    connect_timeout_s = float(config.get("connect_timeout_seconds", min(2.0, timeout_s)))
    return HttpRuntimeExecutorTransport(
        base_url=endpoint,
        credential_env_var=credential_env_var,
        connect_timeout_s=connect_timeout_s,
        read_timeout_s=float(config.get("read_timeout_seconds", timeout_s)),
        total_timeout_s=float(
            config.get("total_timeout_seconds", max(timeout_s, connect_timeout_s + timeout_s))
        ),
        max_response_bytes=int(config.get("max_response_bytes", 1_048_576)),
        max_retries=int(config.get("max_retries", 1)),
        retry_delay_s=float(config.get("retry_delay_seconds", 0.1)),
        max_redirects=int(config.get("max_redirects", 0)),
    )


class HttpRuntimeExecutorTransport:
    """Bounded and fail-closed implementation of the executor HTTP contract."""

    def __init__(
        self,
        *,
        base_url: str,
        credential_env_var: str = "",
        session: Any | None = None,
        connect_timeout_s: float = 1.0,
        read_timeout_s: float = 5.0,
        total_timeout_s: float = 8.0,
        max_response_bytes: int = 1_048_576,
        max_retries: int = 2,
        retry_delay_s: float = 0.05,
        max_redirects: int = 2,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        self._credential_env_var = str(credential_env_var or "").strip()
        self._session = session if session is not None else requests.Session()
        self._connect_timeout_s = _positive_float(connect_timeout_s, "connect_timeout_s")
        self._read_timeout_s = _positive_float(read_timeout_s, "read_timeout_s")
        self._total_timeout_s = _positive_float(total_timeout_s, "total_timeout_s")
        self._max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self._max_retries = max(0, int(max_retries))
        self._retry_delay_s = max(0.0, float(retry_delay_s))
        self._max_redirects = max(0, int(max_redirects))
        self._clock = clock
        self._sleep = sleep

    def submit(self, request: RuntimeExecutorSubmitRequest) -> RuntimeExecutorSubmitResult:
        idempotency_key = _required_text(request.idempotency_key, "idempotency_key")
        correlation_id = _required_text(request.correlation_id, "correlation_id")
        if not isinstance(request.handoff, Mapping) or not request.handoff:
            raise RuntimeExecutorTransportError(
                "invalid_request", "handoff must be a non-empty object"
            )
        payload = {
            "handoff": _json_object_copy(request.handoff, "handoff"),
            # The external v1 wire contract retains its legacy aliases.  The
            # provider adapter is the only layer that translates canonical IDs.
            "conversation_session_id": str(request.thread_id or ""),
            "originating_turn_id": str(request.turn_id or ""),
        }
        try:
            body = self._request_json(
                "POST",
                "/v1/tasks",
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                payload=payload,
                submission=True,
            )
            parsed = _parse_task_response(body, correlation_id=correlation_id)
        except AmbiguousRuntimeSubmissionError:
            raise
        except RuntimeExecutorTransportError as exc:
            if exc.kind in {
                "correlation_mismatch",
                "invalid_json",
                "invalid_response",
                "invalid_status",
                "remote_task_id_mismatch",
                "response_too_large",
            }:
                raise AmbiguousRuntimeSubmissionError(
                    "runtime accepted the submission request but returned an unusable response; "
                    "reconcile using the same idempotency key",
                    status_code=exc.status_code,
                ) from exc
            raise
        return RuntimeExecutorSubmitResult(
            **parsed,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def get_status(self, request: RuntimeExecutorStatusRequest) -> RuntimeExecutorStatusUpdate:
        remote_task_id = _required_text(request.remote_task_id, "remote_task_id")
        correlation_id = _required_text(request.correlation_id, "correlation_id")
        query = urlencode({"cursor": str(request.cursor)}) if request.cursor else ""
        path = f"/v1/tasks/{quote(remote_task_id, safe='')}"
        if query:
            path = f"{path}?{query}"
        body = self._request_json("GET", path, correlation_id=correlation_id)
        parsed = _parse_task_response(
            body,
            correlation_id=correlation_id,
            expected_remote_task_id=remote_task_id,
        )
        return RuntimeExecutorStatusUpdate(**parsed, correlation_id=correlation_id)

    def cancel(self, request: RuntimeExecutorCancelRequest) -> RuntimeExecutorCancelResult:
        remote_task_id = _required_text(request.remote_task_id, "remote_task_id")
        idempotency_key = _required_text(request.idempotency_key, "idempotency_key")
        correlation_id = _required_text(request.correlation_id, "correlation_id")
        path = f"/v1/tasks/{quote(remote_task_id, safe='')}/cancel"
        body = self._request_json(
            "POST",
            path,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            payload={"reason": str(request.reason or "")},
        )
        parsed = _parse_task_response(
            body,
            correlation_id=correlation_id,
            expected_remote_task_id=remote_task_id,
        )
        return RuntimeExecutorCancelResult(
            **parsed,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        correlation_id: str,
        idempotency_key: str = "",
        payload: Mapping[str, Any] | None = None,
        submission: bool = False,
    ) -> dict[str, Any]:
        headers = self._headers(
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )
        deadline = self._clock() + self._total_timeout_s
        url = urljoin(f"{self._base_url}/", path.lstrip("/"))
        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._request_once(
                    method,
                    url,
                    headers=headers,
                    payload=payload,
                    deadline=deadline,
                )
            except RuntimeExecutorTransportError as exc:
                last_status = exc.status_code
                if not exc.retryable or attempt >= self._max_retries:
                    if submission and (exc.retryable or exc.ambiguous):
                        raise AmbiguousRuntimeSubmissionError(
                            "runtime submission outcome is unknown; reconcile using the same idempotency key",
                            status_code=last_status,
                        ) from exc
                    raise
                remaining = deadline - self._clock()
                if remaining <= 0:
                    break
                self._sleep(min(self._retry_delay_s * (2**attempt), remaining))
        if submission:
            raise AmbiguousRuntimeSubmissionError(
                "runtime submission exceeded its total deadline; reconcile using the same idempotency key",
                status_code=last_status,
            )
        raise RuntimeExecutorTransportError(
            "total_timeout",
            "runtime executor request exceeded its total deadline",
            status_code=last_status,
            retryable=True,
        )

    def _request_once(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
        deadline: float,
    ) -> dict[str, Any]:
        redirects = 0
        current_url = url
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise RuntimeExecutorTransportError(
                    "total_timeout",
                    "runtime executor request exceeded its total deadline",
                    retryable=True,
                )
            timeout = (
                min(self._connect_timeout_s, remaining),
                min(self._read_timeout_s, remaining),
            )
            kwargs: dict[str, Any] = {
                "headers": dict(headers),
                "timeout": timeout,
                "allow_redirects": False,
                "stream": True,
            }
            if payload is not None:
                kwargs["json"] = dict(payload)
            try:
                response = self._session.request(method, current_url, **kwargs)
            except requests.Timeout as exc:
                raise RuntimeExecutorTransportError(
                    "timeout", "runtime executor request timed out", retryable=True
                ) from exc
            except requests.RequestException as exc:
                raise RuntimeExecutorTransportError(
                    "network_error", "runtime executor network request failed", retryable=True
                ) from exc
            except (OSError, TimeoutError) as exc:
                raise RuntimeExecutorTransportError(
                    "network_error", "runtime executor network request failed", retryable=True
                ) from exc

            status_code = int(getattr(response, "status_code", 0) or 0)
            if status_code in _REDIRECT_HTTP_STATUSES:
                location = str(getattr(response, "headers", {}).get("Location") or "").strip()
                _close_response(response)
                if not location:
                    raise RuntimeExecutorTransportError(
                        "invalid_redirect", "runtime executor redirect omitted Location"
                    )
                redirect_url = urljoin(current_url, location)
                if not _same_origin(current_url, redirect_url):
                    raise RuntimeExecutorTransportError(
                        "cross_host_redirect", "runtime executor cross-host redirect was rejected"
                    )
                redirects += 1
                if redirects > self._max_redirects:
                    raise RuntimeExecutorTransportError(
                        "too_many_redirects", "runtime executor exceeded the redirect limit"
                    )
                if method != "GET" and status_code not in {307, 308}:
                    raise RuntimeExecutorTransportError(
                        "unsafe_redirect", "runtime executor refused a method-changing redirect"
                    )
                current_url = redirect_url
                continue

            if not 200 <= status_code < 300:
                _close_response(response)
                retryable = status_code in _TRANSIENT_HTTP_STATUSES
                raise RuntimeExecutorTransportError(
                    "http_error",
                    f"runtime executor returned HTTP {status_code}",
                    status_code=status_code,
                    retryable=retryable,
                    ambiguous=method == "POST" and retryable,
                )
            try:
                raw = _read_bounded_response(
                    response,
                    max_bytes=self._max_response_bytes,
                    deadline=deadline,
                    clock=self._clock,
                )
            finally:
                _close_response(response)
            if not raw:
                raise RuntimeExecutorTransportError(
                    "invalid_response", "runtime executor returned an empty response"
                )
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeExecutorTransportError(
                    "invalid_json", "runtime executor returned malformed JSON"
                ) from exc
            if not isinstance(body, dict):
                raise RuntimeExecutorTransportError(
                    "invalid_response", "runtime executor response must be a JSON object"
                )
            return body

    def _headers(self, *, correlation_id: str, idempotency_key: str) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Correlation-ID": correlation_id,
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if self._credential_env_var:
            credential = os.environ.get(self._credential_env_var, "").strip()
            if not credential:
                raise RuntimeExecutorTransportError(
                    "missing_credentials",
                    f"runtime executor credential environment variable {self._credential_env_var!r} is unset",
                )
            headers["Authorization"] = f"Bearer {credential}"
        return headers


def _parse_task_response(
    body: Mapping[str, Any],
    *,
    correlation_id: str,
    expected_remote_task_id: str = "",
) -> dict[str, Any]:
    remote_task_id = _required_text(
        body.get("remote_task_id") or body.get("task_id"), "remote_task_id"
    )
    if expected_remote_task_id and remote_task_id != expected_remote_task_id:
        raise RuntimeExecutorTransportError(
            "correlation_mismatch", "runtime executor returned a different remote task id"
        )
    response_correlation = str(body.get("correlation_id") or correlation_id).strip()
    if response_correlation != correlation_id:
        raise RuntimeExecutorTransportError(
            "correlation_mismatch", "runtime executor returned a different correlation id"
        )
    raw_updates = body.get("updates", [])
    if not isinstance(raw_updates, list):
        raise RuntimeExecutorTransportError(
            "invalid_response", "runtime executor updates must be an array"
        )
    updates = tuple(_parse_update(item) for item in raw_updates)
    observed_at = _optional_number(body.get("observed_at"), "observed_at")
    return {
        "remote_task_id": remote_task_id,
        "status": _normalize_status(body.get("status")),
        "cursor": str(body.get("cursor") or ""),
        "result_summary": str(body.get("result_summary") or ""),
        "updates": updates,
        "observed_at": observed_at,
    }


def _parse_update(value: Any) -> RuntimeExecutorUpdate:
    if not isinstance(value, dict):
        raise RuntimeExecutorTransportError(
            "invalid_response", "runtime executor update must be an object"
        )
    payload = value.get("payload", {})
    if not isinstance(payload, dict):
        raise RuntimeExecutorTransportError(
            "invalid_response", "runtime executor update payload must be an object"
        )
    return RuntimeExecutorUpdate(
        event_id=_required_text(value.get("event_id"), "updates.event_id"),
        status=_normalize_status(value.get("status")),
        message=str(value.get("message") or ""),
        cursor=str(value.get("cursor") or ""),
        observed_at=_optional_number(value.get("observed_at"), "updates.observed_at"),
        payload=dict(payload),
    )


def _normalize_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    status = _STATUS_ALIASES.get(status, status)
    if status not in _ALLOWED_STATUSES:
        raise RuntimeExecutorTransportError(
            "invalid_status", f"runtime executor returned unsupported status {status!r}"
        )
    return status


def _read_bounded_response(
    response: Any,
    *,
    max_bytes: int,
    deadline: float,
    clock: Callable[[], float],
) -> bytes:
    headers = getattr(response, "headers", {})
    content_length = str(headers.get("Content-Length") or "").strip()
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise RuntimeExecutorTransportError(
                "invalid_response", "runtime executor returned invalid Content-Length"
            ) from exc
        if declared_size < 0 or declared_size > max_bytes:
            raise RuntimeExecutorTransportError(
                "response_too_large", "runtime executor response exceeded the size limit"
            )
    chunks: list[bytes] = []
    size = 0
    iterator = getattr(response, "iter_content", None)
    if callable(iterator):
        parts = iterator(chunk_size=min(65_536, max_bytes + 1))
    else:
        parts = (bytes(getattr(response, "content", b"")),)
    for part in parts:
        if clock() > deadline:
            raise RuntimeExecutorTransportError(
                "total_timeout",
                "runtime executor response exceeded its total deadline",
                retryable=True,
            )
        if not part:
            continue
        chunk = bytes(part)
        size += len(chunk)
        if size > max_bytes:
            raise RuntimeExecutorTransportError(
                "response_too_large", "runtime executor response exceeded the size limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_base_url(value: str) -> str:
    base_url = str(value or "").strip().rstrip("/")
    parsed = urlsplit(base_url)
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must be an absolute origin URL without credentials or query")
    if parsed.scheme == "https":
        return base_url
    if parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
        return base_url
    raise ValueError("base_url must use HTTPS except for an explicit loopback host")


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip("[]").lower()
    return normalized in {"localhost", "127.0.0.1", "::1"}


def _same_origin(first: str, second: str) -> bool:
    left = urlsplit(first)
    right = urlsplit(second)
    return (
        left.scheme.lower(),
        left.hostname,
        left.port or (443 if left.scheme.lower() == "https" else 80),
    ) == (
        right.scheme.lower(),
        right.hostname,
        right.port or (443 if right.scheme.lower() == "https" else 80),
    )


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeExecutorTransportError("invalid_request", f"{field_name} is required")
    return text


def _optional_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeExecutorTransportError(
            "invalid_response", f"runtime executor {field_name} must be numeric"
        )
    return float(value)


def _json_object_copy(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=_json_frozen_value)
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise RuntimeExecutorTransportError(
            "invalid_request", f"{field_name} must contain only JSON-compatible values"
        ) from exc
    if not isinstance(copied, dict):
        raise RuntimeExecutorTransportError("invalid_request", f"{field_name} must be an object")
    return copied


def _json_frozen_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _positive_float(value: float, field_name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _positive_int(value: int, field_name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _close_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


__all__ = [
    "HttpRuntimeExecutorTransport",
    "build_runtime_executor_transport",
]
