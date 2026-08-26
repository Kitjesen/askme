from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
import requests

from askme.ports.runtime_executor import (
    AmbiguousRuntimeSubmissionError,
    RuntimeExecutorCancelRequest,
    RuntimeExecutorStatusRequest,
    RuntimeExecutorSubmitRequest,
    RuntimeExecutorTransport,
    RuntimeExecutorTransportError,
)
from askme.providers.runtime_executor import HttpRuntimeExecutorTransport


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        body: Any = None,
        *,
        raw: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._raw = raw if raw is not None else json.dumps(body or {}).encode()
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        del chunk_size
        yield self._raw

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _task_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "remote_task_id": "remote-1",
        "status": "queued",
        "cursor": "cursor-1",
        "result_summary": "",
        "observed_at": 123.5,
        "updates": [],
    }
    body.update(overrides)
    return body


def _transport(session: Any, **kwargs: Any) -> HttpRuntimeExecutorTransport:
    return HttpRuntimeExecutorTransport(
        base_url="https://runtime.example.test",
        session=session,
        retry_delay_s=0,
        **kwargs,
    )


def _submit_request() -> RuntimeExecutorSubmitRequest:
    return RuntimeExecutorSubmitRequest(
        handoff={"handoff_id": "handoff-1", "plan_id": "plan-1"},
        idempotency_key="idem-1",
        correlation_id="corr-1",
        thread_id="session-1",
        turn_id="turn-1",
    )


def test_protocol_and_dtos_are_stable_and_immutable() -> None:
    transport = _transport(_FakeSession([_FakeResponse(body=_task_body())]))
    assert isinstance(transport, RuntimeExecutorTransport)
    request = _submit_request()
    assert request.thread_id == "session-1"
    assert request.turn_id == "turn-1"
    assert not hasattr(request, "conversation_session_id")
    assert not hasattr(request, "originating_turn_id")
    with pytest.raises(FrozenInstanceError):
        request.correlation_id = "changed"  # type: ignore[misc]


def test_submit_serializes_contract_and_resolves_credentials_at_call_time(monkeypatch) -> None:
    session = _FakeSession([_FakeResponse(body=_task_body())])
    transport = _transport(session, credential_env_var="ASKME_TEST_RUNTIME_TOKEN")
    monkeypatch.setenv("ASKME_TEST_RUNTIME_TOKEN", "secret-value")

    result = transport.submit(_submit_request())

    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == "https://runtime.example.test/v1/tasks"
    assert kwargs["json"] == {
        "handoff": {"handoff_id": "handoff-1", "plan_id": "plan-1"},
        "conversation_session_id": "session-1",
        "originating_turn_id": "turn-1",
    }
    assert kwargs["headers"]["Authorization"] == "Bearer secret-value"
    assert kwargs["headers"]["Idempotency-Key"] == "idem-1"
    assert kwargs["headers"]["X-Correlation-ID"] == "corr-1"
    assert kwargs["allow_redirects"] is False
    assert kwargs["stream"] is True
    assert result.remote_task_id == "remote-1"
    assert result.idempotency_key == "idem-1"


def test_missing_configured_credential_fails_before_network(monkeypatch) -> None:
    monkeypatch.delenv("ASKME_TEST_MISSING_TOKEN", raising=False)
    session = _FakeSession([])
    transport = _transport(session, credential_env_var="ASKME_TEST_MISSING_TOKEN")

    with pytest.raises(RuntimeExecutorTransportError, match="environment variable") as raised:
        transport.submit(_submit_request())

    assert raised.value.kind == "missing_credentials"
    assert session.calls == []


def test_status_uses_encoded_task_id_and_cursor_and_normalizes_updates() -> None:
    response = _task_body(
        remote_task_id="remote / 1",
        status="working",
        cursor="next",
        result_summary="halfway",
        updates=[
            {
                "event_id": "event-1",
                "status": "working",
                "message": "moving",
                "cursor": "next",
                "observed_at": 124,
                "payload": {
                    "progress": 0.5,
                    "observation": {"type": "position", "area": "A"},
                    "artifacts": [{"type": "image_ref", "uri": "s3://evidence/a.jpg"}],
                },
            }
        ],
    )
    session = _FakeSession([_FakeResponse(body=response)])

    result = _transport(session).get_status(
        RuntimeExecutorStatusRequest(
            remote_task_id="remote / 1",
            correlation_id="corr-1",
            cursor="old cursor",
        )
    )

    assert session.calls[0][1].endswith("/v1/tasks/remote%20%2F%201?cursor=old+cursor")
    assert result.status == "executing"
    assert result.result_summary == "halfway"
    assert result.updates[0].payload == {
        "progress": 0.5,
        "observation": {"type": "position", "area": "A"},
        "artifacts": ({"type": "image_ref", "uri": "s3://evidence/a.jpg"},),
    }


def test_cancel_uses_its_own_idempotency_key() -> None:
    session = _FakeSession([_FakeResponse(body=_task_body(status="canceled"))])
    result = _transport(session).cancel(
        RuntimeExecutorCancelRequest(
            remote_task_id="remote-1",
            idempotency_key="cancel-1",
            correlation_id="corr-2",
            reason="operator request",
        )
    )

    method, url, kwargs = session.calls[0]
    assert (method, url) == (
        "POST",
        "https://runtime.example.test/v1/tasks/remote-1/cancel",
    )
    assert kwargs["headers"]["Idempotency-Key"] == "cancel-1"
    assert kwargs["json"] == {"reason": "operator request"}
    assert result.status == "cancelled"


def test_transient_submit_retry_reuses_exact_idempotency_key() -> None:
    session = _FakeSession(
        [_FakeResponse(status_code=503), _FakeResponse(body=_task_body(status="submitted"))]
    )

    result = _transport(session).submit(_submit_request())

    assert result.status == "submitted"
    assert len(session.calls) == 2
    assert {call[2]["headers"]["Idempotency-Key"] for call in session.calls} == {"idem-1"}
    assert session.calls[0][2]["json"] == session.calls[1][2]["json"]


def test_submit_exhaustion_is_explicitly_ambiguous() -> None:
    session = _FakeSession([requests.Timeout("slow"), requests.Timeout("slow")])
    transport = _transport(session, max_retries=1)

    with pytest.raises(AmbiguousRuntimeSubmissionError) as raised:
        transport.submit(_submit_request())

    assert raised.value.ambiguous is True
    assert raised.value.retryable is True
    assert len(session.calls) == 2


def test_client_error_is_not_retried_or_marked_ambiguous() -> None:
    session = _FakeSession([_FakeResponse(status_code=422)])

    with pytest.raises(RuntimeExecutorTransportError) as raised:
        _transport(session).submit(_submit_request())

    assert raised.value.status_code == 422
    assert raised.value.retryable is False
    assert raised.value.ambiguous is False
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (_FakeResponse(raw=b"not-json"), "invalid_json"),
        (_FakeResponse(raw=b"[]"), "invalid_response"),
        (_FakeResponse(body=_task_body(status="invented")), "invalid_status"),
        (_FakeResponse(body=_task_body(correlation_id="wrong")), "correlation_mismatch"),
    ],
)
def test_invalid_submit_responses_are_ambiguous_after_remote_acceptance(
    response: _FakeResponse, kind: str
) -> None:
    with pytest.raises(AmbiguousRuntimeSubmissionError) as raised:
        _transport(_FakeSession([response])).submit(_submit_request())
    assert raised.value.ambiguous is True
    assert isinstance(raised.value.__cause__, RuntimeExecutorTransportError)
    assert raised.value.__cause__.kind == kind


def test_response_size_is_bounded_even_without_content_length() -> None:
    response = _FakeResponse(raw=b"x" * 33)
    with pytest.raises(AmbiguousRuntimeSubmissionError) as raised:
        _transport(_FakeSession([response]), max_response_bytes=32).submit(_submit_request())
    assert isinstance(raised.value.__cause__, RuntimeExecutorTransportError)
    assert raised.value.__cause__.kind == "response_too_large"


def test_base_url_requires_https_except_explicit_loopback() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        HttpRuntimeExecutorTransport(base_url="http://runtime.example.test")
    with pytest.raises(ValueError, match="credentials"):
        HttpRuntimeExecutorTransport(base_url="https://user:pass@example.test")
    HttpRuntimeExecutorTransport(base_url="http://127.0.0.1:8123")
    HttpRuntimeExecutorTransport(base_url="http://[::1]:8123")


class _LoopbackHandler(BaseHTTPRequestHandler):
    mode = "normal"
    requests_seen: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        if self.mode == "timeout":
            time.sleep(0.2)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        self.requests_seen.append(
            {"method": "POST", "path": self.path, "headers": dict(self.headers), "raw": raw}
        )
        if self.mode == "redirect":
            self.send_response(307)
            self.send_header("Location", "http://example.test/v1/tasks")
            self.end_headers()
            return
        if self.mode == "malformed":
            self._write(200, b"{broken")
            return
        if self.mode == "oversize":
            self._write(200, b"x" * 256)
            return
        self._write(200, json.dumps(_task_body(status="submitted")).encode())

    def _write(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@contextmanager
def _loopback_server(mode: str = "normal") -> Iterator[tuple[str, type[_LoopbackHandler]]]:
    handler = type("Handler", (_LoopbackHandler,), {"mode": mode, "requests_seen": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_loopback_http_server_proves_wire_serialization_and_headers(monkeypatch) -> None:
    monkeypatch.setenv("ASKME_LOOPBACK_TOKEN", "wire-secret")
    with _loopback_server() as (base_url, handler):
        transport = HttpRuntimeExecutorTransport(
            base_url=base_url,
            credential_env_var="ASKME_LOOPBACK_TOKEN",
            retry_delay_s=0,
        )
        result = transport.submit(_submit_request())

    seen = handler.requests_seen[0]
    assert seen["path"] == "/v1/tasks"
    assert json.loads(seen["raw"]) == {
        "handoff": {"handoff_id": "handoff-1", "plan_id": "plan-1"},
        "conversation_session_id": "session-1",
        "originating_turn_id": "turn-1",
    }
    assert seen["headers"]["Authorization"] == "Bearer wire-secret"
    assert seen["headers"]["Idempotency-Key"] == "idem-1"
    assert result.status == "submitted"


@pytest.mark.parametrize(
    ("mode", "expected_kind", "max_bytes", "ambiguous"),
    [
        ("malformed", "invalid_json", 1024, True),
        ("oversize", "response_too_large", 64, True),
        ("redirect", "cross_host_redirect", 1024, False),
    ],
)
def test_loopback_http_server_rejects_bad_wire_responses(
    mode: str, expected_kind: str, max_bytes: int, ambiguous: bool
) -> None:
    with _loopback_server(mode) as (base_url, _handler):
        transport = HttpRuntimeExecutorTransport(
            base_url=base_url,
            max_response_bytes=max_bytes,
            retry_delay_s=0,
        )
        with pytest.raises(RuntimeExecutorTransportError) as raised:
            transport.submit(_submit_request())
    if ambiguous:
        assert isinstance(raised.value, AmbiguousRuntimeSubmissionError)
        assert isinstance(raised.value.__cause__, RuntimeExecutorTransportError)
        assert raised.value.__cause__.kind == expected_kind
    else:
        assert raised.value.kind == expected_kind


def test_loopback_http_server_enforces_basic_read_timeout() -> None:
    with _loopback_server("timeout") as (base_url, _handler):
        transport = HttpRuntimeExecutorTransport(
            base_url=base_url,
            connect_timeout_s=0.05,
            read_timeout_s=0.05,
            total_timeout_s=0.12,
            max_retries=0,
        )
        with pytest.raises(AmbiguousRuntimeSubmissionError):
            transport.submit(_submit_request())
