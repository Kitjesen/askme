from __future__ import annotations

import asyncio
import json

from fastapi.responses import JSONResponse

from askme.api.services.http_helpers import (
    accepted_keyword_args,
    clean_secret,
    is_remote_bind_host,
    json_snapshot_response,
    maybe_await,
    public_error_payload,
    require_json_object,
    snapshot_payload,
)


def test_public_error_payload_keeps_stable_envelope() -> None:
    payload = public_error_payload(
        "not_ready",
        message="Service is not ready.",
        reason="missing_config",
        next_action="Configure the provider.",
    )

    assert payload == {
        "ok": False,
        "error": "not_ready",
        "message": "Service is not ready.",
        "reason": "missing_config",
        "next_action": "Configure the provider.",
    }


def test_require_json_object_accepts_mapping_bodies() -> None:
    body = {"record_id": "know_1"}

    assert require_json_object(body) is body


def test_require_json_object_rejects_non_object_bodies() -> None:
    try:
        require_json_object(["not", "an", "object"])
    except ValueError as exc:
        assert str(exc) == "JSON object body required"
    else:
        raise AssertionError("non-object JSON body should fail")


def test_clean_secret_normalizes_blank_values() -> None:
    assert clean_secret(None) is None
    assert clean_secret("   ") is None
    assert clean_secret("  token  ") == "token"


def test_accepted_keyword_args_filters_to_callable_signature() -> None:
    def limited(alpha: int) -> int:
        return alpha

    def flexible(**kwargs: int) -> dict[str, int]:
        return kwargs

    kwargs = {"alpha": 1, "beta": 2}

    assert accepted_keyword_args(limited, kwargs) == {"alpha": 1}
    assert accepted_keyword_args(flexible, kwargs) == kwargs


def test_maybe_await_accepts_sync_and_async_values() -> None:
    async def value() -> str:
        return "ready"

    assert asyncio.run(maybe_await("sync")) == "sync"
    assert asyncio.run(maybe_await(value())) == "ready"


def test_snapshot_helpers_return_no_store_json_response() -> None:
    response = json_snapshot_response(lambda: {"status": "ok"}, "health")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert json.loads(response.body) == {"status": "ok"}


def test_snapshot_payload_converts_provider_exceptions() -> None:
    def broken() -> dict[str, str]:
        raise RuntimeError("boom")

    response = snapshot_payload(broken, "broken")

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert response.headers["Cache-Control"] == "no-store"
    assert json.loads(response.body) == {"status": "error", "error": "boom"}


def test_is_remote_bind_host_distinguishes_loopback_and_remote_hosts() -> None:
    assert is_remote_bind_host("") is True
    assert is_remote_bind_host("0.0.0.0") is True
    assert is_remote_bind_host("::") is True
    assert is_remote_bind_host("localhost") is False
    assert is_remote_bind_host("127.0.0.1") is False
    assert is_remote_bind_host("[::1]") is False
    assert is_remote_bind_host("10.0.0.12") is True
    assert is_remote_bind_host("api.example.com") is True
