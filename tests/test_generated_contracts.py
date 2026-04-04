"""Tests for EndpointSpec and AskmeEdgeVoiceContract."""

from __future__ import annotations

import pytest

from askme.voice.generated_contracts import AskmeEdgeVoiceContract, EndpointSpec


class TestEndpointSpec:
    def test_path_no_params(self):
        spec = EndpointSpec("GET", "/api/v1/health")
        assert spec.path() == "/api/v1/health"

    def test_path_with_params(self):
        spec = EndpointSpec("GET", "/api/v1/items/{item_id}")
        assert spec.path(item_id="42") == "/api/v1/items/42"

    def test_frozen_immutable(self):
        spec = EndpointSpec("GET", "/api/v1/test")
        with pytest.raises((AttributeError, TypeError)):
            spec.method = "POST"

    def test_include_idempotency_default_false(self):
        spec = EndpointSpec("GET", "/api/v1/test")
        assert spec.include_idempotency is False

    def test_response_key_default_none(self):
        spec = EndpointSpec("POST", "/api/v1/test")
        assert spec.response_key is None


class TestAskmeEdgeVoiceContract:
    def test_health_is_get(self):
        assert AskmeEdgeVoiceContract.HEALTH.method == "GET"

    def test_health_path(self):
        assert AskmeEdgeVoiceContract.HEALTH.path() == "/api/v1/health"

    def test_process_turn_is_post(self):
        assert AskmeEdgeVoiceContract.PROCESS_TURN.method == "POST"

    def test_process_turn_idempotency(self):
        assert AskmeEdgeVoiceContract.PROCESS_TURN.include_idempotency is True
