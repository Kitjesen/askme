"""Tests for temporal memory tool provider injection."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

from askme.tools.core.tool_registry import ToolRegistry
from askme.tools.spatial.temporal_query_tool import (
    TemporalQueryTool,
    register_temporal_tools,
)


class _FakeTemporalMemory:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {
            "observations": [
                {
                    "label": "person",
                    "ts": time.time(),
                    "pos_x": 1.2,
                    "pos_y": 2.3,
                    "confidence": 0.91,
                }
            ]
        }
        self.calls: list[dict[str, Any]] = []

    def query_temporal_observations(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(params)
        return self.result


def test_temporal_query_uses_injected_memory_client_without_env(
    monkeypatch,
) -> None:
    monkeypatch.delenv("NAV_GATEWAY_URL", raising=False)
    client = _FakeTemporalMemory()
    tool = TemporalQueryTool(temporal_memory_client=client)

    with patch("urllib.request.urlopen") as mock_open:
        result = tool.execute(
            label="person",
            since="30m",
            near_x=1.23456,
            near_y=2.34567,
            radius=3.45678,
            limit=500,
        )

    mock_open.assert_not_called()
    assert "person" in result
    assert client.calls == [
        {
            "since": "30m",
            "limit": 100,
            "label": "person",
            "near_x": 1.235,
            "near_y": 2.346,
            "radius": 3.457,
        }
    ]


def test_temporal_query_registration_accepts_memory_client() -> None:
    registry = ToolRegistry()
    client = _FakeTemporalMemory()

    register_temporal_tools(registry, temporal_memory_client=client)

    tool = registry.get("temporal_query")
    assert tool is not None
    assert "person" in tool.execute(label="person")
    assert client.calls


def test_temporal_query_injected_error_is_readable() -> None:
    tool = TemporalQueryTool(
        temporal_memory_client=_FakeTemporalMemory(result={"error": "not configured"})
    )

    result = tool.execute(label="person")

    assert "not configured" in result
