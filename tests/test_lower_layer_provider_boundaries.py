from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def test_provider_facade_exports_spatial_factories_not_concrete_client() -> None:
    import askme.providers as providers
    from askme.ports import NavigationPort, TemporalMemoryPort

    assert "build_navigation" in providers.__all__
    assert "build_temporal_memory" in providers.__all__
    assert "NavGatewayClient" not in providers.__all__
    assert not hasattr(providers, "NavGatewayClient")

    navigation = providers.build_navigation({})
    temporal_memory = providers.build_temporal_memory({})

    assert isinstance(navigation, NavigationPort)
    assert isinstance(temporal_memory, TemporalMemoryPort)


def test_navigation_status_enriches_ready_status_with_current_odometry() -> None:
    import askme.providers as providers

    payloads = {
        "http://127.0.0.1:5050/api/v1/navigation/status": {
            "state": "IDLE",
            "has_odometry": True,
        },
        "http://127.0.0.1:5050/api/v1/state": {
            "odometry": {"x": 1.25, "y": -0.5, "frame_id": "map"},
        },
    }
    requested: list[str] = []

    def fake_urlopen(url: str, timeout: int):
        requested.append(url)
        response = MagicMock()
        response.read.return_value = json.dumps(payloads[url]).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        return response

    navigation = providers.build_navigation({"base_url": "http://127.0.0.1:5050"})
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = navigation.status()

    assert result["odometry"] == {
        "x": 1.25,
        "y": -0.5,
        "frame_id": "map",
    }
    assert requested == list(payloads)
