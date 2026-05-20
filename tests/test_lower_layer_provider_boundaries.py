from __future__ import annotations


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
