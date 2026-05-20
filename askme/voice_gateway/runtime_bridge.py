"""Compatibility facade for the provider-owned voice runtime bridge.

New code should construct this capability through
``askme.providers.build_voice_runtime_bridge`` and consume it through
``VoiceGatewayService`` instead of importing ``VoiceRuntimeBridge`` from
``askme.voice_gateway``. This module remains for historical imports such as
``askme.voice.runtime_bridge`` and for compatibility monkeypatch tests.
"""

from __future__ import annotations

from askme.providers import voice_runtime as _impl

VoiceRuntimeBridge = _impl.VoiceRuntimeBridge

__all__ = ["VoiceRuntimeBridge"]
