"""Compatibility facade for :mod:`askme.voice_gateway.runtime_bridge`.

New code should not import a runtime bridge from voice orchestration. Build it
through ``askme.providers.build_voice_runtime_bridge`` and pass it into
``askme.voice_gateway.VoiceGatewayService``.
"""

from __future__ import annotations

from askme.voice_gateway.runtime_bridge import VoiceRuntimeBridge

__all__ = ["VoiceRuntimeBridge"]
