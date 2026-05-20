"""Default provider-backed backend registrations.

This module owns registrations for concrete service, hardware, and local
adapter implementations. Product/runtime startup should keep importing
``askme.interfaces.register_defaults`` for compatibility; that facade calls
``register_default_provider_backends`` from here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_default_provider_backends() -> None:
    """Register default lower-layer implementations into backend registries."""

    _register_llm_backends()
    _register_voice_backends()
    _register_bus_backends()
    _register_detector_backends()


def _register_llm_backends() -> None:
    from askme.interfaces.llm import llm_registry
    from askme.llm.core.client import LLMClient

    for name in (
        "openai_compatible",
        "openai",
        "minimax",
        "doubao",
        "dashscope",
        "deepseek",
        "zhipu",
        "fake",
    ):
        llm_registry.register(name)(LLMClient)


def _register_voice_backends() -> None:
    from askme.interfaces.asr import asr_registry
    from askme.interfaces.tts import tts_registry
    from askme.voice.input.asr import ASREngine
    from askme.voice.output.tts import TTSEngine

    asr_registry.register("sherpa")(ASREngine)
    tts_registry.register("minimax")(TTSEngine)

    try:
        from askme.voice.input.cloud_asr import CloudASR
    except ImportError:
        logger.debug("CloudASR not available (missing dependencies), skipping registration")
    else:
        asr_registry.register("cloud")(CloudASR)


def _register_bus_backends() -> None:
    from askme.interfaces.bus import bus_registry
    from askme.robot.telemetry.mock_pulse import MockPulse
    from askme.robot.telemetry.pulse import Pulse

    bus_registry.register("pulse")(Pulse)
    bus_registry.register("mock")(MockPulse)


def _register_detector_backends() -> None:
    # ChangeDetector is still registered as a discovery stub. It does not yet
    # implement the full DetectorBackend ABC; BackendRegistry keeps this soft
    # until the on-device detector adapter is introduced.
    from askme.interfaces.detector import detector_registry

    try:
        from askme.perception.change_detector import ChangeDetector
    except ImportError:
        logger.debug("ChangeDetector not available, skipping registration")
    else:
        detector_registry.register("change_detector")(ChangeDetector)


__all__ = ["register_default_provider_backends"]
