"""Register all default backend implementations into their registries.

Import this module once at startup to populate the registries.
Existing classes predate the ABC interfaces, so the registry uses a soft
type check (warn, not raise) until all classes inherit their interface.

Usage::

    import askme.interfaces.register_defaults  # noqa: F401
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
from askme.interfaces.llm import llm_registry  # noqa: E402

from askme.llm.client import LLMClient  # noqa: E402

llm_registry.register("minimax")(LLMClient)

# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------
from askme.interfaces.asr import asr_registry  # noqa: E402

from askme.voice.asr import ASREngine  # noqa: E402

asr_registry.register("sherpa")(ASREngine)

try:
    from askme.voice.cloud_asr import CloudASR  # noqa: E402

    asr_registry.register("cloud")(CloudASR)
except ImportError:
    logger.debug("CloudASR not available (missing dependencies), skipping registration")

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
from askme.interfaces.tts import tts_registry  # noqa: E402

from askme.voice.tts import TTSEngine  # noqa: E402

tts_registry.register("minimax")(TTSEngine)

# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------
from askme.interfaces.bus import bus_registry  # noqa: E402

from askme.robot.pulse import Pulse  # noqa: E402

bus_registry.register("pulse")(Pulse)

from askme.robot.mock_pulse import MockPulse  # noqa: E402

bus_registry.register("mock")(MockPulse)

try:
    from askme.robot.cyclone_pulse import CyclonePulse  # noqa: E402

    bus_registry.register("cyclonedds")(CyclonePulse)
except ImportError:
    logger.debug("CyclonePulse not available (missing cyclonedds), skipping")

# ---------------------------------------------------------------------------
# Detector — no implementations yet (ChangeDetector is a perception processor,
#             not a DetectorBackend).  BPU YOLO backend will be added when the
#             on-device model wrapper is created.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Navigator — no implementations yet.  LingTu bridge wrapper will be added
#             when the gRPC client is wrapped as a NavigatorBackend.
# ---------------------------------------------------------------------------
