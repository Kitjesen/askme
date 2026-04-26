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

# ---------------------------------------------------------------------------
# Detector — ChangeDetector registered as a stub.  It does not implement the
#            full DetectorBackend ABC (no detect()/model_name), but the soft
#            type check allows registration for discovery purposes.
#            A proper BPU YOLO backend will be added when the on-device model
#            wrapper is created.
# ---------------------------------------------------------------------------
from askme.interfaces.detector import detector_registry  # noqa: E402

try:
    from askme.perception.change_detector import ChangeDetector  # noqa: E402

    detector_registry.register("change_detector")(ChangeDetector)
except ImportError:
    logger.debug("ChangeDetector not available, skipping registration")

# ---------------------------------------------------------------------------
# Reaction
# ---------------------------------------------------------------------------
from askme.interfaces.reaction import reaction_registry  # noqa: E402
from askme.pipeline.reaction_engine import HybridReaction, RuleBasedReaction  # noqa: E402

reaction_registry.register("hybrid")(HybridReaction)
reaction_registry.register("rules")(RuleBasedReaction)

# ---------------------------------------------------------------------------
# Navigator — no NavigatorBackend adapter exists yet.  LingTu gRPC bridge
#             will be registered here when wrapped as a NavigatorBackend.
# ---------------------------------------------------------------------------
