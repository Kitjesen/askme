"""Backend interfaces — ABC definitions for all pluggable subsystems.

Each interface defines WHAT a subsystem does, not HOW.
Implementations live in their respective packages and register via @registry.register().
"""

from askme.interfaces.llm import LLMBackend, llm_registry
from askme.interfaces.asr import ASRBackend, asr_registry
from askme.interfaces.tts import TTSBackend, tts_registry
from askme.interfaces.detector import DetectorBackend, detector_registry
from askme.interfaces.navigator import NavigatorBackend, navigator_registry
from askme.interfaces.bus import BusBackend, bus_registry
from askme.interfaces.reaction import ReactionBackend, reaction_registry

__all__ = [
    "LLMBackend", "llm_registry",
    "ASRBackend", "asr_registry",
    "TTSBackend", "tts_registry",
    "DetectorBackend", "detector_registry",
    "NavigatorBackend", "navigator_registry",
    "BusBackend", "bus_registry",
    "ReactionBackend", "reaction_registry",
]
