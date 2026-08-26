"""Runtime modules for askme.

Re-exports module classes for convenient imports::

    from askme.runtime.modules import LLMModule, PipelineModule, VoiceModule

The re-exports are lazy so importing one concrete module does not import the
whole runtime stack or initialize optional hardware bindings.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "CognitionModule": "askme.runtime.modules.cognition_module",
    "ControlModule": "askme.runtime.modules.control_module",
    "ExecutorModule": "askme.runtime.modules.executor_module",
    "HealthModule": "askme.runtime.modules.health_module",
    "LEDModule": "askme.runtime.modules.led_module",
    "LLMModule": "askme.runtime.modules.llm_module",
    "MemoryModule": "askme.runtime.modules.memory_module",
    "MissionModule": "askme.runtime.modules.mission_module",
    "PerceptionModule": "askme.runtime.modules.perception_module",
    "PipelineModule": "askme.runtime.modules.pipeline_module",
    "ProactiveModule": "askme.runtime.modules.proactive_module",
    "PulseModule": "askme.runtime.modules.pulse_module",
    "ReactionModule": "askme.runtime.modules.reaction_module",
    "RuntimeHandoffModule": "askme.runtime.modules.runtime_handoff_module",
    "SafetyModule": "askme.runtime.modules.safety_module",
    "SkillModule": "askme.runtime.modules.skill_module",
    "TelegramModule": "askme.runtime.modules.telegram_module",
    "WarmSessionModule": "askme.runtime.modules.warm_session_module",
    "TextModule": "askme.runtime.modules.text_module",
    "ToolsModule": "askme.runtime.modules.tools_module",
    "VoiceModule": "askme.runtime.modules.voice_module",
}

__all__ = [
    "CognitionModule",
    "LLMModule",
    "ToolsModule",
    "PulseModule",
    "MemoryModule",
    "MissionModule",
    "PerceptionModule",
    "SafetyModule",
    "PipelineModule",
    "SkillModule",
    "ExecutorModule",
    "VoiceModule",
    "TextModule",
    "ControlModule",
    "LEDModule",
    "ProactiveModule",
    "ReactionModule",
    "RuntimeHandoffModule",
    "HealthModule",
    "TelegramModule",
    "WarmSessionModule",
]


def __getattr__(name: str) -> Any:
    """Resolve runtime module classes on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
