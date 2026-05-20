"""Processing pipeline package.

The root package owns compatibility for historical imports such as
``askme.pipeline.brain_pipeline``. New code should import from the
responsibility-specific subpackages:

- ``askme.pipeline.core`` for turn execution and prompt/stream processing.
- ``askme.pipeline.channels`` for text and voice loop entrypoints.
- ``askme.pipeline.field`` for field operations and delivery readiness.
- ``askme.pipeline.skills`` for skill dispatch and gatekeeping.
- ``askme.pipeline.reactions`` for reaction and proactive alert agents.
- ``askme.pipeline.proactive`` for multi-turn clarification and confirmation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from askme.compat.imports import install_legacy_aliases

_LEGACY_MODULE_ALIASES = {
    "askme.pipeline.alert_dispatcher": "askme.pipeline.field.alert_dispatcher",
    "askme.pipeline.brain_pipeline": "askme.pipeline.core.brain_pipeline",
    "askme.pipeline.commands": "askme.pipeline.channels.commands",
    "askme.pipeline.external_turns": "askme.pipeline.channels.external_turns",
    "askme.pipeline.field_deployment_readiness": (
        "askme.pipeline.field.field_deployment_readiness"
    ),
    "askme.pipeline.field_ingest_adapters": "askme.pipeline.field.field_ingest_adapters",
    "askme.pipeline.field_ingest_bridge": "askme.pipeline.field.field_ingest_bridge",
    "askme.pipeline.field_operations": "askme.pipeline.field.field_operations",
    "askme.pipeline.field_scenarios": "askme.pipeline.field.field_scenarios",
    "askme.pipeline.field_site_profile": "askme.pipeline.field.field_site_profile",
    "askme.pipeline.frames": "askme.pipeline.core.frames",
    "askme.pipeline.hooks": "askme.pipeline.core.hooks",
    "askme.pipeline.incident_alerts": "askme.pipeline.field.incident_alerts",
    "askme.pipeline.persona": "askme.pipeline.core.persona",
    "askme.pipeline.planner_agent": "askme.pipeline.skills.planner_agent",
    "askme.pipeline.proactive_agent": "askme.pipeline.reactions.proactive_agent",
    "askme.pipeline.product_launch_readiness": "askme.pipeline.field.product_launch_readiness",
    "askme.pipeline.prompt_builder": "askme.pipeline.core.prompt_builder",
    "askme.pipeline.protocols": "askme.pipeline.core.protocols",
    "askme.pipeline.rag_policy": "askme.pipeline.core.rag_policy",
    "askme.pipeline.reaction_engine": "askme.pipeline.reactions.reaction_engine",
    "askme.pipeline.skill_dispatcher": "askme.pipeline.skills.skill_dispatcher",
    "askme.pipeline.skill_gate": "askme.pipeline.skills.skill_gate",
    "askme.pipeline.state_led_bridge": "askme.pipeline.reactions.state_led_bridge",
    "askme.pipeline.stream_processor": "askme.pipeline.core.stream_processor",
    "askme.pipeline.text_loop": "askme.pipeline.channels.text_loop",
    "askme.pipeline.tool_executor": "askme.pipeline.core.tool_executor",
    "askme.pipeline.trace": "askme.pipeline.core.trace",
    "askme.pipeline.turn_executor": "askme.pipeline.core.turn_executor",
    "askme.pipeline.utils": "askme.pipeline.core.utils",
    "askme.pipeline.voice_loop": "askme.pipeline.channels.voice_loop",
}

_LAZY_EXPORTS = {
    "BrainPipeline": ("askme.pipeline.core.brain_pipeline", "BrainPipeline"),
    "CommandHandler": ("askme.pipeline.channels.commands", "CommandHandler"),
    "TextLoop": ("askme.pipeline.channels.text_loop", "TextLoop"),
    "VoiceLoop": ("askme.pipeline.channels.voice_loop", "VoiceLoop"),
}

__all__ = sorted(_LAZY_EXPORTS)

_OPTIONAL_DEPENDENCY_FALLBACKS = {
    "VoiceLoop": frozenset({"sherpa_onnx", "sounddevice"}),
}


install_legacy_aliases(__name__, _LEGACY_MODULE_ALIASES)


def __getattr__(name: str) -> Any:
    legacy_module = _LEGACY_MODULE_ALIASES.get(f"{__name__}.{name}")
    if legacy_module:
        value = import_module(legacy_module)
        globals()[name] = value
        return value

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module(module_name), attr_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or exc.name not in _OPTIONAL_DEPENDENCY_FALLBACKS.get(
            name,
            frozenset(),
        ):
            raise
        value = None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
