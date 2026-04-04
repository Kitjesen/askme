"""Runtime modules for askme.

Re-exports all modules for convenient imports::

    from askme.runtime.modules import LLMModule, PipelineModule, VoiceModule

Hardware-dependent modules (VoiceModule, TextModule, etc.) require audio
drivers (sounddevice) that may not be installed in dev/CI environments.
Each hardware module is guarded with try/except so the package stays
importable without the hardware stack.
"""

from askme.runtime.modules.llm_module import LLMModule
from askme.runtime.modules.tools_module import ToolsModule
from askme.runtime.modules.pulse_module import PulseModule
from askme.runtime.modules.memory_module import MemoryModule
from askme.runtime.modules.perception_module import PerceptionModule
from askme.runtime.modules.safety_module import SafetyModule
from askme.runtime.modules.pipeline_module import PipelineModule
from askme.runtime.modules.skill_module import SkillModule
from askme.runtime.modules.executor_module import ExecutorModule

# Hardware modules — require audio/ASR drivers not always installed
_HARDWARE_MODULES = [
    ("VoiceModule", "askme.runtime.modules.voice_module"),
    ("TextModule", "askme.runtime.modules.text_module"),
    ("ControlModule", "askme.runtime.modules.control_module"),
    ("LEDModule", "askme.runtime.modules.led_module"),
    ("ProactiveModule", "askme.runtime.modules.proactive_module"),
    ("ReactionModule", "askme.runtime.modules.reaction_module"),
    ("HealthModule", "askme.runtime.modules.health_module"),
]

import importlib as _importlib

VoiceModule = TextModule = ControlModule = None  # type: ignore[assignment,misc]
LEDModule = ProactiveModule = ReactionModule = HealthModule = None  # type: ignore[assignment,misc]

for _cls_name, _mod_path in _HARDWARE_MODULES:
    try:
        _mod = _importlib.import_module(_mod_path)
        globals()[_cls_name] = getattr(_mod, _cls_name)
    except ModuleNotFoundError:
        pass  # hardware drivers not available — module stays None

del _importlib, _HARDWARE_MODULES, _cls_name, _mod_path

__all__ = [
    "LLMModule",
    "ToolsModule",
    "PulseModule",
    "MemoryModule",
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
    "HealthModule",
]
