"""Pre-built runtime compositions for askme deployment modes.

Usage::

    from askme.runtime.compositions import VOICE_RUNTIME
    app = await VOICE_RUNTIME.build(cfg)
    await app.start()
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import (
    ControlModule,
    ExecutorModule,
    HealthModule,
    LEDModule,
    LLMModule,
    MemoryModule,
    PerceptionModule,
    PipelineModule,
    ProactiveModule,
    PulseModule,
    SafetyModule,
    SkillModule,
    TextModule,
    ToolsModule,
    VoiceModule,
)

VOICE_RUNTIME = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(PulseModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(PerceptionModule)
    + Runtime.use(SafetyModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(ExecutorModule)
    + Runtime.use(VoiceModule)
    + Runtime.use(TextModule)
    + Runtime.use(ProactiveModule)
    + Runtime.use(HealthModule)
)

TEXT_RUNTIME = VOICE_RUNTIME.without(VoiceModule)

EDGE_ROBOT_RUNTIME = (
    VOICE_RUNTIME
    + Runtime.use(ControlModule)
    + Runtime.use(LEDModule)
)
