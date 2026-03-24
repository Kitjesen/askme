"""Text — 纯文本交互模式.

No voice IO, no audio hardware. For SSH sessions and development.

Usage::

    python -m askme.blueprints.text
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import (
    ExecutorModule,
    HealthModule,
    LLMModule,
    MemoryModule,
    PipelineModule,
    ProactiveModule,
    PulseModule,
    SafetyModule,
    SkillModule,
    TextModule,
    ToolsModule,
)

text = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(PulseModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(SafetyModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(ExecutorModule)
    + Runtime.use(TextModule)
    + Runtime.use(ProactiveModule)
    + Runtime.use(HealthModule)
)

__all__ = ["text"]

if __name__ == "__main__":
    from askme.blueprints._runner import run_blueprint

    run_blueprint(text, "Text")
