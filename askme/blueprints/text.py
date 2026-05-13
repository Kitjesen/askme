"""Text — 纯文本交互.

核心能力，无语音硬件。SSH/开发用。5 个模块。

Usage::

    python -m askme.blueprints.text
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import (
    CognitionModule,
    ExecutorModule,
    HealthModule,
    LLMModule,
    MemoryModule,
    MissionModule,
    PipelineModule,
    RuntimeHandoffModule,
    SkillModule,
    TextModule,
    ToolsModule,
)

text = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(MissionModule)
    + Runtime.use(CognitionModule)
    + Runtime.use(RuntimeHandoffModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(ExecutorModule)
    + Runtime.use(TextModule)
    + Runtime.use(HealthModule)
)

__all__ = ["text"]

if __name__ == "__main__":
    from askme.blueprints._runner import run_blueprint

    run_blueprint(text, "Text")
