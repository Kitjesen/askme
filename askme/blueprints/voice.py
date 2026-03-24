"""Voice — 纯语音 AI 助手.

核心能力：听、理解、回答、说。6 个模块。

Usage::

    python -m askme.blueprints.voice
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import (
    LLMModule,
    MemoryModule,
    PipelineModule,
    SkillModule,
    TextModule,
    VoiceModule,
)

voice = (
    Runtime.use(LLMModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(VoiceModule)
    + Runtime.use(TextModule)
)

__all__ = ["voice"]

if __name__ == "__main__":
    from askme.blueprints._runner import run_blueprint

    run_blueprint(voice, "Voice")
