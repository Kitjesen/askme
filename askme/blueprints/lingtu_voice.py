"""LingTu Voice Navigation Adapter blueprint.

This is a site-specific voice runtime for LingTu navigation. It deliberately
omits Thunder control, safety, and LED modules to avoid port and authority
conflicts with the LingTu navigation service.

Run:
    python -m askme.blueprints.lingtu_voice
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import (
    LLMModule,
    MemoryModule,
    PipelineModule,
    SkillModule,
    TextModule,
    ToolsModule,
    VoiceModule,
)
from askme.runtime.modules.telegram_module import TelegramModule

lingtu_voice = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(VoiceModule)
    + Runtime.use(TextModule)
    + Runtime.use(TelegramModule)
)

__all__ = ["lingtu_voice"]

if __name__ == "__main__":
    from askme.blueprints._runner import run_blueprint

    run_blueprint(lingtu_voice, "LingTu Voice Navigation Adapter")
