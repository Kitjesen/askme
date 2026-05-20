"""灵途语音导航适配器蓝图。

用于灵途导航项目的站点专用语音运行时。该蓝图刻意不包含机器人控制、
安全和灯光模块，避免与灵途导航服务产生端口或控制权冲突。

启动：
    python -m askme.blueprints.presets.lingtu_voice
"""

from askme.runtime.core.module import Runtime
from askme.runtime.modules import (
    LLMModule,
    MemoryModule,
    PipelineModule,
    SkillModule,
    TelegramModule,
    TextModule,
    ToolsModule,
    VoiceModule,
)

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
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(lingtu_voice, "灵途语音导航适配器")
