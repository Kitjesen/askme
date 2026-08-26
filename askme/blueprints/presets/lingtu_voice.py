"""灵途语音导航适配器蓝图。

用于灵途导航项目的站点专用语音运行时。该蓝图刻意不包含机器人控制、
安全和灯光模块，避免与灵途导航服务产生端口或控制权冲突。

启动：
    python -m askme.blueprints.presets.lingtu_voice
"""

from askme.blueprints.catalog.data import LINGTU_VOICE_MODULES

_LABEL = "灵途语音导航适配器"


def _build_lingtu_voice():
    from askme.blueprints.runner.runner import compose_runtime
    from askme.runtime.modules import (
        LLMModule,
        MemoryModule,
        PipelineModule,
        SkillModule,
        TelegramModule,
        TextModule,
        ToolsModule,
        VoiceModule,
        WarmSessionModule,
    )

    return compose_runtime(
        (
            LLMModule,
            ToolsModule,
            MemoryModule,
            PipelineModule,
            SkillModule,
            VoiceModule,
            WarmSessionModule,
            TextModule,
            TelegramModule,
        )
    )


__all__ = ["lingtu_voice"]

if __name__ == "__main__":
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(_build_lingtu_voice, _LABEL, module_names=LINGTU_VOICE_MODULES)
else:
    lingtu_voice = _build_lingtu_voice()
