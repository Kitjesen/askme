"""LingTu Voice — 专为 LingTu 导航部署的 blueprint.

在 S100P 上运行：本地语音 + Telegram Bot + LLM 大脑。
不包含 Thunder cortex runtime 控制模块（control/safety/LED），
避免与 LingTu 服务端口冲突。

导航通过 move_tool 的 _go_to_lingtu() 路由到 LingTu REST API：
  NAV_GATEWAY_URL=http://localhost:5050

Usage::

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

    run_blueprint(lingtu_voice, "LingTu Voice")
