"""Voice — 语音巡检模式.

The default Thunder blueprint: voice AI assistant with perception,
safety gating, skill dispatch, and proactive patrol.

Usage::

    python -m askme.blueprints.voice
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import (
    ExecutorModule,
    HealthModule,
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

voice = (
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

__all__ = ["voice"]

if __name__ == "__main__":
    import asyncio
    import signal

    from askme.config import load_config

    async def main():
        cfg = load_config()
        app = await voice.build(cfg)

        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)

        await app.start()
        print(f"Voice running — {len(app.modules)} modules")
        print(f"Wired: {len(app.wired_ports)} ports")
        await stop.wait()
        await app.stop()

    asyncio.run(main())
