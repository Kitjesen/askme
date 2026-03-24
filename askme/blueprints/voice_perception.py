"""Voice + Perception — 语音助手 + 场景感知.

核心 + 感知层。能听能说能看能主动反应。9 个模块。

Usage::

    python -m askme.blueprints.voice_perception
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import (
    PerceptionModule,
    PulseModule,
    ReactionModule,
    SafetyModule,
)

from askme.blueprints.voice import voice

voice_perception = (
    voice
    + Runtime.use(PulseModule)
    + Runtime.use(PerceptionModule)
    + Runtime.use(SafetyModule)
    + Runtime.use(ReactionModule)
)

__all__ = ["voice_perception"]

if __name__ == "__main__":
    from askme.blueprints._runner import run_blueprint

    run_blueprint(voice_perception, "Voice+Perception")
