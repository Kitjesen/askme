"""Processing pipeline — decoupled from app.py assembly."""

from askme.pipeline.brain_pipeline import BrainPipeline
from askme.pipeline.commands import CommandHandler
from askme.pipeline.text_loop import TextLoop

# VoiceLoop depends on hardware drivers (sherpa_onnx, sounddevice) that may
# not be installed in dev / CI environments.  Guard the import so that
# `import askme.pipeline` succeeds in those contexts.
try:
    from askme.pipeline.voice_loop import VoiceLoop
except ModuleNotFoundError:
    VoiceLoop = None  # type: ignore[assignment,misc]

__all__ = ["BrainPipeline", "CommandHandler", "TextLoop", "VoiceLoop"]
