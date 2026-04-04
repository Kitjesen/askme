"""Voice sub-package -- ASR, VAD, KWS, TTS engines and AudioAgent."""

# Hardware-dependent modules (sherpa_onnx, sounddevice, etc.) may not be
# installed in dev / CI environments.  Guard all hardware imports so that
# the rest of the package stays importable without audio/ASR drivers.
try:
    from .asr import ASREngine
except ModuleNotFoundError:
    ASREngine = None  # type: ignore[assignment,misc]

try:
    from .kws import KWSEngine
except ModuleNotFoundError:
    KWSEngine = None  # type: ignore[assignment,misc]

try:
    from .vad import VADEngine
except ModuleNotFoundError:
    VADEngine = None  # type: ignore[assignment,misc]

try:
    from .audio_agent import AudioAgent
except ModuleNotFoundError:
    AudioAgent = None  # type: ignore[assignment,misc]

try:
    from .tts import TTSEngine
except ModuleNotFoundError:
    TTSEngine = None  # type: ignore[assignment,misc]

from .stream_splitter import StreamSplitter

__all__ = [
    "ASREngine",
    "AudioAgent",
    "KWSEngine",
    "StreamSplitter",
    "TTSEngine",
    "VADEngine",
]
