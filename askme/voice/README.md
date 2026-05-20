# AskMe Voice Package Layout

`askme.voice` is organized by runtime responsibility. The package root keeps
small compatibility facades for older imports such as `askme.voice.tts`; new
code should import from the responsibility-specific subpackages below.

## Subpackages

- `core`: pure contracts, trace helpers, stream splitting, punctuation.
- `input`: microphone input, ASR, cloud ASR, VAD, KWS, audio preprocessing.
- `output`: TTS engines, audio routing, voice profiles, sound cues.
- `interaction`: compatibility facades for interaction imports that now live in
  `robot_interaction`.
- `orchestration`: `AudioAgent` plus compatibility facades for historical
  runtime bridge imports.
- `diagnostics`: device discovery, calibration, readiness, smoke checks.

## Import Rule

Use canonical imports for new code:

```python
from askme.voice.output import TTSEngine
from askme.voice.orchestration import AudioAgent
from askme.voice.input import ASREngine
```

Legacy imports remain supported for compatibility:

```python
from askme.voice.tts import TTSEngine
```

For robot interaction policy, new code should use `askme.robot_interaction`.
For runtime bridge construction, new code should use
`askme.providers.build_voice_runtime_bridge()` and pass the bridge into
`askme.voice_gateway.VoiceGatewayService`.
