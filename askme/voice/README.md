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
- `lab`: operator-facing, target-hardware Voice Lab state and evidence collection.
- `realtime`: provider-neutral speech-to-speech contracts, safety coordination,
  and optional cloud realtime adapters.

## Voice Turn Evidence

`VoiceTurnTimeline` is the canonical observational timeline for voice turns.
It records privacy-safe milestones such as listen start, ASR final, fallback
selection, interruption detection, render start/stop, and upstream close.
Conversation Core still owns Thread, Turn, Generation, and committed history;
the voice timeline records evidence supplied by those owners and by media
adapters.

The public contract is intentionally small:

```python
from askme.voice.core import VoiceTurnTimeline
```

- Use `record(...)` to append one bounded observation.
- Use `snapshot(...)` to read a deterministic, sequence-ordered view.
- Use the default in-memory store on latency-sensitive audio paths.
- Attach JSONL persistence or OpenTelemetry export downstream, outside the
  audio callback path.

See `docs/VOICE_TIMELINE.md` for the timeline scope model, privacy rules, and
export contract.

## False Interruption Recovery

`AudioAgent` no longer treats a VAD-confirmed barge-in as immediate
cancellation. The recovery contract is:

1. VAD detects possible speech while TTS is active.
2. TTS attempts a lossless playback hold for the current playback generation.
3. ASR validates the captured speech.
4. If ASR rejects it as noise, empty input, wake filtering, or another false
   interruption, the exact held generation resumes.
5. If ASR admits a real interruption, the hold is aborted and the normal
   turn-cancellation path runs.

Only the sounddevice callback playback path can guarantee a lossless hold. Other
output transports fail closed and continue through the existing cancellation or
recovery behavior.

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
