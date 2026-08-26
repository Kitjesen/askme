# Voice turn timeline

`VoiceTurnTimeline` is the product voice evidence contract. It is observational:
it records what happened around a voice turn, but it does not own conversation
truth, cancellation truth, provider session truth, or memory admission.

## Owners

- Conversation Core owns Thread, Turn, Generation, and committed history.
- Voice orchestration owns media observations such as VAD, ASR, TTS, fallback,
  interruption, and realtime provider lifecycle milestones.
- Memory consumes committed conversation events separately. It must not infer
  durable memory from timeline observations.
- Telemetry adapters consume timeline records downstream. They do not decide
  product readiness.

## Public contract

Use the lazy export from `askme.voice.core`:

```python
from askme.voice.core import VoiceTurnTimeline
```

The public runtime surface is:

- `VoiceTurnTimeline.record(event)` records one observation and returns a
  receipt.
- `VoiceTurnTimeline.snapshot(query)` returns sequence-ordered evidence for a
  bounded query.

Keep the hot audio path in memory:

```python
timeline = VoiceTurnTimeline()
```

Use `JsonlVoiceTimelineStore` only where synchronous file append and `fsync`
latency are acceptable. Use `OpenTelemetryVoiceTimelineExporter` only as a
downstream adapter with a tracer configured by the application composition
root.

## Scope model

Every record has a `voice_turn_id`. Stable identifiers may be added later:

- `thread_id`
- `turn_id`
- `trace_id`

The timeline enforces one stable value per `voice_turn_id`. Reusing an event ID
with different payload evidence raises a conflict.

Generation and provider session identifiers are event-local:

- `generation_id`
- `provider_session_id`

They describe the specific observation that saw them. They must not replace the
stable voice turn identity.

## Stages

Canonical stages include:

- `listen_started`, `first_audio_frame`, `speech_start`, `speech_end`
- `endpoint_committed`, `asr_final`
- `turn_correlated`, `turn_admitted`
- `llm_requested`, `first_llm_payload`, `first_semantic`, `first_clause`
- `tts_first_pcm`, `speaker_render_started`, `speaker_render_stopped`
- `speaker_physical_started`, `speaker_physical_stopped`
- `interrupt_detected`, `interrupt_confirmed`, `interrupt_dismissed`
- `fallback_selected`, `upstream_closed`, `turn_finished`, `error`

`upstream_closed` means the upstream voice provider or streaming lane closed.
`turn_finished` is reserved for Conversation Core settlement.

## Privacy and export

Timeline identities are restricted to bounded ASCII tokens. Attributes are
allowlisted, bounded, and limited to scalar telemetry values. Raw transcripts,
audio payloads, prompts, tool arguments, secrets, and contact details do not
belong in timeline attributes.

The OpenTelemetry adapter exports each timeline record as a short observation
span. It uses timeline IDs as correlation attributes only; it does not forge
parent trace context from `trace_id`.

## Readiness boundary

Timeline events can explain latency, fallback, interruption, and degradation.
They are not acoustic proof. Product full-duplex readiness still requires a
target-device physical hardware report with automatic capture/reference timing
as described in `docs/FULL_DUPLEX_VOICE.md`.
