# AskMe Product Voice Upgrade

Date: 2026-07-12

## Decision

AskMe keeps its existing Python voice loop and adopts proven product patterns
incrementally. Replacing the whole runtime with another framework would create
a second session, tool, safety, and robot integration stack before the current
one is stable.

For browser, mobile, telephone, SIP, or weak-network deployments, evaluate a
WebRTC transport layer separately. The local robot/desktop path remains a
streaming cascade:

`audio -> VAD/turn detection -> ASR -> session/RAG -> LLM -> streaming TTS`

## Upstream Research

| Project | License | Current assessment | Adopted idea |
| --- | --- | --- | --- |
| [LiveKit Agents](https://github.com/livekit/agents) | Apache-2.0; some models have separate terms | Active product-grade WebRTC/SIP agent runtime | Session lifecycle, adaptive interruption, provider fallback, event telemetry |
| [Pipecat](https://github.com/pipecat-ai/pipecat) | BSD-2-Clause | Active, Python-first, closest architectural reference | Typed frame pipeline, cancellation/backpressure, metrics, WebRTC boundary |
| [TEN Framework](https://github.com/TEN-framework/ten-framework) | Requires license review | Active but substantially heavier than AskMe | Extension isolation and graph observability only |
| [Vocode](https://github.com/vocodedev/vocode-core) | MIT | Maintenance is behind the leading projects | Conversation mechanics reference; do not add as a new core dependency |
| [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | Suitable local VAD; AskMe already ships a Silero ONNX model | Local speech start/end signal |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | Strong local STT fallback candidate | Isolated offline STT service, not a full dialogue framework |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | MIT | Strong edge/CPU STT candidate | Native edge fallback |
| [openWakeWord](https://github.com/dscripka/openWakeWord) | Apache-2.0 | Useful but slower release cadence | Optional wake-word provider behind AskMe's KWS port |
| [Wyoming protocol services](https://github.com/rhasspy/wyoming) | MIT | Useful process-isolation pattern | Run local ASR/TTS/wake components out of process |

The archived [rhasspy/piper](https://github.com/rhasspy/piper) should not become
a new dependency. Its active successor
[OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) is GPL-3.0, so a
commercial distribution needs a deliberate license decision or process-level
boundary.

## Product Patterns

1. Treat each user turn as a session-scoped event timeline. Never obtain turn
   evidence from process-global `last_*` state.
2. Combine VAD with transcript/semantic turn signals. Fixed silence alone is
   insufficient for short commands and natural pauses.
3. Support barge-in, false-interruption resume, and cancellation through every
   STT, LLM, tool, and TTS stage.
4. Record what was actually played by TTS, not only the model's original text.
5. Measure speech start/end, ASR partial/final, LLM first token, TTS first audio,
   end-to-end response latency, interruptions, fallbacks, and reconnects.
6. Expose provider, device, circuit-breaker, and fallback state explicitly.
7. Use WebRTC for remote production clients; reserve raw WebSocket audio for
   prototypes or controlled networks.

## Implemented In This Pass

- Tool calls preserve `conversation_session_id` through execution and follow-up.
- RAG evidence is captured per turn and written to the correct session.
- Memory retrieval failures fail closed instead of silently producing an
  ungrounded factual answer.
- Voice admission receives live mission and actor context from safety/runtime.
- Runtime bridge failures and local fallback are visible in status and turn
  metadata.
- MiniMax TTS can fall back to the bundled local model.
- The default product path uses local sherpa-onnx ASR/TTS; cloud ASR and
  MiniMax TTS remain opt-in until their online probes pass.
- Text and voice reasoning use the verified DeepSeek V4 Flash provider with
  thinking disabled for real-time turns and V4 Pro as same-provider fallback.
- Product config enables the explicit `你好小穹` wake word.
- Voice health exposes product-readiness blockers rather than declaring ASR/TTS
  healthy solely because provider objects exist.

## Acceptance Gate For Real Dialogue

All items are required before calling a deployment "real product dialogue":

1. `python -m askme.cli runtime voice-health --live --json` reports no product
   readiness blockers and opens the intended input/output devices.
2. Local ASR transcribes a recorded utterance and local TTS produces and plays
   non-zero audio samples.
3. The selected LLM provider completes a real streamed turn. A fake provider
   does not satisfy this gate.
4. A microphone-to-speaker run proves wake word, ASR, LLM, TTS, echo control,
   and user barge-in on the target machine.
5. Concurrent sessions cannot exchange messages, tool results, or RAG evidence.
6. Runtime-required deployments reject or explicitly degrade when the runtime
   bridge is unavailable; local-only deployments may use a recorded fallback.
7. A 30-minute soak has no stuck playback, runaway wake triggers, device loss,
   unbounded queues, or unrecovered provider circuits.

## Next Architecture Stage

Introduce a small internal voice-event contract (`speech_started`,
`partial_transcript`, `turn_final`, `llm_delta`, `tts_audio`, `interrupted`,
`fallback_used`, `reconnected`). Keep providers behind ports. Add LiveKit or
Pipecat only when a remote WebRTC/SIP product surface is approved.
