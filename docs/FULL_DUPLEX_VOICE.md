# Full-duplex robot voice

This voice stack does not require ROS2. The robot uses one local media plane
for the sound card and keeps cloud inference behind the existing provider
interfaces.

## Runtime shape

```text
combined speaker + microphone
        |
local AEC / VAD / endpointing / barge-in
        |---------------- local KWS, fixed commands, cached PCM
        |
streaming cloud ASR -> streaming LLM -> streaming cloud TTS
        |
final speaker PCM --------> AEC render reference
```

Cloud services provide recognition, reasoning and natural speech. Local code
still owns the latency-critical and safety-critical work: echo cancellation,
speech start/end detection, playback stop, current-turn cancellation, E-STOP,
offline commands and explicit degradation.

Voice turn evidence is recorded through `VoiceTurnTimeline`. Keep the runtime
audio path on the in-memory timeline store; JSONL persistence and OpenTelemetry
export are downstream adapters, not callback-path dependencies. The timeline can
explain latency, fallback, interruption and degradation, but it does not prove
physical acoustic readiness.

The optional speech-to-speech lane is exposed through the explicit
`RealtimeVoiceFrontendPort`. Provider approval, generation binding, discard,
and playback abort are no longer discovered through ad-hoc runtime attribute
checks. Local turn admission stays separate from provider/session mechanics,
and frontends without realtime support continue on the cascade path.

## Configuration

```yaml
voice:
  full_duplex:
    enabled: true
    echo_control: auto
    echo_control_verified: false
    aec_sample_rate_hz: 16000
    aec_delay_ms: 40
  tts:
    render_reference_queue_size: 8
    render_reference_max_lag_ms: 120
```

`echo_control` has three safe values:

- `auto` or `native`: require the optional native WebRTC APM adapter. If it
  cannot load, startup remains half-duplex.
- `hardware`: use only when the combined speakerphone's built-in AEC has been
  verified with a speaker-only negative test and a real-person barge-in test,
  then set `echo_control_verified: true` for that exact device/driver profile.
- `system`: use only when the selected operating-system capture source already
  applies echo cancellation, such as a verified PipeWire echo-cancel source;
  it also requires `echo_control_verified: true`.

An amplitude gate is not AEC and never makes the readiness check pass. When
full-duplex is active, the legacy echo gate and post-playback input cooldown
are disabled. When readiness fails, they remain available for half-duplex.

The final speaker PCM is delivered to AEC in 10 ms frames on a monotonic
render clock. Every normal turn, interruption and shutdown advances its epoch,
discards stale reference work and resets the AEC timeline. Queue overflow,
callback failure, excessive render-clock lag, native AEC failure, or an audio
device that rejects simultaneous input/output immediately restores exclusive
half-duplex routing and the previous echo gate/cooldown. Health counters are
available in `tts.status_snapshot()["render_reference"]`.

Microphone callback starvation is tolerated for two one-second intervals. A
third timeout is classified as device loss and triggers a real stop/open
cycle. Reconnect closes partially opened PortAudio handles, verifies that the
new input is actually open, and uses bounded exponential retry. A lifecycle
generation fence and final cleanup lock prevent a late reconnect worker from
reopening the microphone after runtime shutdown.

`aec_delay_ms: 40` is only a conservative starting value. Measure and tune it
for the actual speaker/microphone/driver combination; software scheduling
cannot provide exact DAC hardware timestamps on every output transport.

## Interruption and cancellation contract

Possible interruptions are validated before cancellation:

1. VAD reports possible barge-in while playback is active.
2. Playback attempts a lossless generation-scoped hold.
3. ASR validates the captured speech.
4. False input resumes the exact held playback generation.
5. Admitted user speech aborts the hold and commits the interruption.

This prevents speaker echo, noise, or rejected wake/filter output from
unnecessarily truncating assistant speech. The lossless hold is available only
on callback-driven playback transports; unsupported transports fail closed.

Confirmed real barge-in then performs one turn-scoped transaction:

1. atomically set the current interaction and pipeline cancellation events;
2. invalidate the current playback generation so late audio is rejected;
3. stop and drain current playback;
4. stop consuming future LLM tokens and suppress undispatched tool work;
5. omit unplayed generated text from assistant history;
6. leave already-dispatched physical robot actions unchanged.

E-STOP is a separate sticky token. Barge-in, session reset and runtime restart
cannot clear it.

## Cloud-first latency policy

- Start cloud ASR when local VAD confirms speech start.
- Start the LLM immediately on ASR final.
- Send complete short clauses to TTS while LLM streaming continues.
- Keep the MiniMax TTS WebSocket warm and serialize task ownership.
- Use the local phrase cache and fixed-command path before any cloud request.
- Keep the local sherpa-onnx path for wake words, short commands and outages.

The canonical MiniMax endpoint is `https://api.minimaxi.com/v1`. The
low-latency voice profile intentionally keeps `MiniMax-M2.7-highspeed`: the
official catalog publishes about 100 output tokens/s for that model. MiniMax-M3
is the newer quality/long-context tier, but it should replace the voice model
only after a target-region TTFT benchmark. TTS uses the current
`speech-2.8-turbo` model over `wss://api.minimaxi.com/ws/v1/t2a_v2`.

For deployments in mainland China, the configured domestic providers remain
the production default. OpenAI Realtime can be implemented as a provider for
officially supported deployment regions, but it does not replace local AEC or
local safety handling.

## Acceptance gates

Record the operating system, audio device/driver, input and output device IDs,
sample rates, Python version and AEC backend for every run.

- user speech start to speaker stop: p95 <= 250 ms, p99 <= 400 ms;
- user speech end to physical first sound: p95 <= 1.2 s, p99 <= 1.8 s;
- speaker-only playback does not trigger barge-in;
- a real person speaking over playback is detected consistently;
- interrupted turns never save unplayed assistant text;
- local wake, E-STOP and core fixed commands continue during cloud outage;
- AEC or simultaneous device-open failure emits a structured reason and
  remains half-duplex.

Run at least 20 speaker-only trials, 20 real-person overlap trials, and 20
speech-end-to-physical-first-sound trials on the target combined
speaker/microphone. A development computer without the native
`_askme_webrtc_apm` extension will correctly report `aec_unavailable` and stay
half-duplex; passing unit tests alone is not evidence of acoustic readiness.

Use the interactive target-hardware evaluator while the robot voice runtime and
health server are running:

```powershell
python scripts/eval/evaluate_full_duplex_hardware.py `
  --config config.board.yaml `
  --status-source http://127.0.0.1:8765/health `
  --output artifacts/voice/full-duplex-hardware.json
```

The evaluator refreshes the health snapshot after every trial, writes progress
after every answer, and fails if full duplex degrades, the status source is
lost, any scenario has fewer than 20 trials, speaker-only playback causes any
false interruption, human overlap detection is inconsistent, p95/p99
speaker-stop latency exceeds 250/400 ms, or p95/p99 physical-first-sound
latency exceeds 1200/1800 ms. Each snapshot must report top-level
`status: ok`, `pipeline_ok: true`, and fresh `snapshot_at`/`recorded_at`
timestamps; missing, stale, or future-dated health data fails closed. It also
requires the OS, room,
device/driver, input/output IDs and sample rates, Python version and AEC backend
in the report. `--latency-mode entry` and the default Enter-to-Enter stopwatch
are now explicitly `manual` diagnostics and return a failed product gate even
when the entered numbers are below threshold. WASAPI/loopback timing is
`render_chain` only. A passing `askme.full_duplex_hardware.v2` report requires
automatic capture/reference timestamps on the same monotonic clock, valid
calibration, zero dropped frames, and at least 20 `physical_acoustic` trials for
both speaker stop and first sound. Overlap stop additionally requires an
`isolated_speaker_monitor`; the ordinary shared room microphone cannot prove
speaker silence while a person is talking. The repository exposes
`build_instrumented_trial_evidence()` for a future capture adapter but does not
yet implement that physical adapter.

Voice Lab can collect server-owned execution evidence for a prepared trial with
this bodyless mutation:

```text
POST /api/voice/lab/runs/{run_id}/trials/{attempt_id}/execute
Idempotency-Key: <stable-key>
If-Match: <run-version>
```

The execute endpoint calls the server-side trial evidence provider and persists
the resulting timeline, fallback, interrupt, AEC, and residual-audio fields on
the run. This evidence is useful for operator diagnosis and report assembly.
The default HTTP composition does not yet install a runtime trial-evidence
provider, so this mutation deliberately returns `503` instead of opening a
second microphone or fabricating evidence. A deployment must inject an adapter
over its existing `VoiceModule`, audio owner, and shared timeline before the
button becomes executable.
Manual entries, algorithm telemetry, and server-owned runtime evidence remain
labeled by evidence kind. Only a physical target-hardware run with valid
automatic capture/reference timestamps can satisfy the product hardware gate.

For `hardware` or `system` echo control, the runtime cannot enter full duplex
until `echo_control_verified: true`. Treat that flag as a controlled lab
candidate assertion for the exact device/driver profile: never promote it to a
deployment configuration unless this evaluator produces a passing report, and
restore it to `false` after any failed or interrupted run. Native WebRTC APM is
instead proven by the live `native_aec_ready` status and does not use this flag.

Unit and loopback tests validate ownership, cancellation and media contracts.
They do not prove acoustic performance; final release still requires the
target combined speaker/microphone in the deployment room.

Normalize all available latency evidence before making a product-readiness
claim:

```powershell
python scripts/eval/report_voice_latency.py `
  --fast-path artifacts/perf/voice_fast_path_latest.json `
  --hardware artifacts/voice/full-duplex-hardware.json `
  --online-smoke artifacts/voice/online-smoke.json `
  --voice-health artifacts/voice/voice-health.json `
  --scenario artifacts/voice_e2e/scenario-evaluation.json `
  --out artifacts/voice/voice-latency-report.json
```

Every normalized metric is labeled `measured`, `projected`, or `simulated`
and includes its measurement scope. Projected budgets and simulated scenarios
can guide optimization but cannot make the report pass. Missing measured
physical-first-sound or barge-in-stop evidence produces
`insufficient_evidence` and a non-zero CLI exit code.
