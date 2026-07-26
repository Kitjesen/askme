# Askme

[![CI](https://github.com/inovxio/askme/actions/workflows/ci.yml/badge.svg)](https://github.com/inovxio/askme/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-4.1.0-green.svg)](pyproject.toml)

[中文 README](README.md)

Askme is a field-delivery and robot interaction platform for solution providers and system integrators. It combines voice, text, customer knowledge, field events, runtime handoff, acceptance evidence, and audit trails into one repeatable demo-to-pilot delivery entry point.

Version: 4.1.0. The canonical project README is [README.md](README.md).

## What this project is

Askme is not a generic chatbot and it does not replace the robot chassis controller. The product boundary is:

- Own the conversational, delivery, evidence, and audit layer.
- Keep real execution inside Runtime, Safety, Hardware, and customer-specific adapters.
- Treat customer signoff and production readiness as separate gates.

## Current capabilities

| Area | Status | Notes |
| --- | --- | --- |
| Voice task center | Available | ASR -> LLM -> TTS cascade, interruption handling, and text fallback. |
| Conversation Core | Phase 1, migration period | Thread / Turn / Generation ledger with session alias normalization and fail-closed conflict handling. |
| Robot field delivery | Pilot-ready | Voice, perception freshness, field events, runtime handoff, and control adapter boundaries. |
| Customer knowledge | Closed loop | Upload, preview, approval, indexing, retrieval evidence, and expiration handling. |
| Field events | Product path | Fall, stuck robot, motor fault, illegal parking, smoke/fire, and related onsite evidence flows. |
| MCP service | Available | Controlled tools and resources for MCP clients. |
| Enterprise audit | Available | Skill audit log, unified audit timeline, and exportable evidence packages. |

## Conversation Core and realtime voice

Askme separates a long-lived product conversation from a short-lived cloud realtime session:

```text
Person -> Thread -> Turn -> Generation -> Provider Session
```

- `Thread` is the user-visible logical conversation and survives provider reconnects.
- `Turn` is one auditable interaction from finalized user input to delivered robot response.
- `Generation` records one model/provider attempt inside a turn, including retries and replacement attempts.
- `Provider Session` is a temporary ASR, LLM, TTS, or speech-to-speech connection.

Phase 1 writes confirmed user text, delivered assistant responses, failed/cancelled turns, and multiple generation attempts into one local JSONL turn ledger. The legacy `ConversationManager` and voice gateway still keep compatibility projections during migration, so Conversation Core is the normative source for new turns but not yet the only physical history store.

Vision is currently an on-demand camera/VLM branch, not an always-on unified multimodal model. A visual question can capture the current frame and return a short answer, and explicit `auto_capture` can add a scene description to an ordinary prompt. The primary `config.yaml` disables `vision.enabled` and `vision.vlm_enabled` by default, while the board profile `config.board.yaml` explicitly enables both; the effective state therefore depends on the selected deployment profile. Raw image/snapshot IDs and capture timestamps are not yet linked back to the committed Turn/Generation, so continuous visual memory and product-grade auditable multimodality are not claimed.

Volcengine/Doubao realtime speech-to-speech support is optional and disabled by default under `voice.realtime`. The central route and two-phase `prepare → durable Turn/Generation → release PCM` safety patch now passes offline regression, and the legacy one-step release path fails closed. Robot commands, emergency stop, approvals/tools, and visual queries remain on the local cascade. `general_chat` is still not production-ready because this environment lacks online credentials, shadow privacy/stability evidence, and target-hardware acoustic acceptance; rollout remains `split → shadow → small general_chat`. See [docs/VOLCENGINE_REALTIME_VOICE.md](docs/VOLCENGINE_REALTIME_VOICE.md) and [docs/FULL_DUPLEX_VOICE.md](docs/FULL_DUPLEX_VOICE.md).

## Quick start

Install the package in editable mode:

```powershell
pip install -e ".[dev]"
```

Run the text runtime:

```powershell
python -m askme.blueprints.presets.text
```

Run the voice task center:

```powershell
python -m askme.blueprints.presets.voice
```

Run the MCP server:

```powershell
python -m askme.mcp.server
```

Docker and deployment paths are documented in [README.md](README.md), [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md), and [docker/README.md](docker/README.md).

## Latency evidence and product-readiness caveat

The earlier warm-system baseline used a first-stream-chunk TTFT that could include an empty delta. A corrected 20-sample measurement now distinguishes non-empty model text from a usable semantic clause:

| Stage | Baseline |
| --- | ---: |
| ASR endpoint confirmation | about 300 ms; prior methodology, target-device remeasurement pending |
| LLM first non-empty content, full runtime prompt | p50 961.8 ms / p95 1140.7 ms, n=20 |
| LLM first useful semantic clause | p50 1168.4 ms / p95 1378.8 ms, n=20 |
| First-clause <=10-character compliance | 95%, n=20 |
| False `[SILENT]` rate on direct questions | 0%, n=20 after persona fix |
| MiniMax warm provider-first PCM | p50 270.71 ms / p95 376.08 ms, n=20; physical audio and provider A/B pending |

The product target is p95 first useful audible response <= 1.2 s. The corrected LLM-only semantic-clause p95 already exceeds that end-to-end target; sequential prompt runs are not a controlled A/B and no prompt-only latency gain is claimed. Do not claim product-grade latency until `scripts/eval/report_voice_latency.py` contains measured evidence from the target hardware. The required target-hardware gates include physical first sound, interruption-to-physical-stop, false trigger rate, sample counts, device IDs, and AEC/full-duplex status.

Gate A code work is now in place: non-blocking warm-up of the actual voice model, isolated phrase-cache priming restricted to stable runtime keys, a v2 acoustic cache signature, a shared MiniMax SSE/WebSocket first/later-packet state machine, and an exact E-STOP endpoint that runs before mute, approval, and conversational admission gates. A 1.5 s long-tail fuse may play separate thinking feedback; empty deltas no longer extinguish it, cancellation/real payloads fence it, and semantic speech cancels an active cue before queuing TTS. That cue is perceived-wait feedback and never semantic-first-audio evidence. The configured 36/54 ms TTS thresholds remain experimental; normal intents retain 300 ms endpointing, while exact E-STOP can close at about 160 ms after all stability guards. The unified report refuses to pass fewer than 20 samples, malformed evidence, or incomplete target-hardware reports. These are software-readiness results, not measured physical latency gains.

The MiniMax online TTS collector is now implemented for the fixed 20-case corpus with phrase caches disabled and no physical playback. In the latest auditable 2026-07-19 run, MiniMax speech-2.8-turbo over WebSocket measured warm provider-first-PCM p50 270.71 ms / p95 376.08 ms and buffer-commit p50 277.01 ms / p95 379.32 ms; cold per-case connections with a 4.5 s case delay measured provider-first-PCM p50 631.75 ms / p95 2294.78 ms and buffer-commit p50 652.09 ms / p95 2314.36 ms. The cold path has a large long tail, and one no-delay cold rerun produced only 13/20 passed samples with 7 provider failures, so the product path must keep warm WebSocket reuse and background prewarm instead of opening a new TTS session per turn. Startup and immediate/pending runtime provider switches now prewarm in the background; replacement/shutdown cancels stale work and harvests uncooperative daemon workers with a 0.5 s total budget. This is still not physical first sound: the unified report remains `insufficient_evidence` until target-hardware `physical_first_nonzero_ms` and `barge_in_to_speaker_stop_ms` are measured. The collector now generates unique output names by default, refuses to overwrite existing evidence, and supports `--case-delay-ms` so cold/new-connection reruns can respect provider RPM limits.

Volcengine TTS V3 is now wired at the offline/software boundary: protocol codec, concurrency-safe WebSocket client, `TTSEngine` backend, configuration, health checks, dashboard state, online smoke path, and fixed 20-case collector. MiniMax remains the default TTS backend. Volcengine `resource_id` and `speaker` are account-specific and are not hard-coded; for Volcengine TTS the project `model` field maps to the `X-Api-Resource-Id` product/resource selector. This environment does not currently provide `VOLCENGINE_TTS_*` credentials, so there is no matching Volcengine online measured dataset and no TTS winner has been selected. Runtime state in `data/voice/system_control.json` can override YAML, so provider switching should use the control API or an explicit persistent-state update.

Target-hardware reports now use fail-closed `askme.full_duplex_hardware.v2`. `--latency-mode entry` and stopwatch runs are `manual` diagnostics, while WASAPI/loopback is reported separately as `render_chain`; neither can populate physical-first-sound or physical-stop gates. Product acceptance requires at least 20 strict `physical_acoustic` trials for each gate with automatic capture/reference, one monotonic clock, valid calibration, and zero dropped frames. Overlap stop timing additionally requires an `isolated_speaker_monitor` separate from the human-speech reference channel. An automatic physical capture adapter is not implemented yet, so the 20+20+20 hardware gate is not claimed complete.

## Development and verification

Default fast tests exclude slow tests through `pyproject.toml`:

```powershell
pytest
```

Common quality checks:

```powershell
ruff check .
mypy askme/
```

For voice readiness and latency evidence, use the project scripts and docs rather than hand-written claims:

- [docs/FULL_DUPLEX_VOICE.md](docs/FULL_DUPLEX_VOICE.md)
- [docs/VOICE_REALTIME_PRODUCT_REPORT_2026-07-18.md](docs/VOICE_REALTIME_PRODUCT_REPORT_2026-07-18.md)
- [docs/VOICE_LATENCY_EXECUTION_PLAN_2026-07-19.md](docs/VOICE_LATENCY_EXECUTION_PLAN_2026-07-19.md)
- [scripts/eval/report_voice_latency.py](scripts/eval/report_voice_latency.py)

## Contributing, security, and license

- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Architecture overview: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Repository layout: [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md)
- License: [MIT](LICENSE)
