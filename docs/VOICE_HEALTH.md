# Voice Health Check

`voice-health` is the software-only preflight for the askme voice stack. It checks
configuration shape, local ASR/VAD/KWS/TTS model files, importable voice
dependencies, runtime-bridge config, and the offline status snapshot contract.

It does not open the microphone, play audio, call cloud ASR/TTS, or contact the
NOVA Dog runtime. Use live bench scripts for hardware and network exercise.

```bash
python -m askme runtime voice-health
python -m askme runtime voice-health --json
python scripts/voice_health.py --json
```

The `--live` flag marks the report as a live preflight request while keeping the
same no-hardware behavior:

```bash
python -m askme runtime voice-health --live --json
```

Top-level booleans are intended for automation:

- `config_ok`: `voice:` config exists.
- `models_ok`: required local model files are present, with KWS treated as OK
  when no wake words are configured.
- `asr_ok`, `vad_ok`, `kws_ok`, `tts_ok`: subsystem readiness.
- `runtime_bridge_ok`: bridge config is internally consistent; disabled bridge
  is valid.
- `health_snapshot_ok`: offline snapshot still matches the live
  `AudioAgent.status_snapshot()` field contract.
- `hardware_required`: true only when `--live` was requested.

Exit code is `0` when the offline voice stack is ready and `1` when the report
is degraded.
