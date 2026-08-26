# Voice latency corpora

`tts_zh_20_v1.json` is the fixed Chinese corpus for MiniMax/Volcengine TTS
latency comparisons. A decision-grade run must:

1. use every `case_id` exactly once per provider and preserve the same text;
2. alternate providers by case instead of running all of one provider first;
3. bypass phrase caches and label cold/warm connection state in the raw capture;
4. record provider first PCM from the client receive boundary, not playback;
5. keep playback first nonzero and first-word integrity as separate measurements;
6. emit `evidence_type: measured` only for observed timestamps.

Normalize each provider run as `askme.voice_latency_experiment.v1` with
`stage: tts`, `corpus_id: askme-tts-zh-20-v1`, and samples such as:

```json
{
  "case_id": "tts-zh-01",
  "provider_first_pcm_ms": 0,
  "buffer_commit_ms": 0,
  "physical_first_nonzero_ms": 0
}
```

Run both inputs through the fail-closed report:

```powershell
uv run --no-sync python scripts/eval/measure_minimax_tts_latency.py `
  --mode warm `
  --out artifacts/voice/minimax-tts-warm.json

uv run --no-sync python scripts/eval/report_voice_latency.py `
  --experiment artifacts/voice/minimax-tts.json `
  --experiment artifacts/voice/volcengine-tts.json `
  --out artifacts/voice/latency-report.json
```

Zeroes above are schema placeholders, not measured evidence. Do not publish a
provider winner until the online adapters, account resources, audio quality,
and target-hardware checks have all passed.
