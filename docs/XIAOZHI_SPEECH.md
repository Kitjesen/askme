# Xiaozhi speech helpers

AskMe speaks through `askme.voice.output.tts.TTSEngine`. Three helpers match the
open-source Xiaozhi server (`xinnan-tech/xiaozhi-esp32-server`) without taking
over its WebSocket stack. They live in `askme.voice.speech` and are already
called from TTS and the voice gateway.

Markdown cleanup is unchanged. Pronunciation runs after that cleanup and before
the TTS call. Emoji stripping for speech is unchanged; emotion is read from the
raw reply first.

## Pronunciation (`correct_words`)

Config key: `voice.tts.correct_words` in `config.yaml` and `config.board.yaml`.

Each entry is `原文|读法`. Keys are matched longest-first, with regex
metacharacters escaped. The synthesized text uses 读法. Captions and
`sentence_start` keep 原文 via `restore_display_text` / `TTSEngine.display_text_for`.

An empty list turns the built-in examples off. To add a word without editing
code, append a line to the yaml list or set:

```bash
ASKME_TTS_CORRECT_WORDS='方总|方总;AskMe|阿斯克米'
```

Entries are separated by semicolons or newlines and are added after the yaml
list. The built-in examples are 小算, 大算, 聚龙科创e谷, and 巨龙科创e谷.
算/蒜 are both suàn, so the spoken form only changes engines that misread 算.
伊 is the spoken name of the Latin letter in 科创e谷. ASR aliases such as
小蒜 and 小帅 stay in `voice.address_detection.aliases`; they are not TTS
replacements.

`correct_stream_chunk` is the optional streaming matcher (words grouped by
first character, longest first). The live TTS path applies the complete-string
API to each segment after the existing coalesce step. A token split that falls
outside that window is not rewritten until a segment contains the whole word.
Call `StreamCorrector` from the board if it splits inside a word, and `flush`
at the end of the utterance.

## Device emotion

`extract_emotion` returns the first emoji in the official map. 😂 is `funny`.
No mapped emoji, including an unknown emoji or a reply with no emoji at all,
becomes 🙂 / `happy`.

`build_llm_emotion_message` returns:

```json
{"type": "llm", "text": "🙂", "emotion": "happy", "session_id": "<session_id>"}
```

`push_device_emotion` serializes that object and calls `sender(json)` or
`sender.send(json)`. This repo has no device socket. `TTSEngine.speak` builds
the payload as speech is queued and sends it when
`set_device_emotion_sender` was called. `VoiceGatewayService.push_device_emotion`
is the same hook for a later board adapter. If a sender is registered on the
gateway, a recorded assistant reply is pushed as the turn is committed.
Register the socket on one side so the device does not receive two messages.

`device_session_id` under `voice.tts` is optional. The gateway uses the
conversation session id when it pushes from a recorded turn.

## TTS retry

`synthesize_with_retry` tries the synthesize call up to 5 times. Runtime
exceptions and empty results are retried. `TypeError` and `ValueError` are
not. Each failed attempt logs a warning with the original display text. Success
logs how many retries were used. After the last failure the helper logs an
error and returns `TTSRetryResult(ok=False)` with `TTSSynthesisFailure`. It
does not sleep and does not raise that failure.

`TTSEngine._generate_audio` wraps the local, Edge, MiniMax, and configured
Volcengine calls. A Volcengine setup that is missing PCM credentials is not
retried. Partial provider audio is still not followed by a second voice, same
as before. The cloud fallback runs only after the primary attempts are
exhausted, and that fallback call is retried on its own.
