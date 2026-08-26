#!/usr/bin/env python3
"""Interactive push-to-talk demo for AskMe realtime voice providers."""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.realtime_contracts import (
    RealtimeVoiceEvent,
    RealtimeVoiceEventType,
    RealtimeVoiceSessionContext,
)
from askme.voice.realtime.factory import build_realtime_dialogue


@dataclass
class _FlushMarker:
    done: threading.Event = field(default_factory=threading.Event)


_STOP = object()
PROVIDER_RECONNECT_EXIT = 75
MAX_PROVIDER_RECONNECTS = 3

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
CHANNELS = 1
CHUNK_MS = 20
POST_PLAYBACK_ACOUSTIC_SETTLE_S = 0.15
PROVIDER_KEY_ENV = {
    "qwen3_5_omni": "DASHSCOPE_API_KEY",
    "volcengine_duplex": "VOLCENGINE_S2S_API_KEY",
}


def _device_identifier(value: str) -> int | str:
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push-to-talk PC demo for AskMe realtime speech providers.",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDER_KEY_ENV),
        default="qwen3_5_omni",
        help="Realtime provider (default: qwen3_5_omni).",
    )
    parser.add_argument("--model", help="Override the provider's default model.")
    parser.add_argument("--voice", help="Override the provider's default voice.")
    parser.add_argument(
        "--workspace-id",
        help="Qwen workspace ID (default: DASHSCOPE_WORKSPACE_ID).",
    )
    parser.add_argument(
        "--region",
        choices=("cn-beijing", "ap-southeast-1"),
        help="Qwen workspace region (default: DASHSCOPE_REGION or cn-beijing).",
    )
    parser.add_argument(
        "--input-device",
        type=_device_identifier,
        help="sounddevice input index or name.",
    )
    parser.add_argument(
        "--output-device",
        type=_device_identifier,
        help="sounddevice output index or name.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit without connecting.",
    )
    return parser


def provider_config(
    args: argparse.Namespace,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Build the repository factory input without logging credentials."""

    realtime: dict[str, Any] = {
        "enabled": True,
        "mode": "general_chat",
        "provider": args.provider,
        "api_key": environ.get(PROVIDER_KEY_ENV[args.provider], "").strip(),
        "input_sample_rate": INPUT_SAMPLE_RATE,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "output_format": "pcm_s16le",
        "chunk_ms": CHUNK_MS,
    }
    if args.model:
        realtime["model"] = args.model
    if args.voice:
        realtime["voice"] = args.voice
    if args.provider == "qwen3_5_omni":
        realtime["workspace_id"] = (
            args.workspace_id or environ.get("DASHSCOPE_WORKSPACE_ID", "")
        ).strip()
        realtime["region"] = (
            args.region or environ.get("DASHSCOPE_REGION", "cn-beijing")
        ).strip().lower() or "cn-beijing"
    return {"voice": {"realtime": realtime}}


def build_session_context(*, recovery: bool = False) -> RealtimeVoiceSessionContext:
    session_id = f"pc-voice-demo-{uuid.uuid4().hex}"
    return RealtimeVoiceSessionContext(
        session_id=session_id,
        dialog_id="" if recovery else session_id,
        input_mode="push_to_talk",
        input_sample_rate=INPUT_SAMPLE_RATE,
        output_sample_rate=OUTPUT_SAMPLE_RATE,
        output_format="pcm_s16le",
        allow_tool_calls=False,
        allow_hardware_dispatch=False,
    )


def _default_device(sounddevice_module: Any, direction: str) -> int | str | None:
    raw = getattr(getattr(sounddevice_module, "default", None), "device", None)
    offset = 0 if direction == "input" else 1
    if isinstance(raw, (int, str)):
        value: Any = raw
    elif isinstance(raw, Sequence):
        try:
            value = raw[offset]
        except IndexError:
            return None
    else:
        return None
    if value in (None, -1):
        return None
    if isinstance(value, (int, str)):
        return value
    return None


def _device_supports(
    sounddevice_module: Any,
    direction: str,
    device: int | str,
) -> bool:
    checker = getattr(sounddevice_module, f"check_{direction}_settings")
    sample_rate = INPUT_SAMPLE_RATE if direction == "input" else OUTPUT_SAMPLE_RATE
    try:
        checker(
            device=device,
            channels=CHANNELS,
            dtype="int16",
            samplerate=sample_rate,
        )
    except Exception:
        return False
    return True


def _choose_audio_device(
    sounddevice_module: Any,
    direction: str,
    requested: int | str | None,
) -> int | str:
    if requested is not None:
        if _device_supports(sounddevice_module, direction, requested):
            return requested
        sample_rate = INPUT_SAMPLE_RATE if direction == "input" else OUTPUT_SAMPLE_RATE
        raise RuntimeError(
            f"Audio {direction} device {requested!r} does not support "
            f"{sample_rate} Hz mono int16. Run with --list-devices and choose "
            "a compatible device (on this PC, try --input-device 1 --output-device 3)."
        )

    devices = list(sounddevice_module.query_devices())
    channel_key = f"max_{direction}_channels"
    candidates: list[int | str] = []
    default = _default_device(sounddevice_module, direction)
    if default is not None:
        candidates.append(default)
    candidates.extend(
        index
        for index, info in enumerate(devices)
        if int(info.get(channel_key, 0) or 0) > 0 and index not in candidates
    )
    for candidate in candidates:
        if _device_supports(sounddevice_module, direction, candidate):
            return candidate
    sample_rate = INPUT_SAMPLE_RATE if direction == "input" else OUTPUT_SAMPLE_RATE
    raise RuntimeError(
        f"No audio {direction} device supports {sample_rate} Hz mono int16. "
        "Run with --list-devices, then pass --input-device/--output-device explicitly."
    )


def choose_audio_devices(
    sounddevice_module: Any,
    input_device: int | str | None,
    output_device: int | str | None,
) -> tuple[int | str, int | str]:
    """Select devices only after probing the demo's native PCM rates."""

    return (
        _choose_audio_device(sounddevice_module, "input", input_device),
        _choose_audio_device(sounddevice_module, "output", output_device),
    )


def list_audio_devices(
    sounddevice_module: Any,
    *,
    emit: Callable[[str], None] = print,
) -> None:
    """Print a compact inventory with compatibility at the demo's PCM rates."""

    devices = list(sounddevice_module.query_devices())
    default_input = _default_device(sounddevice_module, "input")
    default_output = _default_device(sounddevice_module, "output")
    emit("Audio devices (required: input 16 kHz, output 24 kHz, mono int16):")
    for index, info in enumerate(devices):
        input_channels = int(info.get("max_input_channels", 0) or 0)
        output_channels = int(info.get("max_output_channels", 0) or 0)
        input_ok = input_channels > 0 and _device_supports(sounddevice_module, "input", index)
        output_ok = output_channels > 0 and _device_supports(sounddevice_module, "output", index)
        defaults = (
            "".join(
                (
                    "I" if index == default_input else "",
                    "O" if index == default_output else "",
                )
            )
            or "-"
        )
        name = " ".join(str(info.get("name", "unknown")).splitlines()).strip()
        emit(
            f"  [{index}] {name} | default={defaults} | "
            f"in={input_channels} (16k={'yes' if input_ok else 'no'}) | "
            f"out={output_channels} (24k={'yes' if output_ok else 'no'})"
        )


class EventConsumer:
    """Consume normalized provider events and own PC playback/turn signals."""

    def __init__(
        self,
        session: Any,
        output_stream: Any,
        *,
        emit: Callable[[str], None] = print,
    ) -> None:
        self._session = session
        self._output_stream = output_stream
        self._emit = emit
        self._commit_at: float | None = None
        self._latency_reported = False
        self._input_partial = ""
        self._input_printed = False
        self._output_parts: list[str] = []
        self._usage: Mapping[str, Any] | None = None
        self._played_audio_this_turn = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.response_done = threading.Event()
        self.response_done.set()
        self.turn_done = threading.Event()
        self.turn_done.set()
        self.failed = threading.Event()
        self.failure_error = ""

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="realtime-demo-event-consumer",
            daemon=True,
        )
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def close(self, *, timeout: float = 2.0) -> bool:
        self.request_stop()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        if thread is not None and thread.is_alive():
            return False
        self._thread = None
        return True

    def begin_turn(self) -> None:
        self._commit_at = None
        self._latency_reported = False
        self._input_partial = ""
        self._input_printed = False
        self._output_parts.clear()
        self._usage = None
        self._played_audio_this_turn = False
        self.failure_error = ""
        self.response_done.clear()
        self.turn_done.clear()

    def mark_commit(self, *, now: float | None = None) -> None:
        self._commit_at = time.perf_counter() if now is None else now
        self._latency_reported = False

    def cancel_commit(self) -> None:
        self._commit_at = None

    def handle_event(
        self,
        event: RealtimeVoiceEvent,
        *,
        now: float | None = None,
    ) -> None:
        event_type = event.event_type
        if event_type is RealtimeVoiceEventType.INPUT_TRANSCRIPT_DELTA:
            if event.transcript:
                self._input_partial = event.transcript
            return
        if event_type is RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL:
            transcript = event.transcript.strip() or self._input_partial.strip()
            if transcript:
                self._emit(f"[you] {transcript}")
                self._input_printed = True
            return
        if event_type is RealtimeVoiceEventType.OUTPUT_TEXT_DELTA:
            if event.text:
                if event.metadata.get("authoritative_final"):
                    self._output_parts[:] = [event.text]
                else:
                    self._output_parts.append(event.text)
            return
        if event_type is RealtimeVoiceEventType.USAGE:
            usage = event.metadata.get("usage")
            self._usage = usage if isinstance(usage, Mapping) else {}
            return
        if event_type is RealtimeVoiceEventType.RESPONSE_DONE:
            if not self._input_printed and self._input_partial.strip():
                self._emit(f"[you] {self._input_partial.strip()}")
            output_text = event.text.strip() or "".join(self._output_parts).strip()
            if output_text:
                self._emit(f"[assistant] {output_text}")
            if self._usage is not None:
                rendered_usage = json.dumps(
                    dict(self._usage),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                self._emit(f"[usage] {rendered_usage}")
            self._commit_at = None
            self.response_done.set()
            return
        if event_type is RealtimeVoiceEventType.ERROR:
            error = " ".join((event.error or "provider_error").splitlines()).strip()
            self._emit(f"[error] {error or 'provider_error'}")
            self.failure_error = error or "provider_error"
            self.failed.set()
            self.response_done.set()
            self.turn_done.set()
            return
        if event_type is RealtimeVoiceEventType.INTERRUPTED:
            self._emit("[session] response interrupted")
            self.response_done.set()
            return
        if event_type is RealtimeVoiceEventType.SESSION_CLOSED:
            if not self._stop.is_set():
                self._emit("[error] provider session closed")
                self.failure_error = "provider_session_closed"
                self.failed.set()
            self.response_done.set()
            self.turn_done.set()
            return
        if event_type is not RealtimeVoiceEventType.OUTPUT_AUDIO:
            return
        audio = event.audio
        if audio is None or audio.sample_rate != OUTPUT_SAMPLE_RATE or audio.channels != CHANNELS:
            self._emit("[error] provider returned an unsupported output audio shape")
            self.failure_error = "unsupported_output_audio_shape"
            self.failed.set()
            self.response_done.set()
            self.turn_done.set()
            return
        try:
            self._output_stream.write(audio.pcm)
        except Exception as exc:
            self._emit(f"[error] audio playback failed: {type(exc).__name__}")
            self.failure_error = f"audio_playback_{type(exc).__name__}"
            self.failed.set()
            self.response_done.set()
            self.turn_done.set()
            return
        self._played_audio_this_turn = True
        if self._commit_at is not None and not self._latency_reported:
            observed_at = time.perf_counter() if now is None else now
            elapsed_ms = max(0.0, (observed_at - self._commit_at) * 1000.0)
            self._emit(f"[latency] commit-to-first-PCM: {elapsed_ms:.1f} ms")
            self._latency_reported = True

    def finish_turn_playback(self) -> bool:
        """Drain physical playback on the interaction thread before reopening input."""

        if self.failed.is_set():
            self.turn_done.set()
            return False
        if not self._played_audio_this_turn:
            self.turn_done.set()
            return True
        stop = getattr(self._output_stream, "stop", None)
        start = getattr(self._output_stream, "start", None)
        if not callable(stop) or not callable(start):
            self.turn_done.set()
            return True
        try:
            stop(ignore_errors=False)
            time.sleep(POST_PLAYBACK_ACOUSTIC_SETTLE_S)
            start()
        except Exception as exc:
            self._emit(f"[error] audio playback drain failed: {type(exc).__name__}")
            self.failure_error = f"audio_playback_drain_{type(exc).__name__}"
            self.failed.set()
        self.turn_done.set()
        return not self.failed.is_set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._session.next_event(timeout=0.1)
            except Exception as exc:
                if not self._stop.is_set():
                    self._emit(f"[error] provider event consumer failed: {type(exc).__name__}")
                    self.failure_error = f"event_consumer_{type(exc).__name__}"
                    self.failed.set()
                    self.response_done.set()
                    self.turn_done.set()
                return
            if event is not None:
                self.handle_event(event)


class AudioSender:
    """Move microphone PCM from a realtime callback to a bounded queue."""

    def __init__(self, session: Any, *, queue_size: int = 25) -> None:
        self._session = session
        self._queue: queue.Queue[bytes | _FlushMarker | object] = queue.Queue(
            maxsize=max(1, queue_size)
        )
        self._thread: threading.Thread | None = None
        self._failed = threading.Event()
        self._failure = ""
        self._accepting = True
        self.dropped_frames = 0
        self.callback_status_events = 0

    @property
    def failure(self) -> str:
        return self._failure

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="realtime-demo-audio-sender",
            daemon=True,
        )
        self._thread.start()

    def audio_callback(
        self,
        indata: Any,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Copy one PortAudio buffer without making a provider call."""

        del frames, time_info
        if status:
            self.callback_status_events += 1
        if not self._accepting:
            self.dropped_frames += 1
            return
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            self.dropped_frames += 1

    def flush(self, *, timeout: float = 5.0) -> bool:
        """Wait until all PCM queued before this call has reached the session."""

        thread = self._thread
        if thread is None or not thread.is_alive():
            return self._queue.empty() and not self._failed.is_set()
        marker = _FlushMarker()
        try:
            self._queue.put(marker, timeout=max(0.0, timeout))
        except queue.Full:
            return False
        return marker.done.wait(timeout=max(0.0, timeout)) and not self._failed.is_set()

    def close(self, *, timeout: float = 5.0) -> None:
        self._accepting = False
        thread = self._thread
        if thread is None:
            return
        self.flush(timeout=timeout)
        try:
            self._queue.put(_STOP, timeout=max(0.0, timeout))
        except queue.Full:
            return
        thread.join(timeout=max(0.0, timeout))
        self._thread = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                if isinstance(item, _FlushMarker):
                    item.done.set()
                    continue
                if self._failed.is_set():
                    continue
                if not isinstance(item, bytes):
                    self._failure = "invalid_audio_queue_item"
                    self._failed.set()
                    continue
                frame = VoiceMediaFrame(
                    pcm=item,
                    sample_rate=INPUT_SAMPLE_RATE,
                    channels=CHANNELS,
                )
                try:
                    accepted = self._session.offer_audio(frame)
                except Exception as exc:  # provider boundary
                    self._failure = type(exc).__name__
                    self._failed.set()
                else:
                    if not accepted:
                        self._failure = "offer_audio_rejected"
                        self._failed.set()
            finally:
                self._queue.task_done()


def _load_sounddevice() -> Any:
    try:
        import sounddevice
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is not installed; install the project dependencies first"
        ) from exc
    return sounddevice


def _last_session_error(session: Any) -> str:
    try:
        snapshot = session.status_snapshot()
    except Exception:
        return "provider_error"
    error = str(snapshot.get("last_error") or "provider_error")
    return " ".join(error.splitlines()).strip() or "provider_error"


def provider_failure_is_reconnectable(error: str) -> bool:
    """Limit automatic retries to transient provider transport/server failures."""

    if error in {
        "finish_input_commit_timeout",
        "finish_input_mute_timeout",
        "provider_connection_closed",
        "provider_error_45000003",
        "provider_send_timeout",
        "provider_session_closed",
    }:
        return True
    return error.startswith("provider_error_5")


def _consumer_failure_exit(consumer: EventConsumer) -> int:
    return (
        PROVIDER_RECONNECT_EXIT
        if provider_failure_is_reconnectable(consumer.failure_error)
        else 1
    )


def provider_start_hint(
    provider: str,
    error: str,
    *,
    workspace_configured: bool,
) -> str:
    """Return an actionable, secret-free hint for a known startup failure."""

    if provider != "qwen3_5_omni" or error != "provider_connection_closed":
        return ""
    checks = "检查百炼账户余额、欠费状态和 Qwen3.5-Omni 模型权限"
    if workspace_configured:
        return checks + "，并确认 API Key 与 Workspace 属于同一区域。"
    return "先配置 DASHSCOPE_WORKSPACE_ID；然后" + checks + "。"


def _read_command(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


def run_push_to_talk(
    sounddevice_module: Any,
    session: Any,
    sender: AudioSender,
    consumer: EventConsumer,
    *,
    input_device: int | str,
    emit: Callable[[str], None] = print,
) -> int:
    """Run sequential capture/commit/playback turns on one provider session."""

    emit("Ready. This is half-duplex: record one turn, then wait for its response.")
    while not consumer.failed.is_set():
        command = _read_command("\n[Enter] start talking, [q] quit: ")
        if command == "q":
            return 0
        if command:
            emit("Type q to quit, or press Enter to start recording.")
            continue

        consumer.begin_turn()
        dropped_before = sender.dropped_frames
        status_before = sender.callback_status_events
        quit_after_turn = False
        try:
            with sounddevice_module.RawInputStream(
                samplerate=INPUT_SAMPLE_RATE,
                blocksize=INPUT_SAMPLE_RATE * CHUNK_MS // 1000,
                device=input_device,
                channels=CHANNELS,
                dtype="int16",
                callback=sender.audio_callback,
            ):
                stop_command = _read_command("[recording] Speak now; press Enter to stop: ")
                quit_after_turn = stop_command == "q"
        except Exception as exc:
            emit(f"[error] microphone stream failed: {type(exc).__name__}")
            emit("Run --list-devices and select a target-rate-compatible input device.")
            return 1

        if consumer.failed.is_set():
            return _consumer_failure_exit(consumer)

        dropped = sender.dropped_frames - dropped_before
        status_events = sender.callback_status_events - status_before
        if dropped:
            emit(f"[warning] microphone queue dropped {dropped} frame(s)")
        if status_events:
            emit(f"[warning] PortAudio reported {status_events} callback status event(s)")

        if not sender.flush(timeout=10.0):
            if consumer.failed.is_set():
                return _consumer_failure_exit(consumer)
            error = sender.failure or "audio_sender_flush_timeout"
            emit(f"[error] microphone upload failed: {error}")
            return 1

        consumer.mark_commit()
        try:
            committed = session.finish_input()
        except Exception as exc:
            consumer.cancel_commit()
            if consumer.failed.is_set():
                return _consumer_failure_exit(consumer)
            emit(f"[error] input commit failed: {type(exc).__name__}")
            return 1
        if not committed:
            consumer.cancel_commit()
            session_error = _last_session_error(session)
            if consumer.failed.is_set():
                return _consumer_failure_exit(consumer)
            if provider_failure_is_reconnectable(session_error):
                return PROVIDER_RECONNECT_EXIT
            emit(f"[error] input commit failed: {session_error}")
            return 1

        emit("[waiting] response...")
        while not consumer.response_done.wait(timeout=0.1):
            pass
        if consumer.failed.is_set():
            return _consumer_failure_exit(consumer)
        if not consumer.finish_turn_playback():
            return _consumer_failure_exit(consumer)
        if quit_after_turn:
            return 0
    return _consumer_failure_exit(consumer)


def run_demo(
    args: argparse.Namespace,
    sounddevice_module: Any,
    *,
    recovery: bool = False,
) -> int:
    config = provider_config(args, os.environ)
    realtime = config["voice"]["realtime"]
    key_env = PROVIDER_KEY_ENV[args.provider]
    if not realtime["api_key"]:
        print(f"[error] {key_env} is not set in the environment or repository .env")
        return 2
    if args.provider == "qwen3_5_omni" and not realtime.get("workspace_id"):
        print(
            "[error] DASHSCOPE_WORKSPACE_ID is unset. Current Qwen access "
            "requires a Workspace ID and its regional endpoint."
        )
        return 2

    try:
        input_device, output_device = choose_audio_devices(
            sounddevice_module,
            args.input_device,
            args.output_device,
        )
    except Exception as exc:
        print(f"[error] {exc}")
        return 2

    session = build_realtime_dialogue(config)
    if session is None:
        print("[error] provider configuration is invalid; check provider, model, and region")
        return 2

    print(f"Audio route: input={input_device!r}, output={output_device!r}")
    try:
        output_stream = sounddevice_module.RawOutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            device=output_device,
            channels=CHANNELS,
            dtype="int16",
        )
    except Exception as exc:
        print(f"[error] speaker stream failed: {type(exc).__name__}")
        print("Run --list-devices and select a target-rate-compatible output device.")
        return 1

    try:
        output_stream.start()
    except Exception as exc:
        print(f"[error] speaker stream failed: {type(exc).__name__}")
        print("Run --list-devices and select a target-rate-compatible output device.")
        try:
            output_stream.close()
        except Exception:
            pass
        return 1

    sender: AudioSender | None = None
    consumer: EventConsumer | None = None
    try:
        session_context = build_session_context(recovery=recovery)
        try:
            started = session.start(session_context)
        except Exception as exc:
            print(f"[error] provider start failed: {type(exc).__name__}")
            return 1
        if not started:
            start_error = _last_session_error(session)
            print(f"[error] provider start failed: {start_error}")
            hint = provider_start_hint(
                args.provider,
                start_error,
                workspace_configured=bool(realtime.get("workspace_id")),
            )
            if hint:
                print(f"[hint] {hint}")
            return 1

        consumer = EventConsumer(session, output_stream)
        sender = AudioSender(session)
        consumer.start()
        sender.start()
        print(f"Connected to {args.provider}.")
        return run_push_to_talk(
            sounddevice_module,
            session,
            sender,
            consumer,
            input_device=input_device,
        )
    finally:
        if sender is not None:
            sender.close()
        if consumer is not None:
            consumer.request_stop()
        try:
            session.close("pc_demo_shutdown")
        except Exception:
            pass
        consumer_closed = True
        if consumer is not None:
            consumer_closed = consumer.close()
        if consumer_closed:
            try:
                output_stream.stop()
            except Exception:
                pass
            try:
                output_stream.close()
            except Exception:
                pass
        else:
            print("[error] event consumer did not stop; output stream left open")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(REPO_ROOT / ".env", override=False)
    try:
        sounddevice_module = _load_sounddevice()
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 2

    if args.list_devices:
        try:
            list_audio_devices(sounddevice_module)
        except Exception as exc:
            print(f"[error] audio device query failed: {type(exc).__name__}")
            return 1
        return 0

    reconnects = 0
    while True:
        try:
            outcome = run_demo(
                args,
                sounddevice_module,
                recovery=reconnects > 0,
            )
        except KeyboardInterrupt:
            print("\nStopping realtime voice demo.")
            return 130
        if outcome != PROVIDER_RECONNECT_EXIT:
            return outcome
        reconnects += 1
        if reconnects > MAX_PROVIDER_RECONNECTS:
            print("[error] provider disconnected repeatedly; giving up after 3 reconnects")
            return 1
        print(
            f"[reconnect] provider session disconnected; reconnecting "
            f"({reconnects}/{MAX_PROVIDER_RECONNECTS})..."
        )
        time.sleep(min(2.0, 0.5 * reconnects))


if __name__ == "__main__":
    raise SystemExit(main())
