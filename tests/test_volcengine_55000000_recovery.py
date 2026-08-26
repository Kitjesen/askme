from __future__ import annotations

import json
import queue
import threading

import scripts.demo.realtime_voice_chat as voice_chat
from askme.voice.core.realtime_contracts import (
    RealtimeVoiceEvent,
    RealtimeVoiceEventType,
    RealtimeVoiceSessionContext,
)
from askme.voice.realtime.volcengine_duplex import (
    VolcengineDuplexConfig,
    VolcengineDuplexDialogue,
)


def test_raw_55000000_event_degrades_the_provider_session() -> None:
    class _HandshakeResponse:
        headers = {"X-Tt-Logid": "provider-log-55000000"}

    class _WebSocket:
        def __init__(self) -> None:
            self.incoming: queue.Queue[str] = queue.Queue()
            self.incoming.put(
                json.dumps({"type": "session.created", "session": {"id": "dialog-1"}})
            )
            self.incoming.put(
                json.dumps(
                    {
                        "type": "error",
                        "error": {"code": "55000000", "message": "internal error"},
                    }
                )
            )
            self.timeout = 0.1
            self.handshake_response = _HandshakeResponse()

        def send(self, payload: str) -> None:
            del payload

        def recv(self) -> str:
            try:
                return self.incoming.get(timeout=self.timeout)
            except queue.Empty as exc:
                raise TimeoutError from exc

        def settimeout(self, timeout: float) -> None:
            self.timeout = timeout

        def close(self) -> None:
            return None

    websocket = _WebSocket()
    session = VolcengineDuplexDialogue(
        VolcengineDuplexConfig(enabled=True, api_key="configured"),
        connection_factory=lambda *args, **kwargs: websocket,
    )

    assert session.start(RealtimeVoiceSessionContext(session_id="local-1")) is True
    error = session.next_event(timeout=1.0)

    assert error is not None
    assert error.event_type is RealtimeVoiceEventType.ERROR
    assert error.error == "provider_error_55000000"
    assert error.metadata["provider_message"] == "internal error"
    snapshot = session.status_snapshot()
    assert snapshot["state"] == "degraded"
    assert snapshot["active"] is False
    assert snapshot["connected"] is False
    assert snapshot["log_id"] == "provider-log-55000000"
    session.close("test")


def test_55000000_closes_old_session_and_returns_to_ready_with_fresh_identity(
    monkeypatch,
    capsys,
) -> None:
    lifecycle: list[str] = []

    class _Session:
        def __init__(self, name: str, *, provider_error: str = "") -> None:
            self.name = name
            self.provider_error = provider_error
            self.context = None
            self.committed = threading.Event()
            self.event_sent = False
            self.close_calls = 0

        def start(self, context) -> bool:
            self.context = context
            lifecycle.append(f"{self.name}:start")
            return True

        def next_event(self, timeout=None):
            del timeout
            if self.event_sent or not self.committed.wait(timeout=0.01):
                return None
            self.event_sent = True
            if self.provider_error:
                return RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.ERROR,
                    error=self.provider_error,
                )
            return RealtimeVoiceEvent(
                event_type=RealtimeVoiceEventType.RESPONSE_DONE,
                text="recovered",
            )

        def offer_audio(self, frame) -> bool:
            del frame
            return True

        def finish_input(self) -> bool:
            self.committed.set()
            return True

        def status_snapshot(self):
            return {"last_error": self.provider_error if self.event_sent else ""}

        def close(self, reason) -> None:
            del reason
            self.close_calls += 1
            lifecycle.append(f"{self.name}:close")

    class _OutputStream:
        def __init__(self, name: str) -> None:
            self.name = name
            self.started = False
            self.stopped = False
            self.closed = False

        def start(self) -> None:
            self.started = True
            lifecycle.append(f"{self.name}:output-start")

        def write(self, pcm) -> None:
            del pcm

        def stop(self) -> None:
            self.stopped = True
            lifecycle.append(f"{self.name}:output-stop")

        def close(self) -> None:
            self.closed = True
            lifecycle.append(f"{self.name}:output-close")

    class _RawInputStream:
        def __init__(self, *, callback, **kwargs) -> None:
            del kwargs
            self.callback = callback

        def __enter__(self):
            self.callback(b"\x01\x00" * 320, 320, None, None)
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback

    class _SoundDevice:
        def __init__(self) -> None:
            self.outputs: list[_OutputStream] = []
            self.RawInputStream = _RawInputStream

        def check_input_settings(self, **kwargs) -> None:
            del kwargs

        def check_output_settings(self, **kwargs) -> None:
            del kwargs

        def RawOutputStream(self, **kwargs):
            del kwargs
            stream = _OutputStream(f"output-{len(self.outputs) + 1}")
            self.outputs.append(stream)
            return stream

    first = _Session("first", provider_error="provider_error_55000000")
    second = _Session("second")
    sessions = iter((first, second))
    prompts = iter(("", "", "", "", "q"))
    sounddevice = _SoundDevice()

    monkeypatch.setenv("VOLCENGINE_S2S_API_KEY", "configured")
    monkeypatch.setattr(voice_chat, "_load_sounddevice", lambda: sounddevice)
    monkeypatch.setattr(
        voice_chat,
        "build_realtime_dialogue",
        lambda config: next(sessions),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(prompts))
    monkeypatch.setattr(voice_chat.time, "sleep", lambda _: None)

    assert (
        voice_chat.main(
            [
                "--provider",
                "volcengine_duplex",
                "--input-device",
                "1",
                "--output-device",
                "3",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert output.count("Ready. This is half-duplex") == 2
    assert output.count("[error] provider_error_55000000") == 1
    assert output.count("[reconnect] provider session disconnected") == 1
    assert first.close_calls == 1
    assert second.close_calls == 1
    assert first.context is not None
    assert second.context is not None
    assert first.context.session_id != second.context.session_id
    assert first.context.dialog_id == first.context.session_id
    assert second.context.dialog_id == ""
    assert lifecycle.index("first:close") < lifecycle.index("second:start")
    assert lifecycle.index("output-1:output-close") < lifecycle.index("second:start")
    assert len(sounddevice.outputs) == 2
    assert all(
        stream.started and stream.stopped and stream.closed
        for stream in sounddevice.outputs
    )


def test_zero_frame_recording_is_not_committed_to_provider(monkeypatch) -> None:
    emitted: list[str] = []

    class _Session:
        def __init__(self, consumer: _Consumer) -> None:
            self.consumer = consumer
            self.finish_calls = 0

        def finish_input(self) -> bool:
            self.finish_calls += 1
            self.consumer.response_done.set()
            return True

    class _Sender:
        dropped_frames = 0
        callback_status_events = 0
        accepted_frames = 0
        failure = ""

        def audio_callback(self, *args) -> None:
            del args

        def flush(self, *, timeout: float) -> bool:
            del timeout
            return True

    class _Consumer:
        def __init__(self) -> None:
            self.failed = threading.Event()
            self.response_done = threading.Event()
            self.turn_done = threading.Event()
            self.failure_error = ""

        def begin_turn(self) -> None:
            self.response_done.clear()
            self.turn_done.clear()

        def abandon_turn(self) -> None:
            self.response_done.set()
            self.turn_done.set()

        def mark_commit(self) -> None:
            return None

        def cancel_commit(self) -> None:
            return None

        def finish_turn_playback(self) -> bool:
            return True

    class _InputStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback

    class _SoundDevice:
        @staticmethod
        def RawInputStream(**kwargs):
            del kwargs
            return _InputStream()

    prompts = iter(("", "", "q"))
    monkeypatch.setattr("builtins.input", lambda prompt: next(prompts))
    consumer = _Consumer()
    session = _Session(consumer)

    assert (
        voice_chat.run_push_to_talk(
            _SoundDevice(),
            session,
            _Sender(),
            consumer,
            input_device=1,
            emit=emitted.append,
        )
        == 0
    )

    assert session.finish_calls == 0
    assert any("no microphone audio captured" in line for line in emitted)


def test_provider_error_prints_safe_log_id_and_turn_audio_counters() -> None:
    emitted: list[str] = []

    class _Session:
        snapshot = {
            "log_id": "provider-log-55000000",
            "sent_audio_frames": 8,
            "dropped_input_frames": 1,
        }

        def status_snapshot(self):
            return dict(self.snapshot)

    session = _Session()
    consumer = voice_chat.EventConsumer(session, object(), emit=emitted.append)
    consumer.begin_turn()
    session.snapshot["sent_audio_frames"] = 11
    session.snapshot["dropped_input_frames"] = 2

    consumer.handle_event(
        RealtimeVoiceEvent(
            event_type=RealtimeVoiceEventType.ERROR,
            error="provider_error_55000000",
            metadata={"provider_message": "internal error"},
        )
    )

    assert emitted == [
        "[error] provider_error_55000000",
        "[diagnostic] provider_log_id=provider-log-55000000 "
        "turn_audio_frames=3 dropped_input_frames=1 "
        'provider_message="internal error"',
    ]
