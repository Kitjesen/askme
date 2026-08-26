from __future__ import annotations

import threading

import scripts.demo.realtime_voice_chat as voice_chat
from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.realtime_contracts import RealtimeVoiceEvent, RealtimeVoiceEventType
from scripts.demo.realtime_voice_chat import (
    AudioSender,
    EventConsumer,
    build_parser,
    build_session_context,
    choose_audio_devices,
    provider_config,
    provider_start_hint,
)


class _Session:
    def __init__(self) -> None:
        self.offered = []

    def offer_audio(self, frame):
        self.offered.append(frame)
        return True


def test_audio_callback_never_performs_network_io() -> None:
    session = _Session()
    sender = AudioSender(session, queue_size=2)

    sender.audio_callback(b"\x01\x00" * 320, 320, None, None)

    assert session.offered == []


def test_sender_worker_forwards_pcm_and_flushes_before_commit() -> None:
    session = _Session()
    sender = AudioSender(session, queue_size=2)
    pcm = b"\x02\x00" * 320

    sender.start()
    sender.audio_callback(pcm, 320, None, None)

    assert sender.flush(timeout=1.0)
    sender.close()
    assert len(session.offered) == 1
    assert session.offered[0].pcm == pcm
    assert session.offered[0].sample_rate == 16_000
    assert session.offered[0].channels == 1
    assert sender.accepted_frames == 1
    assert sender.accepted_bytes == len(pcm)


def test_cli_defaults_to_qwen_and_uses_the_dashscope_key() -> None:
    args = build_parser().parse_args([])

    config = provider_config(
        args,
        {
            "DASHSCOPE_API_KEY": "dashscope-secret",
            "DASHSCOPE_WORKSPACE_ID": "workspace-123",
            "DASHSCOPE_REGION": "ap-southeast-1",
        },
    )

    realtime = config["voice"]["realtime"]
    assert args.provider == "qwen3_5_omni"
    assert realtime["provider"] == "qwen3_5_omni"
    assert realtime["api_key"] == "dashscope-secret"
    assert realtime["workspace_id"] == "workspace-123"
    assert realtime["region"] == "ap-southeast-1"
    assert realtime["input_sample_rate"] == 16_000
    assert realtime["output_sample_rate"] == 24_000


def test_demo_session_is_manual_ptt_without_action_capabilities() -> None:
    context = build_session_context()

    assert context.input_mode == "push_to_talk"
    assert context.input_sample_rate == 16_000
    assert context.output_sample_rate == 24_000
    assert context.allow_tool_calls is False
    assert context.allow_hardware_dispatch is False


def test_qwen_close_failure_explains_workspace_and_account_checks() -> None:
    hint = provider_start_hint(
        "qwen3_5_omni",
        "provider_connection_closed",
        workspace_configured=False,
    )

    assert "DASHSCOPE_WORKSPACE_ID" in hint
    assert "余额" in hint
    assert "模型权限" in hint


def test_default_devices_fall_back_to_the_first_target_rate_compatible_pair() -> None:
    class _Default:
        device = (2, 3)

    class _SoundDevice:
        default = _Default()

        @staticmethod
        def query_devices():
            return [
                {"name": "compatible mic", "max_input_channels": 1, "max_output_channels": 0},
                {"name": "compatible speaker", "max_input_channels": 0, "max_output_channels": 2},
                {
                    "name": "incompatible default mic",
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                },
                {
                    "name": "incompatible default speaker",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
            ]

        @staticmethod
        def check_input_settings(*, device, **kwargs):
            if device == 2:
                raise RuntimeError("unsupported rate")

        @staticmethod
        def check_output_settings(*, device, **kwargs):
            if device == 3:
                raise RuntimeError("unsupported rate")

    assert choose_audio_devices(_SoundDevice(), None, None) == (0, 1)


def test_first_output_pcm_is_played_and_reports_commit_latency() -> None:
    class _Output:
        def __init__(self) -> None:
            self.writes = []

        def write(self, pcm):
            self.writes.append(pcm)

    output = _Output()
    lines = []
    consumer = EventConsumer(None, output, emit=lines.append)
    pcm = b"\x03\x00" * 480
    consumer.begin_turn()
    consumer.mark_commit(now=10.0)

    consumer.handle_event(
        RealtimeVoiceEvent(
            event_type=RealtimeVoiceEventType.OUTPUT_AUDIO,
            audio=VoiceMediaFrame(pcm=pcm, sample_rate=24_000, channels=1),
        ),
        now=10.125,
    )

    assert output.writes == [pcm]
    assert lines == ["[latency] commit-to-first-PCM: 125.0 ms"]


def test_event_consumer_prints_transcript_text_and_usage_for_each_turn() -> None:
    lines = []
    consumer = EventConsumer(None, object(), emit=lines.append)
    consumer.begin_turn()

    consumer.handle_event(
        RealtimeVoiceEvent(
            event_type=RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL,
            transcript="hello",
            is_final=True,
        )
    )
    consumer.handle_event(
        RealtimeVoiceEvent(
            event_type=RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
            text="world",
        )
    )
    consumer.handle_event(
        RealtimeVoiceEvent(
            event_type=RealtimeVoiceEventType.USAGE,
            metadata={"usage": {"input_tokens": 3, "output_tokens": 4}},
        )
    )
    consumer.handle_event(RealtimeVoiceEvent(event_type=RealtimeVoiceEventType.RESPONSE_DONE))
    assert consumer.response_done.is_set()
    assert consumer.turn_done.is_set() is False
    assert consumer.finish_turn_playback()

    assert lines == [
        "[you] hello",
        "[assistant] world",
        '[usage] {"input_tokens": 3, "output_tokens": 4}',
    ]
    assert consumer.turn_done.is_set()


def test_event_consumer_uses_authoritative_final_text_without_duplication() -> None:
    lines = []
    consumer = EventConsumer(None, object(), emit=lines.append)
    consumer.begin_turn()

    consumer.handle_event(
        RealtimeVoiceEvent(
            event_type=RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
            text="你",
        )
    )
    consumer.handle_event(
        RealtimeVoiceEvent(
            event_type=RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
            text="好",
        )
    )
    consumer.handle_event(
        RealtimeVoiceEvent(
            event_type=RealtimeVoiceEventType.OUTPUT_TEXT_DELTA,
            text="你好。",
            metadata={"authoritative_final": True},
        )
    )
    consumer.handle_event(
        RealtimeVoiceEvent(
            event_type=RealtimeVoiceEventType.RESPONSE_DONE,
            text="你好。",
        )
    )

    assert lines == ["[assistant] 你好。"]


def test_response_done_drains_speaker_before_next_turn_is_released(monkeypatch) -> None:
    class _Output:
        def __init__(self) -> None:
            self.calls = []
            self.stop_started = threading.Event()
            self.release_stop = threading.Event()

        def stop(self, *, ignore_errors) -> None:
            assert ignore_errors is False
            self.calls.append("stop")
            self.stop_started.set()
            assert self.release_stop.wait(timeout=1.0)

        def start(self) -> None:
            self.calls.append("start")

        def write(self, pcm) -> None:
            del pcm
            self.calls.append("write")

    output = _Output()
    consumer = EventConsumer(None, output, emit=lambda _: None)
    consumer.begin_turn()
    monkeypatch.setattr(voice_chat, "POST_PLAYBACK_ACOUSTIC_SETTLE_S", 0.0)

    consumer.handle_event(
        RealtimeVoiceEvent(
            event_type=RealtimeVoiceEventType.OUTPUT_AUDIO,
            audio=VoiceMediaFrame(
                pcm=b"\x01\x00" * 480,
                sample_rate=24_000,
                channels=1,
            ),
        )
    )

    consumer.handle_event(
        RealtimeVoiceEvent(event_type=RealtimeVoiceEventType.RESPONSE_DONE)
    )
    assert consumer.response_done.is_set()
    assert consumer.turn_done.is_set() is False
    drain_thread = threading.Thread(target=consumer.finish_turn_playback)
    drain_thread.start()

    assert output.stop_started.wait(timeout=1.0)
    assert consumer.turn_done.is_set() is False
    output.release_stop.set()
    drain_thread.join(timeout=1.0)
    assert output.calls == ["write", "stop", "start"]
    assert consumer.turn_done.is_set()


def test_consumer_close_timeout_retains_thread_for_safe_retry() -> None:
    class _BlockingSession:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def next_event(self, timeout=None):
            del timeout
            self.entered.set()
            self.release.wait(timeout=1.0)
            return None

    session = _BlockingSession()
    consumer = EventConsumer(session, object(), emit=lambda _: None)
    consumer.start()
    assert session.entered.wait(timeout=1.0)

    assert consumer.close(timeout=0.01) is False
    session.release.set()
    assert consumer.close(timeout=1.0) is True


def test_provider_failure_requests_reconnect_instead_of_ending_demo() -> None:
    class _Consumer:
        failed = threading.Event()
        failure_error = "provider_connection_closed"

    _Consumer.failed.set()

    assert (
        voice_chat.run_push_to_talk(
            object(),
            object(),
            object(),
            _Consumer(),
            input_device=1,
            emit=lambda _: None,
        )
        == 75
    )


def test_local_audio_failure_does_not_request_provider_reconnect() -> None:
    class _Consumer:
        failed = threading.Event()
        failure_error = "audio_playback_PortAudioError"

    _Consumer.failed.set()

    assert (
        voice_chat.run_push_to_talk(
            object(),
            object(),
            object(),
            _Consumer(),
            input_device=1,
            emit=lambda _: None,
        )
        == 1
    )


def test_only_transport_and_provider_5xx_failures_are_reconnectable() -> None:
    assert voice_chat.provider_failure_is_reconnectable("provider_connection_closed")
    assert voice_chat.provider_failure_is_reconnectable("provider_error_50700000")
    assert voice_chat.provider_failure_is_reconnectable("provider_error_45000003")
    assert voice_chat.provider_failure_is_reconnectable("provider_send_timeout")
    assert not voice_chat.provider_failure_is_reconnectable("provider_error_45000004")
    assert not voice_chat.provider_failure_is_reconnectable("provider_receive_error")
    assert not voice_chat.provider_failure_is_reconnectable("provider_frame_error")
    assert not voice_chat.provider_failure_is_reconnectable("provider_event_error")
    assert not voice_chat.provider_failure_is_reconnectable("provider_send_error")
    assert not voice_chat.provider_failure_is_reconnectable("provider_send_payload_error")
    assert not voice_chat.provider_failure_is_reconnectable(
        "audio_playback_PortAudioError"
    )


def test_main_restarts_demo_after_reconnectable_provider_failure(monkeypatch) -> None:
    outcomes = iter((75, 0))
    calls = []
    recovery_flags = []
    monkeypatch.setattr(voice_chat, "_load_sounddevice", lambda: object())
    monkeypatch.setattr(
        voice_chat,
        "run_demo",
        lambda args, sounddevice_module, *, recovery=False: (
            calls.append(sounddevice_module),
            recovery_flags.append(recovery),
            next(outcomes),
        )[-1],
    )
    monkeypatch.setattr(voice_chat.time, "sleep", lambda _: None)

    assert voice_chat.main([]) == 0
    assert len(calls) == 2
    assert recovery_flags == [False, True]


def test_main_cleans_failed_live_session_and_rebuilds_before_next_prompt(
    monkeypatch,
    capsys,
) -> None:
    class _Session:
        def __init__(self, *, disconnect: bool) -> None:
            self.disconnect = disconnect
            self.allow_error = threading.Event()
            self.error_delivered = threading.Event()
            self.error_sent = False
            self.committed = threading.Event()
            self.response_sent = False
            self.offer_calls = 0
            self.close_calls = 0
            self.context = None

        def start(self, context) -> bool:
            self.context = context
            return True

        def next_event(self, timeout=None):
            del timeout
            if (
                self.disconnect
                and not self.error_sent
                and self.allow_error.wait(timeout=0.01)
            ):
                self.error_sent = True
                self.error_delivered.set()
                return RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.ERROR,
                    error="provider_connection_closed",
                )
            if (
                not self.disconnect
                and not self.response_sent
                and self.committed.wait(timeout=0.01)
            ):
                self.response_sent = True
                return RealtimeVoiceEvent(
                    event_type=RealtimeVoiceEventType.RESPONSE_DONE,
                    text="recovered",
                )
            return None

        def offer_audio(self, frame) -> bool:
            del frame
            self.offer_calls += 1
            return not self.disconnect

        def status_snapshot(self):
            return {
                "last_error": (
                    "provider_connection_closed" if self.disconnect else ""
                )
            }

        def finish_input(self) -> bool:
            self.committed.set()
            return True

        def close(self, reason) -> None:
            del reason
            self.close_calls += 1

    class _OutputStream:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False
            self.closed = False

        def start(self) -> None:
            self.started = True

        def write(self, pcm) -> None:
            del pcm

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    first_session = _Session(disconnect=True)
    second_session = _Session(disconnect=False)

    class _RawInputStream:
        def __init__(self, *, callback, **kwargs) -> None:
            del kwargs
            self.callback = callback

        def __enter__(self):
            assert first_session.error_delivered.wait(timeout=1.0)
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
            stream = _OutputStream()
            self.outputs.append(stream)
            return stream

    sounddevice = _SoundDevice()
    sessions = iter((first_session, second_session))
    prompts = iter(("", "", "", "", "q"))
    prompt_count = 0

    def answer_prompt(prompt):
        nonlocal prompt_count
        del prompt
        prompt_count += 1
        if prompt_count == 1:
            first_session.allow_error.set()
        return next(prompts)

    monkeypatch.setenv("VOLCENGINE_S2S_API_KEY", "configured")
    monkeypatch.setattr(voice_chat, "_load_sounddevice", lambda: sounddevice)
    monkeypatch.setattr(voice_chat, "build_realtime_dialogue", lambda config: next(sessions))
    monkeypatch.setattr("builtins.input", answer_prompt)
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
    assert output.count("[error] provider_connection_closed") == 1
    assert output.count("[reconnect] provider session disconnected") == 1
    assert "microphone upload failed" not in output
    assert first_session.offer_calls == 1
    assert second_session.offer_calls == 1
    assert second_session.response_sent is True
    assert first_session.close_calls == 1
    assert second_session.close_calls == 1
    assert first_session.context is not None
    assert second_session.context is not None
    assert first_session.context.session_id != second_session.context.session_id
    assert first_session.context.dialog_id == first_session.context.session_id
    assert second_session.context.dialog_id == ""
    assert len(sounddevice.outputs) == 2
    assert all(stream.started and stream.stopped and stream.closed for stream in sounddevice.outputs)
