from __future__ import annotations

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
