"""Protocol tests for Volcengine's end-to-end RealtimeAPI."""

from __future__ import annotations

import pytest

from askme.voice.realtime.protocol import (
    Compression,
    MessageType,
    ProtocolError,
    RealtimeEvent,
    Serialization,
    decode_frame,
    encode_frame,
)


def test_start_connection_matches_the_official_v1_frame() -> None:
    frame = encode_frame(RealtimeEvent.START_CONNECTION, {})

    assert frame == bytes((0x11, 0x14, 0x10, 0x00, 0, 0, 0, 1, 0, 0, 0, 2, 123, 125))
    decoded = decode_frame(frame)
    assert decoded.event is RealtimeEvent.START_CONNECTION
    assert decoded.payload == {}
    assert decoded.session_id is None


def test_start_session_round_trips_session_id_and_json_payload() -> None:
    session_id = "75a6126e-427f-49a1-a2c1-621143cb9db3"
    payload = {"dialog": {"bot_name": "豆包", "extra": {"model": "1.2.1.1"}}}

    frame = encode_frame(RealtimeEvent.START_SESSION, payload, session_id=session_id)
    decoded = decode_frame(frame)

    assert frame[:4] == bytes((0x11, 0x14, 0x10, 0x00))
    assert decoded.message_type is MessageType.FULL_CLIENT_REQUEST
    assert decoded.serialization is Serialization.JSON
    assert decoded.event is RealtimeEvent.START_SESSION
    assert decoded.session_id == session_id
    assert decoded.payload == payload


def test_task_request_round_trips_raw_audio() -> None:
    audio = b"\x01\x02\x03\x04"

    frame = encode_frame(RealtimeEvent.TASK_REQUEST, audio, session_id="session-1")
    decoded = decode_frame(frame)

    assert frame[:4] == bytes((0x11, 0x24, 0x00, 0x00))
    assert decoded.message_type is MessageType.AUDIO_ONLY_REQUEST
    assert decoded.serialization is Serialization.RAW
    assert decoded.event is RealtimeEvent.TASK_REQUEST
    assert decoded.payload == audio


def test_asr_info_is_mapped_to_a_server_json_event() -> None:
    payload = {"question_id": "question-1"}

    frame = encode_frame(RealtimeEvent.ASR_INFO, payload, session_id="session-1")
    decoded = decode_frame(frame)

    assert frame[:4] == bytes((0x11, 0x94, 0x10, 0x00))
    assert decoded.message_type is MessageType.FULL_SERVER_RESPONSE
    assert decoded.event is RealtimeEvent.ASR_INFO
    assert decoded.payload == payload


def test_tts_response_round_trips_raw_audio_from_the_server() -> None:
    audio = b"OggS\x00\x02"

    frame = encode_frame(RealtimeEvent.TTS_RESPONSE, audio, session_id="session-1")
    decoded = decode_frame(frame)

    assert frame[:4] == bytes((0x11, 0xB4, 0x00, 0x00))
    assert decoded.message_type is MessageType.AUDIO_ONLY_RESPONSE
    assert decoded.event is RealtimeEvent.TTS_RESPONSE
    assert decoded.payload == audio


@pytest.mark.parametrize(
    ("event", "event_id", "message_type", "needs_session"),
    [
        (RealtimeEvent.FINISH_CONNECTION, 2, MessageType.FULL_CLIENT_REQUEST, False),
        (RealtimeEvent.CONNECTION_STARTED, 50, MessageType.FULL_SERVER_RESPONSE, False),
        (RealtimeEvent.CONNECTION_FAILED, 51, MessageType.FULL_SERVER_RESPONSE, False),
        (RealtimeEvent.CONNECTION_FINISHED, 52, MessageType.FULL_SERVER_RESPONSE, False),
        (RealtimeEvent.FINISH_SESSION, 102, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.SESSION_STARTED, 150, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.SESSION_FINISHED, 152, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.SESSION_FAILED, 153, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.USAGE_RESPONSE, 154, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.UPDATE_CONFIG, 201, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.CONFIG_UPDATED, 251, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.TTS_SENTENCE_START, 350, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.TTS_SENTENCE_END, 351, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.END_ASR, 400, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.ASR_RESPONSE, 451, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.ASR_ENDED, 459, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.CHAT_TTS_TEXT, 500, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.CHAT_TEXT_QUERY, 501, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.CHAT_RAG_TEXT, 502, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.CONVERSATION_CREATE, 510, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.CONVERSATION_UPDATE, 511, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.CONVERSATION_RETRIEVE, 512, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.CONVERSATION_TRUNCATE, 513, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.CONVERSATION_DELETE, 514, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.CLIENT_INTERRUPT, 515, MessageType.FULL_CLIENT_REQUEST, True),
        (RealtimeEvent.CHAT_RESPONSE, 550, MessageType.FULL_SERVER_RESPONSE, True),
        (
            RealtimeEvent.CHAT_TEXT_QUERY_CONFIRMED,
            553,
            MessageType.FULL_SERVER_RESPONSE,
            True,
        ),
        (RealtimeEvent.CHAT_ENDED, 559, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.CONVERSATION_CREATED, 567, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.CONVERSATION_UPDATED, 568, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.CONVERSATION_RETRIEVED, 569, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.CONVERSATION_TRUNCATED, 570, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.CONVERSATION_DELETED, 571, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.TTS_ENDED, 359, MessageType.FULL_SERVER_RESPONSE, True),
        (RealtimeEvent.DIALOG_COMMON_ERROR, 599, MessageType.FULL_SERVER_RESPONSE, True),
    ],
)
def test_required_realtime_events_have_official_ids_and_directions(
    event: RealtimeEvent,
    event_id: int,
    message_type: MessageType,
    needs_session: bool,
) -> None:
    session_id = "session-1" if needs_session else None

    decoded = decode_frame(encode_frame(event, {}, session_id=session_id))

    assert int(event) == event_id
    assert decoded.event is event
    assert decoded.message_type is message_type


def test_unknown_event_id_is_preserved_for_forward_compatibility() -> None:
    frame = encode_frame(
        777,
        {"future": True},
        session_id="session-1",
        message_type=MessageType.FULL_SERVER_RESPONSE,
    )

    decoded = decode_frame(frame)

    assert decoded.event == 777
    assert type(decoded.event) is int
    assert decoded.message_type is MessageType.FULL_SERVER_RESPONSE
    assert decoded.payload == {"future": True}


def test_sequence_flags_round_trip_a_final_server_event() -> None:
    frame = encode_frame(
        RealtimeEvent.ASR_RESPONSE,
        {"results": []},
        session_id="session-1",
        sequence=-1,
    )

    decoded = decode_frame(frame)

    assert frame[1] == 0x97
    assert decoded.flags == 0x7
    assert decoded.sequence == -1
    assert decoded.is_final is True


def test_final_flag_without_sequence_round_trips() -> None:
    frame = encode_frame(
        RealtimeEvent.TTS_ENDED,
        {},
        session_id="session-1",
        final=True,
    )

    decoded = decode_frame(frame)

    assert frame[1] == 0x96
    assert decoded.sequence is None
    assert decoded.is_final is True


def test_connect_event_can_carry_the_optional_connection_id() -> None:
    frame = encode_frame(
        RealtimeEvent.START_CONNECTION,
        {},
        connection_id="connect-1",
    )

    decoded = decode_frame(frame)

    assert decoded.connection_id == "connect-1"
    assert decoded.session_id is None
    assert decoded.payload == {}


def test_error_frame_preserves_provider_code_and_payload() -> None:
    frame = encode_frame(
        None,
        {"error": "resource not granted"},
        message_type=MessageType.ERROR,
        error_code=45000001,
    )

    decoded = decode_frame(frame)

    assert frame[:4] == bytes((0x11, 0xF0, 0x10, 0x00))
    assert decoded.message_type is MessageType.ERROR
    assert decoded.event is None
    assert decoded.error_code == 45000001
    assert decoded.payload == {"error": "resource not granted"}


def test_gzip_payload_round_trips_when_header_requests_compression() -> None:
    payload = {"results": [{"text": "你好", "is_interim": True}]}

    frame = encode_frame(
        RealtimeEvent.ASR_RESPONSE,
        payload,
        session_id="session-1",
        compression=Compression.GZIP,
    )
    decoded = decode_frame(frame)

    assert frame[2] == 0x11
    assert decoded.compression is Compression.GZIP
    assert decoded.payload == payload


def test_malformed_or_truncated_frames_fail_closed() -> None:
    valid = encode_frame(RealtimeEvent.START_CONNECTION, {})
    empty_session_id = (
        bytes((0x11, 0x14, 0x10, 0x00))
        + (100).to_bytes(4, "big")
        + (0).to_bytes(4, "big")
        + (2).to_bytes(4, "big")
        + b"{}"
    )
    malformed = [
        b"",
        bytes((0x21,)) + valid[1:],
        valid[:3] + b"\x01" + valid[4:],
        valid[:-1],
        valid + b"\x00",
        empty_session_id,
    ]

    for frame in malformed:
        with pytest.raises(ProtocolError):
            decode_frame(frame)
