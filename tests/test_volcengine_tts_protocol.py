import gzip

import pytest

from askme.voice.output.volcengine_tts_protocol import (
    Compression,
    EventType,
    MessageFlag,
    MessageType,
    Serialization,
    VolcProtocolError,
    decode_server_frame,
    encode_cancel_session,
    encode_client_event_frame,
    encode_start_connection,
    encode_start_session,
    encode_task_request,
)


def _server_header(
    *,
    message_type: MessageType = MessageType.FULL_SERVER_RESPONSE,
    flags: MessageFlag | int = MessageFlag.WITH_EVENT,
    serialization: Serialization = Serialization.JSON,
    compression: Compression = Compression.NONE,
    reserved: int = 0,
) -> bytes:
    return bytes(
        [
            0x11,
            (message_type << 4) | int(flags),
            (serialization << 4) | compression,
            reserved,
        ]
    )


def _u32(value: int) -> bytes:
    return value.to_bytes(4, "big", signed=False)


def _i32(value: int) -> bytes:
    return value.to_bytes(4, "big", signed=True)


def test_start_connection_golden_bytes() -> None:
    assert encode_start_connection() == bytes.fromhex("1114100000000001000000027b7d")


def test_start_session_json_roundtrip_with_session_id() -> None:
    encoded = encode_start_session(
        "session-1",
        {"event": EventType.START_SESSION, "req_params": {"text": ""}},
    )

    assert encoded.startswith(bytes.fromhex("1114100000000064"))
    assert b"session-1" in encoded
    assert encoded.endswith(b'{"event":100,"req_params":{"text":""}}')


def test_task_request_supports_gzip_json_payload() -> None:
    encoded = encode_task_request(
        "s1",
        {"event": EventType.TASK_REQUEST, "req_params": {"text": "你好"}},
        compression=Compression.GZIP,
    )

    assert encoded[:4] == bytes.fromhex("11141100")
    session_len_start = 8
    payload_len_start = session_len_start + 4 + 2
    compressed_len = int.from_bytes(encoded[payload_len_start : payload_len_start + 4], "big")
    compressed = encoded[payload_len_start + 4 :]

    assert compressed_len == len(compressed)
    assert gzip.decompress(compressed) == (
        b'{"event":200,"req_params":{"text":"\xe4\xbd\xa0\xe5\xa5\xbd"}}'
    )


def test_cancel_session_helper_uses_cancel_event() -> None:
    encoded = encode_cancel_session("s1")

    assert encoded[:8] == bytes.fromhex("1114100000000065")
    assert b"s1" in encoded


def test_decode_full_server_response_with_connect_id() -> None:
    connect_id = b"conn-1"
    payload = b'{"ok":true}'
    frame = (
        _server_header()
        + _i32(EventType.CONNECTION_STARTED)
        + _u32(len(connect_id))
        + connect_id
        + _u32(len(payload))
        + payload
    )

    decoded = decode_server_frame(frame)

    assert decoded.message_type == MessageType.FULL_SERVER_RESPONSE
    assert decoded.event == EventType.CONNECTION_STARTED
    assert decoded.connect_id == "conn-1"
    assert decoded.session_id is None
    assert decoded.payload_json == {"ok": True}


def test_decode_connection_event_without_connect_id() -> None:
    payload = b'{"ok":true}'
    frame = (
        _server_header()
        + _i32(EventType.CONNECTION_STARTED)
        + _u32(len(payload))
        + payload
    )

    decoded = decode_server_frame(frame)

    assert decoded.event == EventType.CONNECTION_STARTED
    assert decoded.connect_id is None
    assert decoded.payload == payload


def test_decode_unknown_connection_scoped_event_does_not_read_session_id() -> None:
    payload = b"{}"
    unknown_connection_event = 88
    frame = _server_header() + _i32(unknown_connection_event) + _u32(len(payload)) + payload

    decoded = decode_server_frame(frame)

    assert decoded.event == unknown_connection_event
    assert decoded.session_id is None
    assert decoded.payload == payload


def test_decode_full_server_response_with_unknown_event_preserved() -> None:
    session_id = b"s1"
    payload = b"{}"
    unknown_event = 9876
    frame = (
        _server_header()
        + _i32(unknown_event)
        + _u32(len(session_id))
        + session_id
        + _u32(len(payload))
        + payload
    )

    decoded = decode_server_frame(frame)

    assert decoded.event == unknown_event
    assert decoded.session_id == "s1"
    assert decoded.payload == payload


def test_decode_tts_response_negative_sequence_with_event() -> None:
    session_id = b"s-final"
    payload = b'{"done":true}'
    frame = (
        _server_header(flags=0x07)
        + _i32(-7)
        + _i32(EventType.TTS_RESPONSE)
        + _u32(len(session_id))
        + session_id
        + _u32(len(payload))
        + payload
    )

    decoded = decode_server_frame(frame)

    assert decoded.flags == 0x07
    assert decoded.sequence_flag == MessageFlag.NEGATIVE_SEQ
    assert decoded.has_event is True
    assert decoded.is_final is True
    assert decoded.sequence == -7
    assert decoded.event == EventType.TTS_RESPONSE
    assert decoded.session_id == "s-final"
    assert decoded.payload_json == {"done": True}


def test_decode_audio_only_server_payload() -> None:
    audio = b"\x01\x02\x03\x04"
    frame = (
        _server_header(
            message_type=MessageType.AUDIO_ONLY_SERVER,
            flags=MessageFlag.NO_SEQ,
            serialization=Serialization.RAW,
        )
        + _u32(len(audio))
        + audio
    )

    decoded = decode_server_frame(frame)

    assert decoded.message_type == MessageType.AUDIO_ONLY_SERVER
    assert decoded.event is None
    assert decoded.payload == audio


def test_decode_error_frame() -> None:
    payload = b'{"message":"bad key"}'
    frame = (
        _server_header(message_type=MessageType.ERROR, flags=MessageFlag.NO_SEQ)
        + _u32(401)
        + _u32(len(payload))
        + payload
    )

    decoded = decode_server_frame(frame)

    assert decoded.message_type == MessageType.ERROR
    assert decoded.error_code == 401
    assert decoded.payload == payload


def test_decode_error_frame_rejects_event_or_sequence_flags() -> None:
    payload = b"{}"
    frame = (
        _server_header(message_type=MessageType.ERROR, flags=MessageFlag.WITH_EVENT)
        + _u32(500)
        + _i32(EventType.CONNECTION_FAILED)
        + _u32(len(payload))
        + payload
    )

    with pytest.raises(VolcProtocolError, match="ERROR frames"):
        decode_server_frame(frame)


@pytest.mark.parametrize(
    "frame",
    [
        b"",
        bytes.fromhex("10141000"),
        bytes.fromhex("111f1000"),
        _server_header(reserved=1) + _i32(EventType.CONNECTION_STARTED) + _u32(0),
        bytes.fromhex("1114100000000001"),
        bytes.fromhex("1114100000000064000000047331"),
    ],
)
def test_decode_rejects_truncated_or_illegal_headers(frame: bytes) -> None:
    with pytest.raises(VolcProtocolError):
        decode_server_frame(frame)


def test_client_codec_rejects_non_event_flags() -> None:
    with pytest.raises(VolcProtocolError):
        encode_client_event_frame(EventType.START_CONNECTION, {}, flags=MessageFlag.NO_SEQ)


def test_client_codec_rejects_session_id_on_connection_events() -> None:
    with pytest.raises(VolcProtocolError):
        encode_client_event_frame(EventType.START_CONNECTION, {}, session_id="s1")


def test_client_codec_requires_session_id_for_session_events() -> None:
    with pytest.raises(VolcProtocolError):
        encode_client_event_frame(EventType.START_SESSION, {})
