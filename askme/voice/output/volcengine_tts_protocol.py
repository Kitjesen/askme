"""Pure codec for Volcengine/Doubao bidirectional TTS WebSocket frames.

The module intentionally contains no networking, credentials, or provider
business logic.  It only serializes the binary protocol envelope used by the
TTS V3 bidirectional WebSocket API.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class VolcProtocolError(ValueError):
    """Raised when a Volcengine TTS binary frame is malformed."""


class Version(IntEnum):
    VERSION1 = 1


class HeaderSize(IntEnum):
    HEADER_SIZE_4 = 1
    HEADER_SIZE_8 = 2
    HEADER_SIZE_12 = 3
    HEADER_SIZE_16 = 4


class MessageType(IntEnum):
    INVALID = 0
    FULL_CLIENT_REQUEST = 0b0001
    AUDIO_ONLY_CLIENT = 0b0010
    FULL_SERVER_RESPONSE = 0b1001
    AUDIO_ONLY_SERVER = 0b1011
    FRONT_END_RESULT_SERVER = 0b1100
    ERROR = 0b1111


class MessageFlag(IntEnum):
    NO_SEQ = 0
    POSITIVE_SEQ = 0b0001
    LAST_NO_SEQ = 0b0010
    NEGATIVE_SEQ = 0b0011
    WITH_EVENT = 0b0100


class Serialization(IntEnum):
    RAW = 0
    JSON = 0b0001
    THRIFT = 0b0011
    CUSTOM = 0b1111


class Compression(IntEnum):
    NONE = 0
    GZIP = 0b0001
    CUSTOM = 0b1111


class EventType(IntEnum):
    NONE = 0

    START_CONNECTION = 1
    FINISH_CONNECTION = 2
    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52

    START_SESSION = 100
    CANCEL_SESSION = 101
    FINISH_SESSION = 102
    SESSION_STARTED = 150
    SESSION_CANCELED = 151
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    USAGE_RESPONSE = 154

    TASK_REQUEST = 200
    UPDATE_CONFIG = 201
    AUDIO_MUTED = 250

    SAY_HELLO = 300
    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352
    TTS_ENDED = 359


START_CONNECTION = EventType.START_CONNECTION
START_SESSION = EventType.START_SESSION
TASK_REQUEST = EventType.TASK_REQUEST
FINISH_SESSION = EventType.FINISH_SESSION
CANCEL_SESSION = EventType.CANCEL_SESSION

_CONNECTION_EVENTS = {
    EventType.START_CONNECTION,
    EventType.FINISH_CONNECTION,
    EventType.CONNECTION_STARTED,
    EventType.CONNECTION_FAILED,
    EventType.CONNECTION_FINISHED,
}

_CONNECT_ID_EVENTS = {
    EventType.CONNECTION_STARTED,
    EventType.CONNECTION_FAILED,
    EventType.CONNECTION_FINISHED,
}

_SEQUENCED_FLAGS = {MessageFlag.POSITIVE_SEQ, MessageFlag.NEGATIVE_SEQ}
_SEQUENCED_TYPES = {
    MessageType.FULL_CLIENT_REQUEST,
    MessageType.AUDIO_ONLY_CLIENT,
    MessageType.FULL_SERVER_RESPONSE,
    MessageType.AUDIO_ONLY_SERVER,
    MessageType.FRONT_END_RESULT_SERVER,
}


@dataclass(frozen=True)
class VolcTTSFrame:
    """Decoded Volcengine TTS binary frame."""

    version: int
    header_size_words: int
    message_type: int
    flags: int
    sequence_flag: int
    has_event: bool
    serialization: int
    compression: int
    event: int | None
    payload: bytes
    session_id: str | None = None
    connect_id: str | None = None
    sequence: int | None = None
    error_code: int | None = None
    header_extensions: bytes = b""

    @property
    def is_final(self) -> bool:
        return self.sequence_flag in {MessageFlag.LAST_NO_SEQ, MessageFlag.NEGATIVE_SEQ}

    @property
    def payload_json(self) -> Any:
        if self.serialization != Serialization.JSON:
            raise VolcProtocolError("payload is not marked as JSON")
        payload = _decompress_payload(bytes(self.payload), Compression(self.compression))
        return json.loads(payload.decode("utf-8"))


def encode_client_event_frame(
    event: EventType | int,
    payload: bytes | dict[str, Any] | None = None,
    *,
    session_id: str | None = None,
    message_type: MessageType = MessageType.FULL_CLIENT_REQUEST,
    flags: MessageFlag = MessageFlag.WITH_EVENT,
    serialization: Serialization = Serialization.JSON,
    compression: Compression = Compression.NONE,
) -> bytes:
    """Encode a client event frame with big-endian integer fields."""

    if int(flags) != MessageFlag.WITH_EVENT:
        raise VolcProtocolError("client event frames must use WITH_EVENT in this codec")
    event_value = int(event)
    is_connection_event = _is_connection_event(event_value)
    if is_connection_event and session_id:
        raise VolcProtocolError("connection events must not include a session_id")
    if not is_connection_event and not session_id:
        raise VolcProtocolError("session events require a non-empty session_id")
    payload_bytes = _encode_payload(payload, serialization, compression)
    frame = bytearray()
    frame.extend(
        (
            (Version.VERSION1 << 4) | HeaderSize.HEADER_SIZE_4,
            (message_type << 4) | flags,
            (serialization << 4) | compression,
            0,
        )
    )
    frame.extend(_i32(int(event)))
    if not is_connection_event:
        session_bytes = (session_id or "").encode("utf-8")
        frame.extend(_u32(len(session_bytes)))
        frame.extend(session_bytes)
    frame.extend(_u32(len(payload_bytes)))
    frame.extend(payload_bytes)
    return bytes(frame)


def encode_start_connection(payload: bytes | dict[str, Any] | None = None) -> bytes:
    return encode_client_event_frame(EventType.START_CONNECTION, payload or {})


def encode_start_session(
    session_id: str,
    payload: bytes | dict[str, Any],
    *,
    compression: Compression = Compression.NONE,
) -> bytes:
    return encode_client_event_frame(
        EventType.START_SESSION,
        payload,
        session_id=session_id,
        compression=compression,
    )


def encode_task_request(
    session_id: str,
    payload: bytes | dict[str, Any],
    *,
    compression: Compression = Compression.NONE,
) -> bytes:
    return encode_client_event_frame(
        EventType.TASK_REQUEST,
        payload,
        session_id=session_id,
        compression=compression,
    )


def encode_finish_session(session_id: str) -> bytes:
    return encode_client_event_frame(EventType.FINISH_SESSION, {}, session_id=session_id)


def encode_cancel_session(session_id: str) -> bytes:
    return encode_client_event_frame(EventType.CANCEL_SESSION, {}, session_id=session_id)


def decode_server_frame(data: bytes) -> VolcTTSFrame:
    """Decode one server frame and strictly validate length prefixes."""

    reader = _Reader(data)
    if len(data) < 4:
        raise VolcProtocolError("frame is shorter than the 4-byte protocol header")

    first = reader.u8()
    version = first >> 4
    header_size_words = first & 0x0F
    if version != Version.VERSION1:
        raise VolcProtocolError(f"unsupported protocol version: {version}")
    if header_size_words < HeaderSize.HEADER_SIZE_4:
        raise VolcProtocolError(f"invalid header size words: {header_size_words}")

    second = reader.u8()
    message_type = second >> 4
    flags = second & 0x0F
    try:
        message_type_enum = MessageType(message_type)
    except ValueError as exc:
        raise VolcProtocolError(f"unsupported message type: {message_type}") from exc
    if flags & ~0x07:
        raise VolcProtocolError(f"unsupported message flags: {flags}")
    sequence_flag = flags & 0x03
    has_event = bool(flags & MessageFlag.WITH_EVENT)

    third = reader.u8()
    serialization = third >> 4
    compression = third & 0x0F
    _validate_nibble_enum(serialization, Serialization, "serialization")
    _validate_nibble_enum(compression, Compression, "compression")
    reserved = reader.u8()
    if reserved != 0:
        raise VolcProtocolError(f"reserved header byte must be 0, got {reserved}")

    header_size_bytes = header_size_words * 4
    header_extensions = reader.bytes(header_size_bytes - 4) if header_size_bytes > 4 else b""

    sequence: int | None = None
    error_code: int | None = None
    if message_type_enum == MessageType.ERROR and flags != MessageFlag.NO_SEQ:
        raise VolcProtocolError("ERROR frames must not include sequence or event flags")
    if message_type_enum in _SEQUENCED_TYPES and sequence_flag in _SEQUENCED_FLAGS:
        sequence = reader.i32()
    elif message_type_enum == MessageType.ERROR:
        error_code = reader.u32()
    elif message_type_enum not in {
        MessageType.FULL_SERVER_RESPONSE,
        MessageType.AUDIO_ONLY_SERVER,
        MessageType.ERROR,
    }:
        raise VolcProtocolError(f"unsupported server message type: {message_type}")

    event: int | None = None
    session_id: str | None = None
    connect_id: str | None = None
    if has_event:
        event = reader.i32()
        if not _is_connection_event(event):
            session_id = reader.utf8_with_u32_length()
        elif _event_may_have_connect_id(event):
            connect_id = reader.optional_connect_id_before_payload()

    payload_length = reader.u32()
    payload = reader.bytes(payload_length)
    reader.finish()

    return VolcTTSFrame(
        version=version,
        header_size_words=header_size_words,
        message_type=message_type,
        flags=flags,
        sequence_flag=sequence_flag,
        has_event=has_event,
        serialization=serialization,
        compression=compression,
        event=event,
        session_id=session_id,
        connect_id=connect_id,
        sequence=sequence,
        error_code=error_code,
        payload=payload,
        header_extensions=header_extensions,
    )


def _encode_payload(
    payload: bytes | dict[str, Any] | None,
    serialization: Serialization,
    compression: Compression,
) -> bytes:
    if payload is None:
        raw = b""
    elif isinstance(payload, bytes):
        raw = payload
    elif serialization == Serialization.JSON:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    else:
        raise VolcProtocolError("non-bytes payloads require JSON serialization")
    return _compress_payload(raw, compression)


def _compress_payload(payload: bytes, compression: Compression) -> bytes:
    if compression == Compression.NONE:
        return payload
    if compression == Compression.GZIP:
        return gzip.compress(payload)
    raise VolcProtocolError(f"unsupported payload compression: {compression}")


def _decompress_payload(payload: bytes, compression: Compression) -> bytes:
    if compression == Compression.NONE:
        return payload
    if compression == Compression.GZIP:
        return gzip.decompress(payload)
    raise VolcProtocolError(f"unsupported payload compression: {compression}")


def _is_connection_event(event: EventType | int) -> bool:
    return int(event) < 100


def _event_may_have_connect_id(event: int) -> bool:
    try:
        event_value: EventType | int = EventType(event)
    except ValueError:
        event_value = event
    return event_value in _CONNECT_ID_EVENTS


def _validate_nibble_enum(value: int, enum_type: type[IntEnum], label: str) -> None:
    try:
        enum_type(value)
    except ValueError as exc:
        raise VolcProtocolError(f"unsupported {label}: {value}") from exc


def _i32(value: int) -> bytes:
    return int(value).to_bytes(4, "big", signed=True)


def _u32(value: int) -> bytes:
    return int(value).to_bytes(4, "big", signed=False)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self._data = memoryview(data)
        self._pos = 0

    def u8(self) -> int:
        return self.bytes(1)[0]

    def i32(self) -> int:
        return int.from_bytes(self.bytes(4), "big", signed=True)

    def u32(self) -> int:
        return int.from_bytes(self.bytes(4), "big", signed=False)

    def bytes(self, length: int) -> bytes:
        if length < 0:
            raise VolcProtocolError("negative field length")
        end = self._pos + length
        if end > len(self._data):
            raise VolcProtocolError(
                f"truncated frame: need {length} bytes at offset {self._pos}, "
                f"only {len(self._data) - self._pos} available"
            )
        chunk = bytes(self._data[self._pos : end])
        self._pos = end
        return chunk

    def utf8_with_u32_length(self) -> str:
        return self.bytes(self.u32()).decode("utf-8")

    def optional_connect_id_before_payload(self) -> str | None:
        remaining = len(self._data) - self._pos
        if remaining < 4:
            raise VolcProtocolError("missing payload length")

        payload_len = self.peek_u32(self._pos)
        if payload_len == remaining - 4:
            return None

        if remaining < 8:
            raise VolcProtocolError("truncated connection event payload")
        connect_len = payload_len
        payload_len_offset = self._pos + 4 + connect_len
        if payload_len_offset + 4 > len(self._data):
            raise VolcProtocolError("truncated connect_id field")
        following_payload_len = self.peek_u32(payload_len_offset)
        expected_remaining = 4 + connect_len + 4 + following_payload_len
        if expected_remaining != remaining:
            raise VolcProtocolError("cannot disambiguate connect_id and payload length")
        self._pos += 4
        connect_id = self.bytes(connect_len).decode("utf-8")
        return connect_id

    def peek_u32(self, offset: int) -> int:
        end = offset + 4
        if end > len(self._data):
            raise VolcProtocolError("truncated uint32 field")
        return int.from_bytes(self._data[offset:end], "big", signed=False)

    def finish(self) -> None:
        if self._pos != len(self._data):
            raise VolcProtocolError(
                f"unexpected trailing bytes: {len(self._data) - self._pos}"
            )
