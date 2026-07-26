"""Binary codec for Volcengine's end-to-end RealtimeAPI v1.

This module is deliberately transport-free: it only converts between Python
values and a single WebSocket binary message.
"""

from __future__ import annotations

import gzip
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class ProtocolError(ValueError):
    """Raised when a realtime frame violates the v1 wire contract."""


class MessageType(IntEnum):
    """Wire-level realtime message types."""

    FULL_CLIENT_REQUEST = 0x1
    AUDIO_ONLY_REQUEST = 0x2
    FULL_SERVER_RESPONSE = 0x9
    AUDIO_ONLY_RESPONSE = 0xB
    ERROR = 0xF


class Serialization(IntEnum):
    """Payload serialization encoded in byte two of the header."""

    RAW = 0x0
    JSON = 0x1


class Compression(IntEnum):
    """Payload compression encoded in byte two of the header."""

    NONE = 0x0
    GZIP = 0x1


class RealtimeEvent(IntEnum):
    """Known event identifiers in the Volcengine realtime protocol."""

    START_CONNECTION = 1
    FINISH_CONNECTION = 2
    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52
    START_SESSION = 100
    FINISH_SESSION = 102
    SESSION_STARTED = 150
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    USAGE_RESPONSE = 154
    TASK_REQUEST = 200
    UPDATE_CONFIG = 201
    CONFIG_UPDATED = 251
    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352
    TTS_ENDED = 359
    END_ASR = 400
    ASR_INFO = 450
    ASR_RESPONSE = 451
    ASR_ENDED = 459
    CHAT_TTS_TEXT = 500
    CHAT_TEXT_QUERY = 501
    CHAT_RAG_TEXT = 502
    CONVERSATION_CREATE = 510
    CONVERSATION_UPDATE = 511
    CONVERSATION_RETRIEVE = 512
    CONVERSATION_TRUNCATE = 513
    CONVERSATION_DELETE = 514
    CLIENT_INTERRUPT = 515
    CHAT_RESPONSE = 550
    CHAT_TEXT_QUERY_CONFIRMED = 553
    CHAT_ENDED = 559
    CONVERSATION_CREATED = 567
    CONVERSATION_UPDATED = 568
    CONVERSATION_RETRIEVED = 569
    CONVERSATION_TRUNCATED = 570
    CONVERSATION_DELETED = 571
    DIALOG_COMMON_ERROR = 599


_SERVER_JSON_EVENTS = frozenset(
    {
        RealtimeEvent.CONNECTION_STARTED,
        RealtimeEvent.CONNECTION_FAILED,
        RealtimeEvent.CONNECTION_FINISHED,
        RealtimeEvent.SESSION_STARTED,
        RealtimeEvent.SESSION_FINISHED,
        RealtimeEvent.SESSION_FAILED,
        RealtimeEvent.USAGE_RESPONSE,
        RealtimeEvent.CONFIG_UPDATED,
        RealtimeEvent.TTS_SENTENCE_START,
        RealtimeEvent.TTS_SENTENCE_END,
        RealtimeEvent.TTS_ENDED,
        RealtimeEvent.ASR_INFO,
        RealtimeEvent.ASR_RESPONSE,
        RealtimeEvent.ASR_ENDED,
        RealtimeEvent.CHAT_RESPONSE,
        RealtimeEvent.CHAT_TEXT_QUERY_CONFIRMED,
        RealtimeEvent.CHAT_ENDED,
        RealtimeEvent.CONVERSATION_CREATED,
        RealtimeEvent.CONVERSATION_UPDATED,
        RealtimeEvent.CONVERSATION_RETRIEVED,
        RealtimeEvent.CONVERSATION_TRUNCATED,
        RealtimeEvent.CONVERSATION_DELETED,
        RealtimeEvent.DIALOG_COMMON_ERROR,
    }
)


@dataclass(frozen=True, slots=True)
class RealtimeFrame:
    """A decoded realtime protocol frame."""

    event: RealtimeEvent | int | None
    payload: Any
    session_id: str | None = None
    connection_id: str | None = None
    message_type: MessageType | int = MessageType.FULL_CLIENT_REQUEST
    flags: int = 0x4
    serialization: Serialization | int = Serialization.JSON
    compression: Compression | int = Compression.NONE
    sequence: int | None = None
    is_final: bool = False
    error_code: int | None = None


def encode_frame(
    event: RealtimeEvent | int | None,
    payload: Mapping[str, Any] | bytes | None = None,
    *,
    session_id: str | None = None,
    connection_id: str | None = None,
    message_type: MessageType | int | None = None,
    sequence: int | None = None,
    final: bool = False,
    error_code: int | None = None,
    compression: Compression | int = Compression.NONE,
) -> bytes:
    """Encode one client event using the official v1 layout."""
    if event in {RealtimeEvent.TASK_REQUEST, RealtimeEvent.TTS_RESPONSE}:
        if payload is not None and not isinstance(payload, bytes):
            raise ProtocolError("audio event payload must be raw bytes")
        inferred_message_type = (
            MessageType.AUDIO_ONLY_REQUEST
            if event == RealtimeEvent.TASK_REQUEST
            else MessageType.AUDIO_ONLY_RESPONSE
        )
        serialization = Serialization.RAW
        payload_bytes = payload or b""
    else:
        if isinstance(payload, bytes):
            raise ProtocolError("JSON event payload cannot be raw bytes")
        inferred_message_type = (
            MessageType.FULL_SERVER_RESPONSE
            if event in _SERVER_JSON_EVENTS
            else MessageType.FULL_CLIENT_REQUEST
        )
        serialization = Serialization.JSON
        payload_bytes = json.dumps(
            dict(payload or {}), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    try:
        resolved_compression = Compression(compression)
    except ValueError as exc:
        raise ProtocolError(f"unsupported realtime compression: {compression}") from exc
    if resolved_compression is Compression.GZIP:
        payload_bytes = gzip.compress(payload_bytes)
    resolved_message_type: MessageType | int
    if message_type is None:
        resolved_message_type = (
            MessageType.ERROR if error_code is not None else inferred_message_type
        )
    else:
        resolved_message_type = int(message_type)
    if not 0 <= int(resolved_message_type) <= 0xF:
        raise ProtocolError("message_type must fit in four bits")

    optional = bytearray()
    if resolved_message_type == MessageType.ERROR:
        if error_code is None or not 0 <= error_code <= 0xFFFFFFFF:
            raise ProtocolError("error frames require a uint32 error_code")
        if event is not None or session_id is not None or connection_id is not None:
            raise ProtocolError("error frames cannot carry event identifiers")
        if sequence is not None or final:
            raise ProtocolError("error frames cannot carry sequence flags")
        flags = 0x0
        optional.extend(struct.pack(">I", error_code))
    else:
        if error_code is not None:
            raise ProtocolError("non-error frames cannot carry an error_code")
        if event is None:
            raise ProtocolError("non-error frames require an event")
        flags = 0x4

    if sequence is not None:
        if sequence == 0:
            raise ProtocolError("sequence must be positive or negative, not zero")
        if final and sequence > 0:
            raise ProtocolError("a positive sequence cannot also be final")
        flags |= 0x1 if sequence > 0 else 0x3
        optional.extend(struct.pack(">i", sequence))
    elif final:
        flags |= 0x2

    event_id = None if event is None else int(event)
    if event_id is not None:
        optional.extend(struct.pack(">I", event_id))
    if event_id is not None and event_id < 100:
        if session_id is not None:
            raise ProtocolError("connect events cannot carry a session_id")
        if connection_id is not None:
            if not connection_id:
                raise ProtocolError("connection_id cannot be empty")
            connection_bytes = connection_id.encode("utf-8")
            optional.extend(struct.pack(">I", len(connection_bytes)))
            optional.extend(connection_bytes)
    elif event_id is not None:
        if connection_id is not None:
            raise ProtocolError("session events cannot carry a connection_id")
        if not session_id:
            raise ProtocolError("session events require a non-empty session_id")
        session_bytes = session_id.encode("utf-8")
        optional.extend(struct.pack(">I", len(session_bytes)))
        optional.extend(session_bytes)
    header = bytes(
        (
            0x11,
            (int(resolved_message_type) << 4) | flags,
            (int(serialization) << 4) | int(resolved_compression),
            0x00,
        )
    )
    return header + bytes(optional) + struct.pack(">I", len(payload_bytes)) + payload_bytes


def decode_frame(data: bytes) -> RealtimeFrame:
    """Decode one event from a complete WebSocket message."""
    if len(data) < 12:
        raise ProtocolError("realtime frame is truncated")
    version = data[0] >> 4
    header_size = (data[0] & 0x0F) * 4
    if version != 1 or header_size != 4 or data[3] != 0:
        raise ProtocolError("realtime frame has an invalid v1 header")
    try:
        message_type: MessageType | int = MessageType(data[1] >> 4)
    except ValueError:
        message_type = data[1] >> 4
    flags = data[1] & 0x0F
    try:
        serialization: Serialization | int = Serialization(data[2] >> 4)
    except ValueError:
        serialization = data[2] >> 4
    try:
        compression: Compression | int = Compression(data[2] & 0x0F)
    except ValueError:
        compression = data[2] & 0x0F
    if message_type != MessageType.ERROR and not flags & 0x4:
        raise ProtocolError("realtime event frame is missing its event flag")

    offset = header_size
    error_code: int | None = None
    if message_type == MessageType.ERROR:
        if len(data) < offset + 4:
            raise ProtocolError("realtime frame error code is truncated")
        error_code = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
    sequence_flag = flags & 0x3
    sequence: int | None = None
    if sequence_flag in {0x1, 0x3}:
        if len(data) < offset + 4:
            raise ProtocolError("realtime frame sequence is truncated")
        sequence = struct.unpack(">i", data[offset : offset + 4])[0]
        offset += 4
        if sequence_flag == 0x1 and sequence <= 0:
            raise ProtocolError("non-final realtime sequence must be positive")
        if sequence_flag == 0x3 and sequence >= 0:
            raise ProtocolError("final realtime sequence must be negative")
    event_id: int | None = None
    if flags & 0x4:
        if len(data) < offset + 4:
            raise ProtocolError("realtime frame event id is truncated")
        event_id = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
    session_id: str | None = None
    connection_id: str | None = None
    if event_id is not None and event_id >= 100:
        if len(data) < offset + 4:
            raise ProtocolError("realtime frame session id length is truncated")
        session_size = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        if session_size == 0:
            raise ProtocolError("realtime frame session id cannot be empty")
        if len(data) < offset + session_size:
            raise ProtocolError("realtime frame session id is truncated")
        try:
            session_id = data[offset : offset + session_size].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("realtime frame session id is not UTF-8") from exc
        offset += session_size
    elif event_id is not None and len(data) >= offset + 4:
        first_size = struct.unpack(">I", data[offset : offset + 4])[0]
        remaining_after_size = len(data) - (offset + 4)
        if first_size != remaining_after_size:
            offset += 4
            if len(data) < offset + first_size + 4:
                raise ProtocolError("realtime frame connection id is truncated")
            try:
                connection_id = data[offset : offset + first_size].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ProtocolError("realtime frame connection id is not UTF-8") from exc
            offset += first_size
    if len(data) < offset + 4:
        raise ProtocolError("realtime frame payload size is truncated")
    payload_size = struct.unpack(">I", data[offset : offset + 4])[0]
    offset += 4
    if len(data) != offset + payload_size:
        raise ProtocolError("realtime frame payload length does not match")
    payload_bytes = data[offset:]
    if compression == Compression.GZIP:
        try:
            payload_bytes = gzip.decompress(payload_bytes)
        except (OSError, EOFError) as exc:
            raise ProtocolError("realtime frame contains invalid gzip data") from exc
    elif compression != Compression.NONE:
        raise ProtocolError(f"unsupported realtime compression: {compression}")
    if serialization == Serialization.JSON:
        try:
            payload = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("realtime frame contains invalid JSON") from exc
    elif serialization == Serialization.RAW:
        payload = payload_bytes
    else:
        raise ProtocolError(f"unsupported realtime serialization: {serialization}")
    if event_id is None:
        event: RealtimeEvent | int | None = None
    else:
        try:
            event = RealtimeEvent(event_id)
        except ValueError:
            event = event_id
    return RealtimeFrame(
        event=event,
        payload=payload,
        session_id=session_id,
        connection_id=connection_id,
        message_type=message_type,
        flags=flags,
        serialization=serialization,
        compression=compression,
        sequence=sequence,
        is_final=sequence_flag in {0x2, 0x3},
        error_code=error_code,
    )


__all__ = [
    "Compression",
    "ProtocolError",
    "MessageType",
    "RealtimeEvent",
    "RealtimeFrame",
    "Serialization",
    "decode_frame",
    "encode_frame",
]
