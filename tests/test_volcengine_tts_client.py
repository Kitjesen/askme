from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from askme.voice.output.volcengine_tts_client import (
    VolcengineTTSClient,
    VolcengineTTSClientError,
    VolcengineTTSConfig,
)
from askme.voice.output.volcengine_tts_protocol import (
    EventType,
    MessageFlag,
    MessageType,
    Serialization,
)


def _server_event(
    event: EventType,
    *,
    session_id: str | None = None,
    payload: bytes = b"{}",
    message_type: MessageType = MessageType.FULL_SERVER_RESPONSE,
    flags: int = int(MessageFlag.WITH_EVENT),
) -> bytes:
    frame = bytearray([0x11, (int(message_type) << 4) | flags, 0x10, 0])
    frame.extend(int(event).to_bytes(4, "big", signed=True))
    if int(event) >= 100:
        session = (session_id or "").encode()
        frame.extend(len(session).to_bytes(4, "big"))
        frame.extend(session)
    frame.extend(len(payload).to_bytes(4, "big"))
    frame.extend(payload)
    return bytes(frame)


def _audio_only(payload: bytes) -> bytes:
    return bytes(
        [
            0x11,
            int(MessageType.AUDIO_ONLY_SERVER) << 4,
            int(Serialization.RAW) << 4,
            0,
        ]
    ) + len(payload).to_bytes(4, "big") + payload


def _session_audio(payload: bytes, *, session_id: str) -> bytes:
    return _server_event(
        EventType.TTS_RESPONSE,
        session_id=session_id,
        payload=payload,
        message_type=MessageType.AUDIO_ONLY_SERVER,
    )


def _client_event(frame: bytes) -> int:
    return int.from_bytes(frame[4:8], "big", signed=True)


def _client_payload_json(frame: bytes) -> Any:
    event = _client_event(frame)
    offset = 8
    if event >= 100:
        session_len = int.from_bytes(frame[offset : offset + 4], "big")
        offset += 4 + session_len
    payload_len = int.from_bytes(frame[offset : offset + 4], "big")
    offset += 4
    payload = frame[offset : offset + payload_len]
    return __import__("json").loads(payload.decode("utf-8"))


class FakeWebSocket:
    def __init__(self, incoming: list[bytes]) -> None:
        self.incoming = list(incoming)
        self.sent: list[bytes] = []
        self.closed = False
        self.aborted = False
        self.shutdown_called = False
        self.connected = True
        self.timeouts: list[float] = []

    def send_binary(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self) -> bytes:
        if not self.incoming:
            raise RuntimeError("no fake frame")
        return self.incoming.pop(0)

    def close(self) -> None:
        self.closed = True
        self.connected = False

    def abort(self) -> None:
        self.aborted = True
        self.closed = True
        self.connected = False

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.closed = True
        self.connected = False

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)


class BlockingWebSocket(FakeWebSocket):
    def __init__(self, incoming: list[bytes]) -> None:
        super().__init__(incoming)
        self.blocking_recv_started = threading.Event()

    def recv(self) -> bytes:
        if self.incoming:
            return self.incoming.pop(0)
        self.blocking_recv_started.set()
        while not self.closed:
            time.sleep(0.01)
        raise RuntimeError("socket closed")


class BlockingHandshakeWebSocket(FakeWebSocket):
    def __init__(self) -> None:
        super().__init__([])
        self.handshake_recv_started = threading.Event()
        self.release_handshake = threading.Event()

    def recv(self) -> bytes:
        self.handshake_recv_started.set()
        while not self.closed and not self.release_handshake.is_set():
            time.sleep(0.01)
        if self.closed:
            raise RuntimeError("socket closed")
        return _server_event(EventType.CONNECTION_STARTED)


class DelayedCloseWebSocket(BlockingWebSocket):
    def __init__(self, incoming: list[bytes]) -> None:
        super().__init__(incoming)
        self.closed_observed = threading.Event()
        self.release_closed_recv = threading.Event()

    def recv(self) -> bytes:
        if self.incoming:
            return self.incoming.pop(0)
        self.blocking_recv_started.set()
        while not self.closed:
            time.sleep(0.01)
        self.closed_observed.set()
        if not self.release_closed_recv.wait(timeout=1.0):
            raise RuntimeError("test did not release closed recv")
        raise RuntimeError("socket closed")


class BlockingCloseWebSocket(FakeWebSocket):
    def __init__(self, incoming: list[bytes]) -> None:
        super().__init__(incoming)
        self.abort_started = threading.Event()
        self.release_abort = threading.Event()

    def abort(self) -> None:
        self.abort_started.set()
        if not self.release_abort.wait(timeout=1.0):
            raise RuntimeError("test did not release abort")
        super().abort()


class FakeFactory:
    def __init__(self, *connections: FakeWebSocket) -> None:
        self.connections = list(connections)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeWebSocket:
        self.calls.append({"url": url, **kwargs})
        if not self.connections:
            raise RuntimeError("no fake connection")
        return self.connections.pop(0)


def _config(**kwargs: Any) -> VolcengineTTSConfig:
    values = {
        "endpoint": "wss://example.invalid/v3/tts/bidirection",
        "api_key": "test-api-key",
        "resource_id": "test-resource",
        "speaker": "test-speaker",
        "sample_rate": 24000,
        "audio_format": "pcm",
        **kwargs,
    }
    return VolcengineTTSConfig(**values)


def _client(
    ws: FakeWebSocket,
    *,
    config: VolcengineTTSConfig | None = None,
    session_id: str = "session-1",
) -> tuple[VolcengineTTSClient, FakeFactory]:
    factory = FakeFactory(ws)
    client = VolcengineTTSClient(
        config or _config(),
        connection_factory=factory,
        connect_id_factory=lambda: "connect-1",
        session_id_factory=lambda: session_id,
    )
    return client, factory


def _pause_after_handshake(
    client: VolcengineTTSClient,
) -> tuple[threading.Event, threading.Event]:
    handshake_complete = threading.Event()
    release_publish = threading.Event()
    open_started_connection = client._open_started_connection

    def paused_open_started_connection(operation_epoch: int) -> Any:
        ws = open_started_connection(operation_epoch)
        should_pause = not handshake_complete.is_set()
        if should_pause:
            handshake_complete.set()
            if not release_publish.wait(timeout=1.0):
                raise RuntimeError("test did not release connection publication")
        return ws

    client._open_started_connection = paused_open_started_connection  # type: ignore[method-assign]
    return handshake_complete, release_publish


def _pause_after_socket_open(
    client: VolcengineTTSClient,
) -> tuple[threading.Event, threading.Event]:
    socket_opened = threading.Event()
    release_tracking = threading.Event()
    open_connection = client._open_connection

    def paused_open_connection() -> Any:
        ws = open_connection()
        socket_opened.set()
        if not release_tracking.wait(timeout=1.0):
            raise RuntimeError("test did not release connection tracking")
        return ws

    client._open_connection = paused_open_connection  # type: ignore[method-assign]
    return socket_opened, release_tracking


def test_build_headers_prefers_api_key_and_never_returns_secret_snapshot() -> None:
    client = VolcengineTTSClient(_config(), connection_factory=FakeFactory())

    headers = client.build_headers("connect-1")

    assert "X-Api-Key: test-api-key" in headers
    assert "X-Api-Resource-Id: test-resource" in headers
    assert "X-Api-Connect-Id: connect-1" in headers
    assert not any(header.startswith("Authorization:") for header in headers)


def test_build_headers_supports_legacy_app_access_key() -> None:
    client = VolcengineTTSClient(
        _config(api_key="", app_id="app-1", access_key="legacy-key"),
        connection_factory=FakeFactory(),
    )

    headers = client.build_headers("connect-1")

    assert "X-Api-App-ID: app-1" in headers
    assert "X-Api-Access-Key: legacy-key" in headers
    assert not any(header.startswith("X-Api-Key:") for header in headers)


def test_default_resource_id_tracks_seed_tts_2() -> None:
    assert VolcengineTTSConfig(api_key="k", speaker="s").resource_id == "seed-tts-2.0"


def test_connection_and_session_timeouts_are_applied_to_their_phases() -> None:
    ws = FakeWebSocket([_server_event(EventType.CONNECTION_STARTED)])
    client, factory = _client(
        ws,
        config=_config(
            timeout=9.0,
            connect_timeout=1.25,
            session_timeout=2.5,
        ),
    )

    result = client.prewarm()

    assert result == {"ok": True, "status": "opened"}
    assert factory.calls[0]["timeout"] == 1.25
    assert ws.timeouts == [2.5]


def test_legacy_timeout_applies_to_connect_and_session_phases() -> None:
    ws = FakeWebSocket([_server_event(EventType.CONNECTION_STARTED)])
    client, factory = _client(ws, config=_config(timeout=3.5))

    result = client.prewarm()

    assert result == {"ok": True, "status": "opened"}
    assert factory.calls[0]["timeout"] == 3.5
    assert ws.timeouts == [3.5]


def test_prewarm_opens_and_reuses_connection() -> None:
    ws = FakeWebSocket([_server_event(EventType.CONNECTION_STARTED)])
    client, factory = _client(ws)

    first = client.prewarm()
    second = client.prewarm()

    assert first == {"ok": True, "status": "opened"}
    assert second == {"ok": True, "status": "reused"}
    assert len(factory.calls) == 1
    assert [_client_event(data) for data in ws.sent] == [EventType.START_CONNECTION]


def test_prewarm_handshake_does_not_block_real_synthesis() -> None:
    prewarm_ws = BlockingHandshakeWebSocket()
    synth_ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
            _server_event(EventType.SESSION_FINISHED, session_id="session-1"),
        ]
    )
    factory = FakeFactory(prewarm_ws, synth_ws)
    client = VolcengineTTSClient(
        _config(),
        connection_factory=factory,
        connect_id_factory=lambda: "connect-1",
        session_id_factory=lambda: "session-1",
    )
    prewarm_result: list[dict[str, Any]] = []
    prewarm_thread = threading.Thread(target=lambda: prewarm_result.append(client.prewarm()))
    prewarm_thread.start()
    assert prewarm_ws.handshake_recv_started.wait(timeout=1.0)

    result = client.synthesize("真实请求", on_audio=lambda _chunk: None)

    assert result.status == "finished"
    assert len(factory.calls) == 2
    prewarm_ws.release_handshake.set()
    prewarm_thread.join(timeout=1.0)
    assert prewarm_result == [{"ok": True, "status": "superseded_by_live_session"}]
    assert prewarm_ws.closed is True


def test_interrupt_unblocks_blocking_synthesis_handshake() -> None:
    ws = BlockingHandshakeWebSocket()
    client, _ = _client(ws)
    result: list[str] = []

    thread = threading.Thread(
        target=lambda: result.append(
            client.synthesize("握手中断", on_audio=lambda _chunk: None).status
        )
    )
    thread.start()
    assert ws.handshake_recv_started.wait(timeout=1.0)

    client.interrupt()
    thread.join(timeout=1.0)

    assert result == ["cancelled"]
    assert ws.aborted is True


def test_interrupt_unblocks_blocking_prewarm_handshake() -> None:
    ws = BlockingHandshakeWebSocket()
    client, _ = _client(ws)
    result: list[dict[str, Any]] = []

    thread = threading.Thread(target=lambda: result.append(client.prewarm()))
    thread.start()
    assert ws.handshake_recv_started.wait(timeout=1.0)

    client.interrupt()
    thread.join(timeout=1.0)

    assert result == [
        {"ok": False, "status": "cancelled", "reason": "interrupted"}
    ]
    assert ws.aborted is True


def test_interrupt_returns_before_blocking_connection_factory_finishes() -> None:
    ws = FakeWebSocket([_server_event(EventType.CONNECTION_STARTED)])
    connect_started = threading.Event()
    release_connect = threading.Event()

    def blocking_factory(_url: str, **_kwargs: Any) -> FakeWebSocket:
        connect_started.set()
        if not release_connect.wait(timeout=2.0):
            raise RuntimeError("test did not release connect")
        return ws

    client = VolcengineTTSClient(
        _config(connect_timeout=5.0),
        connection_factory=blocking_factory,
        connect_id_factory=lambda: "connect-1",
        session_id_factory=lambda: "session-1",
    )
    result: list[str] = []
    synth_thread = threading.Thread(
        target=lambda: result.append(
            client.synthesize("连接中断", on_audio=lambda _chunk: None).status
        )
    )
    synth_thread.start()
    assert connect_started.wait(timeout=1.0)

    started_at = time.monotonic()
    client.interrupt()
    synth_thread.join(timeout=0.5)
    elapsed = time.monotonic() - started_at

    assert result == ["cancelled"]
    assert elapsed < 0.5
    release_connect.set()
    deadline = time.monotonic() + 1.0
    while not ws.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ws.closed is True
    assert ws.sent == []


def test_interrupt_closes_synthesis_socket_before_active_publish() -> None:
    ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
            _server_event(EventType.SESSION_FINISHED, session_id="session-1"),
        ]
    )
    client, _ = _client(ws)
    handshake_complete, release_publish = _pause_after_handshake(client)
    result: list[str] = []
    thread = threading.Thread(
        target=lambda: result.append(
            client.synthesize("发布竞态", on_audio=lambda _chunk: None).status
        )
    )
    thread.start()
    assert handshake_complete.wait(timeout=1.0)

    client.interrupt()
    release_publish.set()
    thread.join(timeout=1.0)

    assert result == ["cancelled"]
    assert ws.closed is True
    assert [_client_event(frame) for frame in ws.sent] == [EventType.START_CONNECTION]


def test_interrupt_after_socket_open_cancels_before_provider_handshake() -> None:
    ws = FakeWebSocket([_server_event(EventType.CONNECTION_STARTED)])
    client, _ = _client(ws)
    socket_opened, release_tracking = _pause_after_socket_open(client)
    result: list[str] = []
    thread = threading.Thread(
        target=lambda: result.append(
            client.synthesize("建连竞态", on_audio=lambda _chunk: None).status
        )
    )
    thread.start()
    assert socket_opened.wait(timeout=1.0)

    client.interrupt()
    release_tracking.set()
    thread.join(timeout=1.0)

    assert result == ["cancelled"]
    assert ws.closed is True
    assert ws.sent == []


def test_close_closes_synthesis_socket_before_active_publish() -> None:
    ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
            _server_event(EventType.SESSION_FINISHED, session_id="session-1"),
        ]
    )
    client, _ = _client(ws)
    handshake_complete, release_publish = _pause_after_handshake(client)
    result: list[str] = []
    thread = threading.Thread(
        target=lambda: result.append(
            client.synthesize("关停竞态", on_audio=lambda _chunk: None).status
        )
    )
    thread.start()
    assert handshake_complete.wait(timeout=1.0)

    client.close()
    release_publish.set()
    thread.join(timeout=1.0)

    assert result == ["cancelled"]
    assert ws.closed is True
    assert [_client_event(frame) for frame in ws.sent] == [EventType.START_CONNECTION]


def test_interrupt_closes_prewarm_socket_before_active_publish() -> None:
    ws = FakeWebSocket([_server_event(EventType.CONNECTION_STARTED)])
    client, _ = _client(ws)
    handshake_complete, release_publish = _pause_after_handshake(client)
    result: list[dict[str, Any]] = []
    thread = threading.Thread(target=lambda: result.append(client.prewarm()))
    thread.start()
    assert handshake_complete.wait(timeout=1.0)

    client.interrupt()
    release_publish.set()
    thread.join(timeout=1.0)

    assert result == [
        {"ok": False, "status": "cancelled", "reason": "interrupted"}
    ]
    assert ws.closed is True
    assert [_client_event(frame) for frame in ws.sent] == [EventType.START_CONNECTION]


def test_close_closes_prewarm_socket_before_active_publish() -> None:
    ws = FakeWebSocket([_server_event(EventType.CONNECTION_STARTED)])
    client, _ = _client(ws)
    handshake_complete, release_publish = _pause_after_handshake(client)
    result: list[dict[str, Any]] = []
    thread = threading.Thread(target=lambda: result.append(client.prewarm()))
    thread.start()
    assert handshake_complete.wait(timeout=1.0)

    client.close()
    release_publish.set()
    thread.join(timeout=1.0)

    assert result == [
        {"ok": False, "status": "cancelled", "reason": "interrupted"}
    ]
    assert ws.closed is True
    assert [_client_event(frame) for frame in ws.sent] == [
        EventType.START_CONNECTION,
        EventType.FINISH_CONNECTION,
    ]


def test_synthesize_handshakes_reuses_connection_and_invokes_pcm_callback() -> None:
    ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
            _server_event(EventType.TTS_RESPONSE, session_id="session-1", payload=b"pcm-1"),
            _session_audio(b"pcm-2", session_id="session-1"),
            _server_event(EventType.SESSION_FINISHED, session_id="session-1"),
            _server_event(EventType.SESSION_STARTED, session_id="session-2"),
            _server_event(EventType.TTS_RESPONSE, session_id="session-2", payload=b"pcm-3"),
            _server_event(EventType.SESSION_FINISHED, session_id="session-2"),
        ]
    )
    sessions = iter(["session-1", "session-2"])
    factory = FakeFactory(ws)
    client = VolcengineTTSClient(
        _config(),
        connection_factory=factory,
        connect_id_factory=lambda: "connect-1",
        session_id_factory=lambda: next(sessions),
    )
    chunks: list[bytes] = []

    first = client.synthesize("你好", on_audio=chunks.append)
    second = client.synthesize("再见", on_audio=chunks.append)

    assert first.audio_chunks == 2
    assert first.audio_bytes == len(b"pcm-1pcm-2")
    assert second.audio_chunks == 1
    assert chunks == [b"pcm-1", b"pcm-2", b"pcm-3"]
    assert len(factory.calls) == 1
    sent_events = [_client_event(data) for data in ws.sent]
    assert sent_events == [
        EventType.START_CONNECTION,
        EventType.START_SESSION,
        EventType.TASK_REQUEST,
        EventType.FINISH_SESSION,
        EventType.START_SESSION,
        EventType.TASK_REQUEST,
        EventType.FINISH_SESSION,
    ]


def test_eventless_audio_frame_fails_closed_without_emitting_pcm() -> None:
    ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
            _audio_only(b"stale-pcm"),
        ]
    )
    client, _ = _client(ws)
    chunks: list[bytes] = []

    with pytest.raises(VolcengineTTSClientError, match="session-scoped TTS_RESPONSE"):
        client.synthesize("你好", on_audio=chunks.append)

    assert chunks == []
    assert ws.closed is True


def test_task_payload_contains_text_speaker_and_audio_params() -> None:
    ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
            _server_event(EventType.SESSION_FINISHED, session_id="session-1"),
        ]
    )
    client, _ = _client(ws)

    client.synthesize("请跟我来", on_audio=lambda _chunk: None)

    task = _client_payload_json(ws.sent[2])
    assert task["event"] == EventType.TASK_REQUEST
    assert task["req_params"]["text"] == "请跟我来"
    assert task["req_params"]["speaker"] == "test-speaker"
    assert task["req_params"]["audio_params"] == {
        "format": "pcm",
        "sample_rate": 24000,
    }


@pytest.mark.parametrize(
    ("event", "payload"),
    [
        (EventType.SESSION_STARTED, b"{}"),
        (EventType.TTS_RESPONSE, b"wrong-audio"),
        (EventType.SESSION_FINISHED, b"{}"),
    ],
)
def test_session_scoped_events_must_match_current_session(
    event: EventType,
    payload: bytes,
) -> None:
    ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(event, session_id="wrong-session", payload=payload),
        ]
    )
    client, _ = _client(ws, session_id="session-1")
    chunks: list[bytes] = []

    with pytest.raises(VolcengineTTSClientError, match="mismatched session_id"):
        client.synthesize("你好", on_audio=chunks.append)

    assert chunks == []
    assert ws.closed is True


def test_cancel_session_when_interrupted_before_task() -> None:
    ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
        ]
    )
    client, _ = _client(ws)
    calls = 0

    def should_continue() -> bool:
        nonlocal calls
        calls += 1
        return calls < 2

    result = client.synthesize(
        "停一下",
        on_audio=lambda _chunk: None,
        should_continue=should_continue,
    )

    assert result.status == "cancelled"
    sent_events = [_client_event(data) for data in ws.sent]
    assert sent_events[-1] == EventType.CANCEL_SESSION
    assert ws.closed is True


def test_cancelled_session_drops_connection_before_next_session() -> None:
    first_ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
        ]
    )
    second_ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-2"),
            _server_event(EventType.SESSION_FINISHED, session_id="session-2"),
        ]
    )
    factory = FakeFactory(first_ws, second_ws)
    sessions = iter(["session-1", "session-2"])
    client = VolcengineTTSClient(
        _config(),
        connection_factory=factory,
        connect_id_factory=lambda: "connect-1",
        session_id_factory=lambda: next(sessions),
    )
    calls = 0

    def should_continue() -> bool:
        nonlocal calls
        calls += 1
        return calls < 2

    cancelled = client.synthesize(
        "停一下",
        on_audio=lambda _chunk: None,
        should_continue=should_continue,
    )
    finished = client.synthesize("继续", on_audio=lambda _chunk: None)

    assert cancelled.status == "cancelled"
    assert finished.status == "finished"
    assert first_ws.closed is True
    assert len(factory.calls) == 2


def test_recv_after_predicate_turns_false_does_not_emit_stale_pcm() -> None:
    ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
            _server_event(EventType.TTS_RESPONSE, session_id="session-1", payload=b"stale"),
        ]
    )
    client, _ = _client(ws)
    calls = 0
    chunks: list[bytes] = []

    def should_continue() -> bool:
        nonlocal calls
        calls += 1
        return calls < 4

    result = client.synthesize(
        "你好",
        on_audio=chunks.append,
        should_continue=should_continue,
    )

    assert result.status == "cancelled"
    assert chunks == []
    assert ws.closed is True
    assert _client_event(ws.sent[-1]) == EventType.CANCEL_SESSION


def test_interrupt_unblocks_blocking_recv_from_another_thread() -> None:
    ws = BlockingWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
        ]
    )
    client, _ = _client(ws)
    result: list[str] = []

    thread = threading.Thread(
        target=lambda: result.append(
            client.synthesize("长文本", on_audio=lambda _chunk: None).status
        )
    )
    thread.start()
    assert ws.blocking_recv_started.wait(timeout=1.0)

    client.interrupt()
    thread.join(timeout=1.0)

    assert result == ["cancelled"]
    assert ws.closed is True
    assert ws.aborted is True
    assert _client_event(ws.sent[-1]) == EventType.CANCEL_SESSION


def test_prewarm_does_not_clear_active_synthesis_interrupt() -> None:
    synth_ws = DelayedCloseWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
        ]
    )
    prewarm_ws = FakeWebSocket([_server_event(EventType.CONNECTION_STARTED)])
    factory = FakeFactory(synth_ws, prewarm_ws)
    client = VolcengineTTSClient(
        _config(),
        connection_factory=factory,
        connect_id_factory=lambda: "connect-1",
        session_id_factory=lambda: "session-1",
    )
    result: list[str] = []
    thread = threading.Thread(
        target=lambda: result.append(
            client.synthesize("长文本", on_audio=lambda _chunk: None).status
        )
    )
    thread.start()
    assert synth_ws.blocking_recv_started.wait(timeout=1.0)

    client.interrupt()
    assert synth_ws.closed_observed.wait(timeout=1.0)
    prewarm_result = client.prewarm()
    synth_ws.release_closed_recv.set()
    thread.join(timeout=1.0)

    assert prewarm_result == {
        "ok": False,
        "status": "superseded",
        "reason": "synthesis_started",
    }
    assert result == ["cancelled"]
    assert prewarm_ws.closed is True


def test_blocking_candidate_close_does_not_delay_active_interrupt() -> None:
    prewarm_ws = BlockingCloseWebSocket(
        [_server_event(EventType.CONNECTION_STARTED)]
    )
    synth_ws = BlockingWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
        ]
    )
    factory = FakeFactory(prewarm_ws, synth_ws)
    client = VolcengineTTSClient(
        _config(),
        connection_factory=factory,
        connect_id_factory=lambda: "connect-1",
        session_id_factory=lambda: "session-1",
    )
    handshake_complete, release_publish = _pause_after_handshake(client)
    prewarm_thread = threading.Thread(target=client.prewarm, daemon=True)
    prewarm_thread.start()
    assert handshake_complete.wait(timeout=1.0)

    synth_result: list[str] = []
    synth_thread = threading.Thread(
        target=lambda: synth_result.append(
            client.synthesize("长文本", on_audio=lambda _chunk: None).status
        ),
        daemon=True,
    )
    synth_thread.start()
    assert synth_ws.blocking_recv_started.wait(timeout=1.0)

    release_publish.set()
    assert prewarm_ws.abort_started.wait(timeout=1.0)
    interrupt_done = threading.Event()
    interrupt_thread = threading.Thread(
        target=lambda: (client.interrupt(), interrupt_done.set()),
        daemon=True,
    )
    interrupt_thread.start()

    assert interrupt_done.wait(timeout=0.2)
    assert synth_ws.closed is True
    prewarm_ws.release_abort.set()
    interrupt_thread.join(timeout=1.0)
    prewarm_thread.join(timeout=1.0)
    synth_thread.join(timeout=1.0)

    assert synth_result == ["cancelled"]


def test_new_synthesis_runs_after_previous_operation_was_interrupted() -> None:
    interrupted_ws = BlockingWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-1"),
        ]
    )
    next_ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_STARTED, session_id="session-2"),
            _server_event(EventType.SESSION_FINISHED, session_id="session-2"),
        ]
    )
    sessions = iter(["session-1", "session-2"])
    factory = FakeFactory(interrupted_ws, next_ws)
    client = VolcengineTTSClient(
        _config(),
        connection_factory=factory,
        connect_id_factory=lambda: "connect-1",
        session_id_factory=lambda: next(sessions),
    )
    first_result: list[str] = []
    thread = threading.Thread(
        target=lambda: first_result.append(
            client.synthesize("第一轮", on_audio=lambda _chunk: None).status
        )
    )
    thread.start()
    assert interrupted_ws.blocking_recv_started.wait(timeout=1.0)

    client.interrupt()
    thread.join(timeout=1.0)
    second_result = client.synthesize("第二轮", on_audio=lambda _chunk: None)

    assert first_result == ["cancelled"]
    assert second_result.status == "finished"
    assert len(factory.calls) == 2


def test_error_closes_socket_and_redacts_credentials() -> None:
    secret = "super-secret-key"
    ws = FakeWebSocket([_server_event(EventType.CONNECTION_STARTED)])

    def failing_factory(url: str, **kwargs: Any) -> FakeWebSocket:
        _ = url, kwargs
        raise RuntimeError(f"credential={secret}")

    client = VolcengineTTSClient(
        _config(api_key=secret),
        connection_factory=failing_factory,
        connect_id_factory=lambda: "connect-1",
    )

    with pytest.raises(VolcengineTTSClientError) as exc:
        client.synthesize("你好", on_audio=lambda _chunk: None)

    assert secret not in str(exc.value)
    assert "[redacted]" in str(exc.value)
    assert ws.closed is False


def test_session_failed_closes_reusable_socket() -> None:
    ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            _server_event(EventType.SESSION_FAILED, session_id="session-1"),
        ]
    )
    client, _ = _client(ws)

    with pytest.raises(VolcengineTTSClientError, match="synthesis failed"):
        client.synthesize("你好", on_audio=lambda _chunk: None)

    assert ws.closed is True


def test_close_sends_finish_connection_best_effort() -> None:
    ws = FakeWebSocket([_server_event(EventType.CONNECTION_STARTED)])
    client, _ = _client(ws)
    client._ensure_connection()

    client.close()

    assert ws.closed is True
    assert _client_event(ws.sent[-1]) == EventType.FINISH_CONNECTION


def test_config_requires_modern_or_legacy_credentials() -> None:
    with pytest.raises(VolcengineTTSClientError, match="credentials"):
        VolcengineTTSConfig(api_key="", app_id="", access_key="", speaker="s").validate()


def test_config_requires_speaker() -> None:
    with pytest.raises(VolcengineTTSClientError, match="speaker"):
        VolcengineTTSConfig(api_key="k", speaker="").validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout", 0.0),
        ("connect_timeout", 0.0),
        ("session_timeout", -1.0),
    ],
)
def test_config_requires_positive_timeouts(field: str, value: float) -> None:
    with pytest.raises(VolcengineTTSClientError, match=field):
        _config(**{field: value}).validate()


def test_recv_text_frame_fails_closed() -> None:
    ws = FakeWebSocket(
        [
            _server_event(EventType.CONNECTION_STARTED),
            "not-binary",  # type: ignore[list-item]
        ]
    )
    client, _ = _client(ws)

    with pytest.raises(VolcengineTTSClientError, match="binary"):
        client.synthesize("你好", on_audio=lambda _chunk: None)
    assert ws.closed is True


def test_on_audio_must_be_callable() -> None:
    ws = FakeWebSocket([_server_event(EventType.CONNECTION_STARTED)])
    client, _ = _client(ws)

    with pytest.raises(TypeError):
        client.synthesize("你好", on_audio=None)  # type: ignore[arg-type]
