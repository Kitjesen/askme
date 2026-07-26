from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
import pytest

from askme.voice.output.tts import TTSEngine
from askme.voice.output.volcengine_tts_client import (
    VolcengineTTSClientError,
    VolcengineTTSSynthesisResult,
)


def _pcm(samples: int, value: int = 1200) -> bytes:
    return (np.ones(samples, dtype="<i2") * value).tobytes()


def _base_config(**overrides: Any) -> dict[str, Any]:
    return {
        "backend": "volc",
        "phrase_cache_enabled": False,
        "sample_rate": 24000,
        "output_tail_silence_seconds": 0.0,
        "volcengine_tts_ws_url": "wss://volc.invalid/tts",
        "volcengine_tts_api_key": "secret-volc-key",
        "volcengine_tts_resource_id": "seed-tts-test",
        "volcengine_tts_speaker": "speaker-a",
        "volcengine_tts_model": "seed-tts-test-model",
        "volcengine_tts_sample_rate": 24000,
        "volcengine_tts_audio_format": "pcm",
        **overrides,
    }


class FakeVolcClient:
    def __init__(
        self,
        *,
        script: list[bytes | BaseException] | None = None,
        prewarm_started: threading.Event | None = None,
        release_prewarm: threading.Event | None = None,
        name: str = "client",
    ) -> None:
        self.script = list(script or [])
        self.prewarm_started = prewarm_started
        self.release_prewarm = release_prewarm
        self.name = name
        self.closed = False
        self.interrupted = False
        self.synthesize_calls: list[str] = []
        self.prewarm_calls = 0

    def prewarm(self) -> dict[str, Any]:
        self.prewarm_calls += 1
        if self.prewarm_started is not None:
            self.prewarm_started.set()
        if self.release_prewarm is not None:
            assert self.release_prewarm.wait(timeout=2.0)
        return {"ok": True, "status": "opened"}

    def synthesize(self, text: str, *, on_audio, should_continue=None):
        self.synthesize_calls.append(text)
        should_continue = should_continue or (lambda: True)
        audio_chunks = 0
        audio_bytes = 0
        for item in self.script:
            if not should_continue():
                return VolcengineTTSSynthesisResult("session", audio_chunks, audio_bytes, "cancelled")
            if isinstance(item, BaseException):
                raise item
            on_audio(item)
            audio_chunks += 1
            audio_bytes += len(item)
        return VolcengineTTSSynthesisResult("session", audio_chunks, audio_bytes, "finished")

    def interrupt(self) -> None:
        self.interrupted = True

    def close(self) -> None:
        self.closed = True


class FakeVolcFactory:
    def __init__(self, *clients: FakeVolcClient) -> None:
        self.clients = list(clients)
        self.created: list[FakeVolcClient] = []

    def new(self) -> FakeVolcClient:
        if not self.clients:
            raise AssertionError("no fake Volc client left")
        client = self.clients.pop(0)
        self.created.append(client)
        return client


def _install_factory(monkeypatch: pytest.MonkeyPatch, engine: TTSEngine, factory: FakeVolcFactory) -> None:
    monkeypatch.setattr(engine, "_new_volcengine_client", factory.new)


def _buffered_samples(engine: TTSEngine) -> np.ndarray:
    with engine._buffer_lock:
        chunks = list(engine.tts_buffer)
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


def test_backend_alias_dispatches_to_volcengine_and_queues_pcm(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TTSEngine(_base_config())
    client = FakeVolcClient(script=[_pcm(120)])
    _install_factory(monkeypatch, engine, FakeVolcFactory(client))
    try:
        generation = engine._get_generation()
        generated = engine._generate_audio("你好", generation)

        assert generated == "volcengine"
        assert engine.backend == "volcengine"
        audio = _buffered_samples(engine)
        assert len(audio) == 120
        assert np.max(audio) == pytest.approx(1200 / 32768.0)
        assert client.synthesize_calls == ["你好"]
    finally:
        engine.shutdown()


def test_volcengine_resamples_provider_pcm(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TTSEngine(
        _base_config(sample_rate=16000, volcengine_tts_sample_rate=32000)
    )
    client = FakeVolcClient(script=[_pcm(3200)])
    _install_factory(monkeypatch, engine, FakeVolcFactory(client))
    try:
        assert engine._generate_volcengine("resample", engine._get_generation()) is True

        audio = _buffered_samples(engine)
        assert 1500 <= len(audio) <= 1700
    finally:
        engine.shutdown()


def test_missing_volcengine_config_falls_back_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TTSEngine(_base_config(volcengine_tts_api_key="", volcengine_tts_app_id=""))
    fallback_calls: list[str] = []
    monkeypatch.setattr(
        engine,
        "_use_cloud_tts_fallback",
        lambda text, generation: fallback_calls.append(text) or "edge",
    )
    try:
        generated = engine._generate_audio("fallback", engine._get_generation())

        assert generated == "edge"
        assert fallback_calls == ["fallback"]
        assert not engine._has_buffered_audio()
    finally:
        engine.shutdown()


def test_non_pcm_volcengine_config_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TTSEngine(_base_config(volcengine_tts_audio_format="mp3"))
    fallback_calls: list[str] = []
    monkeypatch.setattr(
        engine,
        "_use_cloud_tts_fallback",
        lambda text, generation: fallback_calls.append(text) or "edge",
    )
    try:
        assert engine._generate_audio("mp3", engine._get_generation()) == "edge"
        assert fallback_calls == ["mp3"]
    finally:
        engine.shutdown()


def test_volcengine_failure_before_audio_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TTSEngine(_base_config())
    client = FakeVolcClient(script=[VolcengineTTSClientError("boom")])
    _install_factory(monkeypatch, engine, FakeVolcFactory(client))
    fallback_calls: list[str] = []
    monkeypatch.setattr(
        engine,
        "_use_cloud_tts_fallback",
        lambda text, generation: fallback_calls.append(text) or "edge",
    )
    try:
        generated = engine._generate_audio("before", engine._get_generation())

        assert generated == "edge"
        assert fallback_calls == ["before"]
        assert client.interrupted is True
        assert client.closed is True
    finally:
        engine.shutdown()


def test_volcengine_failure_after_partial_audio_never_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TTSEngine(_base_config())
    client = FakeVolcClient(
        script=[_pcm(80), VolcengineTTSClientError("provider failed after audio")]
    )
    _install_factory(monkeypatch, engine, FakeVolcFactory(client))
    fallback_calls: list[str] = []
    monkeypatch.setattr(
        engine,
        "_use_cloud_tts_fallback",
        lambda text, generation: fallback_calls.append(text) or "edge",
    )
    try:
        generated = engine._generate_audio("partial", engine._get_generation())

        assert generated is None
        assert fallback_calls == []
        assert len(_buffered_samples(engine)) == 80
    finally:
        engine.shutdown()


def test_volcengine_drain_interrupts_and_closes_live_client(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TTSEngine(_base_config())
    client = FakeVolcClient(script=[_pcm(64)])
    _install_factory(monkeypatch, engine, FakeVolcFactory(client))
    try:
        assert engine._generate_volcengine("live", engine._get_generation()) is True
        engine.drain_buffers()

        assert client.interrupted is True
        assert client.closed is True
        assert not engine._has_buffered_audio()
    finally:
        engine.shutdown()


def test_status_snapshot_hides_volcengine_secret() -> None:
    engine = TTSEngine(_base_config())
    try:
        snapshot = engine.status_snapshot()

        assert snapshot["backend"] == "volcengine"
        assert snapshot["volcengine"]["configured"] is True
        assert snapshot["volcengine"]["resource_id"] == "seed-tts-test"
        assert snapshot["volcengine"]["speaker"] == "speaker-a"
        assert "secret-volc-key" not in repr(snapshot)
    finally:
        engine.shutdown()


def test_client_config_uses_split_timeouts_and_documented_payload_only() -> None:
    engine = TTSEngine(
        _base_config(
            volcengine_tts_connect_timeout_seconds=1.5,
            volcengine_tts_session_timeout_seconds=8.0,
        )
    )
    changed_connect_timeout = TTSEngine(
        _base_config(
            volcengine_tts_connect_timeout_seconds=2.5,
            volcengine_tts_session_timeout_seconds=8.0,
        )
    )
    try:
        client_config = engine._volcengine_client_config()

        assert client_config.connect_timeout == 1.5
        assert client_config.session_timeout == 8.0
        assert client_config.extra_req_params == {}
        assert (
            engine._volcengine_configuration_signature()
            != changed_connect_timeout._volcengine_configuration_signature()
        )
    finally:
        engine.shutdown()
        changed_connect_timeout.shutdown()


def test_phrase_cache_signature_covers_volcengine_acoustic_settings(tmp_path) -> None:
    base = _base_config(phrase_cache_dir=str(tmp_path))
    first = TTSEngine(base)
    changed_speaker = TTSEngine({**base, "volcengine_tts_speaker": "speaker-b"})
    changed_rate = TTSEngine({**base, "volcengine_tts_sample_rate": 16000})
    try:
        key = first._phrase_cache_storage_key("好的。", "quick")
        assert key != changed_speaker._phrase_cache_storage_key("好的。", "quick")
        assert key != changed_rate._phrase_cache_storage_key("好的。", "quick")
    finally:
        first.shutdown()
        changed_speaker.shutdown()
        changed_rate.shutdown()


def test_voice_profile_switch_preserves_volcengine_speaker_without_provider_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TTSEngine, "start_playback", lambda self: None)
    engine = TTSEngine(_base_config())
    try:
        result = engine.set_voice_profile_payload({"profile_id": "visitor_friendly"})

        assert result["updated"] is True
        assert engine._volcengine_tts_speaker == "speaker-a"
        assert result["applied_settings"]["voice_id"] == "speaker-a"
        assert engine.status_snapshot()["volcengine"]["speaker"] == "speaker-a"
    finally:
        engine.shutdown()


def test_voice_profile_switch_uses_explicit_volcengine_speaker_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TTSEngine, "start_playback", lambda self: None)
    engine = TTSEngine(
        _base_config(
            voice_profiles={
                "visitor_friendly": {"volcengine_voice_id": "speaker-b"}
            }
        )
    )
    try:
        result = engine.set_voice_profile_payload({"profile_id": "visitor_friendly"})

        assert result["updated"] is True
        assert result["profile"]["volcengine_voice_id"] == "speaker-b"
        assert result["applied_settings"]["voice_id"] == "speaker-b"
        assert engine.status_snapshot()["volcengine"]["speaker"] == "speaker-b"
    finally:
        engine.shutdown()


def test_persisted_voice_profile_restores_explicit_volcengine_speaker(tmp_path) -> None:
    state_path = tmp_path / "voice-profile.json"
    state_path.write_text(
        json.dumps({"active_profile": "visitor_friendly"}),
        encoding="utf-8",
    )
    engine = TTSEngine(
        _base_config(
            voice_profile="patrol_default",
            voice_profile_state_path=str(state_path),
            voice_profiles={
                "visitor_friendly": {"volcengine_voice_id": "speaker-persisted"}
            },
        )
    )
    try:
        snapshot = engine.status_snapshot()

        assert snapshot["minimax"]["active_profile"] == "visitor_friendly"
        assert snapshot["volcengine"]["speaker"] == "speaker-persisted"
        assert snapshot["minimax"]["active_profile_settings"]["voice_id"] == (
            "speaker-persisted"
        )
    finally:
        engine.shutdown()


def test_volcengine_prewarm_disabled_skips_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TTSEngine(_base_config(volcengine_tts_live_session_prewarm_enabled=False))
    factory = FakeVolcFactory(FakeVolcClient())
    _install_factory(monkeypatch, engine, factory)
    try:
        assert engine.prewarm_provider_session() == {
            "ok": False,
            "status": "skipped",
            "reason": "disabled",
        }
        assert factory.created == []
    finally:
        engine.shutdown()


def test_volcengine_prewarm_opens_and_reuses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TTSEngine(_base_config(volcengine_tts_live_session_prewarm_enabled=True))
    client = FakeVolcClient()
    _install_factory(monkeypatch, engine, FakeVolcFactory(client))
    try:
        opened = engine.prewarm_provider_session()
        reused = engine.prewarm_provider_session()

        assert opened["ok"] is True
        assert opened["status"] == "opened"
        assert reused["ok"] is True
        assert reused["status"] == "reused"
        assert client.prewarm_calls == 1
        assert not engine._has_buffered_audio()
    finally:
        engine.shutdown()


def test_slow_volcengine_prewarm_never_blocks_real_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    candidate = FakeVolcClient(
        prewarm_started=started,
        release_prewarm=release,
        name="candidate",
    )
    live = FakeVolcClient(script=[_pcm(64)], name="live")
    engine = TTSEngine(_base_config(volcengine_tts_live_session_prewarm_enabled=True))
    _install_factory(monkeypatch, engine, FakeVolcFactory(candidate, live))
    prewarm_result: dict[str, Any] = {}
    try:
        thread = threading.Thread(
            target=lambda: prewarm_result.update(engine.prewarm_provider_session())
        )
        thread.start()
        assert started.wait(timeout=1.0)

        generated = engine._generate_audio("real speech", engine._get_generation())
        assert generated == "volcengine"
        assert live.synthesize_calls == ["real speech"]
        assert len(_buffered_samples(engine)) == 64

        release.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert prewarm_result["status"] in {"superseded", "superseded_by_live_session"}
        assert candidate.closed is True
    finally:
        release.set()
        engine.shutdown()


def test_shutdown_interrupts_volcengine_prewarm_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    candidate = FakeVolcClient(
        prewarm_started=started,
        release_prewarm=release,
        name="candidate",
    )
    engine = TTSEngine(_base_config(volcengine_tts_live_session_prewarm_enabled=True))
    _install_factory(monkeypatch, engine, FakeVolcFactory(candidate))
    try:
        thread = threading.Thread(target=engine.prewarm_provider_session)
        thread.start()
        assert started.wait(timeout=1.0)
        engine.shutdown()
        release.set()
        thread.join(timeout=2.0)

        assert candidate.interrupted is True
        assert candidate.closed is True
    finally:
        release.set()
