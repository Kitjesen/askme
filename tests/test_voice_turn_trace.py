from __future__ import annotations

from askme.voice.turn_trace import VoiceTurnTraceRecorder, evaluate_voice_turn_slo

from askme.voice import turn_trace


def test_voice_turn_trace_records_first_stage_occurrence() -> None:
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark("first_audio_frame", chunk_samples=1600)
    recorder.mark("first_audio_frame", chunk_samples=3200)
    recorder.finish("accepted", asr_source="local")

    snapshot = recorder.snapshot()
    latest = snapshot["latest"]
    stages = {stage["name"]: stage for stage in latest["stages"]}

    assert snapshot["current"] is None
    assert latest["status"] == "accepted"
    assert latest["media_transport"] == "local_sounddevice"
    assert latest["total_ms"] is not None
    assert stages["listen_started"]
    assert stages["first_audio_frame"]["metadata"]["chunk_samples"] == 1600
    assert latest["metadata"]["asr_source"] == "local"


def test_voice_turn_trace_tracks_barge_in_count() -> None:
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark_barge_in(peak=2048, rms=300.0)
    recorder.finish("accepted")

    snapshot = recorder.snapshot()
    stages = {stage["name"]: stage for stage in snapshot["latest"]["stages"]}

    assert snapshot["counters"]["barge_in_count"] == 1
    assert stages["barge_in_confirmed"]["metadata"] == {
        "peak": 2048,
        "rms": 300.0,
    }


def test_voice_turn_trace_supersedes_unfinished_turn() -> None:
    recorder = VoiceTurnTraceRecorder()

    first = recorder.start(source="microphone", media_transport="local_sounddevice")
    second = recorder.start(source="microphone", media_transport="local_sounddevice")

    snapshot = recorder.snapshot()

    assert first.voice_turn_id != second.voice_turn_id
    assert snapshot["latest"]["status"] == "superseded"
    assert snapshot["current"]["voice_turn_id"] == second.voice_turn_id


def test_voice_turn_trace_allows_post_recognition_playback_stages() -> None:
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.finish("accepted")
    recorder.mark("tts_playback_started")
    recorder.mark("playback_done")

    stages = {stage["name"] for stage in recorder.snapshot()["latest"]["stages"]}

    assert "tts_playback_started" in stages
    assert "playback_done" in stages


def test_voice_turn_trace_exposes_canonical_latency_bucket_names() -> None:
    assert set(VoiceTurnTraceRecorder.bucket_names()) >= {
        "mic_first_frame_ms",
        "vad_start_ms",
        "vad_end_ms",
        "asr_first_partial_ms",
        "asr_final_ms",
        "intent_route_ms",
        "llm_ttft_ms",
        "llm_done_ms",
        "tts_first_audio_ms",
        "playback_start_ms",
        "playback_done_ms",
        "barge_in_stop_ms",
        "llm_first_payload_ms",
        "llm_first_semantic_text_ms",
        "tts_first_request_ms",
        "tts_first_pcm_ms",
        "render_first_semantic_nonzero_ms",
        "physical_first_audio_ms",
    }


def test_voice_turn_trace_maps_existing_stages_to_latency_buckets(monkeypatch) -> None:
    clock = iter(
        [
            0.00,  # trace start
            0.00,  # listen_started
            0.01,  # first_audio_frame
            0.02,  # vad_start
            0.03,  # asr_first_partial
            0.05,  # vad_end
            0.08,  # asr_final
            0.13,  # intent_route
            0.21,  # llm_ttft
            0.34,  # llm_done
            0.55,  # tts_first_audio
            0.60,  # finish
            0.89,  # tts_playback_started
            1.00,  # playback_done
        ]
    )
    monkeypatch.setattr(turn_trace.time, "monotonic", lambda: next(clock))
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark("first_audio_frame")
    recorder.mark("vad_start")
    recorder.mark("asr_first_partial")
    recorder.mark("vad_end")
    recorder.mark("asr_final")
    recorder.mark("intent_route")
    recorder.mark("llm_ttft")
    recorder.mark("llm_done")
    recorder.mark("tts_first_audio")
    recorder.finish("accepted")
    recorder.mark("tts_playback_started")
    recorder.mark("playback_done")

    snapshot = recorder.snapshot()
    buckets = snapshot["latest"]["latency_buckets"]
    summary = snapshot["latency_summary"]

    assert buckets["mic_first_frame_ms"] == 10.0
    assert buckets["vad_start_ms"] == 20.0
    assert buckets["vad_end_ms"] == 50.0
    assert buckets["asr_first_partial_ms"] == 30.0
    assert buckets["asr_final_ms"] == 80.0
    assert buckets["intent_route_ms"] == 130.0
    assert buckets["llm_ttft_ms"] == 210.0
    assert buckets["llm_done_ms"] == 340.0
    assert buckets["tts_first_audio_ms"] == 550.0
    assert buckets["playback_start_ms"] == 890.0
    assert buckets["playback_done_ms"] == 1000.0
    assert summary["buckets"]["llm_ttft_ms"]["p50_ms"] == 210.0
    assert summary["buckets"]["playback_done_ms"]["p95_ms"] == 1000.0
    assert summary["slowest_bucket"] == "playback_done_ms"
    assert snapshot["slo"]["status"] == "insufficient_evidence"
    assert snapshot["slo"]["ready_to_converse"] is False
    assert "physical_first_audio_ms" in snapshot["slo"]["missing_buckets"]


def test_voice_turn_trace_reports_missing_latency_buckets(monkeypatch) -> None:
    clock = iter([0.00, 0.00, 0.04, 0.05])
    monkeypatch.setattr(turn_trace.time, "monotonic", lambda: next(clock))
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark("first_audio_frame")
    recorder.finish("timeout")

    snapshot = recorder.snapshot()
    latest = snapshot["latest"]
    summary = snapshot["latency_summary"]

    assert latest["latency_buckets"]["mic_first_frame_ms"] == 40.0
    assert "asr_final_ms" in latest["missing_latency_buckets"]
    assert "llm_ttft_ms" in latest["missing_latency_buckets"]
    assert summary["buckets"]["asr_final_ms"]["count"] == 0
    assert summary["buckets"]["asr_final_ms"]["missing_count"] == 1
    assert snapshot["slo"]["status"] == "insufficient_evidence"
    assert snapshot["slo"]["ready_to_converse"] is False
    assert "asr_final_ms" in snapshot["slo"]["missing_buckets"]


def test_voice_turn_trace_keeps_summary_for_tts_playback_failure(monkeypatch) -> None:
    clock = iter([0.00, 0.00, 0.10, 0.20, 0.30, 0.35])
    monkeypatch.setattr(turn_trace.time, "monotonic", lambda: next(clock))
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark("asr_final")
    recorder.finish("accepted")
    recorder.mark("tts_first_audio")
    recorder.mark("tts_playback_started", error="device_unavailable")

    snapshot = recorder.snapshot()
    buckets = snapshot["latest"]["latency_buckets"]
    summary = snapshot["latency_summary"]["buckets"]

    assert buckets["asr_final_ms"] == 100.0
    assert buckets["tts_first_audio_ms"] == 300.0
    assert buckets["playback_start_ms"] == 350.0
    assert buckets["playback_done_ms"] is None
    assert summary["playback_done_ms"]["missing_count"] == 1


def test_voice_turn_trace_maps_barge_in_to_stop_bucket(monkeypatch) -> None:
    clock = iter([0.00, 0.00, 0.12, 0.18])
    monkeypatch.setattr(turn_trace.time, "monotonic", lambda: next(clock))
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark_barge_in(peak=2048)
    recorder.finish("interrupted")

    snapshot = recorder.snapshot()

    assert snapshot["latest"]["latency_buckets"]["barge_in_stop_ms"] == 120.0
    assert snapshot["latency_summary"]["buckets"]["barge_in_stop_ms"]["p95_ms"] == 120.0


def test_voice_turn_trace_default_slo_thresholds_are_exposed() -> None:
    thresholds = VoiceTurnTraceRecorder.default_slo_ms()

    assert thresholds["asr_final_ms"] == 1200.0
    assert thresholds["llm_ttft_ms"] == 900.0
    assert thresholds["tts_first_audio_ms"] == 900.0
    assert thresholds["playback_start_ms"] == 1200.0


def test_voice_turn_slo_fails_on_slow_required_bucket() -> None:
    result = evaluate_voice_turn_slo(
        {
            "latency_buckets": {
                "asr_final_ms": 300.0,
                "llm_ttft_ms": 1400.0,
                "tts_first_audio_ms": 500.0,
                "playback_start_ms": 700.0,
            }
        }
    )

    assert result["status"] == "failed"
    assert result["ready_to_converse"] is False
    assert result["failed_buckets"] == [
        {
            "bucket": "llm_ttft_ms",
            "actual_ms": 1400.0,
            "threshold_ms": 900.0,
        }
    ]


def test_voice_turn_slo_reports_no_turn_as_not_ready() -> None:
    result = evaluate_voice_turn_slo(None)

    assert result["status"] == "no_turn"
    assert result["ready_to_converse"] is False
    assert "asr_final_ms" in result["missing_buckets"]


def test_voice_turn_slo_requires_external_physical_first_audio_evidence() -> None:
    result = evaluate_voice_turn_slo(
        {
            "latency_buckets": {
                "asr_final_ms": 300.0,
                "llm_ttft_ms": 400.0,
                "tts_first_audio_ms": 500.0,
                "playback_start_ms": 650.0,
                "render_first_semantic_nonzero_ms": 700.0,
                "physical_first_audio_ms": None,
            }
        }
    )

    assert result["status"] == "insufficient_evidence"
    assert result["ready_to_converse"] is False
    assert "physical_first_audio_ms" in result["missing_buckets"]


def test_voice_turn_trace_rejects_software_physical_audio_marker() -> None:
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark(
        "physical_first_audio_observed",
        evidence_source="playback_thread",
    )
    recorder.finish("accepted")

    assert recorder.snapshot()["latest"]["latency_buckets"]["physical_first_audio_ms"] is None


def test_voice_turn_trace_accepts_explicit_external_physical_audio_measurement(
    monkeypatch,
) -> None:
    clock = iter([0.0, 0.0, 0.42, 0.5])
    monkeypatch.setattr(turn_trace.time, "monotonic", lambda: next(clock))
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark_physical_first_audio(
        evidence_source="loopback_microphone",
        detector="energy_threshold",
    )
    recorder.finish("accepted")

    latest = recorder.snapshot()["latest"]
    stage = next(
        item for item in latest["stages"] if item["name"] == "physical_first_audio_observed"
    )

    assert latest["latency_buckets"]["physical_first_audio_ms"] == 420.0
    assert stage["metadata"]["evidence_kind"] == "external_sensor"
    assert stage["metadata"]["evidence_source"] == "loopback_microphone"
