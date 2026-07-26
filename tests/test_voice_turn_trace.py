from __future__ import annotations

import pytest
from askme.voice.turn_trace import (
    VoiceTurnTraceRecorder,
    evaluate_voice_turn_slo,
)

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
        "speech_end_to_endpoint_commit_ms",
        "speech_end_to_asr_final_ms",
        "speech_end_to_ack_render_ms",
        "speech_end_to_ack_physical_ms",
        "speech_end_to_feedback_render_ms",
        "speech_end_to_feedback_physical_ms",
        "speech_end_to_first_llm_payload_ms",
        "speech_end_to_tts_first_pcm_ms",
        "speech_end_to_render_first_semantic_ms",
        "speech_end_to_physical_first_semantic_audio_ms",
        "barge_in_to_render_stop_ms",
        "barge_in_to_physical_speaker_stop_ms",
        "barge_in_stop_ms",
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
    assert summary["buckets"]["playback_done_ms"]["p95_ms"] is None
    assert summary["buckets"]["playback_done_ms"]["p95_min_samples"] == 100
    assert summary["slowest_bucket"] == "playback_done_ms"
    assert snapshot["slo"]["status"] == "insufficient_evidence"
    assert snapshot["slo"]["ready_to_converse"] is False


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
    assert "speech_end_to_asr_final_ms" in snapshot["slo"]["missing_buckets"]


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


def test_voice_turn_trace_barge_in_confirmed_is_not_a_stop(monkeypatch) -> None:
    clock = iter([0.00, 0.00, 0.12, 0.18])
    monkeypatch.setattr(turn_trace.time, "monotonic", lambda: next(clock))
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark_barge_in(peak=2048)
    recorder.finish("interrupted")

    snapshot = recorder.snapshot()

    assert snapshot["latest"]["latency_buckets"]["barge_in_stop_ms"] is None
    assert snapshot["latest"]["latency_buckets"]["barge_in_to_render_stop_ms"] is None


def test_voice_turn_trace_derives_barge_in_stop_from_render_stop(monkeypatch) -> None:
    clock = iter([0.00, 0.00, 0.12, 0.18, 0.20])
    monkeypatch.setattr(turn_trace.time, "monotonic", lambda: next(clock))
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark_barge_in(peak=2048)
    recorder.mark("render_speaker_stopped")
    recorder.finish("interrupted")

    buckets = recorder.snapshot()["latest"]["latency_buckets"]

    assert buckets["barge_in_stop_ms"] == 60.0
    assert buckets["barge_in_to_render_stop_ms"] == 60.0
    assert buckets["barge_in_to_physical_speaker_stop_ms"] is None


def test_voice_turn_trace_derives_customer_latency_from_speech_end(monkeypatch) -> None:
    clock = iter(
        [
            0.00,
            0.00,
            0.50,
            0.70,
            0.80,
            0.85,
            0.90,
            0.95,
            1.10,
            1.20,
            1.30,
            1.40,
            1.50,
            1.60,
            1.70,
            1.80,
        ]
    )
    monkeypatch.setattr(turn_trace.time, "monotonic", lambda: next(clock))
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark("speech_last_active_sample")
    recorder.mark("endpoint_committed")
    recorder.mark("asr_final")
    recorder.mark("interaction_admitted")
    recorder.mark_audio("ack_render_started", audio_class="ack", audio_segment_id="ack-1")
    recorder.mark_audio(
        "ack_physical_started",
        audio_class="ack",
        audio_segment_id="ack-1",
        evidence_kind="physical_acoustic",
        instrumented=True,
        clock_id="capture-1",
        provenance_verified=True,
    )
    recorder.mark_audio(
        "feedback_render_started",
        audio_class="feedback",
        audio_segment_id="wait-1",
    )
    recorder.mark_audio(
        "feedback_physical_started",
        audio_class="feedback",
        audio_segment_id="wait-1",
        evidence_kind="physical_acoustic",
        instrumented=True,
        clock_id="capture-1",
        provenance_verified=True,
    )
    recorder.mark("llm_first_payload")
    recorder.mark("llm_first_semantic_text")
    recorder.mark_audio(
        "tts_first_pcm",
        audio_class="semantic",
        audio_segment_id="answer-1",
    )
    recorder.mark_audio(
        "render_first_semantic_nonzero",
        audio_class="semantic",
        audio_segment_id="answer-1",
    )
    recorder.mark_audio(
        "physical_first_semantic_audio",
        audio_class="semantic",
        audio_segment_id="answer-1",
        evidence_kind="physical_acoustic",
        instrumented=True,
        clock_id="capture-1",
        provenance_verified=True,
    )
    recorder.finish("completed")

    snapshot = recorder.snapshot()
    buckets = snapshot["latest"]["derived_latency_buckets"]

    assert buckets["speech_end_to_endpoint_commit_ms"] == 200.0
    assert buckets["speech_end_to_asr_final_ms"] == 300.0
    assert buckets["speech_end_to_turn_admitted_ms"] == 350.0
    assert buckets["speech_end_to_ack_render_ms"] == 400.0
    assert buckets["speech_end_to_ack_physical_ms"] == 450.0
    assert buckets["speech_end_to_feedback_render_ms"] == 600.0
    assert buckets["speech_end_to_feedback_physical_ms"] == 700.0
    assert buckets["speech_end_to_first_llm_payload_ms"] == 800.0
    assert buckets["speech_end_to_first_llm_semantic_text_ms"] == 900.0
    assert buckets["speech_end_to_tts_first_pcm_ms"] == 1000.0
    assert buckets["speech_end_to_render_first_semantic_ms"] == 1100.0
    assert buckets["speech_end_to_physical_first_semantic_audio_ms"] == 1200.0
    assert snapshot["slo"]["status"] == "passed"
    assert snapshot["slo"]["missing_provenance_buckets"] == []


def test_ack_cannot_satisfy_physical_semantic_audio(monkeypatch) -> None:
    clock = iter([0.00, 0.00, 0.50, 0.70, 0.80])
    monkeypatch.setattr(turn_trace.time, "monotonic", lambda: next(clock))
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark("speech_end")
    recorder.mark_audio(
        "ack_physical_started",
        audio_class="ack",
        audio_segment_id="ack-1",
    )
    recorder.mark_audio(
        "physical_first_semantic_audio",
        audio_class="ack",
        audio_segment_id="ack-1",
    )

    snapshot = recorder.snapshot()["current"]

    assert snapshot["derived_latency_buckets"]["speech_end_to_ack_physical_ms"] == 200.0
    assert (
        snapshot["derived_latency_buckets"][
            "speech_end_to_physical_first_semantic_audio_ms"
        ]
        is None
    )


def test_physical_semantic_latency_without_provenance_is_insufficient(monkeypatch) -> None:
    clock = iter([0.00, 0.00, 0.50, 0.70])
    monkeypatch.setattr(turn_trace.time, "monotonic", lambda: next(clock))
    recorder = VoiceTurnTraceRecorder()

    recorder.start(source="microphone", media_transport="local_sounddevice")
    recorder.mark("speech_end")
    recorder.mark_audio(
        "physical_first_semantic_audio",
        audio_class="semantic",
        audio_segment_id="answer-1",
    )
    turn = recorder.snapshot()["current"]
    result = evaluate_voice_turn_slo(
        turn,
        required_buckets=("speech_end_to_physical_first_semantic_audio_ms",),
    )

    assert result["status"] == "insufficient_evidence"
    assert result["missing_buckets"] == []
    assert result["missing_provenance_buckets"] == [
        "speech_end_to_physical_first_semantic_audio_ms"
    ]


def test_audio_class_is_validated() -> None:
    recorder = VoiceTurnTraceRecorder()
    recorder.start(source="microphone", media_transport="local_sounddevice")

    with pytest.raises(ValueError, match="audio_class"):
        recorder.mark_audio("ack_render_started", audio_class="music")


def test_voice_turn_trace_default_slo_thresholds_are_exposed() -> None:
    thresholds = VoiceTurnTraceRecorder.default_slo_ms()

    assert thresholds["speech_end_to_asr_final_ms"] == 1200.0
    assert thresholds["speech_end_to_first_llm_payload_ms"] == 1800.0
    assert thresholds["speech_end_to_tts_first_pcm_ms"] == 1800.0
    assert thresholds["speech_end_to_physical_first_semantic_audio_ms"] == 1800.0


def test_voice_turn_slo_fails_on_slow_required_bucket() -> None:
    buckets = {
        "speech_end_to_endpoint_commit_ms": 300.0,
        "speech_end_to_asr_final_ms": 500.0,
        "speech_end_to_first_llm_payload_ms": 1900.0,
        "speech_end_to_tts_first_pcm_ms": 1200.0,
        "speech_end_to_render_first_semantic_ms": 1400.0,
        "speech_end_to_physical_first_semantic_audio_ms": 1500.0,
    }
    result = evaluate_voice_turn_slo(
        {
            "latency_buckets": buckets,
            "latency_bucket_provenance": {
                "speech_end_to_physical_first_semantic_audio_ms": {
                    "physical_provenance_valid": True,
                }
            },
        }
    )

    assert result["status"] == "failed"
    assert result["ready_to_converse"] is False
    assert result["failed_buckets"] == [
        {
            "bucket": "speech_end_to_first_llm_payload_ms",
            "actual_ms": 1900.0,
            "threshold_ms": 1800.0,
        }
    ]


def test_voice_turn_summary_withholds_tail_percentiles_below_sample_floor() -> None:
    recorder = VoiceTurnTraceRecorder()
    for _ in range(100):
        recorder.start(source="microphone", media_transport="local_sounddevice")
        recorder.mark("speech_end")
        recorder.mark("endpoint_committed")
        recorder.finish("completed")

    bucket = recorder.snapshot()["latency_summary"]["buckets"][
        "speech_end_to_endpoint_commit_ms"
    ]

    assert bucket["count"] == 100
    assert bucket["p95_ms"] is not None
    assert bucket["p99_ms"] is None
    assert bucket["p99_min_samples"] == 300


def test_voice_turn_slo_reports_no_turn_as_not_ready() -> None:
    result = evaluate_voice_turn_slo(None)

    assert result["status"] == "no_turn"
    assert result["ready_to_converse"] is False
    assert "speech_end_to_asr_final_ms" in result["missing_buckets"]
