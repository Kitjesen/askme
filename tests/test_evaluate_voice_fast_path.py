from __future__ import annotations

from scripts.bench.evaluate_voice_fast_path import (
    _acceptance_failures,
    _build_projection,
    _summarize_physical_evidence,
)

ROUTE = {"count": 10, "p50": 0.02, "p95": 0.03, "max": 0.04}
CACHE = {"count": 10, "p50": 0.03, "p95": 0.04, "max": 0.05}


def test_aplay_projection_uses_cold_preroll_not_usb_assumptions() -> None:
    projection = _build_projection(
        tts_config={
            "output_transport": "aplay",
            "aplay_start_buffer_seconds": 0.6,
        },
        candidate_silence_ms=300,
        stable_partial_ms=160,
        route_stats=ROUTE,
        cache_stats=CACHE,
    )

    assert projection["available"] is True
    assert projection["output_transport"] == "aplay"
    assert projection["evidence_type"] == "software_projection"
    assert projection["measured_on_device"] is False
    assert projection["physical_first_sound"] is False
    assert projection["provenance"]["transport_components_ms"] == {
        "aplay_cold_preroll_ms": 1500.0,
    }
    assert projection["p95"] >= 1820.0
    assert "usb_chunk_settle_ms" not in projection["provenance"]


def test_usb_direct_projection_uses_only_configured_transport_guards() -> None:
    projection = _build_projection(
        tts_config={
            "output_transport": "usb_direct",
            "usb_direct_persistent_stream": True,
            "usb_direct_stream_start_grace_seconds": 0.8,
            "usb_direct_speech_leadin_seconds": 0.25,
        },
        candidate_silence_ms=300,
        stable_partial_ms=160,
        route_stats=ROUTE,
        cache_stats=CACHE,
    )

    assert projection["available"] is True
    assert projection["output_transport"] == "usb_direct"
    assert projection["provenance"]["transport_components_ms"] == {
        "usb_direct_speech_leadin_ms": 250.0,
        "usb_direct_stream_start_grace_ms": 800.0,
    }
    assert projection["p95"] >= 1370.0


def test_auto_transport_projection_fails_closed() -> None:
    projection = _build_projection(
        tts_config={"output_transport": "auto"},
        candidate_silence_ms=300,
        stable_partial_ms=160,
        route_stats=ROUTE,
        cache_stats=CACHE,
    )

    assert projection["available"] is False
    assert projection["p50"] is None
    assert projection["p95"] is None
    assert projection["reason"] == "runtime_output_transport_unresolved"
    assert projection["measured_on_device"] is False


def test_acceptance_fails_closed_without_cache_or_physical_evidence() -> None:
    failures = _acceptance_failures(
        tests_passed=True,
        benchmark={
            "false_fast_action_routes": 0,
            "route_ms": ROUTE,
            "cached_pcm_queue_ms": CACHE,
            "cache_evidence": {"ready": False},
            "projected_speech_end_to_first_pcm_ms": {
                "available": True,
                "output_transport": "aplay",
            },
        },
        physical_evidence={"ready": False, "reason": "physical_evidence_missing"},
    )

    assert "cached_phrase_evidence_missing" in failures
    assert "physical_evidence_missing" in failures


def test_physical_evidence_requires_fast_path_semantic_samples_and_transport_match() -> None:
    payload = {
        "output_transport": "aplay",
        "trials": [
            {
                "fast_path": True,
                "heard": True,
                "evidence_kind": "physical_acoustic",
                "audio_class": "semantic",
                "speech_end_to_first_semantic_audio_ms": 820 + index,
            }
            for index in range(20)
        ],
    }

    summary = _summarize_physical_evidence(payload, expected_transport="aplay")
    mismatch = _summarize_physical_evidence(payload, expected_transport="usb_direct")

    assert summary["ready"] is True
    assert summary["count"] == 20
    assert summary["p95"] < 900.0
    assert summary["evidence_type"] == "physical_acoustic_measured"
    assert mismatch["ready"] is False
    assert mismatch["reason"] == "physical_evidence_transport_mismatch"
