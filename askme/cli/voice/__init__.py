"""Voice CLI command handlers extracted from askme.cli."""

from __future__ import annotations

from typing import Any


def _run_voice_health_check(*, live: bool) -> dict[str, Any]:
    from askme.voice.diagnostics.health_check import run_voice_health

    return run_voice_health(live=live)


def _emit_voice_health_payload(payload: dict[str, Any]) -> None:
    from askme.voice.diagnostics.health_check import print_voice_health_summary

    print_voice_health_summary(payload)


def _run_mic_calibration(
    *,
    server: str,
    duration_s: float,
    interval_s: float,
    min_signal_peak: int,
) -> dict[str, Any]:
    from askme.voice.diagnostics.mic_calibration import collect_runtime_mic_calibration

    return collect_runtime_mic_calibration(
        server=server,
        duration_s=duration_s,
        interval_s=interval_s,
        min_signal_peak=min_signal_peak,
    )


def _emit_mic_calibration_payload(payload: dict[str, Any]) -> None:
    from askme.voice.diagnostics.mic_calibration import print_mic_calibration_summary

    print_mic_calibration_summary(payload)


def _run_sunrise_audio_doctor(
    *,
    include_command_probes: bool,
    include_output_probe: bool,
    guard_min_seconds: float,
) -> dict[str, Any]:
    from askme.voice.diagnostics.sunrise_audio_doctor import run_sunrise_audio_doctor

    return run_sunrise_audio_doctor(
        include_command_probes=include_command_probes,
        include_output_probe=include_output_probe,
        guard_min_seconds=guard_min_seconds,
    )


def _emit_sunrise_audio_doctor_payload(payload: dict[str, Any]) -> None:
    from askme.voice.diagnostics.sunrise_audio_doctor import print_sunrise_audio_doctor_summary

    print_sunrise_audio_doctor_summary(payload)


def _run_sunrise_voice_readiness(
    *,
    guard_min_seconds: float,
    include_room_loop: bool,
    room_loop_text: str | None,
    room_loop_expect_prefix: str | None,
    room_loop_trials: int,
    live_tts_room_loop: bool,
    room_loop_asr: str,
    require_cloud_asr: bool,
) -> dict[str, Any]:
    from askme.voice.diagnostics.sunrise_readiness import (
        DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
        DEFAULT_ROOM_LOOP_TEXT,
        run_sunrise_voice_readiness,
    )

    return run_sunrise_voice_readiness(
        guard_min_seconds=guard_min_seconds,
        include_room_loop=include_room_loop,
        room_loop_text=room_loop_text or DEFAULT_ROOM_LOOP_TEXT,
        room_loop_expect_prefix=room_loop_expect_prefix or DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
        room_loop_trials=room_loop_trials,
        live_tts_room_loop=live_tts_room_loop,
        room_loop_asr=room_loop_asr,
        require_cloud_asr=require_cloud_asr,
    )


def _emit_sunrise_voice_readiness_payload(payload: dict[str, Any]) -> None:
    from askme.voice.diagnostics.sunrise_readiness import print_sunrise_voice_readiness_summary

    print_sunrise_voice_readiness_summary(payload)


def _run_s100p_readiness_bundle(
    *,
    output_dir: str | None,
    field: bool,
    guard_min_seconds: float,
    include_room_loop: bool,
    room_loop_text: str | None,
    room_loop_expect_prefix: str | None,
    room_loop_trials: int,
    live_tts_room_loop: bool,
    require_cloud_asr: bool,
    health_url: str,
    change_event_file: str,
    journal_since: str,
    skip_health: bool,
    skip_service_log: bool,
    command_timeout: float,
) -> dict[str, Any]:
    from askme.voice.diagnostics.s100p_readiness_bundle import collect_s100p_readiness_bundle
    from askme.voice.diagnostics.sunrise_readiness import (
        DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
        DEFAULT_ROOM_LOOP_TEXT,
    )

    return collect_s100p_readiness_bundle(
        output_dir,
        field=field,
        guard_min_seconds=guard_min_seconds,
        include_room_loop=include_room_loop,
        room_loop_text=room_loop_text or DEFAULT_ROOM_LOOP_TEXT,
        room_loop_expect_prefix=room_loop_expect_prefix or DEFAULT_ROOM_LOOP_EXPECT_PREFIX,
        room_loop_trials=room_loop_trials,
        live_tts_room_loop=live_tts_room_loop,
        require_cloud_asr=require_cloud_asr,
        health_url=health_url,
        change_event_file=change_event_file,
        journal_since=journal_since,
        skip_health=skip_health,
        skip_service_log=skip_service_log,
        command_timeout=command_timeout,
    )


def _emit_s100p_readiness_bundle_payload(payload: dict[str, Any]) -> None:
    from askme.voice.diagnostics.s100p_readiness_bundle import print_s100p_readiness_bundle_summary

    print_s100p_readiness_bundle_summary(payload)
