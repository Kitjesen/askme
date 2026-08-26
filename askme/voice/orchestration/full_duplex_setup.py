"""Wire verified echo control into the existing local audio frontend."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

from askme.voice.input.aec_bridge import AecPcmBridge
from askme.voice.input.aec_processor import AecProcessor
from askme.voice.input.full_duplex_gate import FullDuplexDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FullDuplexSetupResult:
    enabled: bool
    reason: str
    echo_control: str
    aec_backend: str


@dataclass(frozen=True, slots=True)
class _HalfDuplexSettings:
    post_tts_input_cooldown_s: float | None
    echo_gate_peak: int | None
    processor_echo_gate_peak: int | None


def configure_full_duplex(
    *,
    audio: Any,
    audio_router: Any,
    decision: FullDuplexDecision,
    aec_processor: AecProcessor | None,
    aec_sample_rate_hz: int,
    aec_delay_ms: int = 40,
) -> FullDuplexSetupResult:
    """Install AEC callbacks, then permit capture and playback to overlap."""

    result = FullDuplexSetupResult(
        enabled=False,
        reason=decision.reason,
        echo_control=decision.echo_control,
        aec_backend=decision.aec_backend,
    )
    if not decision.enabled:
        return _publish(audio, result)

    set_mode = getattr(audio_router, "set_mode", None)
    if not callable(set_mode):
        return _publish(
            audio,
            FullDuplexSetupResult(
                enabled=False,
                reason="audio_router_not_switchable",
                echo_control="none",
                aec_backend=decision.aec_backend,
            ),
        )

    half_duplex_settings = _capture_half_duplex_settings(audio)

    bridge: AecPcmBridge | None = None
    render_setter: Any = None
    capture_setter: Any = None
    transport_failure_setter = getattr(
        getattr(audio, "tts", None),
        "set_render_transport_failure_callback",
        None,
    )
    failure_lock = threading.Lock()
    failure_handled = False

    def _fail_closed(reason: str, exc: BaseException) -> FullDuplexSetupResult:
        nonlocal failure_handled
        with failure_lock:
            if failure_handled:
                return FullDuplexSetupResult(
                    enabled=False,
                    reason=reason,
                    echo_control="none",
                    aec_backend=decision.aec_backend,
                )
            failure_handled = True

        logger.error(
            "Full-duplex media failed at runtime (%s); reverting to half-duplex: %s",
            reason,
            exc,
        )
        degraded_result = FullDuplexSetupResult(
            enabled=False,
            reason=reason,
            echo_control="none",
            aec_backend=decision.aec_backend,
        )
        # The safety transition must not wait for USB/process/network cleanup.
        # Capture can observe this state immediately, even when a render/AEC
        # callback is stalled in another thread.
        fail_closed = getattr(audio_router, "fail_closed", None)
        if callable(fail_closed):
            fail_closed()
        else:
            try:
                set_mode("exclusive")
            except Exception as route_exc:
                logger.error("Failed to restore exclusive audio routing: %s", route_exc)
        _restore_half_duplex_settings(audio, half_duplex_settings)
        _publish(audio, degraded_result)
        if callable(capture_setter):
            try:
                capture_setter(None)
            except Exception as setter_exc:
                logger.error("Failed to clear AEC capture seam: %s", setter_exc)
        if callable(render_setter):
            try:
                render_setter(None, reset_existing=False)
            except Exception as setter_exc:
                logger.error("Failed to clear AEC render seam: %s", setter_exc)
        if callable(transport_failure_setter):
            try:
                transport_failure_setter(None)
            except Exception as setter_exc:
                logger.error("Failed to clear render transport seam: %s", setter_exc)

        def _cleanup_failed_media() -> None:
            _call_if_available(audio, "stop_immediately")
            _call_if_available(audio, "drain_buffers")

        threading.Thread(
            target=_cleanup_failed_media,
            name="voice-full-duplex-fail-closed",
            daemon=True,
        ).start()
        return degraded_result

    def _runtime_fail_closed(
        reason: str,
        exc: BaseException | None = None,
    ) -> None:
        _fail_closed(reason, exc or RuntimeError(reason))

    def _aec_fail_closed(exc: BaseException) -> None:
        _fail_closed("aec_runtime_failure", exc)

    def _transport_fail_closed(exc: BaseException) -> None:
        _fail_closed("render_transport_runtime_failure", exc)

    if decision.echo_control == "native":
        render_setter = getattr(getattr(audio, "tts", None), "set_render_reference_callback", None)
        capture_setter = getattr(audio, "set_capture_processor", None)
        if aec_processor is None or not callable(render_setter) or not callable(capture_setter):
            return _publish(
                audio,
                FullDuplexSetupResult(
                    enabled=False,
                    reason="aec_media_seam_unavailable",
                    echo_control="none",
                    aec_backend=decision.aec_backend,
                ),
            )

        bridge = AecPcmBridge(
            aec_processor,
            sample_rate_hz=aec_sample_rate_hz,
            delay_ms=aec_delay_ms,
        )

        def _feed_render(samples: Any, sample_rate_hz: int) -> None:
            bridge.feed_render(samples, sample_rate_hz=sample_rate_hz)

        def _clean_capture(
            samples: Any,
            sample_rate_hz: int,
            tts_active: bool,
        ) -> Any:
            del tts_active
            return bridge.process_capture(samples, sample_rate_hz=sample_rate_hz)

        try:
            render_setter(
                _feed_render,
                on_failure=_aec_fail_closed,
                on_reset=bridge.reset,
            )
            capture_setter(_clean_capture, on_failure=_aec_fail_closed)
        except Exception as exc:
            return _fail_closed("aec_media_seam_setup_failure", exc)
        setattr(audio, "_aec_bridge", bridge)

    if not callable(transport_failure_setter):
        return _publish(
            audio,
            FullDuplexSetupResult(
                enabled=False,
                reason="render_transport_seam_unavailable",
                echo_control="none",
                aec_backend=decision.aec_backend,
            ),
        )
    try:
        transport_failure_setter(_transport_fail_closed)
    except Exception as exc:
        return _fail_closed("render_transport_seam_setup_failure", exc)
    setattr(audio, "_full_duplex_fail_closed", _runtime_fail_closed)
    try:
        set_mode("full_duplex")
    except Exception as exc:
        return _fail_closed("audio_router_activation_failure", exc)
    _disable_half_duplex_gates(audio)
    result = FullDuplexSetupResult(
        enabled=True,
        reason=decision.reason,
        echo_control=decision.echo_control,
        aec_backend=decision.aec_backend,
    )
    return _publish(audio, result)


def _disable_half_duplex_gates(audio: Any) -> None:
    if hasattr(audio, "_post_tts_input_cooldown_s"):
        audio._post_tts_input_cooldown_s = 0.0
    if hasattr(audio, "_echo_gate_peak"):
        audio._echo_gate_peak = 0
    processor = getattr(audio, "_audio_proc", None)
    if processor is not None and hasattr(processor, "_echo_gate_peak"):
        processor._echo_gate_peak = 0


def _capture_half_duplex_settings(audio: Any) -> _HalfDuplexSettings:
    processor = getattr(audio, "_audio_proc", None)
    return _HalfDuplexSettings(
        post_tts_input_cooldown_s=getattr(audio, "_post_tts_input_cooldown_s", None),
        echo_gate_peak=getattr(audio, "_echo_gate_peak", None),
        processor_echo_gate_peak=getattr(processor, "_echo_gate_peak", None),
    )


def _restore_half_duplex_settings(
    audio: Any,
    settings: _HalfDuplexSettings,
) -> None:
    if settings.post_tts_input_cooldown_s is not None:
        audio._post_tts_input_cooldown_s = settings.post_tts_input_cooldown_s
    if settings.echo_gate_peak is not None:
        audio._echo_gate_peak = settings.echo_gate_peak
    processor = getattr(audio, "_audio_proc", None)
    if processor is not None and settings.processor_echo_gate_peak is not None:
        processor._echo_gate_peak = settings.processor_echo_gate_peak


def _call_if_available(target: Any, name: str) -> None:
    callback = getattr(target, name, None)
    if not callable(callback):
        return
    try:
        callback()
    except Exception as exc:
        logger.error("Full-duplex fail-closed step %s failed: %s", name, exc)


def _publish(audio: Any, result: FullDuplexSetupResult) -> FullDuplexSetupResult:
    setattr(audio, "full_duplex_enabled", result.enabled)
    setattr(audio, "full_duplex_status", asdict(result))
    return result
