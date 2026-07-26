from __future__ import annotations

from askme.voice.input.aec_processor import AecStats
from askme.voice.input.full_duplex_gate import decide_full_duplex


def _aec_status(*, available: bool) -> AecStats:
    return AecStats(
        available=available,
        active=available,
        degraded=not available,
        backend="native" if available else "unavailable",
        reason=None if available else "native extension missing",
    )


def test_full_duplex_is_disabled_unless_explicitly_requested() -> None:
    decision = decide_full_duplex({}, aec_status=_aec_status(available=True))

    assert decision.enabled is False
    assert decision.reason == "not_requested"


def test_full_duplex_uses_native_aec_when_available() -> None:
    decision = decide_full_duplex(
        {"enabled": True, "echo_control": "auto"},
        aec_status=_aec_status(available=True),
    )

    assert decision.enabled is True
    assert decision.echo_control == "native"


def test_full_duplex_accepts_explicitly_verified_device_aec() -> None:
    decision = decide_full_duplex(
        {
            "enabled": True,
            "echo_control": "hardware",
            "echo_control_verified": True,
        },
        aec_status=_aec_status(available=False),
    )

    assert decision.enabled is True
    assert decision.echo_control == "hardware"
    assert decision.aec_backend == "hardware"


def test_full_duplex_rejects_unverified_hardware_claim() -> None:
    decision = decide_full_duplex(
        {"enabled": True, "echo_control": "hardware"},
        aec_status=_aec_status(available=False),
    )

    assert decision.enabled is False
    assert decision.reason == "echo_control_unverified"


def test_native_aec_must_be_active_not_only_available() -> None:
    decision = decide_full_duplex(
        {"enabled": True, "echo_control": "auto"},
        aec_status=AecStats(
            available=True,
            active=False,
            degraded=False,
            backend="native",
            reason="not initialized",
        ),
    )

    assert decision.enabled is False
    assert decision.reason == "aec_unavailable"


def test_full_duplex_fails_closed_without_real_echo_control() -> None:
    decision = decide_full_duplex(
        {"enabled": True, "echo_control": "auto"},
        aec_status=_aec_status(available=False),
    )

    assert decision.enabled is False
    assert decision.reason == "aec_unavailable"
