"""Tests for legacy runtime profile resolution."""

from askme.runtime.profiles import (
    EDGE_ROBOT_PROFILE,
    TEXT_PROFILE,
    VOICE_PROFILE,
    legacy_profile_for,
)


def test_legacy_profile_defaults_to_text() -> None:
    profile = legacy_profile_for(voice_mode=False, robot_mode=False)

    assert profile == TEXT_PROFILE
    assert profile.primary_loop == "text"
    assert profile.robot_api is False


def test_legacy_profile_uses_voice_profile() -> None:
    profile = legacy_profile_for(voice_mode=True, robot_mode=False)

    assert profile == VOICE_PROFILE
    assert profile.primary_loop == "voice"


def test_legacy_profile_uses_edge_robot_for_voice_robot() -> None:
    profile = legacy_profile_for(voice_mode=True, robot_mode=True)

    assert profile == EDGE_ROBOT_PROFILE
    assert profile.robot_api is True
    assert profile.led_bridge is True


def test_legacy_profile_enables_robot_api_for_text_robot_mode() -> None:
    profile = legacy_profile_for(voice_mode=False, robot_mode=True)

    assert profile.name == "text"
    assert profile.primary_loop == "text"
    assert profile.robot_api is True
