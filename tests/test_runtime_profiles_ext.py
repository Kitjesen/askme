"""Tests for RuntimeProfile and legacy_profile_for."""

from __future__ import annotations

import pytest

from askme.runtime.profiles import (
    EDGE_ROBOT_PROFILE,
    MCP_PROFILE,
    TEXT_PROFILE,
    VOICE_PROFILE,
    legacy_profile_for,
)

# ── Predefined profiles ───────────────────────────────────────────────────────

class TestPredefinedProfiles:
    def test_voice_profile_name(self):
        assert VOICE_PROFILE.name == "voice"

    def test_text_profile_name(self):
        assert TEXT_PROFILE.name == "text"

    def test_mcp_profile_name(self):
        assert MCP_PROFILE.name == "mcp"

    def test_edge_robot_profile_name(self):
        assert EDGE_ROBOT_PROFILE.name == "edge_robot"

    def test_voice_profile_is_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            VOICE_PROFILE.voice_io = False  # type: ignore

    def test_edge_robot_has_robot_api(self):
        assert EDGE_ROBOT_PROFILE.robot_api is True

    def test_text_profile_no_robot_api(self):
        assert TEXT_PROFILE.robot_api is False

    def test_mcp_profile_has_mcp(self):
        assert MCP_PROFILE.mcp is True

    def test_voice_profile_no_mcp(self):
        assert VOICE_PROFILE.mcp is False


# ── RuntimeProfile.has ────────────────────────────────────────────────────────

class TestHas:
    def test_has_memory_true_for_all(self):
        # "memory" is in _COMMON_COMPONENTS → all profiles include it
        for profile in (VOICE_PROFILE, TEXT_PROFILE, MCP_PROFILE, EDGE_ROBOT_PROFILE):
            assert profile.has("memory") is True, f"{profile.name} should have memory"

    def test_voice_has_operator_io(self):
        assert VOICE_PROFILE.has("operator_io") is True

    def test_text_profile_no_operator_io(self):
        assert TEXT_PROFILE.has("operator_io") is False

    def test_edge_robot_has_robot_io(self):
        assert EDGE_ROBOT_PROFILE.has("robot_io") is True

    def test_text_profile_no_robot_io(self):
        assert TEXT_PROFILE.has("robot_io") is False

    def test_alias_voice_io_resolves(self):
        # "voice_io" is an alias for "operator_io"
        assert VOICE_PROFILE.has("voice_io") == VOICE_PROFILE.has("operator_io")

    def test_alias_skill_runtime_resolves(self):
        # "skill_runtime" → "skills"
        assert VOICE_PROFILE.has("skill_runtime") is True

    def test_edge_robot_has_change_monitor(self):
        assert EDGE_ROBOT_PROFILE.has("change_monitor") is True

    def test_text_profile_no_change_monitor(self):
        assert TEXT_PROFILE.has("change_monitor") is False

    def test_mcp_has_robot_io(self):
        assert MCP_PROFILE.has("robot_io") is True

    def test_empty_components_falls_back_to_bool_fields(self):
        from dataclasses import replace
        # Build a profile with no components set
        profile = replace(TEXT_PROFILE, components=frozenset(), robot_api=True)
        assert profile.has("robot_io") is True


# ── RuntimeProfile.snapshot ───────────────────────────────────────────────────

class TestSnapshot:
    def test_returns_dict(self):
        snap = VOICE_PROFILE.snapshot()
        assert isinstance(snap, dict)

    def test_has_name_field(self):
        snap = VOICE_PROFILE.snapshot()
        assert snap["name"] == "voice"

    def test_components_is_sorted_list(self):
        snap = VOICE_PROFILE.snapshot()
        comps = snap["components"]
        assert isinstance(comps, list)
        assert comps == sorted(comps)

    def test_json_serializable(self):
        import json
        snap = EDGE_ROBOT_PROFILE.snapshot()
        # Should not raise
        json.dumps(snap)


# ── legacy_profile_for ────────────────────────────────────────────────────────

class TestLegacyProfileFor:
    def test_voice_true_robot_false_returns_voice(self):
        profile = legacy_profile_for(voice_mode=True, robot_mode=False)
        assert profile.name == "voice"

    def test_voice_false_robot_false_returns_text(self):
        profile = legacy_profile_for(voice_mode=False, robot_mode=False)
        assert profile.name == "text"

    def test_voice_true_robot_true_returns_edge_robot(self):
        profile = legacy_profile_for(voice_mode=True, robot_mode=True)
        assert profile.name == "edge_robot"

    def test_voice_false_robot_true_returns_text_with_robot_api(self):
        profile = legacy_profile_for(voice_mode=False, robot_mode=True)
        assert profile.robot_api is True
        # But primary loop is text
        assert profile.primary_loop == "text"

    def test_text_robot_combo_has_robot_io(self):
        profile = legacy_profile_for(voice_mode=False, robot_mode=True)
        assert profile.has("robot_io") is True
