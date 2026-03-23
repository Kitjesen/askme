"""Tests for runtime plane builders and dependency resolution."""

from __future__ import annotations

import pytest

from askme.runtime.components import CallableComponent, resolve_start_order


# ---------------------------------------------------------------------------
# resolve_start_order
# ---------------------------------------------------------------------------

class TestResolveStartOrder:
    def test_empty_components(self) -> None:
        assert resolve_start_order({}) == []

    def test_no_dependencies(self) -> None:
        components = {
            "a": CallableComponent(name="a", description="A"),
            "b": CallableComponent(name="b", description="B"),
        }
        order = resolve_start_order(components)
        assert set(order) == {"a", "b"}

    def test_linear_dependency_chain(self) -> None:
        components = {
            "c": CallableComponent(name="c", description="C", depends_on=("b",)),
            "b": CallableComponent(name="b", description="B", depends_on=("a",)),
            "a": CallableComponent(name="a", description="A"),
        }
        order = resolve_start_order(components)
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_diamond_dependency(self) -> None:
        components = {
            "d": CallableComponent(name="d", description="D", depends_on=("b", "c")),
            "b": CallableComponent(name="b", description="B", depends_on=("a",)),
            "c": CallableComponent(name="c", description="C", depends_on=("a",)),
            "a": CallableComponent(name="a", description="A"),
        }
        order = resolve_start_order(components)
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_missing_dependency_is_ignored(self) -> None:
        components = {
            "a": CallableComponent(name="a", description="A", depends_on=("missing",)),
        }
        order = resolve_start_order(components)
        assert order == ["a"]

    def test_cycle_raises_value_error(self) -> None:
        components = {
            "a": CallableComponent(name="a", description="A", depends_on=("b",)),
            "b": CallableComponent(name="b", description="B", depends_on=("a",)),
        }
        with pytest.raises(ValueError, match="cycle"):
            resolve_start_order(components)

    def test_self_cycle_raises_value_error(self) -> None:
        components = {
            "a": CallableComponent(name="a", description="A", depends_on=("a",)),
        }
        with pytest.raises(ValueError, match="cycle"):
            resolve_start_order(components)

    def test_provides_and_profiles_stored(self) -> None:
        comp = CallableComponent(
            name="x",
            description="X",
            provides=("foo", "bar"),
            profiles=("voice", "text"),
        )
        assert comp.provides == ("foo", "bar")
        assert comp.profiles == ("voice", "text")


# ---------------------------------------------------------------------------
# RuntimeProfile.has()
# ---------------------------------------------------------------------------

from askme.runtime.profiles import (
    EDGE_ROBOT_PROFILE,
    MCP_PROFILE,
    TEXT_PROFILE,
    VOICE_PROFILE,
    RuntimeProfile,
    legacy_profile_for,
)


class TestRuntimeProfileHas:
    def test_voice_profile_has_voice_io(self) -> None:
        assert VOICE_PROFILE.has("voice_io") is True

    def test_text_profile_lacks_voice_io(self) -> None:
        assert TEXT_PROFILE.has("voice_io") is False

    def test_edge_robot_has_signal_runtime(self) -> None:
        assert EDGE_ROBOT_PROFILE.has("signal_runtime") is True

    def test_voice_profile_lacks_signal_runtime(self) -> None:
        assert VOICE_PROFILE.has("signal_runtime") is False

    def test_all_profiles_have_memory(self) -> None:
        for profile in [VOICE_PROFILE, TEXT_PROFILE, MCP_PROFILE, EDGE_ROBOT_PROFILE]:
            assert profile.has("memory") is True

    def test_has_always_true_for_common_components(self) -> None:
        assert TEXT_PROFILE.has("skill_runtime") is True
        assert TEXT_PROFILE.has("agent_shell") is True

    def test_snapshot_serializes_components(self) -> None:
        snap = VOICE_PROFILE.snapshot()
        assert isinstance(snap["components"], list)
        assert sorted(snap["components"]) == snap["components"]

    def test_legacy_profile_for_robot_adds_robot_api(self) -> None:
        profile = legacy_profile_for(voice_mode=False, robot_mode=True)
        assert profile.has("robot_api") is True
        assert profile.robot_api is True

    def test_legacy_fallback_for_empty_components(self) -> None:
        """Profile created via replace() without components uses bool fallback."""
        bare = RuntimeProfile(
            name="bare",
            description="test",
            primary_loop="text",
            voice_io=True,
            text_io=True,
            mcp=False,
            robot_api=False,
            health_http=True,
            proactive=False,
            led_bridge=False,
            change_detector=False,
            http_chat=False,
        )
        # components is empty frozenset → falls back to bool fields
        assert bare.has("voice_io") is True
        assert bare.has("robot_api") is False
        assert bare.has("control_plane") is True  # maps to health_http
        # Unknown component → True (always included)
        assert bare.has("memory") is True
