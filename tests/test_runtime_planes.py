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


# ---------------------------------------------------------------------------
# Profile-based component filtering
# ---------------------------------------------------------------------------

class TestProfileComponentFiltering:
    """Verify _build_components filters by profile.has()."""

    def test_profile_components_match_runtime(self) -> None:
        """Components built for a profile are a subset of profile.components."""
        profile = VOICE_PROFILE
        # Simulate the filter logic from _build_components
        all_components = {
            "memory": CallableComponent(name="memory", description="M"),
            "skill_runtime": CallableComponent(name="skill_runtime", description="S"),
            "agent_shell": CallableComponent(name="agent_shell", description="A"),
            "voice_io": CallableComponent(name="voice_io", description="V"),
            "pulse": CallableComponent(name="pulse", description="P"),
            "control_plane": CallableComponent(name="control_plane", description="CP"),
            "robot_api": CallableComponent(name="robot_api", description="R"),
            "vision": CallableComponent(name="vision", description="Vis"),
            "proactive_runtime": CallableComponent(name="proactive_runtime", description="PR"),
            "perception_runtime": CallableComponent(name="perception_runtime", description="PER"),
            "signal_runtime": CallableComponent(name="signal_runtime", description="SIG"),
        }
        if profile.components:
            filtered = {k: v for k, v in all_components.items() if profile.has(k)}
        else:
            filtered = all_components

        assert set(filtered.keys()) <= set(profile.components)

    def test_robot_api_excluded_from_voice_profile(self) -> None:
        """VOICE_PROFILE does not include robot_api."""
        assert VOICE_PROFILE.has("robot_api") is False
        # Simulate: robot_api built by plane but filtered out
        all_components = {
            "memory": CallableComponent(name="memory", description="M"),
            "robot_api": CallableComponent(name="robot_api", description="R"),
        }
        filtered = {k: v for k, v in all_components.items() if VOICE_PROFILE.has(k)}
        assert "robot_api" not in filtered
        assert "memory" in filtered


# ---------------------------------------------------------------------------
# Stop order is reverse of start order
# ---------------------------------------------------------------------------

class TestStopOrderReversesStartOrder:
    """Verify RuntimeAssembly stops components in reverse topological order."""

    @pytest.mark.asyncio
    async def test_stop_order_is_reverse_start_order(self) -> None:
        """start() records order; stop() reverses it."""
        from askme.runtime.assembly import RuntimeAssembly

        log: list[str] = []

        def make_component(name: str, depends_on: tuple[str, ...] = ()) -> CallableComponent:
            return CallableComponent(
                name=name,
                description=name,
                start_hook=lambda n=name: log.append(f"start:{n}"),
                stop_hook=lambda n=name: log.append(f"stop:{n}"),
                depends_on=depends_on,
            )

        components = {
            "a": make_component("a"),
            "b": make_component("b", depends_on=("a",)),
            "c": make_component("c", depends_on=("b",)),
        }

        # Build a minimal RuntimeAssembly with async-safe mock services
        from unittest.mock import AsyncMock, MagicMock

        services = MagicMock()
        services.pipeline.shutdown = AsyncMock()
        services.audio.shutdown = MagicMock()
        services.arm_controller = None
        services.qp_memory = None

        runtime = RuntimeAssembly(
            cfg={},
            app_name="test",
            app_version="0.0.0",
            profile=TEXT_PROFILE,
            services=services,
            voice_mode=False,
            robot_mode=False,
            components=components,
        )

        await runtime.start()
        start_log = [entry for entry in log if entry.startswith("start:")]
        assert start_log == ["start:a", "start:b", "start:c"]

        # Verify _start_order is stored
        assert runtime._start_order == ["a", "b", "c"]

        await runtime.stop()
        stop_log = [entry for entry in log if entry.startswith("stop:")]
        assert stop_log == ["stop:c", "stop:b", "stop:a"]
