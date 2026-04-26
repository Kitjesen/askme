"""Tests for runtime profiles and module system dependency ordering."""

from __future__ import annotations

import pytest

from askme.runtime.module import Module, Runtime
from askme.runtime.profiles import (
    EDGE_ROBOT_PROFILE,
    MCP_PROFILE,
    TEXT_PROFILE,
    VOICE_PROFILE,
    RuntimeProfile,
    legacy_profile_for,
)

# ---------------------------------------------------------------------------
# Minimal test modules for dependency ordering
# ---------------------------------------------------------------------------


class _ModuleA(Module):
    name = "a"
    provides = ()

    def build(self, cfg, registry):
        self._log = cfg.get("_log", [])
        self._log.append("build:a")

    async def start(self):
        self._log.append("start:a")

    async def stop(self):
        self._log.append("stop:a")


class _ModuleB(Module):
    name = "b"
    depends_on = ("a",)
    provides = ()

    def build(self, cfg, registry):
        self._log = cfg.get("_log", [])
        self._log.append("build:b")

    async def start(self):
        self._log.append("start:b")

    async def stop(self):
        self._log.append("stop:b")


class _ModuleC(Module):
    name = "c"
    depends_on = ("b",)
    provides = ()

    def build(self, cfg, registry):
        self._log = cfg.get("_log", [])
        self._log.append("build:c")

    async def start(self):
        self._log.append("start:c")

    async def stop(self):
        self._log.append("stop:c")


# ---------------------------------------------------------------------------
# RuntimeProfile.has()
# ---------------------------------------------------------------------------


class TestRuntimeProfileHas:
    def test_voice_profile_has_operator_io(self) -> None:
        assert VOICE_PROFILE.has("operator_io") is True

    def test_text_profile_lacks_operator_io(self) -> None:
        assert TEXT_PROFILE.has("operator_io") is False

    def test_edge_robot_has_indicators(self) -> None:
        assert EDGE_ROBOT_PROFILE.has("indicators") is True

    def test_voice_profile_lacks_indicators(self) -> None:
        assert VOICE_PROFILE.has("indicators") is False

    def test_all_profiles_have_memory(self) -> None:
        for profile in [VOICE_PROFILE, TEXT_PROFILE, MCP_PROFILE, EDGE_ROBOT_PROFILE]:
            assert profile.has("memory") is True

    def test_has_always_true_for_common_components(self) -> None:
        assert TEXT_PROFILE.has("skills") is True
        assert TEXT_PROFILE.has("executor") is True

    def test_snapshot_serializes_components(self) -> None:
        snap = VOICE_PROFILE.snapshot()
        assert isinstance(snap["components"], list)
        assert sorted(snap["components"]) == snap["components"]

    def test_legacy_profile_for_robot_adds_robot_io(self) -> None:
        profile = legacy_profile_for(voice_mode=False, robot_mode=True)
        assert profile.has("robot_io") is True
        assert profile.robot_api is True

    def test_legacy_fallback_for_empty_components(self) -> None:
        """Profile created directly without components uses bool fallback."""
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
        assert bare.has("operator_io") is True
        assert bare.has("robot_io") is False
        assert bare.has("diagnostics") is True  # maps to health_http
        assert bare.has("voice_io") is True  # legacy alias still resolves
        assert bare.has("robot_api") is False
        assert bare.has("control_plane") is True
        assert bare.has("memory") is True


# ---------------------------------------------------------------------------
# Profile-based component filtering
# ---------------------------------------------------------------------------


class TestProfileComponentFiltering:
    """Verify profile.has() correctly filters component names."""

    def test_robot_io_excluded_from_voice_profile(self) -> None:
        """VOICE_PROFILE does not include robot_io."""
        assert VOICE_PROFILE.has("robot_io") is False
        all_names = {"memory", "robot_io"}
        filtered = {k for k in all_names if VOICE_PROFILE.has(k)}
        assert "robot_io" not in filtered
        assert "memory" in filtered


# ---------------------------------------------------------------------------
# Stop order is reverse of start order (module system)
# ---------------------------------------------------------------------------


class TestStopOrderReversesStartOrder:
    """Verify RuntimeApp stops modules in reverse topological order."""

    @pytest.mark.asyncio
    async def test_stop_order_is_reverse_start_order(self) -> None:
        """start() records order; stop() reverses it."""
        log: list[str] = []

        runtime = (
            Runtime.use(_ModuleA)
            + Runtime.use(_ModuleB)
            + Runtime.use(_ModuleC)
        )
        app = await runtime.build({"_log": log})

        build_log = [e for e in log if e.startswith("build:")]
        assert build_log == ["build:a", "build:b", "build:c"]

        log.clear()
        await app.start()
        start_log = [e for e in log if e.startswith("start:")]
        assert start_log == ["start:a", "start:b", "start:c"]

        log.clear()
        await app.stop()
        stop_log = [e for e in log if e.startswith("stop:")]
        assert stop_log == ["stop:c", "stop:b", "stop:a"]
