"""Tests for CLI helper functions — argument parsing, flags, URL normalization."""

from __future__ import annotations

import argparse

from askme.cli import (
    _looks_like_mcp_request,
    _normalise_server_url,
    _resolve_runtime_flags,
    build_parser,
)

# ── _looks_like_mcp_request ───────────────────────────────────────────────────

class TestLooksLikeMcpRequest:
    def test_transport_flag(self):
        assert _looks_like_mcp_request(["--transport", "stdio"]) is True

    def test_host_flag(self):
        assert _looks_like_mcp_request(["--host", "0.0.0.0"]) is True

    def test_port_flag(self):
        assert _looks_like_mcp_request(["--port", "8080"]) is True

    def test_no_mcp_flags(self):
        assert _looks_like_mcp_request(["--voice", "--robot"]) is False

    def test_empty_args(self):
        assert _looks_like_mcp_request([]) is False

    def test_mixed_args_with_transport(self):
        assert _looks_like_mcp_request(["run", "--transport", "sse"]) is True


# ── _normalise_server_url ─────────────────────────────────────────────────────

class TestNormaliseServerUrl:
    def test_trailing_slash_stripped(self):
        assert _normalise_server_url("http://localhost:8080/") == "http://localhost:8080"

    def test_no_trailing_slash_unchanged(self):
        assert _normalise_server_url("http://localhost:8080") == "http://localhost:8080"

    def test_multiple_trailing_slashes(self):
        assert _normalise_server_url("http://host:80///") == "http://host:80"

    def test_empty_string(self):
        assert _normalise_server_url("") == ""


# ── _resolve_runtime_flags ────────────────────────────────────────────────────

def _ns(**kwargs) -> argparse.Namespace:
    """Create argparse.Namespace with defaults."""
    defaults = {"voice": False, "text": False, "robot": False, "profile": ""}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestResolveRuntimeFlags:
    def test_default_voice_mode_true(self):
        voice, robot = _resolve_runtime_flags(_ns())
        assert voice is True

    def test_text_flag_disables_voice(self):
        voice, robot = _resolve_runtime_flags(_ns(text=True))
        assert voice is False

    def test_voice_flag_enables_voice(self):
        voice, robot = _resolve_runtime_flags(_ns(voice=True))
        assert voice is True

    def test_text_profile_disables_voice(self):
        voice, robot = _resolve_runtime_flags(_ns(profile="text"))
        assert voice is False

    def test_voice_profile_enables_voice(self):
        voice, robot = _resolve_runtime_flags(_ns(profile="voice"))
        assert voice is True

    def test_edge_robot_profile_sets_both(self):
        voice, robot = _resolve_runtime_flags(_ns(profile="edge_robot"))
        assert voice is True
        assert robot is True

    def test_robot_flag_sets_robot_mode(self):
        voice, robot = _resolve_runtime_flags(_ns(robot=True))
        assert robot is True

    def test_no_robot_flag(self):
        voice, robot = _resolve_runtime_flags(_ns(robot=False))
        assert robot is False

    def test_text_flag_overrides_profile_voice(self):
        # --text should override profile=voice
        voice, robot = _resolve_runtime_flags(_ns(text=True, profile="voice"))
        assert voice is False

    def test_voice_flag_overrides_text_flag(self):
        # --voice overrides --text (voice processed after text)
        voice, robot = _resolve_runtime_flags(_ns(text=True, voice=True))
        assert voice is True


# ── build_parser ──────────────────────────────────────────────────────────────

class TestBuildParser:
    def test_returns_argument_parser(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_has_subcommands(self):
        parser = build_parser()
        # Parse with no args — should not raise (defaults to 'run' or similar)
        # Just verify it can parse known subcommands
        args = parser.parse_args(["agent", "send", "hello"])
        assert args is not None

    def test_runtime_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["runtime", "status"])
        assert args is not None

    def test_skills_list_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["skills", "list"])
        assert args is not None

    def test_voice_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--voice"])
        assert getattr(args, "voice", False) is True

    def test_text_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--text"])
        assert getattr(args, "text", False) is True

    def test_robot_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--robot"])
        assert getattr(args, "robot", False) is True
