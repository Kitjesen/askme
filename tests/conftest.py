"""Shared pytest fixtures for askme tests."""

import os
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch, tmp_path):
    """Set minimal environment variables so config.py doesn't fail."""
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-key")
    monkeypatch.setenv("MINIMAX_GROUP_ID", "0")
    monkeypatch.setenv("LOCAL_EMBED_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("TTS_VOICE_ID", "male-qn-qingse")
    monkeypatch.setenv("TTS_SPEED", "1")
    monkeypatch.setenv("TTS_EMOTION", "happy")


@pytest.fixture
def project_root() -> Path:
    """Return the askme project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def app_context():
    """Create a minimal AppContext for MCP tool unit tests."""
    from askme.mcp_server import AppContext

    return AppContext()
