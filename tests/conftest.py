"""Shared pytest fixtures for askme tests."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
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
def tmp_path(project_root: Path) -> Path:
    """Create a writable temp directory inside the repository workspace."""
    base_dir = project_root / "data" / "pytest-tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="case-", dir=base_dir))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def app_context():
    """Create a minimal AppContext for MCP tool unit tests."""
    from askme.mcp_server import AppContext

    return AppContext()
