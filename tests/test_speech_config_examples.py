"""The shipped word list is present in both runtime configs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED = (
    "小算|小蒜",
    "大算|大蒜",
    "聚龙科创e谷|聚龙科创伊谷",
    "巨龙科创e谷|巨龙科创伊谷",
)


@pytest.mark.parametrize("name", ["config.yaml", "config.board.yaml"])
def test_tts_config_documents_the_word_list_and_env_extension(name: str) -> None:
    text = (_ROOT / name).read_text(encoding="utf-8")
    tts_block = text.split("tts:", 1)[1].split("phrase_cache_enabled:", 1)[0]

    assert "correct_words:" in tts_block
    assert "ASKME_TTS_CORRECT_WORDS" in tts_block
    assert "device_session_id:" in tts_block
    for entry in _REQUIRED:
        assert entry in tts_block
