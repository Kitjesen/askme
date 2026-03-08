"""Tests for KWS keyword file generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def _create_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "kws-model"
    model_dir.mkdir()
    for filename in (
        "tokens.txt",
        "encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        "joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
    ):
        (model_dir / filename).write_text("stub", encoding="utf-8")
    return model_dir


def test_kws_rewrites_keywords_file_from_config(tmp_path):
    from askme.voice.kws import KWSEngine

    model_dir = _create_model_dir(tmp_path)

    with patch("askme.voice.kws.sherpa_onnx.KeywordSpotter") as spotter_cls:
        engine = KWSEngine(
            {
                "model_dir": str(model_dir),
                "keywords": ["雷霆", "Thunder"],
            }
        )

    keywords_text = (model_dir / "keywords.txt").read_text(encoding="utf-8")
    assert "雷霆 @雷霆" in keywords_text
    assert "Thunder @Thunder" in keywords_text
    assert engine.available is True
    spotter_cls.assert_called_once()


def test_kws_migrates_legacy_default_keywords(tmp_path):
    from askme.voice.kws import KWSEngine

    model_dir = _create_model_dir(tmp_path)

    with patch("askme.voice.kws.sherpa_onnx.KeywordSpotter"):
        KWSEngine(
            {
                "model_dir": str(model_dir),
                "keywords": ["浣犲ソ", "灏忔櫤"],
            }
        )

    keywords_text = (model_dir / "keywords.txt").read_text(encoding="utf-8")
    assert "雷霆 @雷霆" in keywords_text
    assert "Thunder @Thunder" in keywords_text
