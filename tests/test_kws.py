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
    (model_dir / "tokens.txt").write_text(
        "\n".join(
            [
                "雷霆 1",
                "Thunder 2",
                "你好 3",
                "小智 4",
                "n 5",
                "ǐ 6",
                "h 7",
                "ǎo 8",
                "x 9",
                "i 10",
                "q 11",
                "ióng 12",
                "s 13",
                "uàn 14",
                "iǎo 15",
            ]
        ),
        encoding="utf-8",
    )
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


def test_kws_writes_configured_keywords_directly(tmp_path):
    from askme.voice.kws import KWSEngine

    model_dir = _create_model_dir(tmp_path)

    with patch("askme.voice.kws.sherpa_onnx.KeywordSpotter"):
        KWSEngine(
            {
                "model_dir": str(model_dir),
                "keywords": ["你好", "小智"],
            }
        )

    keywords_text = (model_dir / "keywords.txt").read_text(encoding="utf-8")
    assert "你好 @你好" in keywords_text
    assert "小智 @小智" in keywords_text


def test_kws_empty_keywords_disables(tmp_path):
    from askme.voice.kws import KWSEngine

    model_dir = _create_model_dir(tmp_path)

    engine = KWSEngine(
        {
            "model_dir": str(model_dir),
            "keywords": [],
        }
    )

    assert engine.available is False
    assert engine.spotter is None


def test_kws_accepts_pinyin_tokenized_keyword(tmp_path):
    from askme.voice.kws import KWSEngine

    model_dir = _create_model_dir(tmp_path)
    keyword = "x iǎo s uàn :2.0 #0.20 @小算"

    with patch("askme.voice.kws.sherpa_onnx.KeywordSpotter") as spotter_cls:
        engine = KWSEngine({"model_dir": str(model_dir), "keywords": [keyword]})

    assert (model_dir / "keywords.txt").read_text(encoding="utf-8").strip() == keyword
    assert engine.available is True
    spotter_cls.assert_called_once()


def test_kws_rejects_unsupported_tokens_before_native_init(tmp_path):
    from askme.voice.kws import KWSEngine

    model_dir = _create_model_dir(tmp_path)

    with patch("askme.voice.kws.sherpa_onnx.KeywordSpotter") as spotter_cls:
        engine = KWSEngine(
            {"model_dir": str(model_dir), "keywords": ["unsupported @wake"]}
        )

    assert engine.available is False
    spotter_cls.assert_not_called()
