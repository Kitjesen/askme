"""Tests for SemanticIndex — L5 unified semantic search."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class TestInit:
    def test_default_config(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        assert idx._mem_cfg == {}
        assert idx._rm is None
        assert idx._rm_failed is False
        assert idx.available is False

    def test_custom_config(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex({"foo": "bar"})
        assert idx._mem_cfg == {"foo": "bar"}


class TestEnsureRm:
    def test_returns_false_when_import_fails(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        with patch.dict("sys.modules", {"robotmem": None, "robotmem.sdk": None}):
            result = idx._ensure_rm()
        assert result is False
        assert idx._rm_failed is True

    def test_does_not_retry_after_failure(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        idx._rm_failed = True
        result = idx._ensure_rm()
        assert result is False

    def test_returns_true_when_already_initialised(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        idx._rm = MagicMock()
        assert idx._ensure_rm() is True

    def test_initialises_on_success(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        mock_rm = MagicMock()
        mock_cls = MagicMock(return_value=mock_rm)
        mock_sdk = MagicMock()
        mock_sdk.RobotMemory = mock_cls
        with patch.dict("sys.modules", {"robotmem": MagicMock(), "robotmem.sdk": mock_sdk}):
            result = idx._ensure_rm()
        assert result is True
        assert idx._rm is mock_rm
        assert idx.available is True


class TestContentHash:
    def test_deterministic(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        assert idx._content_hash("hello") == idx._content_hash("hello")

    def test_different_inputs_differ(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        assert idx._content_hash("hello") != idx._content_hash("world")

    def test_length_12(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        assert len(idx._content_hash("x")) == 12


class TestIndexText:
    async def test_empty_text_skipped(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        result = await idx._index_text("  ", "knowledge")
        assert result is False

    async def test_already_indexed_skipped(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        h = idx._content_hash("hello")
        idx._indexed_hashes.add(h)
        result = await idx._index_text("hello", "knowledge")
        assert result is False

    async def test_no_rm_returns_false(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        idx._rm_failed = True
        result = await idx._index_text("text", "knowledge")
        assert result is False

    async def test_success_adds_hash(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        rm = MagicMock()
        rm.learn = MagicMock(return_value=None)
        idx._rm = rm
        result = await idx._index_text("new text", "knowledge", category="cat1")
        assert result is True
        h = idx._content_hash("new text")
        assert h in idx._indexed_hashes
        rm.learn.assert_called_once_with(
            "new text", context={"source": "knowledge", "category": "cat1"}
        )

    async def test_exception_returns_false(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        rm = MagicMock()
        rm.learn.side_effect = RuntimeError("db error")
        idx._rm = rm
        result = await idx._index_text("text", "source")
        assert result is False


class TestIndexKnowledge:
    async def test_returns_zero_if_dir_missing(self, tmp_path):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        result = await idx.index_knowledge(tmp_path / "nonexistent")
        assert result == 0

    async def test_indexes_bullet_lines(self, tmp_path):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        rm = MagicMock()
        rm.learn = MagicMock(return_value=None)
        idx._rm = rm
        md = tmp_path / "test.md"
        md.write_text("# Header\n- fact one\n- fact two\n", encoding="utf-8")
        result = await idx.index_knowledge(tmp_path)
        assert result == 2

    async def test_skips_non_bullet_lines(self, tmp_path):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        rm = MagicMock()
        rm.learn = MagicMock(return_value=None)
        idx._rm = rm
        md = tmp_path / "test.md"
        md.write_text("# Header\nparagraph text\n- only this\n", encoding="utf-8")
        result = await idx.index_knowledge(tmp_path)
        assert result == 1

    async def test_deduplicates_repeated_calls(self, tmp_path):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        rm = MagicMock()
        rm.learn = MagicMock(return_value=None)
        idx._rm = rm
        md = tmp_path / "test.md"
        md.write_text("- fact one\n", encoding="utf-8")
        first = await idx.index_knowledge(tmp_path)
        second = await idx.index_knowledge(tmp_path)
        assert first == 1
        assert second == 0  # already indexed


class TestIndexDigests:
    async def test_returns_zero_if_dir_missing(self, tmp_path):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        result = await idx.index_digests(tmp_path / "nonexistent")
        assert result == 0

    async def test_indexes_digest_content(self, tmp_path):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        rm = MagicMock()
        rm.learn = MagicMock(return_value=None)
        idx._rm = rm
        (tmp_path / "2024-01.md").write_text("digest content here", encoding="utf-8")
        result = await idx.index_digests(tmp_path)
        assert result == 1

    async def test_truncates_to_500_chars(self, tmp_path):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        rm = MagicMock()
        rm.learn = MagicMock(return_value=None)
        idx._rm = rm
        long_text = "x" * 1000
        (tmp_path / "long.md").write_text(long_text, encoding="utf-8")
        await idx.index_digests(tmp_path)
        call_args = rm.learn.call_args[0][0]
        assert len(call_args) == 500


class TestSearch:
    async def test_returns_empty_when_no_rm(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        idx._rm_failed = True
        result = await idx.search("query")
        assert result == []

    async def test_returns_empty_on_no_results(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        rm = MagicMock()
        rm.recall = MagicMock(return_value=[])
        idx._rm = rm
        result = await idx.search("query")
        assert result == []

    async def test_parses_results(self):
        from askme.memory.semantic_index import SemanticIndex
        import json
        idx = SemanticIndex()
        rm = MagicMock()
        rm.recall = MagicMock(return_value=[{
            "content": "温度异常",
            "context": json.dumps({"source": "knowledge", "category": "sensors"}),
            "confidence": 0.92,
        }])
        idx._rm = rm
        result = await idx.search("温度")
        assert len(result) == 1
        assert result[0]["text"] == "温度异常"
        assert result[0]["source"] == "knowledge"
        assert result[0]["category"] == "sensors"
        assert result[0]["score"] == 0.92

    async def test_source_filter(self):
        from askme.memory.semantic_index import SemanticIndex
        import json
        idx = SemanticIndex()
        rm = MagicMock()
        rm.recall = MagicMock(return_value=[
            {"content": "A", "context": json.dumps({"source": "knowledge"}), "confidence": 0.9},
            {"content": "B", "context": json.dumps({"source": "digest"}), "confidence": 0.8},
        ])
        idx._rm = rm
        result = await idx.search("q", source_filter="knowledge")
        assert len(result) == 1
        assert result[0]["text"] == "A"

    async def test_skips_empty_content(self):
        from askme.memory.semantic_index import SemanticIndex
        import json
        idx = SemanticIndex()
        rm = MagicMock()
        rm.recall = MagicMock(return_value=[
            {"content": "", "context": json.dumps({"source": "knowledge"}), "confidence": 0.9},
        ])
        idx._rm = rm
        result = await idx.search("q")
        assert result == []

    async def test_exception_returns_empty(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        rm = MagicMock()
        rm.recall.side_effect = RuntimeError("db error")
        idx._rm = rm
        result = await idx.search("q")
        assert result == []


class TestClose:
    def test_close_clears_rm(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        rm = MagicMock()
        idx._rm = rm
        idx.close()
        assert idx._rm is None
        rm.close.assert_called_once()

    def test_close_swallows_exception(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        rm = MagicMock()
        rm.close.side_effect = RuntimeError("already closed")
        idx._rm = rm
        idx.close()  # should not raise
        assert idx._rm is None

    def test_close_noop_when_not_initialised(self):
        from askme.memory.semantic_index import SemanticIndex
        idx = SemanticIndex()
        idx.close()  # no raise, no effect
