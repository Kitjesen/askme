"""Tests for AssociationGraph — vector-similarity-based memory association."""

from unittest.mock import MagicMock

from askme.memory.association import AssociationGraph


def _make_graph(available=True, search_results=None):
    """Create an AssociationGraph with a mocked VectorStore."""
    mock_store = MagicMock()
    mock_store.available = available
    mock_store.search = MagicMock(return_value=search_results or [])
    graph = AssociationGraph(mock_store)
    return graph, mock_store


class TestFindSimilarSituations:
    def test_delegates_to_vector_store(self):
        results = [
            {"text": "warehouse A temp alert 3 days ago", "score": 0.85, "metadata": {}},
        ]
        graph, store = _make_graph(search_results=results)
        found = graph.find_similar_situations("warehouse A temperature", top_k=3)
        store.search.assert_called_once_with("warehouse A temperature", top_k=3)
        assert len(found) == 1
        assert found[0]["text"] == "warehouse A temp alert 3 days ago"

    def test_empty_description_returns_empty(self):
        graph, store = _make_graph()
        assert graph.find_similar_situations("") == []
        assert graph.find_similar_situations("   ") == []
        store.search.assert_not_called()

    def test_respects_top_k(self):
        graph, store = _make_graph()
        graph.find_similar_situations("test", top_k=7)
        store.search.assert_called_once_with("test", top_k=7)


class TestFindRelatedToEntity:
    def test_delegates_to_vector_store(self):
        results = [
            {"text": "person Zhang detected at gate", "score": 0.9, "metadata": {}},
        ]
        graph, store = _make_graph(search_results=results)
        found = graph.find_related_to_entity("Zhang", top_k=5)
        store.search.assert_called_once_with("Zhang", top_k=5)
        assert len(found) == 1

    def test_empty_entity_returns_empty(self):
        graph, store = _make_graph()
        assert graph.find_related_to_entity("") == []
        store.search.assert_not_called()


class TestAvailability:
    def test_available_reflects_store(self):
        graph, _ = _make_graph(available=True)
        assert graph.available is True

        graph2, _ = _make_graph(available=False)
        assert graph2.available is False


class TestGetAssociationsText:
    def test_formats_results(self):
        results = [
            {"text": "3天前仓库A也报过温度异常", "score": 0.85, "metadata": {}},
            {"text": "仓库A正常巡检", "score": 0.45, "metadata": {}},
        ]
        graph, _ = _make_graph(search_results=results)
        text = graph.get_associations_text("仓库A温度异常")
        assert "85%" in text
        assert "3天前仓库A也报过温度异常" in text
        assert "45%" in text

    def test_filters_low_score(self):
        results = [
            {"text": "irrelevant", "score": 0.2, "metadata": {}},
        ]
        graph, _ = _make_graph(search_results=results)
        text = graph.get_associations_text("something")
        assert text == ""

    def test_empty_when_no_results(self):
        graph, _ = _make_graph(search_results=[])
        text = graph.get_associations_text("anything")
        assert text == ""
