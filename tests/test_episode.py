"""Tests for Episode dataclass and score_importance."""

from askme.brain.episode import Episode, score_importance


class TestEpisode:
    def test_create_episode(self):
        ep = Episode(event_type="command", description="用户说: 去仓库A")
        assert ep.event_type == "command"
        assert ep.description == "用户说: 去仓库A"

    def test_create_with_importance(self):
        ep = Episode(event_type="error", description="LLM超时", importance=0.9)
        assert ep.importance == 0.9

    def test_create_with_context(self):
        ep = Episode(event_type="action", description="导航", context={"target": "仓库A"})
        assert ep.context["target"] == "仓库A"

    def test_default_context_empty(self):
        ep = Episode(event_type="command", description="测试")
        assert ep.context == {} or ep.context is None


class TestScoreImportance:
    def test_returns_float(self):
        score = score_importance("command", "你好")
        assert isinstance(score, float)

    def test_range_0_to_1(self):
        for kind in ["command", "action", "error", "perception", "outcome"]:
            score = score_importance(kind, "测试内容")
            assert 0.0 <= score <= 1.0, f"{kind}: {score} out of range"

    def test_error_higher_than_perception(self):
        err = score_importance("error", "LLM错误: timeout")
        perc = score_importance("perception", "环境正常")
        assert err >= perc

    def test_with_context(self):
        score = score_importance("command", "急停", {"urgent": True})
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_empty_description(self):
        score = score_importance("command", "")
        assert isinstance(score, float)
