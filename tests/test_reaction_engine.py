"""Tests for evaluate_rules, RuleBasedReaction, and HybridReaction."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from askme.pipeline.reaction_engine import (
    RuleBasedReaction,
    evaluate_rules,
)
from askme.schemas.events import ChangeEvent, ChangeEventType
from askme.schemas.reaction import ReactionDecision, ReactionType, SceneContext


# ── Helpers ───────────────────────────────────────────────────────────────────

def _event(event_type: ChangeEventType, **kwargs) -> ChangeEvent:
    return ChangeEvent(
        event_type=event_type,
        timestamp=time.time(),
        subject_class=kwargs.pop("subject_class", "person"),
        **kwargs,
    )


def _ctx(event: ChangeEvent, **kwargs) -> SceneContext:
    return SceneContext(event=event, **kwargs)


# ── evaluate_rules ────────────────────────────────────────────────────────────

class TestEvaluateRules:
    def test_restricted_zone_person_fires_warn(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e, zone_tags=["restricted"])
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "restricted_zone_person"
        assert decision.reaction_type == ReactionType.WARN

    def test_after_hours_fires_alert(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e, is_business_hours=False, zone_tags=[])
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "after_hours_unknown"
        assert decision.reaction_type == ReactionType.ALERT

    def test_wake_word_fires_act(self):
        e = _event(ChangeEventType.OBJECT_APPEARED, subject_class="chair")
        ctx = _ctx(e, wake_word_heard=True)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "wake_word"
        assert decision.reaction_type == ReactionType.ACT

    def test_busy_robot_fires_observe(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e, robot_busy=True, zone_tags=[])
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "busy_ignore"
        assert decision.reaction_type == ReactionType.OBSERVE

    def test_person_passing_fires_ignore(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e, person_duration_s=1.0, person_approaching=False)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_passing"
        assert decision.reaction_type == ReactionType.IGNORE

    def test_person_seen_recently_fires_observe(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e, seen_person_recently=True, minutes_since_last_person=5.0,
                   person_duration_s=5.0, person_approaching=False)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_seen_recently"
        assert decision.reaction_type == ReactionType.OBSERVE

    def test_approaching_person_fires_greet(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e, person_approaching=True, person_distance_m=2.0,
                   person_duration_s=5.0, seen_person_recently=False)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_approaching_greet"
        assert decision.reaction_type == ReactionType.GREET

    def test_entrance_zone_fires_greet(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e, zone_tags=["entrance"], person_duration_s=5.0,
                   person_approaching=False, seen_person_recently=False)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "entrance_greet"
        assert decision.reaction_type == ReactionType.GREET

    def test_person_left_fires_ignore(self):
        e = _event(ChangeEventType.PERSON_LEFT)
        ctx = _ctx(e)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_left"
        assert decision.reaction_type == ReactionType.IGNORE

    def test_object_event_fires_observe(self):
        e = _event(ChangeEventType.OBJECT_APPEARED, subject_class="chair")
        ctx = _ctx(e, wake_word_heard=False, robot_busy=False)
        decision = evaluate_rules(ctx)
        assert decision.reaction_type == ReactionType.OBSERVE

    def test_default_person_appeared_fires_observe(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e, person_duration_s=5.0, person_approaching=False,
                   seen_person_recently=False, zone_tags=[],
                   is_business_hours=True, robot_busy=False, wake_word_heard=False)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_appeared_default"
        assert decision.reaction_type == ReactionType.OBSERVE

    def test_decision_includes_context(self):
        e = _event(ChangeEventType.PERSON_LEFT)
        ctx = _ctx(e)
        decision = evaluate_rules(ctx)
        assert decision.context is ctx

    def test_metadata_copied(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e, zone_tags=["restricted"])
        decision = evaluate_rules(ctx)
        assert "template" in decision.metadata

    def test_restricted_overrides_after_hours(self):
        """Restricted zone has higher priority than after-hours."""
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e, zone_tags=["restricted"], is_business_hours=False)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "restricted_zone_person"

    def test_wake_word_overrides_busy(self):
        """Wake word fires even when robot is busy."""
        e = _event(ChangeEventType.OBJECT_APPEARED, subject_class="chair")
        ctx = _ctx(e, wake_word_heard=True, robot_busy=True)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "wake_word"


# ── RuleBasedReaction ─────────────────────────────────────────────────────────

class TestRuleBasedReaction:
    @pytest.mark.asyncio
    async def test_decide_returns_decision(self):
        reaction = RuleBasedReaction()
        e = _event(ChangeEventType.PERSON_LEFT)
        ctx = _ctx(e)
        decision = await reaction.decide(ctx)
        assert isinstance(decision, ReactionDecision)

    @pytest.mark.asyncio
    async def test_generate_content_uses_template(self):
        reaction = RuleBasedReaction()
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.WARN,
            metadata={"template": "请注意危险"},
        )
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e)
        content = await reaction.generate_content(decision, ctx)
        assert content == "请注意危险"

    @pytest.mark.asyncio
    async def test_generate_content_uses_fallback_template(self):
        reaction = RuleBasedReaction()
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.ASSIST,
            metadata={"template_fallback": "需要帮助吗？"},
        )
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e)
        content = await reaction.generate_content(decision, ctx)
        assert content == "需要帮助吗？"

    @pytest.mark.asyncio
    async def test_generate_content_empty_when_no_template(self):
        reaction = RuleBasedReaction()
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.OBSERVE,
            metadata={},
        )
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e)
        content = await reaction.generate_content(decision, ctx)
        assert content == ""

    @pytest.mark.asyncio
    async def test_execute_ignore_does_not_dispatch(self):
        dispatcher = MagicMock()
        reaction = RuleBasedReaction(alert_dispatcher=dispatcher)
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.IGNORE,
            metadata={},
        )
        await reaction.execute(decision, "")
        dispatcher.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_dispatches_warn(self):
        dispatcher = MagicMock()
        reaction = RuleBasedReaction(alert_dispatcher=dispatcher)
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.WARN,
            metadata={"severity": "warning"},
        )
        await reaction.execute(decision, "警告内容")
        dispatcher.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_logs_to_episodic(self):
        episodic = MagicMock()
        reaction = RuleBasedReaction(episodic=episodic)
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.GREET,
            metadata={},
        )
        await reaction.execute(decision, "你好")
        episodic.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_empty_content_skips_dispatch(self):
        dispatcher = MagicMock()
        reaction = RuleBasedReaction(alert_dispatcher=dispatcher)
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.WARN,
            metadata={},
        )
        await reaction.execute(decision, "")  # empty content
        dispatcher.dispatch.assert_not_called()


# ── SceneContext.to_dict ──────────────────────────────────────────────────────

class TestSceneContextToDict:
    def test_returns_dict(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e)
        d = ctx.to_dict()
        assert isinstance(d, dict)

    def test_contains_event_type(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e)
        d = ctx.to_dict()
        assert d["event_type"] == "person_appeared"

    def test_contains_zone_tags(self):
        e = _event(ChangeEventType.PERSON_APPEARED)
        ctx = _ctx(e, zone_tags=["entrance", "lobby"])
        d = ctx.to_dict()
        assert "entrance" in d["zone_tags"]


# ── ReactionDecision.to_dict ──────────────────────────────────────────────────

class TestReactionDecisionToDict:
    def test_serializes_correctly(self):
        decision = ReactionDecision(
            rule_name="my_rule",
            reaction_type=ReactionType.GREET,
            metadata={"template": "你好"},
        )
        d = decision.to_dict()
        assert d["rule_name"] == "my_rule"
        assert d["reaction_type"] == "greet"
        assert d["metadata"]["template"] == "你好"
