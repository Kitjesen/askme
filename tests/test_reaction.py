"""Tests for the ReactionModule: schemas, rule matrix, backends, module wiring."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.schemas.events import ChangeEvent, ChangeEventType
from askme.schemas.reaction import ReactionDecision, ReactionType, SceneContext
from askme.pipeline.reaction_engine import (
    HybridReaction,
    LLMReaction,
    RuleBasedReaction,
    evaluate_rules,
)


# -- Helpers ------------------------------------------------------------------


def _event(
    event_type: ChangeEventType = ChangeEventType.PERSON_APPEARED,
    subject_class: str = "person",
    distance_m: float | None = None,
) -> ChangeEvent:
    return ChangeEvent(
        event_type=event_type,
        timestamp=time.time(),
        subject_class=subject_class,
        distance_m=distance_m,
    )


def _ctx(
    event_type: ChangeEventType = ChangeEventType.PERSON_APPEARED,
    **kwargs,
) -> SceneContext:
    """Build a SceneContext with sensible defaults, override via kwargs."""
    evt = kwargs.pop("event", None) or _event(event_type)
    defaults = {
        "event": evt,
        "person_count": 1,
        "person_distance_m": 3.0,
        "person_duration_s": 5.0,
        "person_approaching": False,
        "person_stationary": False,
        "hour": 10,
        "is_business_hours": True,
        "zone_name": "",
        "zone_tags": [],
        "seen_person_recently": False,
        "minutes_since_last_person": 999.0,
        "robot_busy": False,
        "wake_word_heard": False,
    }
    defaults.update(kwargs)
    return SceneContext(**defaults)

# -- SceneContext tests --------------------------------------------------------


class TestSceneContext:
    def test_frozen(self):
        ctx = _ctx()
        with pytest.raises(AttributeError):
            ctx.person_count = 99  # type: ignore[misc]

    def test_to_dict(self):
        ctx = _ctx(person_count=2, hour=14)
        d = ctx.to_dict()
        assert d["person_count"] == 2
        assert d["hour"] == 14
        assert d["event_type"] == "person_appeared"

    def test_default_zone_tags_empty(self):
        ctx = _ctx()
        assert ctx.zone_tags == []


# -- ReactionDecision tests ----------------------------------------------------


class TestReactionDecision:
    def test_to_dict(self):
        d = ReactionDecision(
            rule_name="test_rule",
            reaction_type=ReactionType.GREET,
            metadata={"template": "hi"},
        )
        result = d.to_dict()
        assert result["rule_name"] == "test_rule"
        assert result["reaction_type"] == "greet"
        assert result["metadata"]["template"] == "hi"
        assert "timestamp" in result

    def test_frozen(self):
        d = ReactionDecision(rule_name="x", reaction_type=ReactionType.IGNORE)
        with pytest.raises(AttributeError):
            d.rule_name = "y"  # type: ignore[misc]


# -- Rule matrix tests ---------------------------------------------------------


class TestRuleMatrix:
    """Test each rule in the priority-ordered matrix fires correctly."""

    def test_restricted_zone_warns(self):
        ctx = _ctx(zone_tags=["restricted"])
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "restricted_zone_person"
        assert decision.reaction_type == ReactionType.WARN

    def test_after_hours_alerts(self):
        ctx = _ctx(is_business_hours=False)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "after_hours_unknown"
        assert decision.reaction_type == ReactionType.ALERT
        assert decision.metadata.get("escalate") is True

    def test_restricted_zone_beats_after_hours(self):
        ctx = _ctx(is_business_hours=False, zone_tags=["restricted"])
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "restricted_zone_person"

    def test_wake_word_acts(self):
        ctx = _ctx(wake_word_heard=True)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "wake_word"
        assert decision.reaction_type == ReactionType.ACT

    def test_busy_robot_observes(self):
        ctx = _ctx(robot_busy=True)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "busy_ignore"
        assert decision.reaction_type == ReactionType.OBSERVE

    def test_busy_robot_restricted_still_warns(self):
        ctx = _ctx(robot_busy=True, zone_tags=["restricted"])
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "restricted_zone_person"
        assert decision.reaction_type == ReactionType.WARN

    def test_person_passing_ignored(self):
        ctx = _ctx(person_duration_s=1.0, person_approaching=False)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_passing"
        assert decision.reaction_type == ReactionType.IGNORE

    def test_person_seen_recently_observes(self):
        ctx = _ctx(seen_person_recently=True, minutes_since_last_person=5.0)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_seen_recently"
        assert decision.reaction_type == ReactionType.OBSERVE

    def test_person_approaching_greets(self):
        ctx = _ctx(person_approaching=True, person_distance_m=2.5, person_duration_s=5.0)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_approaching_greet"
        assert decision.reaction_type == ReactionType.GREET
        assert decision.metadata.get("use_llm") is True

    def test_person_approaching_far_no_greet(self):
        ctx = _ctx(person_approaching=True, person_distance_m=6.0, person_duration_s=5.0)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_appeared_default"
        assert decision.reaction_type == ReactionType.OBSERVE
    def test_person_lingering_assists(self):
        ctx = _ctx(
            event_type=ChangeEventType.COUNT_CHANGED,
            person_duration_s=150.0,
            person_stationary=True,
        )
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_lingering"
        assert decision.reaction_type == ReactionType.ASSIST

    def test_entrance_greets(self):
        ctx = _ctx(zone_tags=["entrance"], person_duration_s=5.0)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "entrance_greet"
        assert decision.reaction_type == ReactionType.GREET

    def test_default_person_appeared_observes(self):
        ctx = _ctx(person_duration_s=5.0)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_appeared_default"
        assert decision.reaction_type == ReactionType.OBSERVE

    def test_person_left_ignored(self):
        ctx = _ctx(event_type=ChangeEventType.PERSON_LEFT)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "person_left"
        assert decision.reaction_type == ReactionType.IGNORE

    def test_object_change_default(self):
        evt = _event(ChangeEventType.OBJECT_APPEARED, subject_class="chair")
        ctx = _ctx(event=evt)
        decision = evaluate_rules(ctx)
        assert decision.rule_name == "object_change_default"
        assert decision.reaction_type == ReactionType.OBSERVE


# -- RuleBasedReaction tests ---------------------------------------------------


class TestRuleBasedReaction:
    async def test_decide_returns_decision(self):
        engine = RuleBasedReaction()
        ctx = _ctx(zone_tags=["restricted"])
        decision = await engine.decide(ctx)
        assert decision.reaction_type == ReactionType.WARN

    async def test_generate_content_uses_template(self):
        engine = RuleBasedReaction()
        ctx = _ctx(zone_tags=["restricted"])
        decision = await engine.decide(ctx)
        content = await engine.generate_content(decision, ctx)
        assert len(content) > 0

    async def test_generate_content_empty_for_observe(self):
        engine = RuleBasedReaction()
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.OBSERVE,
        )
        ctx = _ctx()
        content = await engine.generate_content(decision, ctx)
        assert content == ""

    async def test_execute_dispatches_for_warn(self):
        dispatcher = MagicMock()
        engine = RuleBasedReaction(alert_dispatcher=dispatcher)
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.WARN,
            metadata={"severity": "warning"},
        )
        await engine.execute(decision, "alert text")
        dispatcher.dispatch.assert_called_once()

    async def test_execute_skips_for_ignore(self):
        dispatcher = MagicMock()
        engine = RuleBasedReaction(alert_dispatcher=dispatcher)
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.IGNORE,
        )
        await engine.execute(decision, "")
        dispatcher.dispatch.assert_not_called()

    async def test_execute_logs_to_episodic(self):
        episodic = MagicMock()
        engine = RuleBasedReaction(episodic=episodic)
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.OBSERVE,
        )
        await engine.execute(decision, "")
        episodic.log.assert_called_once()

# -- HybridReaction tests ------------------------------------------------------


class TestHybridReaction:
    async def test_decide_uses_rules(self):
        engine = HybridReaction()
        ctx = _ctx(person_duration_s=5.0)
        decision = await engine.decide(ctx)
        assert decision.rule_name == "person_appeared_default"

    async def test_generate_content_returns_template_when_no_llm(self):
        engine = HybridReaction()
        ctx = _ctx(zone_tags=["entrance"], person_duration_s=5.0)
        decision = await engine.decide(ctx)
        content = await engine.generate_content(decision, ctx)
        assert len(content) > 0

    async def test_generate_content_calls_llm_when_use_llm(self):
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="LLM generated greeting")
        engine = HybridReaction(llm=mock_llm, content_timeout=2.0)
        ctx = _ctx(person_approaching=True, person_distance_m=2.0, person_duration_s=5.0)
        decision = await engine.decide(ctx)
        assert decision.metadata.get("use_llm") is True
        content = await engine.generate_content(decision, ctx)
        assert content == "LLM generated greeting"
        mock_llm.chat.assert_called_once()

    async def test_generate_content_falls_back_on_llm_timeout(self):
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(side_effect=asyncio.TimeoutError())
        engine = HybridReaction(llm=mock_llm, content_timeout=0.1)
        ctx = _ctx(
            event_type=ChangeEventType.COUNT_CHANGED,
            person_duration_s=150.0,
            person_stationary=True,
        )
        decision = await engine.decide(ctx)
        content = await engine.generate_content(decision, ctx)
        assert len(content) > 0

    async def test_generate_content_empty_for_observe(self):
        engine = HybridReaction()
        ctx = _ctx(person_duration_s=5.0)
        decision = await engine.decide(ctx)
        assert decision.reaction_type == ReactionType.OBSERVE
        content = await engine.generate_content(decision, ctx)
        assert content == ""

    async def test_execute_dispatches(self):
        dispatcher = MagicMock()
        engine = HybridReaction(alert_dispatcher=dispatcher)
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.GREET,
            metadata={},
        )
        await engine.execute(decision, "hello")
        dispatcher.dispatch.assert_called_once()


# -- LLMReaction tests ---------------------------------------------------------


class TestLLMReaction:
    async def test_decide_falls_back_to_rules_without_llm(self):
        engine = LLMReaction()
        ctx = _ctx(zone_tags=["restricted"])
        decision = await engine.decide(ctx)
        assert decision.rule_name == "restricted_zone_person"

    async def test_decide_uses_rules_with_llm(self):
        mock_llm = AsyncMock()
        engine = LLMReaction(llm=mock_llm)
        ctx = _ctx(person_duration_s=5.0)
        decision = await engine.decide(ctx)
        assert decision.rule_name == "person_appeared_default"

    async def test_generate_content_uses_llm(self):
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="LLM says hello")
        engine = LLMReaction(llm=mock_llm, decision_timeout=2.0)
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.GREET,
            metadata={},
        )
        ctx = _ctx()
        content = await engine.generate_content(decision, ctx)
        assert content == "LLM says hello"

    async def test_generate_content_falls_back_without_llm(self):
        engine = LLMReaction()
        decision = ReactionDecision(
            rule_name="test",
            reaction_type=ReactionType.WARN,
            metadata={"template": "warning text"},
        )
        ctx = _ctx()
        content = await engine.generate_content(decision, ctx)
        assert content == "warning text"


# -- ReactionModule build test -------------------------------------------------


class TestReactionModule:
    def test_module_builds_with_empty_registry(self):
        from askme.runtime.module import ModuleRegistry
        from askme.runtime.modules.reaction_module import ReactionModule

        registry = ModuleRegistry()
        mod = ReactionModule()
        mod.build({}, registry)
        assert mod._backend_name == "hybrid"
        assert mod.engine is not None

    def test_module_builds_rules_backend(self):
        from askme.runtime.module import ModuleRegistry
        from askme.runtime.modules.reaction_module import ReactionModule

        registry = ModuleRegistry()
        mod = ReactionModule()
        mod.build({"proactive": {"reaction": {"backend": "rules"}}}, registry)
        assert mod._backend_name == "rules"
        assert isinstance(mod.engine, RuleBasedReaction)

    def test_module_health(self):
        from askme.runtime.module import ModuleRegistry
        from askme.runtime.modules.reaction_module import ReactionModule

        registry = ModuleRegistry()
        mod = ReactionModule()
        mod.build({}, registry)
        health = mod.health()
        assert health["status"] == "ok"
        assert health["backend"] == "hybrid"

    def test_module_name_and_depends(self):
        from askme.runtime.modules.reaction_module import ReactionModule

        assert ReactionModule.name == "reaction"
        assert "perception" in ReactionModule.depends_on
        assert "memory" in ReactionModule.depends_on

