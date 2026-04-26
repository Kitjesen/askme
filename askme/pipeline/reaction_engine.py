"""ReactionEngine -- smart scene-aware reaction system.

Three implementations:
- RuleBasedReaction: Pure rule matrix, no LLM. <1ms decisions. Templates only.
- HybridReaction: Rules decide WHAT, LLM generates WHAT TO SAY. Default.
- LLMReaction: Full LLM decision (future, expensive).

The rule matrix is a list of tuples evaluated top-to-bottom; first match wins.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from askme.interfaces.reaction import ReactionBackend
from askme.schemas.events import ChangeEventType
from askme.schemas.reaction import ReactionDecision, ReactionType, SceneContext

if TYPE_CHECKING:
    from askme.llm.client import LLMClient
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.pipeline.alert_dispatcher import AlertDispatcher

logger = logging.getLogger(__name__)

# -- Rule matrix -- priority-ordered, first match wins ------------------------

_REACTION_RULES: list[
    tuple[str, Callable[[SceneContext], bool], ReactionType, dict[str, Any]]
] = [
    # ---- Safety / Security (highest priority) ----
    (
        "restricted_zone_person",
        lambda ctx: (
            ctx.event.is_person_event
            and ctx.event.event_type == ChangeEventType.PERSON_APPEARED
            and "restricted" in ctx.zone_tags
        ),
        ReactionType.WARN,
        {"severity": "warning", "template": "请注意，此区域需要授权进入。"},
    ),
    (
        "after_hours_unknown",
        lambda ctx: (
            ctx.event.is_person_event
            and ctx.event.event_type == ChangeEventType.PERSON_APPEARED
            and not ctx.is_business_hours
        ),
        ReactionType.ALERT,
        {
            "severity": "warning",
            "template": "非工作时间检测到人员，已通知管理人员。",
            "escalate": True,
        },
    ),
    # ---- Wake word (always respond) ----
    (
        "wake_word",
        lambda ctx: ctx.wake_word_heard,
        ReactionType.ACT,
        {"action": "enter_conversation"},
    ),
    # ---- Robot busy (suppress most reactions) ----
    (
        "busy_ignore",
        lambda ctx: (
            ctx.robot_busy
            and ctx.event.is_person_event
            and "restricted" not in ctx.zone_tags
        ),
        ReactionType.OBSERVE,
        {},
    ),
    # ---- Person passing through (IGNORE) ----
    (
        "person_passing",
        lambda ctx: (
            ctx.event.event_type == ChangeEventType.PERSON_APPEARED
            and ctx.person_duration_s < 3.0
            and not ctx.person_approaching
        ),
        ReactionType.IGNORE,
        {},
    ),
    # ---- Recently seen (suppress duplicate greetings) ----
    (
        "person_seen_recently",
        lambda ctx: (
            ctx.event.event_type == ChangeEventType.PERSON_APPEARED
            and ctx.seen_person_recently
            and ctx.minutes_since_last_person < 10.0
        ),
        ReactionType.OBSERVE,
        {},
    ),
    # ---- Person approaching and close (GREET) ----
    (
        "person_approaching_greet",
        lambda ctx: (
            ctx.event.event_type == ChangeEventType.PERSON_APPEARED
            and ctx.person_approaching
            and ctx.person_distance_m is not None
            and ctx.person_distance_m < 4.0
        ),
        ReactionType.GREET,
        {"use_llm": True},
    ),
    # ---- Person standing a long time (ASSIST) ----
    (
        "person_lingering",
        lambda ctx: (
            ctx.event.event_type != ChangeEventType.PERSON_LEFT
            and ctx.person_duration_s > 120.0
            and ctx.person_stationary
        ),
        ReactionType.ASSIST,
        {"use_llm": True, "template_fallback": "你好，需要帮助吗？"},
    ),
    # ---- Person appeared at entrance (GREET with template) ----
    (
        "entrance_greet",
        lambda ctx: (
            ctx.event.event_type == ChangeEventType.PERSON_APPEARED
            and "entrance" in ctx.zone_tags
        ),
        ReactionType.GREET,
        {"template": "你好，欢迎。"},
    ),
    # ---- Person appeared, generic (OBSERVE only) ----
    (
        "person_appeared_default",
        lambda ctx: ctx.event.event_type == ChangeEventType.PERSON_APPEARED,
        ReactionType.OBSERVE,
        {},
    ),
    # ---- Person left ----
    (
        "person_left",
        lambda ctx: ctx.event.event_type == ChangeEventType.PERSON_LEFT,
        ReactionType.IGNORE,
        {},
    ),
    # ---- Non-person events ----
    (
        "object_change_default",
        lambda ctx: True,
        ReactionType.OBSERVE,
        {},
    ),
]
# Content generation prompt template
_CONTENT_PROMPT = (
    "你是巡检机器人Thunder。"
    "根据以下场景信息，"
    "用一句简短的中文回应。\n"
    "不超过30字。语气友好但专业。\n\n"
    "场景: {scene_summary}\n"
    "时间: {time_str}\n"
    "地点: {zone_name}\n"
    "反应类型: {reaction_type}\n\n"
    "回应:"
)


def evaluate_rules(context: SceneContext) -> ReactionDecision:
    """Evaluate the rule matrix against a SceneContext. First match wins."""
    for rule_name, predicate, reaction_type, metadata in _REACTION_RULES:
        try:
            if predicate(context):
                return ReactionDecision(
                    rule_name=rule_name,
                    reaction_type=reaction_type,
                    metadata=dict(metadata),
                    context=context,
                )
        except Exception as exc:
            logger.warning("[ReactionEngine] Rule %r raised: %s", rule_name, exc)
            continue

    # Fallback -- should never reach here due to catch-all rule
    return ReactionDecision(
        rule_name="fallback",
        reaction_type=ReactionType.OBSERVE,
        metadata={},
        context=context,
    )


def _log_decision(episodic, decision: ReactionDecision, content: str) -> None:
    """Log a reaction decision to episodic memory."""
    msg = "反应决策: " + decision.rule_name + " -> " + decision.reaction_type.value
    if content:
        msg += " -> " + content
    episodic.log("reaction", msg)


def _dispatch(alert_dispatcher, decision: ReactionDecision, content: str) -> None:
    """Dispatch an alert via AlertDispatcher."""
    severity = decision.metadata.get("severity", "info")
    topic = "reaction." + decision.reaction_type.value
    alert_dispatcher.dispatch(
        content,
        severity=severity,
        topic=topic,
        payload=decision.to_dict(),
    )

# -- RuleBasedReaction -- pure rules, no LLM ----------------------------------


class RuleBasedReaction(ReactionBackend):
    """Pure rule-based reaction engine. <1ms decisions. Templates only, no LLM.

    Suitable for offline/low-resource deployments where LLM is unavailable.
    """

    def __init__(
        self,
        *,
        alert_dispatcher: AlertDispatcher | None = None,
        episodic: EpisodicMemory | None = None,
        **kwargs: Any,
    ) -> None:
        self._alert_dispatcher = alert_dispatcher
        self._episodic = episodic

    async def decide(self, context: SceneContext) -> ReactionDecision:
        return evaluate_rules(context)

    async def generate_content(
        self, decision: ReactionDecision, context: SceneContext
    ) -> str:
        """Use template from metadata. No LLM."""
        template = decision.metadata.get("template", "")
        if not template:
            template = decision.metadata.get("template_fallback", "")
        return template

    async def execute(self, decision: ReactionDecision, content: str) -> None:
        """Execute reaction via alert dispatcher."""
        if self._episodic:
            _log_decision(self._episodic, decision, content)

        if decision.reaction_type in (ReactionType.IGNORE, ReactionType.OBSERVE):
            return
        if not content:
            return
        if self._alert_dispatcher:
            _dispatch(self._alert_dispatcher, decision, content)


# -- HybridReaction -- rules decide, LLM speaks -------------------------------


class HybridReaction(ReactionBackend):
    """Hybrid reaction engine: rules decide WHAT, LLM generates WHAT TO SAY.

    Decision is always rule-based (<1ms). LLM is called only when the rule
    metadata includes use_llm=True (GREET/ASSIST). Falls back to
    template if LLM is unavailable or times out.

    This is the default backend.
    """

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        alert_dispatcher: AlertDispatcher | None = None,
        episodic: EpisodicMemory | None = None,
        content_model: str = "",
        content_timeout: float = 5.0,
        **kwargs: Any,
    ) -> None:
        self._llm = llm
        self._alert_dispatcher = alert_dispatcher
        self._episodic = episodic
        self._content_model = content_model
        self._content_timeout = content_timeout

    async def decide(self, context: SceneContext) -> ReactionDecision:
        return evaluate_rules(context)

    async def generate_content(
        self, decision: ReactionDecision, context: SceneContext
    ) -> str:
        """Generate content: LLM for use_llm rules, template otherwise."""
        if decision.reaction_type in (ReactionType.IGNORE, ReactionType.OBSERVE):
            return ""

        template = decision.metadata.get("template", "")
        template_fallback = decision.metadata.get("template_fallback", "")
        use_llm = decision.metadata.get("use_llm", False)

        # If template is set and no LLM requested, use template directly
        if template and not use_llm:
            return template

        # Try LLM content generation
        if use_llm and self._llm:
            try:
                import datetime

                now = datetime.datetime.now()
                time_str = now.strftime("%H:%M")
                zone = context.zone_name or "未知区域"

                prompt = _CONTENT_PROMPT.format(
                    scene_summary=context.event.description_zh(),
                    time_str=time_str,
                    zone_name=zone,
                    reaction_type=decision.reaction_type.value,
                )

                chat_kwargs: dict[str, Any] = {"temperature": 0.7}
                if self._content_model:
                    chat_kwargs["model"] = self._content_model

                result = await asyncio.wait_for(
                    self._llm.chat(
                        [{"role": "user", "content": prompt}],
                        **chat_kwargs,
                    ),
                    timeout=self._content_timeout,
                )
                result = result.strip()
                if result:
                    return result
            except TimeoutError:
                logger.debug("[HybridReaction] LLM content gen timed out")
            except Exception as exc:
                logger.debug("[HybridReaction] LLM content gen failed: %s", exc)

        # Fallback to template
        return template or template_fallback

    async def execute(self, decision: ReactionDecision, content: str) -> None:
        """Execute reaction via alert dispatcher."""
        if self._episodic:
            _log_decision(self._episodic, decision, content)

        if decision.reaction_type in (ReactionType.IGNORE, ReactionType.OBSERVE):
            return
        if not content:
            return
        if self._alert_dispatcher:
            _dispatch(self._alert_dispatcher, decision, content)

# -- LLMReaction -- full LLM decision (future) --------------------------------


class LLMReaction(ReactionBackend):
    """Full LLM-driven reaction engine. Expensive (~3s per event).

    Sends the full SceneContext to the LLM and lets it decide both the
    reaction type and spoken content. Reserved for future high-compute
    deployments. Falls back to rule-based if LLM is unavailable.
    """

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        alert_dispatcher: AlertDispatcher | None = None,
        episodic: EpisodicMemory | None = None,
        decision_model: str = "",
        decision_timeout: float = 8.0,
        **kwargs: Any,
    ) -> None:
        self._llm = llm
        self._alert_dispatcher = alert_dispatcher
        self._episodic = episodic
        self._decision_model = decision_model
        self._decision_timeout = decision_timeout
        self._rule_fallback = RuleBasedReaction(
            alert_dispatcher=alert_dispatcher,
            episodic=episodic,
        )

    async def decide(self, context: SceneContext) -> ReactionDecision:
        """Use LLM for decision, fall back to rules."""
        if not self._llm:
            return await self._rule_fallback.decide(context)
        # For now, delegate to rules. Full LLM decision is future work.
        return evaluate_rules(context)

    async def generate_content(
        self, decision: ReactionDecision, context: SceneContext
    ) -> str:
        """Always use LLM for content generation."""
        if decision.reaction_type in (ReactionType.IGNORE, ReactionType.OBSERVE):
            return ""

        template = decision.metadata.get("template", "")
        if not self._llm:
            return template or decision.metadata.get("template_fallback", "")

        try:
            import datetime

            now = datetime.datetime.now()
            prompt = _CONTENT_PROMPT.format(
                scene_summary=context.event.description_zh(),
                time_str=now.strftime("%H:%M"),
                zone_name=context.zone_name or "未知区域",
                reaction_type=decision.reaction_type.value,
            )

            chat_kwargs: dict[str, Any] = {"temperature": 0.7}
            if self._decision_model:
                chat_kwargs["model"] = self._decision_model

            result = await asyncio.wait_for(
                self._llm.chat(
                    [{"role": "user", "content": prompt}],
                    **chat_kwargs,
                ),
                timeout=self._decision_timeout,
            )
            return result.strip() or template
        except Exception as exc:
            logger.debug("[LLMReaction] Content gen failed: %s", exc)
            return template or decision.metadata.get("template_fallback", "")

    async def execute(self, decision: ReactionDecision, content: str) -> None:
        """Execute reaction via alert dispatcher."""
        if self._episodic:
            _log_decision(self._episodic, decision, content)

        if decision.reaction_type in (ReactionType.IGNORE, ReactionType.OBSERVE):
            return
        if not content:
            return
        if self._alert_dispatcher:
            _dispatch(self._alert_dispatcher, decision, content)

