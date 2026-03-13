"""Category: Interrupt → Reroute (interrupt_payload)

Real risks covered:
  - User says "算了，改去仓库B" mid-slot-collection → old code returned
    proceed=False with no payload → VoiceLoop drops the new intent entirely
    → user must repeat the whole command from scratch (bad UX)
  - "不了解" starts with "不了" but must NOT be treated as an interrupt
    (no separator after prefix → false-positive protection)
  - "算法问题" contains "算" but is not an interrupt prefix → no false positive
  - Empty payload after separator ("算了，") → interrupt detected, payload=""
    (caller treats same as pure bail-out — documented design decision)
  - "先不了，去仓库" — "先不了" is in _INTERRUPT_SIGNALS, checked before "先不",
    separator found → payload correctly extracted (longest-signal-first fix)
  - ConfirmationAgent: "算了，手动操作" during confirm → interrupt_payload="手动操作"
  - Caller (VoiceLoop) reroute pattern: r1.interrupt_payload → dispatch new intent

Status BEFORE interrupt_payload was added:
  - parse_interrupt_payload: function did not exist (AttributeError)
  - interrupt_payload field: ProactiveResult had no such field
  - "先不了，去仓库" silently returned (False, "") — reroute impossible
Status AFTER implementation: ALL PASS.
"""

from __future__ import annotations

import pytest

from askme.pipeline.proactive.clarification_agent import (
    ClarificationPlannerAgent,
    parse_interrupt_payload,
)
from askme.pipeline.proactive.confirm_agent import ConfirmationAgent
from askme.pipeline.proactive.base import ProactiveContext
from askme.pipeline.proactive.orchestrator import ProactiveOrchestrator
from askme.pipeline.proactive.session_state import ClarificationSession, ClarificationState
from askme.skills.skill_model import SkillDefinition, SlotSpec
from unittest.mock import MagicMock


# ── Helpers ────────────────────────────────────────────────────────────────────

class MockAudio:
    def __init__(self, answers=None):
        self.spoken: list[str] = []
        self._answers = list(answers or [])
        self._idx = 0
    def drain_buffers(self): ...
    def speak(self, t): self.spoken.append(t)
    def start_playback(self): ...
    def stop_playback(self): ...
    def wait_speaking_done(self): ...
    def listen_loop(self):
        if self._idx < len(self._answers):
            r = self._answers[self._idx]; self._idx += 1; return r
        return None


def _nav_skill():
    return SkillDefinition(
        name="navigate", voice_trigger="去,导航",
        # pipeline=None: location slot uses trigger-stripping
        required_slots=[SlotSpec(name="destination", type="location", prompt="导航去哪里？")],
    )


def _search_skill():
    return SkillDefinition(
        name="web_search", voice_trigger="搜索",
        required_slots=[SlotSpec(name="query", type="text", prompt="搜什么？")],
    )


def _ctx():
    return ProactiveContext(session=ClarificationSession())


def _make_orch(extra_skills=None):
    """Build orchestrator with pipeline=None so no MagicMock side-effects in tests."""
    sk_nav = _nav_skill()
    sk_search = _search_skill()
    skills = {"navigate": sk_nav, "web_search": sk_search}
    if extra_skills:
        skills.update(extra_skills)

    dispatcher = MagicMock()
    dispatcher.current_mission = None
    dispatcher.get_skill.side_effect = lambda name: skills.get(name)

    return ProactiveOrchestrator.default(pipeline=None, dispatcher=dispatcher)


# ── parse_interrupt_payload: unit tests ───────────────────────────────────────

class TestParseInterruptPayload:
    """parse_interrupt_payload() is the core parsing function."""

    # --- pure bail-out words (exact match) ---

    def test_suan_le_no_payload(self):
        is_int, payload = parse_interrupt_payload("算了")
        assert is_int is True
        assert payload == ""

    def test_bu_le_no_payload(self):
        is_int, payload = parse_interrupt_payload("不了")
        assert is_int is True
        assert payload == ""

    def test_suan_le_ba_no_payload(self):
        is_int, payload = parse_interrupt_payload("算了吧")
        assert is_int is True
        assert payload == ""

    def test_bu_xiang_le_no_payload(self):
        is_int, payload = parse_interrupt_payload("不想了")
        assert is_int is True
        assert payload == ""

    def test_xian_bu_le_no_payload(self):
        is_int, payload = parse_interrupt_payload("先不了")
        assert is_int is True
        assert payload == ""

    def test_whitespace_padded_suan_le(self):
        """Leading/trailing whitespace must be stripped before matching."""
        is_int, payload = parse_interrupt_payload("  算了  ")
        assert is_int is True
        assert payload == ""

    def test_whitespace_padded_with_payload(self):
        """Whitespace around "算了，去仓库B" must not prevent detection."""
        is_int, payload = parse_interrupt_payload("  算了，去仓库B  ")
        assert is_int is True
        assert payload == "去仓库B"

    def test_none_is_not_interrupt(self):
        is_int, payload = parse_interrupt_payload(None)
        assert is_int is False
        assert payload == ""

    def test_empty_string_is_not_interrupt(self):
        is_int, payload = parse_interrupt_payload("")
        assert is_int is False
        assert payload == ""

    # --- interrupt with reroute payload ---

    def test_suan_le_comma_nav(self):
        """'算了，去仓库B' → interrupt with payload '去仓库B'."""
        is_int, payload = parse_interrupt_payload("算了，去仓库B")
        assert is_int is True
        assert payload == "去仓库B", (
            f"Expected '去仓库B', got {payload!r}. "
            "The reroute intent must be extracted after the separator."
        )

    def test_bu_le_comma_nav(self):
        is_int, payload = parse_interrupt_payload("不了，导航去工厂")
        assert is_int is True
        assert payload == "导航去工厂"

    def test_suan_le_ascii_comma(self):
        """ASCII comma also works as separator."""
        is_int, payload = parse_interrupt_payload("算了,改搜北京天气")
        assert is_int is True
        assert payload == "改搜北京天气"

    def test_suan_le_space_separator(self):
        """Space as separator."""
        is_int, payload = parse_interrupt_payload("算了 去仓库A")
        assert is_int is True
        assert payload == "去仓库A"

    def test_empty_payload_after_separator(self):
        """'算了，' — separator present but payload is empty.

        Design decision: treat same as pure bail-out (interrupt_payload="").
        The caller cannot distinguish "算了" from "算了，" — both mean
        'give up, no new intent'. Documented here as the canonical case.
        """
        is_int, payload = parse_interrupt_payload("算了，")
        assert is_int is True
        assert payload == "", (
            "Empty payload after separator must be treated as pure bail-out. "
            "Caller should not attempt reroute when interrupt_payload is ''."
        )

    def test_xian_bu_le_comma_nav(self):
        """'先不了，去仓库' — longest-signal-first fix.

        OLD behaviour (iterate _BAILOUT_PREFIXES with '先不' prefix):
          - '先不了，去仓库'.startswith('先不') → True
          - rest = '了，去仓库', rest[0] = '了' NOT a separator → (False, '')  BUG

        NEW behaviour (iterate _INTERRUPT_SIGNALS sorted longest-first):
          - '先不了' in signals, '先不了，去仓库'.startswith('先不了') → True
          - rest = '，去仓库', rest[0] = '，' IS a separator → (True, '去仓库')  FIXED
        """
        is_int, payload = parse_interrupt_payload("先不了，去仓库")
        assert is_int is True, (
            "'先不了，去仓库' must be detected as interrupt+payload. "
            "Check that _INTERRUPT_SIGNALS are sorted longest-first."
        )
        assert payload == "去仓库"

    def test_xian_bu_prefix_with_payload(self):
        """'先不，去看看仓库' — shorter '先不' prefix works when no longer match."""
        is_int, payload = parse_interrupt_payload("先不，去看看仓库")
        assert is_int is True
        assert payload == "去看看仓库"

    def test_multi_part_payload(self):
        """Payload containing commas is returned verbatim."""
        is_int, payload = parse_interrupt_payload("算了，先去仓库A，再去仓库B")
        assert is_int is True
        assert payload == "先去仓库A，再去仓库B"

    # --- false-positive protection ---

    def test_bu_le_jie_not_interrupt(self):
        """'不了解' — '不了' prefix but next char '解' is not a separator → NOT interrupt.

        Before this protection: '不了' prefix would match → interrupt detected → BUG.
        After: '解' not in _SEPARATORS → not interrupt → correct.
        """
        is_int, payload = parse_interrupt_payload("不了解")
        assert is_int is False, (
            "'不了解' must NOT be treated as an interrupt. "
            "The char after '不了' ('解') is not a separator."
        )

    def test_suan_fa_wenti_not_interrupt(self):
        """'算法问题' contains '算' but is not '算了' → not interrupt."""
        is_int, payload = parse_interrupt_payload("算法问题")
        assert is_int is False

    def test_long_answer_not_interrupt(self):
        is_int, payload = parse_interrupt_payload("不了解这个技术细节")
        assert is_int is False

    def test_regular_answer_not_interrupt(self):
        is_int, payload = parse_interrupt_payload("北京天气预报")
        assert is_int is False
        assert payload == ""

    def test_double_interrupt_payload_is_interrupt_word(self):
        """'算了，算了' — payload itself is an interrupt word.

        Caller should check r.interrupt_payload in _INTERRUPT_SIGNALS and
        not reroute (pure bail-out). This test documents the expected value.
        """
        is_int, payload = parse_interrupt_payload("算了，算了")
        assert is_int is True
        assert payload == "算了"  # raw payload — caller decides what to do with it


# ── Integration: interrupt_payload flows through agent ────────────────────────

class TestInterruptPayloadInAgent:
    """The ProactiveResult must carry interrupt_payload from the agent."""

    async def test_suan_le_with_nav_returns_payload(self):
        """'算了，去仓库B' → result.interrupt_payload == '去仓库B'."""
        agent = ClarificationPlannerAgent()
        sk = _search_skill()
        audio = MockAudio(["算了，去仓库B"])
        result = await agent.interact(sk, "搜索", audio, _ctx())

        assert result.proceed is False
        assert result.cancelled_by == "interrupted"
        assert result.interrupt_payload == "去仓库B", (
            f"Expected '去仓库B' in interrupt_payload, got {result.interrupt_payload!r}. "
            "The reroute text must survive from parse_interrupt_payload to the result."
        )

    async def test_pure_suan_le_payload_is_empty(self):
        audio = MockAudio(["算了"])
        result = await ClarificationPlannerAgent().interact(
            _search_skill(), "搜索", audio, _ctx()
        )
        assert result.proceed is False
        assert result.interrupt_payload == ""

    async def test_bu_le_with_new_intent(self):
        audio = MockAudio(["不了，搜北京天气"])
        result = await ClarificationPlannerAgent().interact(
            _nav_skill(), "去那里", audio, _ctx()
        )
        assert result.proceed is False
        assert result.interrupt_payload == "搜北京天气"

    async def test_xian_bu_le_with_payload(self):
        """'先不了，去仓库' correctly extracted via longest-signal-first."""
        audio = MockAudio(["先不了，去仓库"])
        result = await ClarificationPlannerAgent().interact(
            _search_skill(), "搜索", audio, _ctx()
        )
        assert result.proceed is False
        assert result.interrupt_payload == "去仓库"

    async def test_no_interrupt_payload_on_normal_proceed(self):
        audio = MockAudio(["仓库A"])
        result = await ClarificationPlannerAgent().interact(
            _nav_skill(), "去那里", audio, _ctx()
        )
        assert result.proceed is True
        assert result.interrupt_payload == ""

    async def test_estop_has_no_interrupt_payload(self):
        audio = MockAudio(["急停"])
        result = await ClarificationPlannerAgent().interact(
            _search_skill(), "搜索", audio, _ctx()
        )
        assert result.proceed is False
        assert result.cancelled_by == "estop"
        assert result.interrupt_payload == ""


# ── ConfirmationAgent interrupt_payload ───────────────────────────────────────

class TestConfirmationAgentInterruptPayload:
    """ConfirmationAgent must also support interrupt+payload during confirm stage."""

    async def test_suan_le_with_payload_during_confirm(self):
        """'算了，手动操作' during confirm → proceed=False, payload='手动操作'."""
        sk = SkillDefinition(
            name="robot_grab", description="机械臂抓取",
            voice_trigger="抓取", confirm_before_execute=True,
        )
        audio = MockAudio(["算了，手动操作"])
        result = await ConfirmationAgent().interact(sk, "抓取红色瓶子", audio, _ctx())

        assert result.proceed is False
        assert result.cancelled_by == "interrupted"
        assert result.interrupt_payload == "手动操作", (
            f"Expected payload '手动操作', got {result.interrupt_payload!r}. "
            "ConfirmationAgent must now support interrupt+payload."
        )

    async def test_pure_suan_le_during_confirm_no_payload(self):
        sk = SkillDefinition(
            name="robot_grab", description="抓取",
            voice_trigger="抓取", confirm_before_execute=True,
        )
        audio = MockAudio(["算了"])
        result = await ConfirmationAgent().interact(sk, "抓取红色瓶子", audio, _ctx())
        assert result.proceed is False
        assert result.cancelled_by == "interrupted"
        assert result.interrupt_payload == ""

    async def test_estop_during_confirm_no_payload(self):
        sk = SkillDefinition(
            name="robot_grab", description="抓取",
            voice_trigger="抓取", confirm_before_execute=True,
        )
        audio = MockAudio(["急停"])
        result = await ConfirmationAgent().interact(sk, "抓取红色瓶子", audio, _ctx())
        assert result.proceed is False
        assert result.cancelled_by == "estop"
        assert result.interrupt_payload == ""

    async def test_interrupt_before_yes_no_check(self):
        """'算了' at confirm stage must NOT fall through to '取消' detection.

        ConfirmationAgent checks interrupt BEFORE yes/no words.  '算了' appears
        in _NO_WORDS (cancel vocabulary) but must route to 'interrupted', not
        '用户取消', because parse_interrupt_payload() runs first in the code path.
        Also verifies the session transitions to INTERRUPTED, not CANCELED.
        """
        sk = SkillDefinition(
            name="robot_grab", description="抓取",
            voice_trigger="抓取", confirm_before_execute=True,
        )
        ctx = _ctx()  # capture to assert session state after the call
        audio = MockAudio(["算了"])
        result = await ConfirmationAgent().interact(sk, "抓取红色瓶子", audio, ctx)
        assert result.cancelled_by == "interrupted", (
            f"'算了' during confirm must be 'interrupted', not '用户取消'. "
            f"Got: {result.cancelled_by!r}"
        )
        assert ctx.session.state == ClarificationState.INTERRUPTED, (
            f"session.state must be INTERRUPTED after bail-out interrupt, "
            f"got {ctx.session.state!r}. CANCELED is reserved for ESTOP/non-recoverable stops."
        )


# ── End-to-end through orchestrator ──────────────────────────────────────────

class TestOrchestratorInterruptPayload:
    """interrupt_payload must flow correctly through the full orchestrator chain."""

    async def test_orchestrator_returns_interrupt_payload(self):
        """Orchestrator.run() must pass agent's interrupt_payload up unchanged."""
        orch = _make_orch()
        r = await orch.run("navigate", "去那里", MockAudio(["算了，搜仓库地图"]))
        assert r.proceed is False
        assert r.interrupt_payload == "搜仓库地图", (
            f"Orchestrator must not drop the interrupt_payload from the agent. "
            f"Got: {r.interrupt_payload!r}"
        )

    async def test_proceed_result_has_empty_payload(self):
        """Normal proceed must return interrupt_payload=''."""
        orch = _make_orch()
        r = await orch.run("web_search", "搜索", MockAudio(["北京天气"]))
        assert r.proceed is True
        assert r.interrupt_payload == ""


# ── Caller reroute pattern ─────────────────────────────────────────────────────

class TestCallerReroutePattern:
    """Simulate VoiceLoop: on interrupt + payload, dispatch the new intent."""

    async def test_reroute_via_orchestrator(self):
        """Run 1: interrupt with payload → Run 2: dispatch the payload intent."""
        orch = _make_orch()

        r1 = await orch.run("navigate", "去那里", MockAudio(["算了，搜仓库地图"]))
        assert r1.proceed is False
        assert r1.interrupt_payload == "搜仓库地图"

        # Simulate VoiceLoop: non-empty payload → reroute
        assert r1.interrupt_payload, "Caller must detect non-empty payload and reroute"

        r2 = await orch.run("web_search", r1.interrupt_payload, MockAudio([]))
        assert r2.proceed is True, (
            "The rerouted intent '搜仓库地图' should proceed — query slot is filled."
        )

    async def test_no_payload_caller_does_not_reroute(self):
        orch = _make_orch()
        r = await orch.run("web_search", "搜索", MockAudio(["算了"]))
        assert r.proceed is False
        assert r.interrupt_payload == ""

    async def test_three_independent_reroutes(self):
        """Three separate interrupt+reroute pairs all produce independent payloads.

        Regression test for the longest-signal-first bug:
          OLD: "先不了，去仓库C" matched "先不" prefix → rest="了，去仓库C",
               '了' not a separator → (False, "")  — reroute silently lost.
          NEW: "先不了" is checked first → rest="，去仓库C",
               '，' IS a separator → (True, "去仓库C")  — payload preserved.

        All three payloads must be distinct with no cross-contamination.
        """
        orch = _make_orch()

        r1 = await orch.run("navigate", "去那里", MockAudio(["算了，去仓库A"]))
        r2 = await orch.run("navigate", "去那里", MockAudio(["不了，去仓库B"]))
        r3 = await orch.run("navigate", "去那里", MockAudio(["先不了，去仓库C"]))

        assert r1.interrupt_payload == "去仓库A"
        assert r2.interrupt_payload == "去仓库B"
        assert r3.interrupt_payload == "去仓库C"

        payloads = {r1.interrupt_payload, r2.interrupt_payload, r3.interrupt_payload}
        assert len(payloads) == 3, "All three reroute payloads must be distinct"

    async def test_double_interrupt_payload_not_auto_rerouted(self):
        """Payload '算了' is itself an interrupt word — caller must not blindly reroute.

        The payload being '算了' means the user said "算了，算了" — they were
        emphatic about bailing out, not issuing a new command. The caller
        (VoiceLoop) should check: if parse_interrupt_payload(payload) returns True
        → payload is also an interrupt signal → do not reroute.

        Note: We test the caller pattern using the public parse_interrupt_payload()
        API, not the private _INTERRUPT_SIGNALS set. This keeps the test insulated
        from internal naming changes.
        """
        orch = _make_orch()
        r = await orch.run("web_search", "搜索", MockAudio(["算了，算了"]))
        assert r.proceed is False
        assert r.interrupt_payload == "算了"
        # Caller guard: payload is itself an interrupt → do not reroute
        is_also_int, _ = parse_interrupt_payload(r.interrupt_payload)
        assert is_also_int, (
            "Caller can detect double-interrupt: parse_interrupt_payload(payload) → "
            "(True, '') → no reroute. Got (False, ...) — payload was not recognised "
            "as an interrupt word."
        )


# ── ConfirmationAgent: interrupt takes priority over YES/NO detection ──────────


class TestConfirmInterruptPriorityOverYes:
    """ConfirmationAgent must run interrupt detection BEFORE yes/no word matching.

    '算了' appears in _NO_WORDS (the cancel vocabulary) AND in _INTERRUPT_SIGNALS.
    If yes/no matching ran first, the result would be cancelled_by="用户取消".
    If interrupt matching runs first (correct), the result is cancelled_by="interrupted".

    This test locks in the priority ordering so a future refactor can't accidentally
    swap the check sequence without a test failure.
    """

    async def test_suan_le_is_interrupted_not_yonghu_quxiao(self):
        """'算了' → interrupted (interrupt check precedes NO-word check)."""
        sk = SkillDefinition(
            name="robot_grab", description="抓取",
            voice_trigger="抓取", confirm_before_execute=True,
        )
        result = await ConfirmationAgent().interact(
            sk, "抓取红色瓶子", MockAudio(["算了"]), _ctx()
        )
        assert result.cancelled_by == "interrupted", (
            f"'算了' is in both _NO_WORDS and _INTERRUPT_SIGNALS. "
            f"Interrupt detection must run first → 'interrupted', not '用户取消'. "
            f"Got: {result.cancelled_by!r}"
        )

    async def test_bu_le_is_interrupted_not_yonghu_quxiao(self):
        """'不了' → interrupted (also in NO-word vocabulary, interrupt wins)."""
        sk = SkillDefinition(
            name="robot_grab", description="抓取",
            voice_trigger="抓取", confirm_before_execute=True,
        )
        result = await ConfirmationAgent().interact(
            sk, "抓取红色瓶子", MockAudio(["不了"]), _ctx()
        )
        assert result.cancelled_by == "interrupted", (
            f"'不了' must → 'interrupted' (interrupt wins over '用户取消'). "
            f"Got: {result.cancelled_by!r}"
        )

    async def test_bu_interrupt_with_payload_has_correct_fields(self):
        """'不了，手动操作' → interrupted + payload='手动操作' (not '用户取消')."""
        sk = SkillDefinition(
            name="robot_grab", description="抓取",
            voice_trigger="抓取", confirm_before_execute=True,
        )
        result = await ConfirmationAgent().interact(
            sk, "抓取红色瓶子", MockAudio(["不了，手动操作"]), _ctx()
        )
        assert result.cancelled_by == "interrupted"
        assert result.interrupt_payload == "手动操作"


# ── _INTERRUPT_SIGNALS_BY_LEN sort-order invariant ────────────────────────────


class TestInterruptSignalsSortOrder:
    """_INTERRUPT_SIGNALS_BY_LEN must be sorted longest-first.

    This is an implementation invariant, not just a preference: if a shorter
    signal appears before a longer one that shares its prefix (e.g. "先不" before
    "先不了"), the shorter one will match first and consume characters that belong
    to the longer signal, producing wrong results.

    This test locks the sort order so it can never regress silently.
    """

    def test_interrupt_signals_sorted_longest_first(self):
        """Every element must be >= the element that follows it (by length)."""
        from askme.pipeline.proactive.clarification_agent import (
            _INTERRUPT_SIGNALS,
            _INTERRUPT_SIGNALS_BY_LEN,
        )
        # Same set of signals — no missing or extra entries
        assert set(_INTERRUPT_SIGNALS_BY_LEN) == set(_INTERRUPT_SIGNALS), (
            "_INTERRUPT_SIGNALS_BY_LEN must contain exactly the same signals as "
            "_INTERRUPT_SIGNALS. A signal was added to one but not regenerated in the other."
        )
        # Each element >= next (non-increasing length)
        for i in range(len(_INTERRUPT_SIGNALS_BY_LEN) - 1):
            a = _INTERRUPT_SIGNALS_BY_LEN[i]
            b = _INTERRUPT_SIGNALS_BY_LEN[i + 1]
            assert len(a) >= len(b), (
                f"Sort order violated at index {i}: "
                f"{a!r} (len={len(a)}) < {b!r} (len={len(b)}). "
                "Shorter signal before longer one will cause prefix-match bugs."
            )
