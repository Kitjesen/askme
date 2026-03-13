"""Category 3: Fast / Slow Channel Routing Tests

Fast path: VOICE_TRIGGER → skill executes directly, no LLM
Slow path: GENERAL     → goes through LLM (pipeline.process)

This matters for:
  - Response latency (fast = <500ms, slow = 3-7s)
  - Cost (LLM tokens consumed only on slow path)
  - Accuracy (fast path is deterministic; slow path is probabilistic)

What we verify:
  - Which inputs land on fast vs slow path
  - That question phrasing always goes slow
  - That negation suppression sends to slow
  - That explicit trigger commands go fast
  - That commands with missing slots still go through fast (then clarify)
"""

from __future__ import annotations

import pytest

from askme.brain.intent_router import IntentRouter, IntentType


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _router_with_nav_and_time():
    return IntentRouter(
        voice_triggers={
            "导航到仓库": "navigate",
            "导航": "navigate",
            "几点了": "get_time",
            "现在几点": "get_time",
            "建图": "mapping",
            "搜索": "web_search",
        }
    )


# ── Fast path: VOICE_TRIGGER ──────────────────────────────────────────────────


class TestFastPath:
    """Commands that should skip LLM entirely."""

    def test_exact_trigger_is_fast(self):
        r = _router_with_nav_and_time()
        assert r.route("几点了").type == IntentType.VOICE_TRIGGER

    def test_trigger_with_payload_is_fast(self):
        r = _router_with_nav_and_time()
        assert r.route("帮我导航到仓库").type == IntentType.VOICE_TRIGGER

    def test_bare_navigate_trigger_is_fast(self):
        """Even without destination, VOICE_TRIGGER fires (clarification handles slot)."""
        r = _router_with_nav_and_time()
        intent = r.route("导航")
        assert intent.type == IntentType.VOICE_TRIGGER, (
            "Missing slot does NOT prevent fast-path routing. "
            "Clarification runs AFTER intent routing, not before."
        )

    def test_mapping_trigger_is_fast(self):
        r = _router_with_nav_and_time()
        assert r.route("建图").type == IntentType.VOICE_TRIGGER

    def test_trigger_in_longer_sentence_is_fast(self):
        r = _router_with_nav_and_time()
        intent = r.route("麻烦帮我几点了")
        assert intent.type == IntentType.VOICE_TRIGGER


# ── Slow path: GENERAL → LLM ─────────────────────────────────────────────────


class TestSlowPath:
    """Inputs that MUST go through LLM, never direct skill execution."""

    def test_open_ended_question_is_slow(self):
        r = _router_with_nav_and_time()
        intent = r.route("你好，今天天气怎么样")
        assert intent.type == IntentType.GENERAL

    def test_question_phrasing_forces_slow(self):
        """'导航会失败吗' ends with 吗 → question → must use LLM."""
        r = _router_with_nav_and_time()
        intent = r.route("导航会失败吗")
        assert intent.type == IntentType.GENERAL, (
            "Question about navigation should NOT execute navigate. "
            "This would be a dangerous 'confident wrong' classification."
        )

    def test_question_mark_forces_slow(self):
        r = _router_with_nav_and_time()
        assert r.route("导航到仓库？").type == IntentType.GENERAL
        assert r.route("导航到仓库?").type == IntentType.GENERAL

    def test_question_particle_me_forces_slow(self):
        r = _router_with_nav_and_time()
        assert r.route("导航有没有用么").type == IntentType.GENERAL

    def test_question_particle_ne_forces_slow(self):
        r = _router_with_nav_and_time()
        assert r.route("你能帮我导航呢").type == IntentType.GENERAL


# ── Negation suppression always goes slow ────────────────────────────────────


class TestNegationGoesToSlow:
    """'不要导航' must NOT execute navigate — goes to GENERAL for LLM handling."""

    def test_bu_yao_suppresses_trigger(self):
        r = _router_with_nav_and_time()
        assert r.route("不要导航").type == IntentType.GENERAL

    def test_bie_suppresses_trigger(self):
        r = _router_with_nav_and_time()
        assert r.route("别导航了").type == IntentType.GENERAL

    def test_mei_you_suppresses_trigger(self):
        r = _router_with_nav_and_time()
        assert r.route("没有导航到仓库").type == IntentType.GENERAL

    def test_positive_still_fast(self):
        r = _router_with_nav_and_time()
        assert r.route("导航到仓库").type == IntentType.VOICE_TRIGGER


# ── Channel correctness: which path each command type takes ──────────────────


class TestChannelMatrix:
    """
    Document the fast/slow decision for common command patterns.

    This test serves as living documentation of routing behavior.
    A regression here means user experience changed significantly.
    """

    ROUTER = IntentRouter(
        voice_triggers={
            "导航到仓库":      "navigate",
            "建图":            "mapping",
            "搜索":            "web_search",
            "几点了":          "get_time",
            "停下来":          "stop_speaking",
        }
    )

    # Each entry: (utterance, expected_fast, reason)
    CASES = [
        ("导航到仓库",       True,  "exact trigger"),
        ("帮我导航到仓库",   True,  "embedded trigger"),
        ("建图",             True,  "exact trigger"),
        ("搜索一下",         True,  "trigger '搜索' found as substring in '搜索一下'"),
        ("停下来",           True,  "exact trigger"),
        ("不要停下来",       False, "negated"),
        ("停下来吗",         False, "question particle"),
        ("导航到仓库吗",     False, "question particle"),
        ("今天天气好吗",     False, "completely general"),
        ("帮我查天气",       False, "no matching trigger"),
    ]

    @pytest.mark.parametrize("utterance,expected_fast,reason", CASES)
    def test_routing_matrix(self, utterance: str, expected_fast: bool, reason: str):
        intent = self.ROUTER.route(utterance)
        actual_fast = intent.type == IntentType.VOICE_TRIGGER
        assert actual_fast == expected_fast, (
            f"Utterance={utterance!r}: expected fast={expected_fast} ({reason}), "
            f"got intent.type={intent.type}"
        )


# ── Skill name correctness on fast path ──────────────────────────────────────


class TestFastPathSkillBinding:
    """On fast path, the correct skill must be bound."""

    def test_navigate_skill_bound(self):
        r = IntentRouter(voice_triggers={"导航到仓库": "navigate", "去": "navigate_short"})
        intent = r.route("导航到仓库B")
        assert intent.skill_name == "navigate"

    def test_longest_match_wins(self):
        """'导航到仓库' (6 chars) beats '导航' (2 chars)."""
        r = IntentRouter(voice_triggers={
            "导航": "navigate_generic",
            "导航到仓库": "navigate_warehouse",
        })
        intent = r.route("导航到仓库取货")
        assert intent.skill_name == "navigate_warehouse", (
            "Longest-match must win to prevent '导航' short-match from "
            "incorrectly firing for more specific commands."
        )
