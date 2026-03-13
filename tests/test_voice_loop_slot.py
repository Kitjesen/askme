"""Tests for VoiceLoop._slot_present() — trigger-based slot detection."""

import pytest

from askme.pipeline.voice_loop import VoiceLoop
from askme.skills.skill_model import SkillDefinition


# ── minimal VoiceLoop stub ────────────────────────────────────────────────────

class _FakePipeline:
    def extract_semantic_target(self, text: str) -> str:
        """Minimal replica of the real extract_semantic_target regex logic."""
        import re
        patterns = [
            r"导航到(.{1,20}?)(?:去|了|吧|啊|呢|$)",
            r"带我去(.{1,20}?)(?:去|了|吧|啊|呢|$)",
            r"前往(.{1,20}?)(?:去|了|吧|啊|呢|$)",
            r"去(.{1,20}?)(?:吧|了|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m and m.group(1).strip():
                return m.group(1).strip()
        return text


def _make_loop() -> VoiceLoop:
    """Build a VoiceLoop with the minimum deps needed to call _slot_present()."""
    loop = VoiceLoop.__new__(VoiceLoop)
    loop._pipeline = _FakePipeline()  # type: ignore[attr-defined]
    return loop


def _skill(name: str, *, voice_trigger: str = "", required_prompt: str = "问你什么？") -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description="test",
        voice_trigger=voice_trigger or None,
        required_prompt=required_prompt,
    )


LOOP = _make_loop()


# ── navigate (semantic extraction) ───────────────────────────────────────────

class TestNavigateSlot:
    NAV = _skill("navigate", voice_trigger="导航到,带我去,前往,去")

    def test_bare_trigger_missing(self):
        assert not LOOP._slot_present(self.NAV, "导航到")

    def test_destination_present(self):
        assert LOOP._slot_present(self.NAV, "去厨房")

    def test_destination_present_long(self):
        assert LOOP._slot_present(self.NAV, "导航到仓库B")

    def test_带我去_no_target(self):
        # "带我去" without anything after — no slot
        assert not LOOP._slot_present(self.NAV, "带我去")

    def test_带我去_with_target(self):
        assert LOOP._slot_present(self.NAV, "带我去会议室")


# ── web_search (trigger-based generic) ───────────────────────────────────────

class TestWebSearchSlot:
    WS = _skill(
        "web_search",
        voice_trigger="帮我搜索,搜索一下,搜一下,查一下,查询一下,上网查",
    )

    def test_bare_搜索一下_missing(self):
        """'搜索一下' is exactly the trigger — no query content, slot missing."""
        assert not LOOP._slot_present(self.WS, "搜索一下")

    def test_搜索一下_with_query(self):
        assert LOOP._slot_present(self.WS, "搜索一下明天天气")

    def test_查一下_missing(self):
        assert not LOOP._slot_present(self.WS, "查一下")

    def test_查一下_with_topic(self):
        assert LOOP._slot_present(self.WS, "查一下最新新闻")

    def test_帮我搜索_missing(self):
        assert not LOOP._slot_present(self.WS, "帮我搜索")

    def test_帮我搜索_with_query(self):
        assert LOOP._slot_present(self.WS, "帮我搜索穹沛科技")

    def test_longest_trigger_wins(self):
        """'帮我搜索' (4 chars) beats '搜索' (2 chars) — remainder correctly computed."""
        assert not LOOP._slot_present(self.WS, "帮我搜索")

    def test_one_char_remainder_missing(self):
        """A single trailing char after trigger is not enough."""
        assert not LOOP._slot_present(self.WS, "搜一下的")

    def test_two_char_remainder_present(self):
        assert LOOP._slot_present(self.WS, "搜一下北京")


# ── robot_grab (trigger-based generic) ───────────────────────────────────────

class TestRobotGrabSlot:
    GRAB = _skill("robot_grab", voice_trigger="抓住,抓取,拿起来,放下,松开")

    def test_抓取_bare_missing(self):
        assert not LOOP._slot_present(self.GRAB, "抓取")

    def test_抓取_with_object(self):
        assert LOOP._slot_present(self.GRAB, "抓取那个瓶子")

    def test_拿起来_bare_missing(self):
        assert not LOOP._slot_present(self.GRAB, "拿起来")

    def test_拿起来_with_object(self):
        assert LOOP._slot_present(self.GRAB, "拿起来那个红色积木")


# ── mapping (trigger-based generic) ──────────────────────────────────────────

class TestMappingSlot:
    MAP = _skill("mapping", voice_trigger="建图,开始建图,扫描地图,创建地图")

    def test_建图_bare_missing(self):
        assert not LOOP._slot_present(self.MAP, "建图")

    def test_开始建图_bare_missing(self):
        assert not LOOP._slot_present(self.MAP, "开始建图")

    def test_建图_with_area(self):
        assert LOOP._slot_present(self.MAP, "建图仓库区域")

    def test_扫描地图_with_area(self):
        assert LOOP._slot_present(self.MAP, "扫描地图二楼")


# ── skill with no voice_trigger ───────────────────────────────────────────────

class TestNoTrigger:
    SK = _skill("no_trigger_skill", voice_trigger="")

    def test_short_text_missing(self):
        assert not LOOP._slot_present(self.SK, "")

    def test_longer_text_present(self):
        # falls back to length > len(voice_trigger or "") = len("") = 0
        assert LOOP._slot_present(self.SK, "some content here")
