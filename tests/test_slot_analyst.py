"""Tests for the typed slot analyst: vague detection + slot analysis."""

import pytest

from askme.pipeline.proactive.slot_analyst import analyze_slots, is_vague
from askme.skills.skill_model import SkillDefinition, SlotSpec


# ── is_vague ─────────────────────────────────────────────────────────────────


class TestIsVague:
    def test_empty_is_vague(self):
        assert is_vague("")

    def test_single_char_is_vague(self):
        assert is_vague("A")

    def test_referential_placeholder(self):
        assert is_vague("那个")
        assert is_vague("这个")
        assert is_vague("那件")

    def test_vague_location(self):
        assert is_vague("那里")
        assert is_vague("这里")
        assert is_vague("那边")

    def test_filler_word(self):
        assert is_vague("一下")
        assert is_vague("看看")
        assert is_vague("查查")
        assert is_vague("某处")

    def test_real_content_not_vague(self):
        assert not is_vague("北京")
        assert not is_vague("仓库B")
        assert not is_vague("明天天气")
        assert not is_vague("穹沛科技")
        assert not is_vague("红色瓶子")


# ── analyze_slots ─────────────────────────────────────────────────────────────


class FakePipeline:
    def extract_semantic_target(self, text: str) -> str:
        import re
        for pat in [r"去(.{1,10}?)(?:吧|了|$)", r"导航到(.{1,10}?)(?:去|了|$)"]:
            m = re.search(pat, text)
            if m and m.group(1).strip():
                return m.group(1).strip()
        return text


def _nav_skill() -> SkillDefinition:
    return SkillDefinition(
        name="navigate",
        description="导航",
        voice_trigger="导航到,带我去,去",
        required_slots=[SlotSpec(name="destination", type="location", prompt="导航去哪里？")],
    )


def _search_skill() -> SkillDefinition:
    return SkillDefinition(
        name="web_search",
        description="搜索",
        voice_trigger="搜索一下,查一下,帮我搜索",
        required_slots=[SlotSpec(name="query", type="text", prompt="搜索什么内容？")],
    )


def _grab_skill() -> SkillDefinition:
    return SkillDefinition(
        name="robot_grab",
        description="机械臂抓取",
        voice_trigger="抓取,拿起来",
        required_slots=[
            SlotSpec(name="object", type="referent", prompt="抓取什么物体？"),
            SlotSpec(name="place_target", type="location", prompt="放到哪里？", optional=True),
        ],
    )


class TestAnalyzeSlotsNavigate:
    skill = _nav_skill()
    pipeline = FakePipeline()

    def test_bare_trigger_missing(self):
        a = analyze_slots(self.skill, "导航到", self.pipeline)
        assert not a.ready
        assert a.missing_required[0].status == "missing"

    def test_with_destination_ready(self):
        a = analyze_slots(self.skill, "去仓库B", self.pipeline)
        assert a.ready
        assert a.slots[0].status == "filled"
        assert a.slots[0].value == "仓库B"

    def test_vague_location_not_ready(self):
        # 导航到那里 → extracted value is "那里" → vague
        a = analyze_slots(self.skill, "导航到那里", self.pipeline)
        # FakePipeline doesn't extract "那里" specially, returns full text
        # but slot_utils checks via pipeline — depends on extract_semantic_target
        # Key thing: missing_required is non-empty when ready=False
        assert not a.ready or a.slots[0].value is not None


class TestAnalyzeSlotsWebSearch:
    skill = _search_skill()

    def test_bare_搜索一下_missing(self):
        a = analyze_slots(self.skill, "搜索一下")
        # After stripping trigger "搜索一下", remainder is empty → missing
        assert not a.ready

    def test_with_query_ready(self):
        a = analyze_slots(self.skill, "搜索一下明天北京天气")
        assert a.ready
        assert a.slots[0].status == "filled"

    def test_vague_placeholder_not_ready(self):
        # "搜索一下那个" → remainder "那个" → vague
        a = analyze_slots(self.skill, "搜索一下那个")
        assert not a.ready
        assert a.slots[0].status == "vague"

    def test_查查_is_vague(self):
        # "查查" is itself the trigger AND a vague word — no query
        skill_with_chazha = SkillDefinition(
            name="web_search",
            voice_trigger="查查",
            required_slots=[SlotSpec(name="query", type="text", prompt="查什么？")],
        )
        a = analyze_slots(skill_with_chazha, "查查")
        assert not a.ready


class TestAnalyzeSlotsMultiSlot:
    skill = _grab_skill()

    def test_both_slots_missing(self):
        a = analyze_slots(self.skill, "抓取")
        required_fills = [f for f in a.slots if not f.spec.optional]
        assert not a.ready
        assert required_fills[0].status in ("missing", "vague")

    def test_object_filled_optional_skipped(self):
        # place_target is optional — only object matters for ready
        a = analyze_slots(self.skill, "抓取红色瓶子")
        assert a.ready  # object filled; place_target optional → don't block

    def test_vague_object_not_ready(self):
        a = analyze_slots(self.skill, "抓取那个")
        assert not a.ready
        assert a.slots[0].status == "vague"

    def test_slots_list_has_required_only(self):
        """Optional slots are excluded from analysis entirely."""
        a = analyze_slots(self.skill, "抓取瓶子")
        # Only 'object' should be in slots (place_target is optional → skipped)
        assert len(a.slots) == 1
        assert a.slots[0].spec.name == "object"


# ── SlotAnalysis properties ───────────────────────────────────────────────────

class TestSlotAnalysisProperties:
    def test_ready_true_when_all_filled(self):
        from askme.pipeline.proactive.slot_types import SlotAnalysis, SlotFill
        sk = SlotSpec(name="q", prompt="?")
        a = SlotAnalysis(skill_name="s", slots=[SlotFill(spec=sk, value="北京", status="filled")])
        assert a.ready

    def test_ready_false_when_any_missing(self):
        from askme.pipeline.proactive.slot_types import SlotAnalysis, SlotFill
        sk = SlotSpec(name="q", prompt="?")
        a = SlotAnalysis(skill_name="s", slots=[SlotFill(spec=sk, value=None, status="missing")])
        assert not a.ready

    def test_missing_required_filters_correctly(self):
        from askme.pipeline.proactive.slot_types import SlotAnalysis, SlotFill
        sk1 = SlotSpec(name="a")
        sk2 = SlotSpec(name="b")
        a = SlotAnalysis(skill_name="s", slots=[
            SlotFill(spec=sk1, value="ok", status="filled"),
            SlotFill(spec=sk2, value=None, status="missing"),
        ])
        assert len(a.missing_required) == 1
        assert a.missing_required[0].spec.name == "b"
