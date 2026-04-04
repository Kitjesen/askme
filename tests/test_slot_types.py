"""Tests for slot analysis types: SlotFill, SlotAnalysis."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from askme.pipeline.proactive.slot_types import SlotAnalysis, SlotFill


def _make_spec(required: bool = True, name: str = "dest") -> MagicMock:
    spec = MagicMock()
    spec.name = name
    spec.required = required
    return spec


class TestSlotFill:
    def test_default_status_missing(self):
        sf = SlotFill(spec=_make_spec())
        assert sf.status == "missing"

    def test_default_value_none(self):
        sf = SlotFill(spec=_make_spec())
        assert sf.value is None

    def test_is_ok_when_filled(self):
        sf = SlotFill(spec=_make_spec(), value="仓库A", status="filled")
        assert sf.is_ok is True

    def test_not_ok_when_missing(self):
        sf = SlotFill(spec=_make_spec())
        assert sf.is_ok is False

    def test_not_ok_when_vague(self):
        sf = SlotFill(spec=_make_spec(), value="那边", status="vague")
        assert sf.is_ok is False


class TestSlotAnalysis:
    def test_ready_when_all_filled(self):
        s1 = SlotFill(spec=_make_spec(), value="仓库A", status="filled")
        s2 = SlotFill(spec=_make_spec(name="speed"), value="慢速", status="filled")
        analysis = SlotAnalysis(skill_name="navigate", slots=[s1, s2])
        assert analysis.ready is True

    def test_not_ready_when_any_missing(self):
        s1 = SlotFill(spec=_make_spec(), value="仓库A", status="filled")
        s2 = SlotFill(spec=_make_spec(name="speed"))  # missing
        analysis = SlotAnalysis(skill_name="navigate", slots=[s1, s2])
        assert analysis.ready is False

    def test_not_ready_when_any_vague(self):
        s1 = SlotFill(spec=_make_spec(), value="那边", status="vague")
        analysis = SlotAnalysis(skill_name="navigate", slots=[s1])
        assert analysis.ready is False

    def test_ready_with_no_slots(self):
        analysis = SlotAnalysis(skill_name="patrol")
        assert analysis.ready is True  # vacuously true

    def test_missing_required_returns_unfilled_slots(self):
        s1 = SlotFill(spec=_make_spec(), value="仓库A", status="filled")
        s2 = SlotFill(spec=_make_spec(name="speed"))
        analysis = SlotAnalysis(skill_name="navigate", slots=[s1, s2])
        missing = analysis.missing_required
        assert len(missing) == 1
        assert missing[0] is s2

    def test_missing_required_empty_when_all_filled(self):
        s1 = SlotFill(spec=_make_spec(), value="仓库A", status="filled")
        analysis = SlotAnalysis(skill_name="navigate", slots=[s1])
        assert analysis.missing_required == []

    def test_skill_name_stored(self):
        analysis = SlotAnalysis(skill_name="grab")
        assert analysis.skill_name == "grab"
