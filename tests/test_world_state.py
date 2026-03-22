"""Tests for WorldState and AttentionManager."""

from __future__ import annotations

import time

import pytest

from askme.perception.attention_manager import AttentionManager, AttentionConfig
from askme.perception.world_state import TrackedObject, WorldState
from askme.schemas.events import ChangeEvent, ChangeEventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_type: ChangeEventType,
    subject_class: str = "person",
    confidence: float = 0.9,
    track_id: str = "t_0001",
    bbox: tuple | None = (100, 100, 200, 300),
    distance_m: float | None = None,
    importance: float = 0.0,  # 0.0 → auto from event type
    timestamp: float | None = None,
) -> ChangeEvent:
    return ChangeEvent(
        event_type=event_type,
        timestamp=timestamp if timestamp is not None else time.time(),
        subject_class=subject_class,
        confidence=confidence,
        bbox=bbox,
        track_id=track_id,
        distance_m=distance_m,
        importance=importance,
    )


# ---------------------------------------------------------------------------
# WorldState — sync API (tests run synchronously for simplicity)
# ---------------------------------------------------------------------------

class TestWorldStateAdd:
    def test_add_person(self):
        ws = WorldState()
        event = _make_event(ChangeEventType.PERSON_APPEARED, track_id="t_0001")
        ws.apply_event_sync(event)
        objs = ws.get_objects_sync()
        assert "t_0001" in objs
        assert objs["t_0001"].class_id == "person"

    def test_add_object(self):
        ws = WorldState()
        event = _make_event(
            ChangeEventType.OBJECT_APPEARED, subject_class="chair", track_id="t_0002"
        )
        ws.apply_event_sync(event)
        objs = ws.get_objects_sync()
        assert "t_0002" in objs
        assert objs["t_0002"].class_id == "chair"

    def test_add_records_timestamp(self):
        ws = WorldState()
        ts = 1000.0
        event = _make_event(
            ChangeEventType.PERSON_APPEARED, track_id="t_0001", timestamp=ts
        )
        ws.apply_event_sync(event)
        obj = ws.get_objects_sync()["t_0001"]
        assert obj.first_seen == ts
        assert obj.last_seen == ts

    def test_add_updates_existing(self):
        ws = WorldState()
        t1 = 1000.0
        t2 = 1005.0
        ws.apply_event_sync(
            _make_event(ChangeEventType.PERSON_APPEARED, track_id="t_0001", timestamp=t1)
        )
        ws.apply_event_sync(
            _make_event(ChangeEventType.PERSON_APPEARED, track_id="t_0001", timestamp=t2)
        )
        obj = ws.get_objects_sync()["t_0001"]
        assert obj.first_seen == t1     # unchanged
        assert obj.last_seen == t2      # updated

    def test_add_no_track_id_uses_auto(self):
        ws = WorldState()
        event = _make_event(ChangeEventType.OBJECT_APPEARED, subject_class="cup", track_id="")
        ws.apply_event_sync(event)
        objs = ws.get_objects_sync()
        assert len(objs) == 1
        key = list(objs.keys())[0]
        assert key.startswith("auto_")


class TestWorldStateRemove:
    def test_remove_by_track_id(self):
        ws = WorldState()
        ws.apply_event_sync(_make_event(ChangeEventType.PERSON_APPEARED, track_id="t_0001"))
        ws.apply_event_sync(_make_event(ChangeEventType.PERSON_LEFT, track_id="t_0001"))
        assert ws.get_objects_sync() == {}

    def test_remove_by_class_when_no_track_id(self):
        ws = WorldState()
        ws.apply_event_sync(
            _make_event(ChangeEventType.OBJECT_APPEARED, subject_class="chair", track_id="t_c1")
        )
        ws.apply_event_sync(
            _make_event(ChangeEventType.OBJECT_DISAPPEARED, subject_class="chair", track_id="")
        )
        assert ws.get_objects_sync() == {}

    def test_remove_oldest_when_multiple_same_class(self):
        ws = WorldState()
        ws.apply_event_sync(
            _make_event(
                ChangeEventType.OBJECT_APPEARED, subject_class="chair",
                track_id="t_c1", timestamp=1000.0
            )
        )
        ws.apply_event_sync(
            _make_event(
                ChangeEventType.OBJECT_APPEARED, subject_class="chair",
                track_id="t_c2", timestamp=1005.0
            )
        )
        ws.apply_event_sync(
            _make_event(ChangeEventType.OBJECT_DISAPPEARED, subject_class="chair", track_id="")
        )
        objs = ws.get_objects_sync()
        assert len(objs) == 1
        assert "t_c1" not in objs   # oldest removed
        assert "t_c2" in objs

    def test_remove_nonexistent_no_crash(self):
        ws = WorldState()
        # Should not raise
        ws.apply_event_sync(
            _make_event(ChangeEventType.PERSON_LEFT, track_id="t_ghost")
        )
        assert ws.get_objects_sync() == {}


class TestWorldStateGetPersons:
    def test_get_persons_filters_by_class(self):
        ws = WorldState()
        ws.apply_event_sync(
            _make_event(ChangeEventType.PERSON_APPEARED, subject_class="person", track_id="p1")
        )
        ws.apply_event_sync(
            _make_event(ChangeEventType.OBJECT_APPEARED, subject_class="chair", track_id="c1")
        )
        persons = ws.get_persons_sync()
        assert len(persons) == 1
        assert persons[0].track_id == "p1"

    def test_get_persons_empty_scene(self):
        ws = WorldState()
        assert ws.get_persons_sync() == []


class TestWorldStateSummary:
    def test_empty_scene(self):
        ws = WorldState()
        summary = ws.get_summary_sync()
        assert "无检测目标" in summary

    def test_one_person(self):
        ws = WorldState()
        ws.apply_event_sync(
            _make_event(ChangeEventType.PERSON_APPEARED, subject_class="person", track_id="p1")
        )
        summary = ws.get_summary_sync()
        assert "人" in summary
        assert "当前场景" in summary

    def test_person_with_distance(self):
        ws = WorldState()
        ws.apply_event_sync(
            _make_event(
                ChangeEventType.PERSON_APPEARED, subject_class="person",
                track_id="p1", distance_m=3.5
            )
        )
        summary = ws.get_summary_sync()
        assert "3.5米" in summary

    def test_multiple_objects(self):
        ws = WorldState()
        ws.apply_event_sync(
            _make_event(ChangeEventType.PERSON_APPEARED, subject_class="person", track_id="p1")
        )
        ws.apply_event_sync(
            _make_event(ChangeEventType.OBJECT_APPEARED, subject_class="chair", track_id="c1")
        )
        summary = ws.get_summary_sync()
        assert "人" in summary
        assert "椅子" in summary

    def test_after_removal_summary_updates(self):
        ws = WorldState()
        ws.apply_event_sync(
            _make_event(ChangeEventType.PERSON_APPEARED, subject_class="person", track_id="p1")
        )
        ws.apply_event_sync(
            _make_event(ChangeEventType.PERSON_LEFT, subject_class="person", track_id="p1")
        )
        summary = ws.get_summary_sync()
        assert "无检测目标" in summary

    def test_two_persons(self):
        ws = WorldState()
        ws.apply_event_sync(
            _make_event(ChangeEventType.PERSON_APPEARED, subject_class="person", track_id="p1")
        )
        ws.apply_event_sync(
            _make_event(ChangeEventType.PERSON_APPEARED, subject_class="person", track_id="p2")
        )
        summary = ws.get_summary_sync()
        assert "2名人" in summary


class TestWorldStateEventHistory:
    def test_history_records_events(self):
        ws = WorldState()
        ws.apply_event_sync(_make_event(ChangeEventType.PERSON_APPEARED, track_id="p1"))
        ws.apply_event_sync(_make_event(ChangeEventType.PERSON_LEFT, track_id="p1"))
        history = ws.event_history_sync()
        assert len(history) == 2
        assert history[0].event_type == ChangeEventType.PERSON_APPEARED
        assert history[1].event_type == ChangeEventType.PERSON_LEFT

    def test_history_capped_at_max(self):
        ws = WorldState()
        ws._max_history = 5
        for i in range(10):
            ws.apply_event_sync(
                _make_event(ChangeEventType.OBJECT_APPEARED, track_id=f"t_{i:04d}")
            )
        history = ws.event_history_sync()
        assert len(history) == 5


class TestWorldStateAsync:
    @pytest.mark.asyncio
    async def test_apply_event_async(self):
        ws = WorldState()
        event = _make_event(ChangeEventType.PERSON_APPEARED, track_id="t_0001")
        await ws.apply_event(event)
        objs = await ws.get_objects()
        assert "t_0001" in objs

    @pytest.mark.asyncio
    async def test_get_persons_async(self):
        ws = WorldState()
        await ws.apply_event(
            _make_event(ChangeEventType.PERSON_APPEARED, subject_class="person", track_id="p1")
        )
        persons = await ws.get_persons()
        assert len(persons) == 1

    @pytest.mark.asyncio
    async def test_get_summary_async(self):
        ws = WorldState()
        await ws.apply_event(
            _make_event(ChangeEventType.PERSON_APPEARED, subject_class="person", track_id="p1")
        )
        summary = await ws.get_summary()
        assert "人" in summary

    @pytest.mark.asyncio
    async def test_snapshot(self):
        ws = WorldState()
        await ws.apply_event(
            _make_event(ChangeEventType.PERSON_APPEARED, subject_class="person", track_id="p1")
        )
        snap = await ws.snapshot()
        assert snap["object_count"] == 1
        assert len(snap["objects"]) == 1
        assert "当前场景" in snap["summary"]


# ---------------------------------------------------------------------------
# AttentionManager
# ---------------------------------------------------------------------------

class TestAttentionManagerAlert:
    def test_alert_passes_high_importance(self):
        mgr = AttentionManager()
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        assert mgr.should_alert(event) is True

    def test_alert_blocked_low_importance(self):
        mgr = AttentionManager()
        # COUNT_CHANGED auto-importance = 0.3, below default threshold 0.5
        event = _make_event(ChangeEventType.COUNT_CHANGED, subject_class="chair", importance=0.3)
        assert mgr.should_alert(event) is False

    def test_alert_cooldown_blocks_repeat(self):
        mgr = AttentionManager()
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        assert mgr.should_alert(event) is True
        # Second call immediately — blocked by cooldown
        assert mgr.should_alert(event) is False

    def test_alert_after_cooldown_passes(self):
        mgr = AttentionManager(config={
            "proactive": {"attention": {"cooldowns": {"person_appeared": 0.0}}}
        })
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        assert mgr.should_alert(event) is True
        assert mgr.should_alert(event) is True  # zero cooldown → always passes

    def test_reset_cooldown_re_enables(self):
        mgr = AttentionManager()
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        mgr.should_alert(event)  # consume cooldown
        assert mgr.should_alert(event) is False
        mgr.reset_cooldown(ChangeEventType.PERSON_APPEARED)
        assert mgr.should_alert(event) is True

    def test_reset_all_cooldowns(self):
        mgr = AttentionManager()
        e1 = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        e2 = _make_event(ChangeEventType.PERSON_LEFT, importance=0.7)
        mgr.should_alert(e1)
        mgr.should_alert(e2)
        mgr.reset_all_cooldowns()
        assert mgr.should_alert(e1) is True
        assert mgr.should_alert(e2) is True

    def test_different_event_types_independent_cooldowns(self):
        mgr = AttentionManager()
        e_appeared = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        e_left = _make_event(ChangeEventType.PERSON_LEFT, importance=0.7)
        mgr.should_alert(e_appeared)
        # PERSON_LEFT has its own cooldown — should still pass
        assert mgr.should_alert(e_left) is True


class TestAttentionManagerInvestigate:
    def test_investigate_requires_higher_threshold(self):
        mgr = AttentionManager()
        # PERSON_APPEARED importance=0.8 >= default investigate_threshold 0.7
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        assert mgr.should_investigate(event) is True

    def test_investigate_blocked_below_threshold(self):
        mgr = AttentionManager()
        # PERSON_LEFT auto-importance=0.7, borderline — exactly at threshold
        event = _make_event(ChangeEventType.PERSON_LEFT, importance=0.65)
        assert mgr.should_investigate(event) is False

    def test_investigate_has_longer_cooldown_than_alert(self):
        mgr = AttentionManager()
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        mgr.should_investigate(event)   # consume investigate cooldown
        # Alert cooldown is shorter, but investigate cooldown is 2x
        assert mgr.should_investigate(event) is False

    def test_investigate_and_alert_independent(self):
        mgr = AttentionManager()
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        # Alert and investigate have separate cooldown tracking
        assert mgr.should_alert(event) is True
        assert mgr.should_investigate(event) is True

    def test_investigate_blocked_second_call(self):
        mgr = AttentionManager()
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        mgr.should_investigate(event)
        assert mgr.should_investigate(event) is False


class TestAttentionManagerConfig:
    def test_custom_alert_threshold(self):
        mgr = AttentionManager(config={
            "proactive": {"attention": {"alert_threshold": 0.9}}
        })
        # importance=0.8 < threshold 0.9 → blocked
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        assert mgr.should_alert(event) is False

    def test_custom_investigate_threshold(self):
        mgr = AttentionManager(config={
            "proactive": {"attention": {"investigate_threshold": 0.95}}
        })
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        assert mgr.should_investigate(event) is False

    def test_cooldown_remaining_before_first_alert(self):
        mgr = AttentionManager()
        # Never alerted — cooldown_remaining should be 0
        assert mgr.cooldown_remaining(ChangeEventType.PERSON_APPEARED) == 0.0

    def test_cooldown_remaining_after_alert(self):
        mgr = AttentionManager()
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.8)
        mgr.should_alert(event)
        remaining = mgr.cooldown_remaining(ChangeEventType.PERSON_APPEARED)
        assert remaining > 0.0
        assert remaining <= 10.0  # default cooldown for person_appeared

    def test_status_returns_dict(self):
        mgr = AttentionManager()
        s = mgr.status()
        assert "alert_threshold" in s
        assert "investigate_threshold" in s
        assert "cooldowns_remaining" in s
        assert "person_appeared" in s["cooldowns_remaining"]
