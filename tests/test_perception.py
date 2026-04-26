"""Tests for perception layer: schemas (Detection, Observation, ChangeEvent) + ChangeDetector."""

from __future__ import annotations

import json
import time
from pathlib import Path

from askme.perception.change_detector import ChangeDetector, compute_iou
from askme.schemas.events import ChangeEvent, ChangeEventType
from askme.schemas.observation import Detection, Observation

# ── Detection ────────────────────────────────────────────────────────────────


class TestDetection:
    def test_center(self):
        d = Detection("person", 0.9, (100, 200, 300, 400))
        assert d.center == (200.0, 300.0)

    def test_area(self):
        d = Detection("chair", 0.8, (0, 0, 100, 50))
        assert d.area == 5000.0

    def test_zero_area(self):
        d = Detection("cup", 0.7, (10, 10, 10, 10))
        assert d.area == 0.0

    def test_distance_default_none(self):
        d = Detection("person", 0.9, (0, 0, 1, 1))
        assert d.distance_m is None

    def test_distance_with_value(self):
        d = Detection("person", 0.9, (0, 0, 1, 1), distance_m=2.5)
        assert d.distance_m == 2.5


# ── Observation ──────────────────────────────────────────────────────────────


class TestObservation:
    def test_from_daemon_json_basic(self):
        data = {
            "timestamp": 1710000000.0,
            "infer_ms": 3.2,
            "detections": [
                {"class_id": "person", "confidence": 0.92, "bbox": [100, 200, 300, 400]},
                {"class_id": "chair", "confidence": 0.85, "bbox": [400, 100, 500, 300]},
            ],
        }
        obs = Observation.from_daemon_json(data)
        assert obs.timestamp == 1710000000.0
        assert obs.infer_ms == 3.2
        assert len(obs.detections) == 2
        assert obs.detections[0].class_id == "person"
        assert obs.detections[0].bbox == (100, 200, 300, 400)

    def test_from_daemon_json_empty(self):
        obs = Observation.from_daemon_json({"timestamp": 0})
        assert obs.detections == []

    def test_by_class(self):
        obs = Observation(
            timestamp=0,
            detections=[
                Detection("person", 0.9, (0, 0, 10, 10)),
                Detection("chair", 0.8, (20, 20, 30, 30)),
                Detection("person", 0.7, (40, 40, 50, 50)),
            ],
        )
        groups = obs.by_class()
        assert len(groups["person"]) == 2
        assert len(groups["chair"]) == 1

    def test_count(self):
        obs = Observation(
            timestamp=0,
            detections=[
                Detection("person", 0.9, (0, 0, 10, 10)),
                Detection("person", 0.7, (40, 40, 50, 50)),
            ],
        )
        assert obs.count("person") == 2
        assert obs.count("chair") == 0


# ── ChangeEvent ──────────────────────────────────────────────────────────────


class TestChangeEvent:
    def test_to_dict_roundtrip(self):
        event = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1710000000.0,
            subject_class="person",
            confidence=0.92,
            bbox=(100, 200, 300, 400),
            distance_m=2.3,
            track_id="t_0001",
        )
        d = event.to_dict()
        restored = ChangeEvent.from_dict(d)
        assert restored.event_type == ChangeEventType.PERSON_APPEARED
        assert restored.subject_class == "person"
        assert restored.confidence == 0.92
        assert restored.track_id == "t_0001"
        assert restored.distance_m == 2.3

    def test_auto_importance(self):
        event = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=0, subject_class="person",
        )
        assert event.importance == 0.8

    def test_is_person_event(self):
        assert ChangeEvent(ChangeEventType.PERSON_APPEARED, 0, "person").is_person_event
        assert ChangeEvent(ChangeEventType.PERSON_LEFT, 0, "person").is_person_event
        assert not ChangeEvent(ChangeEventType.OBJECT_APPEARED, 0, "chair").is_person_event

    def test_description_zh(self):
        e = ChangeEvent(ChangeEventType.PERSON_APPEARED, 0, "person", distance_m=3.0)
        assert "人" in e.description_zh()
        assert "3.0" in e.description_zh()

    def test_to_dict_minimal(self):
        event = ChangeEvent(ChangeEventType.OBJECT_APPEARED, 0, "chair", confidence=0.7)
        d = event.to_dict()
        assert "bbox" not in d  # None bbox omitted
        assert "track_id" not in d  # empty track_id omitted

    def test_jsonl_serializable(self):
        event = ChangeEvent(ChangeEventType.PERSON_LEFT, time.time(), "person")
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        restored = ChangeEvent.from_dict(json.loads(line))
        assert restored.event_type == ChangeEventType.PERSON_LEFT


# ── compute_iou ──────────────────────────────────────────────────────────────


class TestComputeIoU:
    def test_perfect_overlap(self):
        assert compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0

    def test_no_overlap(self):
        assert compute_iou((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0

    def test_partial_overlap(self):
        iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.1 < iou < 0.2  # ~14.3%

    def test_contained(self):
        iou = compute_iou((0, 0, 200, 200), (50, 50, 100, 100))
        assert 0.05 < iou < 0.1  # small box inside large

    def test_zero_area_box(self):
        assert compute_iou((10, 10, 10, 10), (0, 0, 100, 100)) == 0.0


# ── ChangeDetector matching ──────────────────────────────────────────────────


class TestChangeDetectorMatching:
    def _make_detector(self, **overrides) -> ChangeDetector:
        cfg = {"proactive": {"change_detector": {"confirm_frames": 1, "disappear_frames": 1, **overrides}}}
        return ChangeDetector(cfg)

    def test_person_appeared(self):
        det = self._make_detector()
        prev = Observation(timestamp=0, detections=[])
        curr = Observation(timestamp=1, detections=[Detection("person", 0.9, (100, 200, 300, 400))])
        changes = det._compare(prev, curr)
        assert len(changes) == 1
        assert changes[0].change_type == "appeared"
        assert changes[0].class_id == "person"

    def test_person_disappeared(self):
        det = self._make_detector()
        prev = Observation(timestamp=0, detections=[Detection("person", 0.9, (100, 200, 300, 400))])
        curr = Observation(timestamp=1, detections=[])
        changes = det._compare(prev, curr)
        assert len(changes) == 1
        assert changes[0].change_type == "disappeared"

    def test_same_position_no_change(self):
        det = self._make_detector()
        d = Detection("person", 0.9, (100, 200, 300, 400))
        prev = Observation(timestamp=0, detections=[d])
        curr = Observation(timestamp=1, detections=[Detection("person", 0.85, (105, 205, 305, 405))])
        changes = det._compare(prev, curr)
        assert len(changes) == 0  # IoU > threshold, matched

    def test_two_classes_independent(self):
        det = self._make_detector()
        prev = Observation(timestamp=0, detections=[
            Detection("person", 0.9, (0, 0, 100, 100)),
        ])
        curr = Observation(timestamp=1, detections=[
            Detection("person", 0.9, (0, 0, 100, 100)),
            Detection("chair", 0.8, (200, 200, 300, 300)),
        ])
        changes = det._compare(prev, curr)
        assert len(changes) == 1
        assert changes[0].class_id == "chair"
        assert changes[0].change_type == "appeared"


# ── ChangeDetector debounce ──────────────────────────────────────────────────


class TestChangeDetectorDebounce:
    def _make_detector(self, confirm=3, disappear=5) -> ChangeDetector:
        cfg = {"proactive": {"change_detector": {
            "confirm_frames": confirm, "disappear_frames": disappear,
        }}}
        return ChangeDetector(cfg)

    def test_confirm_after_n_frames(self):
        det = self._make_detector(confirm=3)
        prev = Observation(timestamp=0, detections=[])
        curr = Observation(timestamp=1, detections=[Detection("person", 0.9, (100, 200, 300, 400))])

        # Frame 1, 2: no event yet
        for i in range(2):
            changes = det._compare(prev, curr)
            events = det._debounce(changes, float(i + 1))
            assert events == [], f"Frame {i+1} should not emit event"

        # Frame 3: confirmed
        changes = det._compare(prev, curr)
        events = det._debounce(changes, 3.0)
        assert len(events) == 1
        assert events[0].event_type == ChangeEventType.PERSON_APPEARED

    def test_intermittent_resets_counter(self):
        det = self._make_detector(confirm=3)
        prev = Observation(timestamp=0, detections=[])
        curr = Observation(timestamp=1, detections=[Detection("person", 0.9, (100, 200, 300, 400))])

        # Frame 1: appeared
        changes = det._compare(prev, curr)
        det._debounce(changes, 1.0)

        # Frame 2: disappeared (intermittent)
        det._debounce([], 2.0)

        # Frame 3, 4: appeared again — counter reset
        for i in range(2):
            changes = det._compare(prev, curr)
            events = det._debounce(changes, float(3 + i))
            assert events == [], f"Frame {3+i} should not emit (counter was reset)"

    def test_disappear_after_n_frames(self):
        det = self._make_detector(disappear=2)
        d = Detection("chair", 0.8, (200, 200, 300, 300))
        prev = Observation(timestamp=0, detections=[d])
        curr = Observation(timestamp=1, detections=[])

        # Frame 1: pending
        changes = det._compare(prev, curr)
        events = det._debounce(changes, 1.0)
        assert events == []

        # Frame 2: confirmed
        changes = det._compare(prev, curr)
        events = det._debounce(changes, 2.0)
        assert len(events) == 1
        assert events[0].event_type == ChangeEventType.OBJECT_DISAPPEARED


# ── ChangeDetector event output ──────────────────────────────────────────────


class TestChangeDetectorOutput:
    def test_emit_events_writes_jsonl(self, tmp_path: Path):
        cfg = {"proactive": {"change_detector": {"event_file": str(tmp_path / "events.jsonl")}}}
        det = ChangeDetector(cfg)
        events = [
            ChangeEvent(ChangeEventType.PERSON_APPEARED, time.time(), "person", confidence=0.9),
        ]
        det._emit_events(events)
        lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "person_appeared"
        assert data["subject_class"] == "person"

    def test_emit_appends(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        cfg = {"proactive": {"change_detector": {"event_file": str(path)}}}
        det = ChangeDetector(cfg)
        e1 = ChangeEvent(ChangeEventType.PERSON_APPEARED, 1.0, "person")
        e2 = ChangeEvent(ChangeEventType.PERSON_LEFT, 2.0, "person")
        det._emit_events([e1])
        det._emit_events([e2])
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
