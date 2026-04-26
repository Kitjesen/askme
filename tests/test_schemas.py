"""Tests for ChangeEvent (schemas/events.py) and Observation/Detection (schemas/observation.py)."""

from __future__ import annotations

from askme.schemas.events import ChangeEvent, ChangeEventType
from askme.schemas.observation import Detection, Observation

# ── ChangeEventType ───────────────────────────────────────────────────────────

class TestChangeEventType:
    def test_all_values_string(self):
        for member in ChangeEventType:
            assert isinstance(member.value, str)

    def test_person_appeared_value(self):
        assert ChangeEventType.PERSON_APPEARED.value == "person_appeared"


# ── ChangeEvent defaults & importance ────────────────────────────────────────

class TestChangeEventDefaults:
    def test_person_appeared_importance(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1.0,
            subject_class="person",
        )
        assert e.importance == 0.8

    def test_person_left_importance(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_LEFT,
            timestamp=1.0,
            subject_class="person",
        )
        assert e.importance == 0.7

    def test_object_appeared_importance(self):
        e = ChangeEvent(
            event_type=ChangeEventType.OBJECT_APPEARED,
            timestamp=1.0,
            subject_class="chair",
        )
        assert e.importance == 0.4

    def test_explicit_importance_not_overridden(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1.0,
            subject_class="person",
            importance=0.99,
        )
        assert e.importance == 0.99

    def test_is_person_event_true(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1.0,
            subject_class="person",
        )
        assert e.is_person_event is True

    def test_is_person_event_false_for_object(self):
        e = ChangeEvent(
            event_type=ChangeEventType.OBJECT_APPEARED,
            timestamp=1.0,
            subject_class="chair",
        )
        assert e.is_person_event is False


# ── ChangeEvent.to_dict ───────────────────────────────────────────────────────

class TestToDict:
    def test_required_keys_present(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1234.0,
            subject_class="person",
        )
        d = e.to_dict()
        assert "event_type" in d
        assert "timestamp" in d
        assert "subject_class" in d

    def test_event_type_is_value_string(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1.0,
            subject_class="person",
        )
        assert e.to_dict()["event_type"] == "person_appeared"

    def test_bbox_included_when_set(self):
        e = ChangeEvent(
            event_type=ChangeEventType.OBJECT_APPEARED,
            timestamp=1.0,
            subject_class="chair",
            bbox=(10.0, 20.0, 100.0, 200.0),
        )
        d = e.to_dict()
        assert "bbox" in d
        assert len(d["bbox"]) == 4

    def test_bbox_not_included_when_none(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1.0,
            subject_class="person",
        )
        assert "bbox" not in e.to_dict()

    def test_distance_m_included_when_set(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1.0,
            subject_class="person",
            distance_m=2.5,
        )
        d = e.to_dict()
        assert "distance_m" in d
        assert d["distance_m"] == 2.5

    def test_counts_included_when_nonzero(self):
        e = ChangeEvent(
            event_type=ChangeEventType.COUNT_CHANGED,
            timestamp=1.0,
            subject_class="person",
            prev_count=1,
            curr_count=3,
        )
        d = e.to_dict()
        assert d["prev_count"] == 1
        assert d["curr_count"] == 3


# ── ChangeEvent.from_dict ─────────────────────────────────────────────────────

class TestFromDict:
    def test_round_trip(self):
        original = ChangeEvent(
            event_type=ChangeEventType.OBJECT_APPEARED,
            timestamp=100.0,
            subject_class="bottle",
            confidence=0.9,
            bbox=(10.0, 20.0, 50.0, 80.0),
            distance_m=1.2,
        )
        restored = ChangeEvent.from_dict(original.to_dict())
        assert restored.event_type == original.event_type
        assert restored.subject_class == original.subject_class

    def test_missing_optional_fields_use_defaults(self):
        d = {"event_type": "person_appeared", "timestamp": 1.0, "subject_class": "person"}
        e = ChangeEvent.from_dict(d)
        assert e.confidence == 0.0
        assert e.bbox is None

    def test_bbox_restored_as_tuple(self):
        d = {
            "event_type": "object_appeared",
            "timestamp": 1.0,
            "subject_class": "chair",
            "bbox": [1.0, 2.0, 3.0, 4.0],
        }
        e = ChangeEvent.from_dict(d)
        assert isinstance(e.bbox, tuple)
        assert len(e.bbox) == 4


# ── ChangeEvent.description_zh ───────────────────────────────────────────────

class TestDescriptionZh:
    def test_person_appeared_contains_person_text(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1.0,
            subject_class="person",
        )
        assert "人" in e.description_zh()

    def test_person_appeared_with_distance(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1.0,
            subject_class="person",
            distance_m=3.0,
        )
        assert "3.0" in e.description_zh()

    def test_count_changed(self):
        e = ChangeEvent(
            event_type=ChangeEventType.COUNT_CHANGED,
            timestamp=1.0,
            subject_class="person",
            prev_count=2,
            curr_count=4,
        )
        desc = e.description_zh()
        assert "2" in desc
        assert "4" in desc

    def test_object_appeared(self):
        e = ChangeEvent(
            event_type=ChangeEventType.OBJECT_APPEARED,
            timestamp=1.0,
            subject_class="chair",
        )
        assert "chair" in e.description_zh()


# ── Detection ─────────────────────────────────────────────────────────────────

class TestDetection:
    def test_center_calculated(self):
        d = Detection(class_id="person", confidence=0.9, bbox=(0.0, 0.0, 100.0, 80.0))
        assert d.center == (50.0, 40.0)

    def test_area_calculated(self):
        d = Detection(class_id="chair", confidence=0.8, bbox=(0.0, 0.0, 10.0, 20.0))
        assert d.area == 200.0

    def test_area_zero_for_degenerate_bbox(self):
        d = Detection(class_id="x", confidence=0.5, bbox=(5.0, 5.0, 5.0, 5.0))
        assert d.area == 0.0

    def test_distance_m_default_none(self):
        d = Detection(class_id="x", confidence=0.5, bbox=(0.0, 0.0, 1.0, 1.0))
        assert d.distance_m is None


# ── Observation ───────────────────────────────────────────────────────────────

class TestObservation:
    def test_empty_detections(self):
        obs = Observation(timestamp=1.0)
        assert obs.detections == []
        assert obs.count("person") == 0

    def test_count_by_class(self):
        obs = Observation(
            timestamp=1.0,
            detections=[
                Detection("person", 0.9, (0.0, 0.0, 10.0, 10.0)),
                Detection("person", 0.8, (5.0, 5.0, 20.0, 20.0)),
                Detection("chair", 0.7, (10.0, 10.0, 30.0, 30.0)),
            ],
        )
        assert obs.count("person") == 2
        assert obs.count("chair") == 1
        assert obs.count("bottle") == 0

    def test_by_class_grouping(self):
        obs = Observation(
            timestamp=1.0,
            detections=[
                Detection("person", 0.9, (0.0, 0.0, 10.0, 10.0)),
                Detection("bottle", 0.5, (1.0, 1.0, 5.0, 5.0)),
            ],
        )
        groups = obs.by_class()
        assert "person" in groups
        assert "bottle" in groups
        assert len(groups["person"]) == 1

    def test_from_daemon_json(self):
        data = {
            "timestamp": 12345.6,
            "infer_ms": 10.5,
            "detections": [
                {
                    "class_id": "person",
                    "confidence": 0.95,
                    "bbox": [10, 20, 100, 200],
                    "distance_m": 2.3,
                }
            ],
        }
        obs = Observation.from_daemon_json(data)
        assert obs.timestamp == 12345.6
        assert obs.infer_ms == 10.5
        assert len(obs.detections) == 1
        d = obs.detections[0]
        assert d.class_id == "person"
        assert d.confidence == 0.95
        assert d.distance_m == 2.3

    def test_from_daemon_json_empty_detections(self):
        obs = Observation.from_daemon_json({"timestamp": 1.0, "detections": []})
        assert obs.detections == []

    def test_from_daemon_json_invalid_bbox_uses_zeros(self):
        data = {
            "timestamp": 1.0,
            "detections": [
                {"class_id": "x", "confidence": 0.5, "bbox": [1, 2]},  # only 2 values
            ],
        }
        obs = Observation.from_daemon_json(data)
        assert obs.detections[0].bbox == (0, 0, 0, 0)
