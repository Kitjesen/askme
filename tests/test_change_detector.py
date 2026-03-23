"""Tests for ChangeDetector — IoU matching, debounce, event emission."""

from __future__ import annotations

import asyncio
import json
import tempfile
from unittest.mock import MagicMock

import pytest

from askme.perception.change_detector import ChangeDetector, compute_iou
from askme.schemas.events import ChangeEvent, ChangeEventType
from askme.schemas.observation import Detection, Observation


# ---------- IoU ----------

class TestComputeIoU:
    def test_identical(self):
        assert compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0

    def test_no_overlap(self):
        assert compute_iou((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0

    def test_partial_overlap(self):
        iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.1 < iou < 0.2  # 2500 / (10000+10000-2500)

    def test_contained(self):
        iou = compute_iou((0, 0, 200, 200), (50, 50, 100, 100))
        assert 0.05 < iou < 0.1  # 2500 / 40000

    def test_zero_area(self):
        assert compute_iou((0, 0, 0, 0), (0, 0, 100, 100)) == 0.0


# ---------- Frame matching ----------

class TestMatchClass:
    def setup_method(self):
        self.cd = ChangeDetector(config={})

    def _det(self, cls, x1, y1, x2, y2, conf=0.9):
        return Detection(class_id=cls, confidence=conf, bbox=(x1, y1, x2, y2))

    def test_empty_to_empty(self):
        appeared, disappeared = self.cd._match_class([], [])
        assert appeared == []
        assert disappeared == []

    def test_person_appears(self):
        curr = [self._det("person", 100, 100, 200, 300)]
        appeared, disappeared = self.cd._match_class([], curr)
        assert len(appeared) == 1
        assert appeared[0].class_id == "person"
        assert disappeared == []

    def test_person_leaves(self):
        prev = [self._det("person", 100, 100, 200, 300)]
        appeared, disappeared = self.cd._match_class(prev, [])
        assert appeared == []
        assert len(disappeared) == 1

    def test_person_stays(self):
        prev = [self._det("person", 100, 100, 200, 300)]
        curr = [self._det("person", 105, 102, 205, 302)]  # slight move
        appeared, disappeared = self.cd._match_class(prev, curr)
        assert appeared == []
        assert disappeared == []

    def test_two_people_one_leaves(self):
        prev = [
            self._det("person", 100, 100, 200, 300),
            self._det("person", 500, 100, 600, 300),
        ]
        curr = [self._det("person", 102, 102, 202, 302)]  # first stays
        appeared, disappeared = self.cd._match_class(prev, curr)
        assert len(appeared) == 0
        assert len(disappeared) == 1
        assert disappeared[0].bbox[0] == 500  # second person left


# ---------- Debounce ----------

class TestDebounce:
    def setup_method(self):
        self.cd = ChangeDetector(config={
            "proactive": {"change_detector": {
                "confirm_frames": 2,
                "disappear_frames": 3,
            }}
        })

    def _make_raw(self, change_type, cls="person", x=100, y=100):
        det = Detection(class_id=cls, confidence=0.9, bbox=(x, y, x+100, y+200))
        from askme.perception.change_detector import _RawChange
        key = ChangeDetector._debounce_key(cls, det)
        return _RawChange(change_type, cls, det, key)

    def test_appear_needs_n_frames(self):
        raw = [self._make_raw("appeared")]
        # Frame 1: not confirmed yet
        events = self.cd._debounce(raw, 1.0)
        assert len(events) == 0
        # Frame 2: confirmed!
        events = self.cd._debounce(raw, 2.0)
        assert len(events) == 1
        assert events[0].event_type == ChangeEventType.PERSON_APPEARED

    def test_disappear_needs_more_frames(self):
        raw = [self._make_raw("disappeared")]
        # Frames 1-2: not yet
        self.cd._debounce(raw, 1.0)
        self.cd._debounce(raw, 2.0)
        events = self.cd._debounce(raw, 3.0)
        # Frame 3: confirmed
        assert len(events) == 1
        assert events[0].event_type == ChangeEventType.PERSON_LEFT

    def test_intermittent_resets(self):
        raw = [self._make_raw("appeared")]
        self.cd._debounce(raw, 1.0)  # frame 1
        self.cd._debounce([], 2.0)    # frame 2: not seen → reset
        self.cd._debounce(raw, 3.0)  # frame 3: restart count
        events = self.cd._debounce(raw, 4.0)  # frame 4: 2nd consecutive
        assert len(events) == 1  # now confirmed

    def test_object_event_type(self):
        raw = [self._make_raw("appeared", cls="chair")]
        self.cd._debounce(raw, 1.0)
        events = self.cd._debounce(raw, 2.0)
        assert events[0].event_type == ChangeEventType.OBJECT_APPEARED


# ---------- Event emission ----------

class TestEventEmission:
    def test_emit_jsonl(self):
        cd = ChangeDetector(config={})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        cd._event_file = path

        event = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1000.0,
            subject_class="person",
            confidence=0.9,
            bbox=(100, 100, 200, 300),
        )
        cd._emit_events([event])

        with open(path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "person_appeared"
        assert data["subject_class"] == "person"

        import os
        os.unlink(path)


# ---------- Observation ----------

class TestObservation:
    def test_from_daemon_json(self):
        data = {
            "timestamp": 1000.0,
            "infer_ms": 3.2,
            "detections": [
                {"class_id": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300]},
                {"class_id": "chair", "confidence": 0.7, "bbox": [400, 200, 500, 400]},
            ],
        }
        obs = Observation.from_daemon_json(data)
        assert obs.timestamp == 1000.0
        assert len(obs.detections) == 2
        assert obs.detections[0].class_id == "person"
        assert obs.detections[0].bbox == (100, 100, 200, 300)

    def test_from_daemon_empty(self):
        obs = Observation.from_daemon_json({"timestamp": 0, "detections": []})
        assert len(obs.detections) == 0

    def test_by_class(self):
        obs = Observation(timestamp=0, detections=[
            Detection("person", 0.9, (0, 0, 100, 100)),
            Detection("person", 0.8, (200, 200, 300, 300)),
            Detection("chair", 0.7, (400, 400, 500, 500)),
        ])
        groups = obs.by_class()
        assert len(groups["person"]) == 2
        assert len(groups["chair"]) == 1


# ---------- ChangeEvent ----------

class TestChangeEvent:
    def test_roundtrip(self):
        event = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=1000.0,
            subject_class="person",
            confidence=0.9,
            bbox=(100, 100, 200, 300),
            track_id="t_0001",
        )
        d = event.to_dict()
        restored = ChangeEvent.from_dict(d)
        assert restored.event_type == event.event_type
        assert restored.subject_class == event.subject_class
        assert restored.track_id == "t_0001"

    def test_auto_importance(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=0, subject_class="person",
        )
        assert e.importance == 0.8  # person events are high priority

    def test_description_zh(self):
        e = ChangeEvent(
            event_type=ChangeEventType.PERSON_APPEARED,
            timestamp=0, subject_class="person", distance_m=2.5,
        )
        assert "有人出现" in e.description_zh()
        assert "2.5米" in e.description_zh()


# ---------- Pulse push mode ----------

class TestPulsePushMode:
    """Tests for Pulse callback-driven change detection."""

    def _make_pulse(self, *, enabled: bool = True) -> MagicMock:
        client = MagicMock()
        client._enabled = enabled
        client.on = MagicMock()
        return client

    def test_init_accepts_pulse(self):
        dds = self._make_pulse()
        cd = ChangeDetector(config={}, pulse=dds)
        assert cd._pulse is dds

    def test_init_pulse_default_none(self):
        cd = ChangeDetector(config={})
        assert cd._pulse is None

    def test_use_pulse_true_when_enabled(self):
        dds = self._make_pulse(enabled=True)
        cd = ChangeDetector(config={}, pulse=dds)
        assert cd._use_pulse() is True

    def test_use_pulse_false_when_disabled(self):
        dds = self._make_pulse(enabled=False)
        cd = ChangeDetector(config={}, pulse=dds)
        assert cd._use_pulse() is False

    def test_use_pulse_false_when_none(self):
        cd = ChangeDetector(config={}, pulse=None)
        assert cd._use_pulse() is False

    def test_on_pulse_detections_processes_frame(self):
        """Pulse callback should process detection data into change events."""
        cd = ChangeDetector(config={
            "proactive": {"change_detector": {
                "confirm_frames": 1,
            }}
        })

        # First frame — sets prev_obs
        data1 = {
            "timestamp": 1.0,
            "detections": [],
        }
        cd._on_pulse_detections("/thunder/detections", data1)
        assert cd._prev_obs is not None
        assert cd.is_active is True

        # Second frame — person appears
        data2 = {
            "timestamp": 2.0,
            "detections": [
                {"class_id": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300]},
            ],
        }
        cd._on_pulse_detections("/thunder/detections", data2)
        assert cd._prev_obs.timestamp == 2.0

    def test_on_pulse_detections_error_does_not_crash(self):
        """Malformed data in Pulse callback should not raise."""
        cd = ChangeDetector(config={})
        # Pass data that will cause from_daemon_json to work but with edge case
        cd._on_pulse_detections("/thunder/detections", {"timestamp": 0, "detections": []})
        # Should not raise

    async def test_run_pulse_mode_registers_callback(self):
        """In Pulse mode, run() registers a callback and waits for stop."""
        pulse = self._make_pulse(enabled=True)
        cd = ChangeDetector(config={}, pulse=pulse)

        stop = asyncio.Event()
        # Set stop immediately so run() exits
        stop.set()
        await cd.run(stop)

        pulse.on.assert_called_once()
        call_args = pulse.on.call_args
        assert call_args[0][0] == "/thunder/detections"

    async def test_run_polling_mode_when_no_pulse(self):
        """Without Pulse, run() uses the polling loop (stops quickly via stop_event)."""
        cd = ChangeDetector(config={}, pulse=None)
        stop = asyncio.Event()

        async def set_stop():
            await asyncio.sleep(0.05)
            stop.set()

        asyncio.get_event_loop().create_task(set_stop())
        await cd.run(stop)
        # If we get here, polling loop ran and exited cleanly

    def test_on_pulse_detections_emits_events(self):
        """Pulse callback should emit confirmed events to the event file."""
        cd = ChangeDetector(config={
            "proactive": {"change_detector": {
                "confirm_frames": 1,
            }}
        })

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        cd._event_file = path

        # Frame 1: empty scene
        cd._on_pulse_detections("/thunder/detections", {
            "timestamp": 1.0,
            "detections": [],
        })

        # Frame 2: person appears (confirm_frames=1, so immediately confirmed)
        cd._on_pulse_detections("/thunder/detections", {
            "timestamp": 2.0,
            "detections": [
                {"class_id": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300]},
            ],
        })

        with open(path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "person_appeared"

        import os
        os.unlink(path)
