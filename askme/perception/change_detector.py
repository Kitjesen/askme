"""ChangeDetector — extract discrete events from continuous YOLO detections.

Reads frame_daemon's detection JSON at ~1Hz, compares consecutive frames
via greedy IoU matching, debounces changes, and emits structured events
to /tmp/askme_events.jsonl for ProactiveAgent consumption.

Usage::

    detector = ChangeDetector(config)
    await detector.run(stop_event)  # runs until stop_event is set
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from askme.schemas.events import ChangeEvent, ChangeEventType
from askme.schemas.observation import Detection, Observation

logger = logging.getLogger(__name__)

_DETECTIONS_PATH = "/tmp/askme_frame_detections.json"
_HEARTBEAT_PATH = "/tmp/askme_frame_daemon.heartbeat"
_DEFAULT_EVENT_FILE = "/tmp/askme_events.jsonl"


@dataclass
class _RawChange:
    """Intermediate change before debounce."""

    change_type: str  # "appeared" or "disappeared"
    class_id: str
    detection: Detection | None  # the detection involved (None for disappeared)
    key: str  # debounce key: "class_id:quadrant"


class ChangeDetector:
    """Detects scene changes by comparing consecutive YOLO frames."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("proactive", {}).get("change_detector", {})

        self._enabled: bool = cfg.get("enabled", True)
        self._read_interval: float = float(cfg.get("read_interval", 1.0))
        self._confirm_frames: int = int(cfg.get("confirm_frames", 3))
        self._disappear_frames: int = int(cfg.get("disappear_frames", 5))
        self._iou_threshold: float = float(cfg.get("iou_threshold", 0.3))
        self._event_file: str = cfg.get("event_file", _DEFAULT_EVENT_FILE)
        self._max_staleness: float = 3.0  # daemon heartbeat max age

        # State
        self._prev_obs: Observation | None = None
        self._pending_appear: dict[str, int] = defaultdict(int)  # key → frame count
        self._pending_disappear: dict[str, int] = defaultdict(int)
        self._pending_appear_det: dict[str, Detection] = {}  # key → last detection
        self._pending_disappear_det: dict[str, Detection] = {}
        self._track_counter: int = 0
        self._active: bool = False  # set True once producing events

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, stop_event: asyncio.Event) -> None:
        if not self._enabled:
            logger.info("[ChangeDetector] Disabled in config.")
            return

        logger.info(
            "[ChangeDetector] Started — interval=%.1fs confirm=%d disappear=%d iou=%.2f",
            self._read_interval, self._confirm_frames, self._disappear_frames, self._iou_threshold,
        )

        while not stop_event.is_set():
            try:
                obs = await asyncio.to_thread(self._read_daemon)
                if obs is not None:
                    self._active = True
                    if self._prev_obs is not None:
                        raw_changes = self._compare(self._prev_obs, obs)
                        events = self._debounce(raw_changes, obs.timestamp)
                        if events:
                            await asyncio.to_thread(self._emit_events, events)
                    self._prev_obs = obs
                else:
                    self._active = False
            except Exception as exc:
                logger.debug("[ChangeDetector] Tick error: %s", exc)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._read_interval)
                break
            except asyncio.TimeoutError:
                pass

        logger.info("[ChangeDetector] Stopped.")

    # ------------------------------------------------------------------
    # Daemon reading
    # ------------------------------------------------------------------

    def _read_daemon(self) -> Observation | None:
        """Read latest detections from frame_daemon. Returns None if stale/missing."""
        # Heartbeat check
        try:
            with open(_HEARTBEAT_PATH, "r") as f:
                ts = float(f.read().strip())
            if time.time() - ts > self._max_staleness:
                return None
        except (FileNotFoundError, ValueError):
            return None

        # Read detections
        try:
            with open(_DETECTIONS_PATH, "r") as f:
                data = json.load(f)
            return Observation.from_daemon_json(data)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    # ------------------------------------------------------------------
    # Frame comparison (greedy IoU matching)
    # ------------------------------------------------------------------

    def _compare(self, prev: Observation, curr: Observation) -> list[_RawChange]:
        """Compare two observations. Returns raw (un-debounced) changes."""
        changes: list[_RawChange] = []

        all_classes = set()
        for d in prev.detections:
            all_classes.add(d.class_id)
        for d in curr.detections:
            all_classes.add(d.class_id)

        for cls in all_classes:
            prev_dets = [d for d in prev.detections if d.class_id == cls]
            curr_dets = [d for d in curr.detections if d.class_id == cls]

            appeared, disappeared = self._match_class(prev_dets, curr_dets)

            for det in appeared:
                key = self._debounce_key(cls, det)
                changes.append(_RawChange("appeared", cls, det, key))

            for det in disappeared:
                key = self._debounce_key(cls, det)
                changes.append(_RawChange("disappeared", cls, det, key))

        return changes

    def _match_class(
        self, prev_dets: list[Detection], curr_dets: list[Detection],
    ) -> tuple[list[Detection], list[Detection]]:
        """Greedy IoU matching for detections of the same class.

        Returns (appeared_in_curr, disappeared_from_prev).
        """
        if not prev_dets and not curr_dets:
            return [], []
        if not prev_dets:
            return curr_dets, []
        if not curr_dets:
            return [], prev_dets

        # Compute all IoU pairs
        pairs: list[tuple[float, int, int]] = []
        for i, p in enumerate(prev_dets):
            for j, c in enumerate(curr_dets):
                iou = compute_iou(p.bbox, c.bbox)
                if iou >= self._iou_threshold:
                    pairs.append((iou, i, j))

        # Greedy: highest IoU first
        pairs.sort(reverse=True)
        matched_prev: set[int] = set()
        matched_curr: set[int] = set()
        for _, i, j in pairs:
            if i not in matched_prev and j not in matched_curr:
                matched_prev.add(i)
                matched_curr.add(j)

        appeared = [curr_dets[j] for j in range(len(curr_dets)) if j not in matched_curr]
        disappeared = [prev_dets[i] for i in range(len(prev_dets)) if i not in matched_prev]
        return appeared, disappeared

    # ------------------------------------------------------------------
    # Debounce
    # ------------------------------------------------------------------

    def _debounce(self, raw_changes: list[_RawChange], timestamp: float) -> list[ChangeEvent]:
        """Apply N-frame confirmation. Returns only confirmed events."""
        events: list[ChangeEvent] = []

        # Track which keys are still active this frame
        active_appear_keys: set[str] = set()
        active_disappear_keys: set[str] = set()

        for change in raw_changes:
            if change.change_type == "appeared":
                active_appear_keys.add(change.key)
                self._pending_appear[change.key] += 1
                if change.detection:
                    self._pending_appear_det[change.key] = change.detection

                if self._pending_appear[change.key] >= self._confirm_frames:
                    det = self._pending_appear_det.get(change.key)
                    self._track_counter += 1
                    is_person = change.class_id == "person"
                    events.append(ChangeEvent(
                        event_type=ChangeEventType.PERSON_APPEARED if is_person
                        else ChangeEventType.OBJECT_APPEARED,
                        timestamp=timestamp,
                        subject_class=change.class_id,
                        confidence=det.confidence if det else 0.0,
                        bbox=det.bbox if det else None,
                        distance_m=det.distance_m if det else None,
                        track_id=f"t_{self._track_counter:04d}",
                    ))
                    # Reset after emitting
                    del self._pending_appear[change.key]
                    self._pending_appear_det.pop(change.key, None)

            elif change.change_type == "disappeared":
                active_disappear_keys.add(change.key)
                self._pending_disappear[change.key] += 1
                if change.detection:
                    self._pending_disappear_det[change.key] = change.detection

                if self._pending_disappear[change.key] >= self._disappear_frames:
                    det = self._pending_disappear_det.get(change.key)
                    is_person = change.class_id == "person"
                    events.append(ChangeEvent(
                        event_type=ChangeEventType.PERSON_LEFT if is_person
                        else ChangeEventType.OBJECT_DISAPPEARED,
                        timestamp=timestamp,
                        subject_class=change.class_id,
                        confidence=det.confidence if det else 0.0,
                        bbox=det.bbox if det else None,
                    ))
                    del self._pending_disappear[change.key]
                    self._pending_disappear_det.pop(change.key, None)

        # Reset counters for keys NOT seen this frame (intermittent = not real)
        stale_appear = [k for k in self._pending_appear if k not in active_appear_keys]
        for k in stale_appear:
            del self._pending_appear[k]
            self._pending_appear_det.pop(k, None)

        stale_disappear = [k for k in self._pending_disappear if k not in active_disappear_keys]
        for k in stale_disappear:
            del self._pending_disappear[k]
            self._pending_disappear_det.pop(k, None)

        return events

    @staticmethod
    def _debounce_key(class_id: str, det: Detection) -> str:
        """Generate a coarse spatial key for debounce grouping.

        Uses 2x2 quadrant of frame (assuming 1280x720) to group nearby detections.
        """
        cx, cy = det.center
        qx = "L" if cx < 640 else "R"
        qy = "T" if cy < 360 else "B"
        return f"{class_id}:{qx}{qy}"

    # ------------------------------------------------------------------
    # Event output
    # ------------------------------------------------------------------

    def _emit_events(self, events: list[ChangeEvent]) -> None:
        """Append events to JSONL file."""
        try:
            with open(self._event_file, "a", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
                f.flush()
            for event in events:
                logger.info("[ChangeDetector] Event: %s", event.description_zh())
        except Exception as exc:
            logger.warning("[ChangeDetector] Failed to write events: %s", exc)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def compute_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union between two (x1, y1, x2, y2) bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0
