"""Tests for typed message dataclasses in askme.schemas.messages."""

from __future__ import annotations

from askme.schemas.messages import (
    CmsState,
    DetectionFrame,
    EstopState,
    ImuSnapshot,
    JointStateSnapshot,
)
from askme.schemas.observation import Detection


# ── EstopState ──────────────────────────────────────────


def test_estop_roundtrip():
    state = EstopState(active=True, timestamp=100.0)
    d = state.to_dict()
    restored = EstopState.from_dict(d)
    assert restored.active is True
    assert restored.timestamp == 100.0


def test_estop_from_dict_defaults():
    restored = EstopState.from_dict({})
    assert restored.active is False
    assert restored.timestamp == 0.0


def test_estop_from_pulse_dict():
    """Pulse sends _ts as timestamp key."""
    restored = EstopState.from_dict({"active": True, "_ts": 55.5})
    assert restored.active is True
    assert restored.timestamp == 55.5


def test_estop_frozen():
    state = EstopState(active=False)
    try:
        state.active = True  # type: ignore[misc]
        assert False, "Should be frozen"
    except AttributeError:
        pass


# ── DetectionFrame ──────────────────────────────────────


def test_detection_frame_roundtrip():
    frame = DetectionFrame(
        timestamp=1.0,
        frame_id=42,
        detections=[
            Detection(class_id="person", confidence=0.95, bbox=(10, 20, 100, 200), distance_m=2.3),
            Detection(class_id="chair", confidence=0.7, bbox=(50, 60, 150, 250)),
        ],
    )
    d = frame.to_dict()
    assert d["frame_id"] == 42
    assert len(d["detections"]) == 2
    assert d["detections"][0]["class_id"] == "person"
    assert d["detections"][0]["distance_m"] == 2.3
    assert d["detections"][1]["distance_m"] is None

    restored = DetectionFrame.from_dict(d)
    assert restored.frame_id == 42
    assert len(restored.detections) == 2
    assert restored.detections[0].class_id == "person"
    assert restored.detections[0].confidence == 0.95
    assert restored.detections[0].distance_m == 2.3
    assert restored.detections[1].class_id == "chair"
    assert restored.detections[1].distance_m is None


def test_detection_frame_from_dict_defaults():
    restored = DetectionFrame.from_dict({})
    assert restored.timestamp == 0.0
    assert restored.frame_id == 0
    assert restored.detections == []


def test_detection_frame_from_pulse_format():
    """Pulse detection JSON uses 'label' not 'class_id'."""
    d = {
        "timestamp": 10.0,
        "frame_id": 5,
        "detections": [
            {"label": "person", "confidence": 0.9, "bbox": [10, 20, 30, 40]},
        ],
    }
    frame = DetectionFrame.from_dict(d)
    assert frame.detections[0].class_id == "person"
    assert frame.detections[0].bbox == (10, 20, 30, 40)


def test_detection_frame_nested_detection_objects():
    """DetectionFrame should correctly produce Detection objects with all properties."""
    d = {
        "timestamp": 5.0,
        "frame_id": 1,
        "detections": [
            {"class_id": "dog", "confidence": 0.8, "bbox": [0, 0, 100, 100], "distance_m": 1.5},
        ],
    }
    frame = DetectionFrame.from_dict(d)
    det = frame.detections[0]
    assert isinstance(det, Detection)
    assert det.center == (50.0, 50.0)
    assert det.area == 10000.0
    assert det.distance_m == 1.5


# ── JointStateSnapshot ─────────────────────────────────


def test_joint_state_roundtrip():
    snap = JointStateSnapshot(
        name=["j1", "j2"],
        position=[0.5, 1.0],
        velocity=[0.1, 0.2],
        effort=[10.0, 20.0],
        timestamp=99.0,
    )
    d = snap.to_dict()
    restored = JointStateSnapshot.from_dict(d)
    assert restored.name == ["j1", "j2"]
    assert restored.position == [0.5, 1.0]
    assert restored.velocity == [0.1, 0.2]
    assert restored.effort == [10.0, 20.0]
    assert restored.timestamp == 99.0


def test_joint_state_from_dict_defaults():
    restored = JointStateSnapshot.from_dict({})
    assert restored.name == []
    assert restored.position == []
    assert restored.velocity == []
    assert restored.effort == []
    assert restored.timestamp == 0.0


def test_joint_state_from_pulse_dict():
    """Pulse sends _ts."""
    d = {"name": ["a"], "position": [1], "velocity": [2], "effort": [3], "_ts": 77.0}
    snap = JointStateSnapshot.from_dict(d)
    assert snap.timestamp == 77.0
    assert snap.name == ["a"]


# ── ImuSnapshot ────────────────────────────────────────


def test_imu_roundtrip():
    imu = ImuSnapshot(
        angular_velocity=(0.1, 0.2, 0.3),
        orientation=(0.0, 0.0, 0.7071, 0.7071),
        timestamp=50.0,
    )
    d = imu.to_dict()
    assert d["angular_velocity"]["x"] == 0.1
    assert d["orientation"]["w"] == 0.7071

    restored = ImuSnapshot.from_dict(d)
    assert restored.angular_velocity == (0.1, 0.2, 0.3)
    assert restored.orientation == (0.0, 0.0, 0.7071, 0.7071)
    assert restored.timestamp == 50.0


def test_imu_from_dict_defaults():
    restored = ImuSnapshot.from_dict({})
    assert restored.angular_velocity == (0.0, 0.0, 0.0)
    assert restored.orientation == (0.0, 0.0, 0.0, 0.0)
    assert restored.timestamp == 0.0


def test_imu_from_pulse_dict():
    d = {
        "angular_velocity": {"x": 1.0, "y": 2.0, "z": 3.0},
        "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
        "_ts": 88.0,
    }
    imu = ImuSnapshot.from_dict(d)
    assert imu.angular_velocity == (1.0, 2.0, 3.0)
    assert imu.orientation == (0.1, 0.2, 0.3, 0.9)
    assert imu.timestamp == 88.0


# ── CmsState ───────────────────────────────────────────


def test_cms_state_roundtrip():
    cms = CmsState(state="Standing", addr="192.168.1.1", timestamp=30.0)
    d = cms.to_dict()
    assert d["state"] == "Standing"
    assert d["addr"] == "192.168.1.1"

    restored = CmsState.from_dict(d)
    assert restored.state == "Standing"
    assert restored.addr == "192.168.1.1"
    assert restored.timestamp == 30.0


def test_cms_state_from_dict_defaults():
    restored = CmsState.from_dict({})
    assert restored.state == "unknown"
    assert restored.addr == ""
    assert restored.timestamp == 0.0


def test_cms_state_no_addr_in_dict():
    """to_dict omits addr when empty."""
    cms = CmsState(state="connected")
    d = cms.to_dict()
    assert "addr" not in d


def test_cms_state_from_pulse_dict():
    d = {"state": "Grounded", "_ts": 22.0}
    cms = CmsState.from_dict(d)
    assert cms.state == "Grounded"
    assert cms.timestamp == 22.0


# ── PubSubBase typed convenience methods ───────────────


def test_pubsub_typed_methods_return_none_when_empty():
    from askme.robot.mock_pulse import MockPulse

    mock = MockPulse()
    assert mock.get_estop() is None
    assert mock.get_detection_frame() is None
    assert mock.get_joints() is None
    assert mock.get_imu_snapshot() is None
    assert mock.get_cms_state() is None


def test_pubsub_get_estop_typed():
    from askme.robot.mock_pulse import MockPulse

    mock = MockPulse()
    mock.publish("/thunder/estop", {"active": True, "_ts": 1.0})
    estop = mock.get_estop()
    assert estop is not None
    assert isinstance(estop, EstopState)
    assert estop.active is True


def test_pubsub_get_detection_frame_typed():
    from askme.robot.mock_pulse import MockPulse

    mock = MockPulse()
    mock.publish("/thunder/detections", {
        "timestamp": 5.0,
        "frame_id": 10,
        "detections": [{"class_id": "person", "confidence": 0.9, "bbox": [1, 2, 3, 4]}],
    })
    frame = mock.get_detection_frame()
    assert frame is not None
    assert isinstance(frame, DetectionFrame)
    assert frame.frame_id == 10
    assert len(frame.detections) == 1


def test_pubsub_get_joints_typed():
    from askme.robot.mock_pulse import MockPulse

    mock = MockPulse()
    mock.publish("/thunder/joint_states", {
        "name": ["j1"], "position": [0.5], "velocity": [0.0], "effort": [1.0], "_ts": 2.0,
    })
    joints = mock.get_joints()
    assert joints is not None
    assert isinstance(joints, JointStateSnapshot)
    assert joints.name == ["j1"]


def test_pubsub_get_imu_snapshot_typed():
    from askme.robot.mock_pulse import MockPulse

    mock = MockPulse()
    mock.publish("/thunder/imu", {
        "angular_velocity": {"x": 0.1, "y": 0.2, "z": 0.3},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    })
    imu = mock.get_imu_snapshot()
    assert imu is not None
    assert isinstance(imu, ImuSnapshot)
    assert imu.orientation[3] == 1.0


def test_pubsub_get_cms_state_typed():
    from askme.robot.mock_pulse import MockPulse

    mock = MockPulse()
    mock.publish("/thunder/cms_state", {"state": "Walking", "addr": "1.2.3.4"})
    cms = mock.get_cms_state()
    assert cms is not None
    assert isinstance(cms, CmsState)
    assert cms.state == "Walking"


def test_is_estop_active_uses_typed_estop():
    """is_estop_active should work via the typed EstopState path."""
    from askme.robot.mock_pulse import MockPulse

    mock = MockPulse()
    assert mock.is_estop_active() is False
    mock.publish("/thunder/estop", {"active": True})
    assert mock.is_estop_active() is True
    mock.publish("/thunder/estop", {"active": False})
    assert mock.is_estop_active() is False
