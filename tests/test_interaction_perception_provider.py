from __future__ import annotations

import json
import time
from pathlib import Path

from askme.perception.interaction_provider import FileInteractionPerceptionProvider
from askme.runtime.module import ModuleRegistry
from askme.runtime.modules.perception_module import PerceptionModule
from askme.voice.perception_context import InteractionPerceptionSnapshot


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_file_interaction_provider_merges_real_algorithm_outputs(tmp_path: Path) -> None:
    now = time.time()
    provider = FileInteractionPerceptionProvider(
        {
            "enabled": True,
            "max_age_s": 2.0,
            "paths": {
                "pose_gaze": _write(
                    tmp_path / "pose.json",
                    {
                        "observed_at": now,
                        "person_facing_robot": True,
                        "person_angle_deg": 8,
                        "objects": [{"label": "person", "distance_m": 1.4, "angle_deg": 8}],
                    },
                ),
                "gesture": _write(
                    tmp_path / "gesture.json",
                    {"observed_at": now, "gesture": "raise_hand"},
                ),
                "sound_source": _write(
                    tmp_path / "doa.json",
                    {"observed_at": now, "sound_source_angle_deg": 9},
                ),
                "audio_visual_association": _write(
                    tmp_path / "association.json",
                    {
                        "observed_at": now,
                        "matched_track_id": "track-1",
                        "speaker_track_id": "track-1",
                        "sound_source_matches_person": True,
                        "association_confidence": 0.88,
                    },
                ),
                "approach_dwell": _write(
                    tmp_path / "dwell.json",
                    {
                        "observed_at": now,
                        "track_id": "track-1",
                        "approach_state": "approaching",
                        "dwell_s": 7.5,
                        "distance_m": 1.4,
                    },
                ),
                "multi_person_arbitration": _write(
                    tmp_path / "arbitration.json",
                    {
                        "observed_at": now,
                        "active_person_track_id": "track-1",
                        "speaker_track_id": "track-1",
                        "person_count": 1,
                    },
                ),
            },
        }
    )

    payload = provider.snapshot(now=now + 0.5)
    snapshot = InteractionPerceptionSnapshot.from_payload(payload, now=now + 0.5)

    assert payload["reason"] == "fresh"
    assert snapshot.fresh is True
    assert snapshot.person_facing_robot is True
    assert snapshot.gesture == "raise_hand"
    assert snapshot.sound_source_matches_person is True
    assert snapshot.sound_source_angle_deg == 9
    assert snapshot.nearest_person_distance_m == 1.4
    assert payload["metadata"]["active_person_track_id"] == "track-1"
    assert payload["metadata"]["freshness_by_sensor"]["pose_gaze"]["status"] == "fresh"


def test_file_interaction_provider_drops_stale_sensor_facts(tmp_path: Path) -> None:
    now = time.time()
    provider = FileInteractionPerceptionProvider(
        {
            "enabled": True,
            "max_age_s": 1.0,
            "paths": {
                "pose_gaze": _write(
                    tmp_path / "pose.json",
                    {"observed_at": now - 10, "person_facing_robot": True},
                ),
                "sound_source": _write(
                    tmp_path / "doa.json",
                    {"observed_at": now, "sound_source_angle_deg": 21},
                ),
            },
        }
    )

    payload = provider.snapshot(now=now + 0.2)
    snapshot = InteractionPerceptionSnapshot.from_payload(payload, now=now + 0.2)

    assert payload["metadata"]["freshness_by_sensor"]["pose_gaze"]["status"] == "stale"
    assert snapshot.person_facing_robot is None
    assert snapshot.sound_source_angle_deg == 21


def test_perception_module_prefers_configured_interaction_provider(tmp_path: Path) -> None:
    now = time.time()
    pose_path = _write(
        tmp_path / "pose.json",
        {
            "observed_at": now,
            "person_facing_robot": True,
            "objects": [{"label": "person", "distance_m": 1.2}],
        },
    )
    mod = PerceptionModule()
    mod.build(
        {
            "perception": {
                "interaction_provider": {
                    "enabled": True,
                    "max_age_s": 5.0,
                    "paths": {"pose_gaze": pose_path},
                }
            }
        },
        ModuleRegistry(),
    )

    payload = mod.interaction_snapshot()

    assert payload["source"] == "interaction_provider"
    assert payload["person_facing_robot"] is True
    assert mod.health()["interaction_provider_enabled"] is True
